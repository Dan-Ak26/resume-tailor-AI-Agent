"""FastAPI backend for the resume tailor.

Run from the resume-tailor folder:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000

Endpoints
  POST /api/runs                 start a run (form: jd, optional resume PDF) -> {"run_id": ...}
  GET  /api/runs/{id}/stream     SSE: live progress, then the drafted bullets for review
  POST /api/runs/{id}/review     send accept/edit/reject decisions -> approved bullets
  GET  /api/sample-jd            contents of data/jobs/jd.txt (for convenience)
  GET  /healthz                  health check (no password needed)

Public by default. Abuse protection for a public deployment:
  - RUNS_PER_HOUR runs per visitor per hour (default 5)
  - RUNS_PER_DAY runs across all visitors per day (default 100)
  - size caps on the job description and resume
Optional: set APP_PASSWORD to put the whole app behind a password.
"""
import base64
import io
import json
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel
from pypdf import PdfReader

from src.graph import ROOT, build_graph

app = FastAPI(title="Resume Tailor")
graph = build_graph()
RUNS: dict[str, dict] = {}  # run_id -> {"jd", "resume_text", "resume_name", "started"}
MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_JD_CHARS = 15_000
MAX_RESUME_CHARS = 20_000

# ---------- Rate limiting (in memory; resets when the server restarts) ----------
RUNS_PER_HOUR = int(os.getenv("RUNS_PER_HOUR", "5"))
RUNS_PER_DAY = int(os.getenv("RUNS_PER_DAY", "100"))
_runs_by_ip: dict[str, deque] = defaultdict(deque)
_all_runs: deque = deque()
_rate_lock = threading.Lock()


def client_ip(request: Request) -> str:
    # Render sits behind a proxy, so the visitor's IP is the first X-Forwarded-For entry
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.time()
    with _rate_lock:
        mine = _runs_by_ip[ip]
        while mine and now - mine[0] > 3600:
            mine.popleft()
        while _all_runs and now - _all_runs[0] > 86400:
            _all_runs.popleft()
        if len(_all_runs) >= RUNS_PER_DAY:
            raise HTTPException(429, "The demo has reached its daily limit. Please try again tomorrow.")
        if len(mine) >= RUNS_PER_HOUR:
            raise HTTPException(429, f"You've reached the limit of {RUNS_PER_HOUR} runs per hour. Please try again later.")
        mine.append(now)
        _all_runs.append(now)
APP_PASSWORD = os.getenv("APP_PASSWORD")  # read after src.graph has loaded .env


@app.middleware("http")
async def require_password(request: Request, call_next):
    """HTTP Basic auth gate so a public deployment can't spend your API credits."""
    if APP_PASSWORD and request.url.path != "/healthz":
        authorized = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                _, _, password = base64.b64decode(header[6:]).decode().partition(":")
                authorized = secrets.compare_digest(password, APP_PASSWORD)
            except Exception:
                authorized = False
        if not authorized:
            return Response(
                "Password required.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Resume Tailor"'},
            )
    return await call_next(request)


# ---------- Request models ----------
class Decision(BaseModel):
    action: Literal["accept", "edit", "reject"]
    text: str | None = None


class ReviewIn(BaseModel):
    decisions: list[Decision]


# ---------- Helpers ----------
def config_for(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def describe(node: str, update: dict) -> dict:
    """Turn a graph node's state update into a progress message for the UI."""
    if node == "parse_jd":
        reqs = update.get("requirements", [])
        return {"node": node, "message": f"Found {len(reqs)} requirements", "requirements": reqs}
    if node == "retrieve":
        return {"node": node, "message": f"Retrieved {len(update.get('evidence', {}))} evidence chunks"}
    if node == "draft_bullets":
        return {"node": node, "message": f"Drafted {len(update.get('bullets', []))} bullets"}
    return {"node": node, "message": ""}


# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/sample-jd")
def sample_jd():
    path = ROOT / "data" / "jobs" / "jd.txt"
    return {"jd": path.read_text(encoding="utf-8") if path.exists() else ""}


@app.post("/api/runs")
def start_run(request: Request, jd: str = Form(...), resume: UploadFile | None = File(None)):
    """Start a run. If a resume PDF is uploaded, bullets are drafted from it
    instead of the bundled data/experience files."""
    if not jd.strip():
        raise HTTPException(400, "Job description is empty.")
    if len(jd) > MAX_JD_CHARS:
        raise HTTPException(400, f"Job description is too long (max {MAX_JD_CHARS:,} characters).")

    resume_text, resume_name = "", ""
    if resume is not None and resume.filename:
        if not resume.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Please upload your resume as a PDF.")
        data = resume.file.read(MAX_PDF_BYTES + 1)
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(400, "That PDF is larger than 5 MB.")
        try:
            resume_text = pdf_to_text(data)
        except Exception:
            raise HTTPException(400, "Couldn't read that PDF. Try exporting it again.")
        if len(resume_text) < 50:
            raise HTTPException(400, "Couldn't find text in that PDF. If it's a scanned image, export a text-based PDF instead.")
        if len(resume_text) > MAX_RESUME_CHARS:
            raise HTTPException(400, "That resume is too long. Please upload a resume of a few pages at most.")
        resume_name = resume.filename

    check_rate_limit(client_ip(request))  # only count requests that passed validation
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {"jd": jd, "resume_text": resume_text, "resume_name": resume_name, "started": False}
    return {"run_id": run_id, "resume_name": resume_name or None}


@app.get("/api/runs/{run_id}/stream")
def stream_run(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    config = config_for(run_id)

    def events():
        yield ": connected\n\n"  # SSE comment so the browser sees the stream open right away
        try:
            # Only execute the graph once; a reconnect just re-sends the review payload.
            if not run["started"]:
                run["started"] = True
                inputs = {"jd": run["jd"], "resume_text": run["resume_text"], "resume_name": run["resume_name"]}
                for chunk in graph.stream(inputs, config, stream_mode="updates"):
                    for node, update in chunk.items():
                        if node != "__interrupt__":
                            yield sse("progress", describe(node, update or {}))

            state = graph.get_state(config)
            if state.next == ("human_review",):
                yield sse("review", {
                    "bullets": state.values["bullets"],
                    "evidence": state.values["evidence"],
                })
            else:
                yield sse("failed", {"message": "Run is not waiting for review."})
        except Exception as exc:  # surface errors (bad API key, etc.) in the UI
            yield sse("failed", {"message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/runs/{run_id}/review")
def review_run(run_id: str, body: ReviewIn):
    config = config_for(run_id)
    state = graph.get_state(config)
    if state.next != ("human_review",):
        raise HTTPException(409, "Run is not waiting for review.")
    if len(body.decisions) != len(state.values["bullets"]):
        raise HTTPException(400, "Need exactly one decision per bullet.")
    if any(d.action == "edit" and not (d.text or "").strip() for d in body.decisions):
        raise HTTPException(400, "Edited bullets can't be empty.")

    result = graph.invoke(Command(resume=[d.model_dump() for d in body.decisions]), config)
    output_path = Path(result["output_path"])
    markdown = output_path.read_text(encoding="utf-8")

    # Privacy: once the review is saved, delete this run's resume text,
    # saved checkpoints, and output file from the server.
    RUNS.pop(run_id, None)
    output_path.unlink(missing_ok=True)
    try:
        graph.checkpointer.delete_thread(run_id)
    except Exception:
        pass  # older langgraph versions may not support delete_thread

    return {"approved": result["approved"], "markdown": markdown}