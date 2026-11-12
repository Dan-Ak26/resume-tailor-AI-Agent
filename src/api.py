"""FastAPI backend for the resume tailor.

Run from the resume-tailor folder:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000

Endpoints
  POST /api/runs                 start a run (body: {"jd": "..."}) -> {"run_id": ...}
  GET  /api/runs/{id}/stream     SSE: live progress, then the drafted bullets for review
  POST /api/runs/{id}/review     send accept/edit/reject decisions -> approved bullets
  GET  /api/sample-jd            contents of data/jobs/jd.txt (for convenience)
"""
import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from src.graph import ROOT, build_graph

app = FastAPI(title="Resume Tailor")
graph = build_graph()
RUNS: dict[str, dict] = {}  # run_id -> {"jd": str, "started": bool}


# ---------- Request models ----------
class StartRun(BaseModel):
    jd: str


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
@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/sample-jd")
def sample_jd():
    path = ROOT / "data" / "jobs" / "jd.txt"
    return {"jd": path.read_text(encoding="utf-8") if path.exists() else ""}


@app.post("/api/runs")
def start_run(body: StartRun):
    if not body.jd.strip():
        raise HTTPException(400, "Job description is empty.")
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {"jd": body.jd, "started": False}
    return {"run_id": run_id}


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
                for chunk in graph.stream({"jd": run["jd"]}, config, stream_mode="updates"):
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
    return {
        "approved": result["approved"],
        "markdown": output_path.read_text(encoding="utf-8"),
        "output_path": str(output_path),
    }