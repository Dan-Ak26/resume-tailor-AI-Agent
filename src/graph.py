"""LangGraph workflow:
parse_jd -> retrieve -> draft_bullets -> human_review (pause) -> finalize
"""
import sqlite3
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.index import ROOT, get_retriever, retriever_from_text

load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=4096)


# ---------- Structured outputs from Claude ----------
class Requirements(BaseModel):
    requirements: list[str] = Field(
        description="The 5-8 most important skills or qualifications in the job description, each a short phrase"
    )


class Bullet(BaseModel):
    text: str = Field(description="Tailored resume bullet, under ~30 words")
    requirement: str = Field(description="The job requirement this bullet addresses")
    rationale: str = Field(
        description="One short sentence explaining why this bullet fits the requirement, "
                    "naming the specific experience from the evidence it draws on"
    )
    evidence_ids: list[str] = Field(description="Evidence IDs (e.g. E1, E3) supporting every fact in the bullet")


class Bullets(BaseModel):
    bullets: list[Bullet]


# ---------- Graph state ----------
class State(TypedDict, total=False):
    jd: str
    resume_text: str  # text of an uploaded resume PDF ("" = use data/experience)
    resume_name: str
    requirements: list[str]
    evidence: dict[str, dict]  # "E1" -> {"source": file name, "text": chunk}
    bullets: list[dict]
    approved: list[dict]
    output_path: str


# ---------- Nodes ----------
def parse_jd(state: State) -> State:
    result = llm.with_structured_output(Requirements, method="json_schema").invoke(
        f"Extract the key requirements from this job description:\n\n{state['jd']}"
    )
    return {"requirements": result.requirements}


def retrieve(state: State) -> State:
    """LlamaIndex: find the experience chunks that match each requirement.

    Uses the uploaded resume if there is one, otherwise the indexed data/experience files.
    """
    if state.get("resume_text"):
        retriever = retriever_from_text(state["resume_text"], state.get("resume_name") or "resume.pdf")
    else:
        retriever = get_retriever(top_k=3)
    evidence, seen = {}, {}
    for req in state["requirements"]:
        for hit in retriever.retrieve(req):
            if hit.node.node_id in seen:
                continue
            eid = f"E{len(evidence) + 1}"
            seen[hit.node.node_id] = eid
            evidence[eid] = {
                "source": hit.node.metadata.get("file_name", "?"),
                "text": hit.node.get_content(),
            }
    return {"evidence": evidence}


DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write resume bullets. Use ONLY facts stated in the evidence. "
     "Never invent numbers, tools, titles, or results. Every bullet must cite the "
     "evidence IDs it relies on. Skip any requirement the evidence does not support. "
     "Each bullet describes ONE project or accomplishment; never stitch unrelated work together. "
     "Never reuse the same accomplishment in two bullets. No first person. "
     "Skip education and skills-list requirements; those belong in other resume sections. "
     "For each bullet, add a one-sentence rationale that tells the reviewer why it matches "
     "the requirement and which experience it comes from. Keep the rationale factual."),
    ("human",
     "Job requirements:\n{requirements}\n\nEvidence:\n{evidence}\n\n"
     "Write two alternative bullets per supported requirement, each using a different "
     "accomplishment where possible."),
])


def draft_bullets(state: State) -> State:
    evidence_text = "\n\n".join(
        f"[{eid}] ({e['source']})\n{e['text']}" for eid, e in state["evidence"].items()
    )
    requirements = "\n".join(f"- {r}" for r in state["requirements"])
    chain = DRAFT_PROMPT | llm.with_structured_output(Bullets, method="json_schema")
    result = chain.invoke({"requirements": requirements, "evidence": evidence_text})

    # Guardrail: drop any bullet without valid citations.
    valid = [
        b.model_dump() for b in result.bullets
        if b.evidence_ids and all(i in state["evidence"] for i in b.evidence_ids)
    ]
    return {"bullets": valid}


def human_review(state: State) -> State:
    """Pause the graph until a human accepts, edits, or rejects each bullet."""
    decisions = interrupt({"bullets": state["bullets"]})
    approved = []
    for bullet, d in zip(state["bullets"], decisions):
        if d["action"] == "accept":
            approved.append(bullet)
        elif d["action"] == "edit":
            approved.append({**bullet, "text": d["text"]})
    return {"approved": approved}


def finalize(state: State, config: RunnableConfig) -> State:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    # One file per run, so different users never overwrite each other's results
    run_id = config["configurable"].get("thread_id", "run")[:8]
    path = out_dir / f"tailored_bullets_{run_id}.md"

    lines = ["# Tailored bullets", ""]
    for b in state["approved"]:
        sources = ", ".join(sorted({state["evidence"][i]["source"] for i in b["evidence_ids"]}))
        lines.append(f"- {b['text']}")
        lines.append(f"  - covers: {b['requirement']} | from: {sources}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_path": str(path)}


# ---------- Build ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("parse_jd", parse_jd)
    g.add_node("retrieve", retrieve)
    g.add_node("draft_bullets", draft_bullets)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "retrieve")
    g.add_edge("retrieve", "draft_bullets")
    g.add_edge("draft_bullets", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)

    conn = sqlite3.connect(str(ROOT / "checkpoints.db"), check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))