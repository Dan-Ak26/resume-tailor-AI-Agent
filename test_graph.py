"""Smoke test for the LangGraph workflow (makes real Claude calls, costs a few cents).

Runs the graph to the human_review pause, checks the state, then resumes with
scripted decisions (accept #1, edit #2, reject the rest) and checks the output.

    python test_graph.py
"""
import uuid
from pathlib import Path

from langgraph.types import Command

from src.graph import ROOT, build_graph

SAMPLE_JD = """ AI Engineer
We're looking for an engineer to build and evaluate LLM-powered applications.
Requirements:
- Strong Python; experience with LLMs, RAG, and agentic workflows
- Experience optimizing inference latency and memory usage
- Building REST APIs (FastAPI or Django) and deploying with Docker
- Designing evaluation and benchmarking pipelines for models
"""


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"PASS: {message}")


def main():
    jd_path = ROOT / "data" / "jobs" / "jd.txt"
    jd = jd_path.read_text(encoding="utf-8").strip() if jd_path.exists() else ""
    if not jd:
        print("jd.txt missing or empty - using built-in sample job description")
        jd = SAMPLE_JD

    graph = build_graph()
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    # ---- Phase 1: run until the human review pause ----
    graph.invoke({"jd": jd}, config)
    state = graph.get_state(config)
    values = state.values

    check(state.next == ("human_review",), "graph paused at human_review")
    check(len(values.get("requirements", [])) > 0, f"parse_jd found {len(values['requirements'])} requirements")
    check(len(values.get("evidence", {})) > 0, f"retrieve found {len(values['evidence'])} evidence chunks")
    check(len(values.get("bullets", [])) > 0, f"draft_bullets wrote {len(values['bullets'])} bullets")
    check(
        all(i in values["evidence"] for b in values["bullets"] for i in b["evidence_ids"]),
        "every bullet cites real evidence",
    )

    print("\nDrafted bullets:")
    for b in values["bullets"]:
        print(f"  - {b['text']}  [{', '.join(b['evidence_ids'])}]")

    # ---- Phase 2: resume with scripted human decisions ----
    bullets = values["bullets"]
    decisions = []
    for i, b in enumerate(bullets):
        if i == 0:
            decisions.append({"action": "accept"})
        elif i == 1:
            decisions.append({"action": "edit", "text": "EDITED: " + b["text"]})
        else:
            decisions.append({"action": "reject"})

    result = graph.invoke(Command(resume=decisions), config)
    print()

    expected = min(len(bullets), 2)
    check(len(result["approved"]) == expected, f"{expected} bullet(s) approved after review")
    if len(bullets) > 1:
        check(result["approved"][1]["text"].startswith("EDITED:"), "edit was applied")
    check(Path(result["output_path"]).exists(), f"output written to {result['output_path']}")
    check(graph.get_state(config).next == (), "graph finished")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()