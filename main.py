"""Run the resume tailor from the command line:
    python main.py
"""
import uuid

from langgraph.types import Command

from src.graph import ROOT, build_graph


def review(bullets: list[dict]) -> list[dict]:
    """Human-in-the-loop: accept, edit, or reject each drafted bullet."""
    decisions = []
    for i, b in enumerate(bullets, 1):
        print(f"\n[{i}]/{len(bullets)}] {b['text']}")
        print(f"  covers: {b['requirement']} | evidence: {', '.join(b['evidence_ids'])}")
        choice = ""
        while choice not in ("a", "e", "r"):
            choice = input("  (a)ccept / (e)dit / (r)eject?").strip().lower()
        if choice == "a":
            decisions.append({"action": "accept"})
        elif choice == "r":
            decisions.append({"action": "reject"})
        else:
            decisions.append({"action": "edit", "text": input("    New text:").strip()})
    return decisions

def main():
    jd = (ROOT / "data" / "jobs" / "jd.txt").read_text(encoding="utf-8")
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Analyzing job description and drafting bullets...")
    graph.invoke({"jd": jd}, config) # runs until the human_review pause

    state = graph.get_state(config)
    print("\nRequirements found:")
    for r in state.values["requirements"]:
        print(f" -{r}")

    decisions = review(state.values["bullets"])
    result = graph.invoke(Command(resume=decisions), config) # resume after review

    print(f"\nSaved {len(result['approved'])} bullets to {result['output_path']}")

if __name__ == "__main__":
    main()