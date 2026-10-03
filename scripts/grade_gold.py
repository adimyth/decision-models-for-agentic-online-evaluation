"""Grade each agent answer against the SimpleQA gold answer with gpt-5.6-luna (SimpleQA's own scheme).

Usage: uv run scripts/grade_gold.py --tag main
Writes results/gold_<tag>.jsonl with grade in {CORRECT, INCORRECT, NOT_ATTEMPTED}.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from langchain_openai import ChatOpenAI  # noqa: E402
from langsmith import tracing_context  # noqa: E402
from jev_online_eval import ledger  # noqa: E402

GRADER_MODEL = "gpt-5.6-luna"
# Condensed from OpenAI simple-evals SimpleQA grader template.
TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

CORRECT: the predicted answer fully contains the important information in the gold target and does not contain any information that contradicts it. Extra hedging or extra detail is fine if the gold target is clearly stated.
INCORRECT: the predicted answer contains a factual statement that contradicts the gold target, or confidently states a different answer.
NOT_ATTEMPTED: the important information in the gold target is not included in the answer, and nothing in the answer contradicts the gold target (e.g. "I could not find reliable evidence").

Question: {question}
Gold target: {gold}
Predicted answer: {answer}

Grade the predicted answer as one of: A (CORRECT), B (INCORRECT), C (NOT_ATTEMPTED). Reply with just the letter."""
MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "results" / f"runs_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    projected = len(rows) * ledger.cost_openai(GRADER_MODEL, 500, 5)
    print(f"[plan] grade {len(rows)} answers; wall clock about {len(rows)*2/8/60:.1f} min")
    ledger.check("openai", projected, "gold grading")
    llm = ChatOpenAI(model=GRADER_MODEL, max_retries=3, timeout=60)

    def grade(r):
        prompt = TEMPLATE.format(question=r["question"], gold=r["gold"], answer=r["answer"][:3000])
        with tracing_context(enabled=False):  # keep grader calls out of the agent project
            resp = llm.invoke(prompt)
        um = resp.usage_metadata or {}
        cost = ledger.cost_openai(GRADER_MODEL, um.get("input_tokens", 0), um.get("output_tokens", 0))
        letter = resp.content.strip()[:1].upper()
        return {"qid": r["qid"], "run_id": r["run_id"], "grade": MAP.get(letter, "UNPARSED"), "raw": resp.content[:50], "cost_usd": cost}

    out = ROOT / "results" / f"gold_{args.tag}.jsonl"
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex, out.open("w") as f:
        for g in ex.map(grade, rows):
            ledger.add("openai", g["cost_usd"])
            f.write(json.dumps(g) + "\n")
    grades = [json.loads(l)["grade"] for l in out.read_text().splitlines()]
    from collections import Counter
    print(f"[done] {Counter(grades)} in {time.time()-t0:.0f}s; openai spent ${ledger.spent('openai'):.4f}")


if __name__ == "__main__":
    main()
