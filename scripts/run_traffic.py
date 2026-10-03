"""Send SimpleQA questions through the agent, traced to LangSmith.

Usage: uv run scripts/run_traffic.py --n 10 --offset 0 --concurrency 4 --tag smoke
Prints projected cost and wall-clock before spending.
"""
import argparse
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from jev_online_eval import ledger  # noqa: E402
from jev_online_eval.agent import AGENT_MODEL, build_agent  # noqa: E402

# Per-question estimates; replaced by measured values from smoke run if present.
EST_COST_PER_Q = 0.006
EST_SECONDS_PER_Q = 40


def load_questions(n: int, offset: int) -> list[dict]:
    df = pd.read_csv(ROOT / "data" / "simple_qa_test_set.csv")
    idx = list(range(len(df)))
    random.Random(20261003).shuffle(idx)
    rows = []
    for i in idx[offset: offset + n]:
        r = df.iloc[i]
        rows.append({"qid": int(i), "question": r["problem"], "gold": r["answer"], "metadata": r["metadata"]})
    return rows


def measured_estimates() -> tuple[float, float]:
    smoke = ROOT / "results" / "runs_smoke.jsonl"
    if not smoke.exists():
        return EST_COST_PER_Q, EST_SECONDS_PER_Q
    rows = [json.loads(l) for l in smoke.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        return EST_COST_PER_Q, EST_SECONDS_PER_Q
    return (sum(r["cost_usd"] for r in rows) / len(rows), sum(r["latency_s"] for r in rows) / len(rows))


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    return content or ""


def run_one(agent, q: dict, tag: str) -> dict:
    run_id = str(uuid.uuid4())
    t0 = time.time()
    rec = {"qid": q["qid"], "run_id": run_id, "question": q["question"], "gold": q["gold"], "tag": tag}
    try:
        out = agent.invoke(
            {"messages": [HumanMessage(content=q["question"])]},
            config={"run_id": run_id, "tags": [tag], "metadata": {"qid": q["qid"], "tag": tag},
                    "recursion_limit": 30},
        )
        msgs = out["messages"]
        in_tok = out_tok = 0
        tool_calls = []
        for m in msgs:
            um = getattr(m, "usage_metadata", None)
            if um:
                in_tok += um.get("input_tokens", 0)
                out_tok += um.get("output_tokens", 0)
            for tc in getattr(m, "tool_calls", []) or []:
                tool_calls.append(tc["name"])
        rec.update(ok=True, answer=_text(msgs[-1].content) if msgs else "", input_tokens=in_tok,
                   output_tokens=out_tok, cost_usd=ledger.cost_openai(AGENT_MODEL, in_tok, out_tok),
                   n_tool_calls=len(tool_calls), tool_calls=tool_calls, n_messages=len(msgs))
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e)[:500], cost_usd=0.0)
    rec["latency_s"] = round(time.time() - t0, 2)
    rec["ended_at"] = time.time()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--yes", action="store_true", help="skip the 3 s pause after printing projections")
    args = ap.parse_args()

    qs = load_questions(args.n, args.offset)
    cost_q, sec_q = measured_estimates()
    projected_cost = cost_q * len(qs)
    projected_wall = sec_q * len(qs) / args.concurrency
    print(f"[plan] {len(qs)} questions, concurrency {args.concurrency}, model {AGENT_MODEL}")
    print(f"[plan] projected wall clock {projected_wall/60:.1f} min (est {sec_q:.0f} s/question)")
    if projected_wall > 25 * 60:
        print("[plan] ABORT: projected wall clock over 25 minutes; split the batch")
        sys.exit(2)
    ledger.check("openai", projected_cost, f"agent traffic '{args.tag}'")
    if not args.yes:
        time.sleep(3)

    agent = build_agent()
    out_path = ROOT / "results" / f"runs_{args.tag}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex, out_path.open("a") as f:
        futs = {ex.submit(run_one, agent, q, args.tag): q for q in qs}
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            ledger.add("openai", rec["cost_usd"])
            done += 1
            status = "ok " if rec["ok"] else "ERR"
            print(f"[{done}/{len(qs)}] {status} {rec['latency_s']:6.1f}s ${rec['cost_usd']:.4f} "
                  f"tools={rec.get('n_tool_calls','-')} q{rec['qid']}: {rec['question'][:60]}")
            if ledger.over_cap("openai"):
                print("[ledger] cap stop line reached; cancelling remaining work")
                for other in futs:
                    other.cancel()
                break
    print(f"[done] {done} runs in {(time.time()-t0)/60:.1f} min; openai spent so far ${ledger.spent('openai'):.4f}")


if __name__ == "__main__":
    main()
