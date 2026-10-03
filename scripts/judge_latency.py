"""Direct judge comparison on the recorded traces: latency, cost, repeatability.

Pulls the root runs from LangSmith, builds the same State both online evaluators see, and
sends it directly to Jev and to gpt-6-luna. --repeats N re-asks the first 20 states N times.

Usage: uv run scripts/judge_latency.py --tag main --repeats 5 --repeat-n 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from langsmith import Client, tracing_context  # noqa: E402
from jev_online_eval import jev_client, ledger, llm_judge  # noqa: E402
from jev_online_eval.questions import QUESTIONS  # noqa: E402
from jev_online_eval.state import state_from_run, state_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--repeat-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "results" / f"runs_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")][: args.limit]
    client = Client()
    print(f"[plan] fetching {len(rows)} runs from LangSmith and building states")
    states = []
    for r in rows:
        run = client.read_run(r["run_id"])
        s = state_from_run(run)
        states.append((r, s, state_tokens(s)))
    avg_tok = sum(t for _, _, t in states) / max(1, len(states))
    n_calls = len(states) + args.repeats * min(args.repeat_n, len(states))
    jev_proj = n_calls * ledger.cost_jev(int(avg_tok))
    llm_proj = n_calls * ledger.cost_openai(llm_judge.JUDGE_MODEL, int(avg_tok) + 600, 80)
    print(f"[plan] avg state {avg_tok:.0f} tokens; {n_calls} calls per judge; wall clock about {n_calls*2/60:.0f} min")
    ledger.check("jev", jev_proj, "direct Jev calls")
    ledger.check("openai", llm_proj, "direct gpt-6-luna judge calls")

    out = ROOT / "results" / f"direct_{args.tag}.jsonl"
    with out.open("w") as f, tracing_context(enabled=False):
        def one(r, s, tok, rep):
            j = jev_client.ask(s, QUESTIONS)
            jtok = j["usage"].get("input_tokens") or tok
            jcost = ledger.cost_jev(jtok) if j["answers"] else 0.0
            ledger.add("jev", jcost)
            l = llm_judge.ask(s)
            lcost = ledger.cost_openai(llm_judge.JUDGE_MODEL, l["usage"]["input_tokens"], l["usage"]["output_tokens"])
            ledger.add("openai", lcost)
            rec = {"qid": r["qid"], "run_id": r["run_id"], "rep": rep, "state_tokens": tok,
                   "jev": {**j, "cost_usd": jcost}, "llm": {**l, "cost_usd": lcost}}
            rec["jev"].pop("raw", None)
            f.write(json.dumps(rec) + "\n"); f.flush()
            return rec

        t0 = time.time()
        for i, (r, s, tok) in enumerate(states):
            rec = one(r, s, tok, 0)
            print(f"[{i+1}/{len(states)}] jev {rec['jev']['latency_s'] or 0:.2f}s  llm {rec['llm']['latency_s']:.2f}s  tokens {tok}")
        for rep in range(1, args.repeats + 1):
            for r, s, tok in states[: args.repeat_n]:
                one(r, s, tok, rep)
            print(f"[repeat {rep}/{args.repeats}] done")
    print(f"[done] {(time.time()-t0)/60:.1f} min; jev ${ledger.spent('jev'):.4f} openai ${ledger.spent('openai'):.4f}")


if __name__ == "__main__":
    main()
