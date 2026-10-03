"""Pull the online evaluators' own runs (latency, tokens, cost) from LangSmith.

Each feedback item carries source_metadata.__run.run_id pointing at the evaluator run in the
'evaluators' project. For the LLM judge the ChatOpenAI child run has tokens and cost; for Jev the
rule run has latency and the feedback metadata has the raw answer.

Usage: uv run scripts/evaluator_runs.py --tag main
Writes results/evaluator_runs_<tag>.jsonl
"""
import argparse
import json
import sys
import warnings
from datetime import timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from langsmith import Client  # noqa: E402
from jev_online_eval import ledger  # noqa: E402

JUDGE_MODEL = "gpt-5.6-luna"


def _utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    fb = [json.loads(l) for l in (ROOT / "results" / f"feedback_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    # one evaluator run per (agent run, judge)
    ev = {}
    for f in fb:
        meta = f.get("source_metadata") or {}
        rid = (meta.get("__run") or {}).get("run_id")
        if not rid or f["key"] == "comment":
            continue
        judge = f["key"].split("_")[0]
        ev.setdefault(rid, {"evaluator_run_id": rid, "run_id": f["run_id"], "judge": judge, "rule_id": meta.get("rule_id")})
    client = Client()
    ids = list(ev)
    print(f"{len(ids)} evaluator runs to fetch")
    got = {}
    for i in range(0, len(ids), 50):
        for r in client.list_runs(id=ids[i:i + 50]):
            got[str(r.id)] = r
    # child llm runs: fetch by trace ids of the evaluator runs
    traces = {str(r.trace_id) for r in got.values()}
    llm_by_trace = {}
    starts = [_utc(r.start_time) for r in got.values() if r.start_time]
    if starts:
        from datetime import timedelta
        proj = next(iter(got.values())).session_id
        for r in client.list_runs(project_id=proj, run_type="llm", start_time=min(starts) - timedelta(minutes=1)):
            if str(r.trace_id) in traces:
                llm_by_trace[str(r.trace_id)] = r
    out = []
    for rid, e in ev.items():
        r = got.get(rid)
        if r is None:
            e["missing"] = True; out.append(e); continue
        st, en = _utc(r.start_time), _utc(r.end_time)
        e.update(start=st.isoformat() if st else None, end=en.isoformat() if en else None,
                 latency_s=(en - st).total_seconds() if (st and en) else None, status=r.status, error=r.error)
        if e["judge"] == "llm":
            l = llm_by_trace.get(str(r.trace_id))
            if l is not None:
                ls, le = _utc(l.start_time), _utc(l.end_time)
                e.update(prompt_tokens=l.prompt_tokens, completion_tokens=l.completion_tokens,
                         model_latency_s=(le - ls).total_seconds() if (ls and le) else None,
                         langsmith_cost=float(l.total_cost) if l.total_cost is not None else None,
                         cost_usd=ledger.cost_openai(JUDGE_MODEL, l.prompt_tokens or 0, l.completion_tokens or 0))
        else:
            # Jev: tokens not exposed by LangSmith; estimate from the rendered state size if present in run inputs
            inp = json.dumps(r.inputs or {})
            est_tokens = int(len(inp) / 3.6)
            e.update(prompt_tokens=est_tokens, prompt_tokens_estimated=True, completion_tokens=0,
                     cost_usd=ledger.cost_jev(est_tokens), outputs=(r.outputs if r.outputs else None))
        out.append(e)
    p = ROOT / "results" / f"evaluator_runs_{args.tag}.jsonl"
    p.write_text("".join(json.dumps(e, default=str) + "\n" for e in out))
    import statistics as s
    for judge in ("jev", "llm"):
        rows = [e for e in out if e["judge"] == judge and e.get("latency_s") is not None]
        if rows:
            lat = sorted(e["latency_s"] for e in rows)
            print(f"{judge}: n={len(rows)} latency p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)-1]:.2f}s "
                  f"tokens~{s.mean(e.get('prompt_tokens') or 0 for e in rows):.0f} cost~${s.mean(e.get('cost_usd') or 0 for e in rows):.5f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
