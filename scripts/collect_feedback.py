"""Collect online-evaluator feedback for recorded runs and measure lag.

Usage: uv run scripts/collect_feedback.py --tag main [--wait 600]
Writes results/feedback_<tag>.jsonl: one row per feedback item with run end time, feedback
creation time, lag seconds, key, score, value, comment and source metadata (Jev raw answer).
"""
import argparse
import json
import sys
import time
import warnings
from datetime import timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from langsmith import Client  # noqa: E402

KEYS = [f"{p}_{q}" for p in ("jev", "llm") for q in ("answered", "grounded", "correct", "confidence", "outcome")]


def collect(client: Client, rows: list[dict]) -> list[dict]:
    out = []
    run_ids = [r["run_id"] for r in rows]
    runs = {}
    for i in range(0, len(run_ids), 50):
        for run in client.list_runs(id=run_ids[i:i + 50], select=["id", "end_time", "start_time", "total_tokens"]):
            runs[str(run.id)] = run
    for i in range(0, len(run_ids), 50):
        for fb in client.list_feedback(run_ids=run_ids[i:i + 50]):
            run = runs.get(str(fb.run_id))
            end = run.end_time if run else None
            if end is not None and end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            created = fb.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            out.append({
                "run_id": str(fb.run_id), "key": fb.key, "score": fb.score, "value": fb.value,
                "comment": fb.comment, "feedback_id": str(fb.id),
                "created_at": created.isoformat() if created else None,
                "run_end": end.isoformat() if end else None,
                "lag_s": (created - end).total_seconds() if (end and created) else None,
                "source_metadata": (fb.feedback_source.metadata if fb.feedback_source else None),
                "source_type": (fb.feedback_source.type if fb.feedback_source else None),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--wait", type=int, default=0, help="poll for up to N seconds until all keys present")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "results" / f"runs_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    client = Client()
    t0 = time.time()
    while True:
        fb = collect(client, rows)
        have = {(f["run_id"], f["key"]) for f in fb}
        expected = len(rows) * len(KEYS)
        n = sum(1 for r in rows for k in KEYS if (r["run_id"], k) in have)
        print(f"[{time.time()-t0:5.0f}s] feedback items {len(fb)}; expected keys present {n}/{expected}")
        if n >= expected or time.time() - t0 > args.wait:
            break
        time.sleep(20)
    out = ROOT / "results" / f"feedback_{args.tag}.jsonl"
    out.write_text("".join(json.dumps(f) + "\n" for f in fb))
    by_key = {}
    for f in fb:
        by_key.setdefault(f["key"], []).append(f)
    for k in sorted(by_key):
        lags = [f["lag_s"] for f in by_key[k] if f["lag_s"] is not None]
        lags.sort()
        p50 = lags[len(lags)//2] if lags else None
        print(f"  {k:16s} n={len(by_key[k]):4d}  lag p50={p50}  max={lags[-1] if lags else None}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
