"""Post the direct-call verdicts of the extra judges onto the LangSmith traces as feedback.

This is the self-hosted path for judges LangSmith does not offer natively (Perplexity Decisions),
and for gpt-6-luna run outside the UI. Every create_feedback call passes extend_trace_retention=False.

Usage: uv run scripts/post_extra_feedback.py --tag main --judges pplx,luna6
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from langsmith import Client  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--judges", default="pplx,luna6")
    args = ap.parse_args()
    client = Client()
    total = 0
    for judge in args.judges.split(","):
        p = ROOT / "results" / f"direct_{judge}_{args.tag}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["rep"] == 0 and r[judge].get("answers")]
        for r in rows:
            for key, a in r[judge]["answers"].items():
                if judge == "pplx":
                    if a["type"] == "noul":
                        kw = {"score": a["noul"]}
                    elif a["type"] == "score":
                        kw = {"score": a["score"]}
                    else:
                        kw = {"value": a["choice"]}
                    src = {"perplexity": a}
                else:
                    if key == "comment":
                        continue
                    kw = {"value": a} if isinstance(a, str) else {"score": float(a)}
                    src = {"model": "gpt-6-luna"}
                client.create_feedback(r["run_id"], key=key, source_info=src, extend_trace_retention=False, **kw)
                total += 1
        print(f"{judge}: posted feedback for {len(rows)} runs")
    print(f"total feedback items posted: {total}")


if __name__ == "__main__":
    main()
