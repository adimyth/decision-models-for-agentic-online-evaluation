"""Direct judge pass for additional judges, mirroring the two online evaluators' rendered state exactly.

Judges: pplx (Perplexity Decisions, same question JSON as Jev) and luna6 (gpt-6-luna with the
llm-online prompt and schema). Writes results/direct_<judge>_<tag>.jsonl in the same row shape as
direct_<tag>.jsonl so analyze.py can treat them alike.

Usage: uv run scripts/judge_direct.py --judge pplx --tag main --repeats 5 --repeat-n 20
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import tiktoken  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langsmith import Client, tracing_context  # noqa: E402
from jev_online_eval import ledger, openai_decisions_client, pplx_client  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")


def load_configs():
    jev = json.loads((ROOT / "results" / "jev-online_prompt.json").read_text())
    llm = json.loads((ROOT / "results" / "llm-online_prompt.json").read_text())
    return (jev["model_config"]["questions"],
            jev["manifest"]["kwargs"]["messages"][0]["kwargs"]["prompt"]["kwargs"]["template"],
            llm["manifest"]["kwargs"]["messages"][0]["kwargs"]["prompt"]["kwargs"]["template"],
            llm["manifest"]["kwargs"]["schema"])


def render(template: str, mapping: dict) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", json.dumps(v, ensure_ascii=False, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["pplx", "luna6", "oai"], required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--repeat-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    jev_q, jev_t, llm_t, llm_schema = load_configs()
    # Perplexity keys: rename jev_* question names to pplx_* so feedback keys stay distinct
    pplx_q = {k.replace("jev_", "pplx_"): v for k, v in jev_q.items()}
    oai_q = {k.replace("jev_", "oai_"): v for k, v in jev_q.items()}
    rows = [json.loads(l) for l in (ROOT / "results" / f"runs_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")][: args.limit]
    client = Client()
    runs = {}
    ids = [r["run_id"] for r in rows]
    for i in range(0, len(ids), 50):
        for run in client.list_runs(id=ids[i:i + 50]):
            runs[str(run.id)] = run
    states = []
    for r in rows:
        run = runs.get(r["run_id"])
        if run is None:
            continue
        inp, out = run.inputs or {}, run.outputs or {}
        if args.judge in ("pplx", "oai"):
            st = render(jev_t, {"input": inp, "output": out})
        else:
            st = render(llm_t, {"var1": inp, "var2": out.get("messages", [])})
        states.append((r, st, len(ENC.encode(st))))
    avg_tok = sum(s[2] for s in states) / max(1, len(states))
    n_calls = len(states) + args.repeats * min(args.repeat_n, len(states))
    if args.judge == "pplx":
        ledger.check("perplexity", n_calls * ledger.cost_pplx(int(avg_tok * 1.2)), "direct Perplexity Decisions calls")
    elif args.judge == "oai":
        ledger.check("openai", n_calls * ledger.cost_openai("gpt-6-luna-decisions", int(avg_tok * 1.3), 0), "direct OpenAI Decisions calls")
    else:
        ledger.check("openai", n_calls * ledger.cost_openai("gpt-6-luna", int(avg_tok) + 100, 100), "direct gpt-6-luna judge calls")
    print(f"[plan] judge {args.judge}: avg state {avg_tok:.0f} tokens; {n_calls} calls; wall clock about {n_calls*3/60:.0f} min")

    if args.judge == "luna6":
        llm = ChatOpenAI(model="gpt-6-luna", max_retries=3, timeout=90, use_responses_api=True).with_structured_output(
            llm_schema, method="json_schema", include_raw=True)

    def ask(st):
        if args.judge == "oai":
            j = openai_decisions_client.ask(st, oai_q)
            tok = j["usage"].get("input_tokens") or 0
            cost = ledger.cost_openai("gpt-6-luna-decisions", tok, 0) if j["answers"] else 0.0
            ledger.add("openai", cost)
            return {**j, "cost_usd": cost}
        if args.judge == "pplx":
            j = pplx_client.ask(st, pplx_q)
            tok = j["usage"].get("input_tokens") or 0
            cost = ledger.cost_pplx(tok) if j["answers"] else 0.0
            ledger.add("perplexity", cost)
            return {**j, "cost_usd": cost}
        t0 = time.perf_counter()
        try:
            res = llm.invoke(st)
        except Exception as e:  # noqa: BLE001
            return {"answers": None, "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_s": time.perf_counter() - t0, "error": repr(e)[:300], "cost_usd": 0.0}
        lat = time.perf_counter() - t0
        um = res["raw"].usage_metadata or {}
        parsed = res["parsed"]
        if parsed:  # rename llm_* -> luna6_* so keys stay distinct
            parsed = {k.replace("llm_", "luna6_"): v for k, v in parsed.items()}
        cost = ledger.cost_openai("gpt-6-luna", um.get("input_tokens", 0), um.get("output_tokens", 0))
        ledger.add("openai", cost)
        return {"answers": parsed, "usage": {"input_tokens": um.get("input_tokens", 0), "output_tokens": um.get("output_tokens", 0)},
                "latency_s": lat, "error": None if parsed else str(res.get("parsing_error"))[:300], "cost_usd": cost}

    out_path = ROOT / "results" / f"direct_{args.judge}_{args.tag}.jsonl"
    t0 = time.time()
    with out_path.open("w") as f, tracing_context(enabled=False):
        def one(r, st, tok, rep):
            a = ask(st)
            rec = {"qid": r["qid"], "run_id": r["run_id"], "rep": rep, "state_tokens": tok, args.judge: a}
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
            return rec
        for i, (r, st, tok) in enumerate(states):
            rec = one(r, st, tok, 0)
            if i % 25 == 0 or i == len(states) - 1:
                a = rec[args.judge]
                print(f"[{i+1}/{len(states)}] {a.get('latency_s') or 0:.2f}s in={a['usage'].get('input_tokens')} ok={bool(a.get('answers'))} {a.get('error') or ''}", flush=True)
            if ledger.over_cap("perplexity") or ledger.over_cap("openai"):
                print("[ledger] stop line reached"); return
        for rep in range(1, args.repeats + 1):
            for r, st, tok in states[: args.repeat_n]:
                one(r, st, tok, rep)
            print(f"[repeat {rep}/{args.repeats}] done", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min; ledger {json.dumps(ledger._load())}")


if __name__ == "__main__":
    main()
