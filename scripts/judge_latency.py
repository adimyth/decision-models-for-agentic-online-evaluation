"""Direct judge comparison replicating the two online evaluators exactly: latency, cost, repeatability.

Pulls each agent root run from LangSmith, renders the same state the LangSmith evaluators render
(the saved hub configs in results/*_prompt.json), and calls Jev and gpt-5.6-luna directly.
--repeats N re-asks the first --repeat-n states N more times.

Usage: uv run scripts/judge_latency.py --tag main --repeats 5 --repeat-n 20
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
from jev_online_eval import jev_client, ledger  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")
JUDGE_MODEL = "gpt-5.6-luna"


def load_configs():
    jev = json.loads((ROOT / "results" / "jev-online_prompt.json").read_text())
    llm = json.loads((ROOT / "results" / "llm-online_prompt.json").read_text())
    jev_questions = jev["model_config"]["questions"]
    jev_template = jev["manifest"]["kwargs"]["messages"][0]["kwargs"]["prompt"]["kwargs"]["template"]
    llm_template = llm["manifest"]["kwargs"]["messages"][0]["kwargs"]["prompt"]["kwargs"]["template"]
    llm_schema = llm["manifest"]["kwargs"]["schema"]
    return jev_questions, jev_template, llm_template, llm_schema


def render(template: str, mapping: dict) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", json.dumps(v, ensure_ascii=False, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--repeat-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    jev_q, jev_t, llm_t, llm_schema = load_configs()
    rows = [json.loads(l) for l in (ROOT / "results" / f"runs_{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")][: args.limit]
    client = Client()
    print(f"[plan] fetching {len(rows)} runs from LangSmith")
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
        jev_state = render(jev_t, {"input": inp, "output": out})
        llm_prompt = render(llm_t, {"var1": inp, "var2": out.get("messages", [])})
        states.append((r, jev_state, llm_prompt, len(ENC.encode(jev_state))))
    avg_tok = sum(s[3] for s in states) / max(1, len(states))
    n_calls = len(states) + args.repeats * min(args.repeat_n, len(states))
    ledger.check("jev", n_calls * ledger.cost_jev(int(avg_tok)), "direct Jev calls")
    ledger.check("openai", n_calls * ledger.cost_openai(JUDGE_MODEL, int(avg_tok) + 700, 90), "direct LLM judge calls")
    print(f"[plan] avg state {avg_tok:.0f} tokens; {n_calls} calls per judge; wall clock about {n_calls*3/60:.0f} min")

    llm = ChatOpenAI(model=JUDGE_MODEL, max_retries=3, timeout=90, use_responses_api=True).with_structured_output(
        llm_schema, method="json_schema", include_raw=True)

    def ask_llm(prompt: str) -> dict:
        t0 = time.perf_counter()
        try:
            res = llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001
            return {"answers": None, "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_s": time.perf_counter() - t0, "error": repr(e)[:300]}
        lat = time.perf_counter() - t0
        um = res["raw"].usage_metadata or {}
        return {"answers": res["parsed"], "usage": {"input_tokens": um.get("input_tokens", 0), "output_tokens": um.get("output_tokens", 0)},
                "latency_s": lat, "error": None if res["parsed"] is not None else str(res.get("parsing_error"))[:300]}

    out_path = ROOT / "results" / f"direct_{args.tag}.jsonl"
    t0 = time.time()
    with out_path.open("w") as f, tracing_context(enabled=False):
        def one(r, jev_state, llm_prompt, tok, rep):
            j = jev_client.ask(jev_state, jev_q)
            jtok = j["usage"].get("input_tokens") or tok
            jcost = ledger.cost_jev(jtok, billed=True) if j["answers"] else 0.0
            ledger.add("jev", jcost)
            l = ask_llm(llm_prompt)
            lcost = ledger.cost_openai(JUDGE_MODEL, l["usage"]["input_tokens"], l["usage"]["output_tokens"])
            ledger.add("openai", lcost)
            rec = {"qid": r["qid"], "run_id": r["run_id"], "rep": rep, "state_tokens": tok,
                   "jev": {k: v for k, v in j.items() if k != "raw"} | {"cost_usd": jcost}, "llm": l | {"cost_usd": lcost}}
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
            return rec

        for i, (r, js, lp, tok) in enumerate(states):
            rec = one(r, js, lp, tok, 0)
            if i % 10 == 0 or i == len(states) - 1:
                print(f"[{i+1}/{len(states)}] jev {rec['jev']['latency_s'] or 0:.2f}s llm {rec['llm']['latency_s']:.2f}s tokens {tok} "
                      f"jev_in={rec['jev']['usage'].get('input_tokens')} llm_in={rec['llm']['usage']['input_tokens']}")
            if ledger.over_cap("jev") or ledger.over_cap("openai"):
                print("[ledger] stop line reached"); return
        for rep in range(1, args.repeats + 1):
            for r, js, lp, tok in states[: args.repeat_n]:
                one(r, js, lp, tok, rep)
            print(f"[repeat {rep}/{args.repeats}] done")
    print(f"[done] {(time.time()-t0)/60:.1f} min; jev ${ledger.spent('jev'):.4f} openai ${ledger.spent('openai'):.4f}")


if __name__ == "__main__":
    main()
