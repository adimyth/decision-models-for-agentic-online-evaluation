"""Analysis: coverage, lag, cost, accuracy vs gold, agreement, misses. Writes results/metrics.json and figures.

Usage: uv run scripts/analyze.py --tag main
Inputs: runs_<tag>.jsonl, feedback_<tag>.jsonl, gold_<tag>.jsonl, evaluator_runs_<tag>.jsonl (optional),
        direct_<tag>.jsonl (optional).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_online_eval import ledger  # noqa: E402

QS = ["answered", "grounded", "correct", "confidence", "outcome"]


def load(name, tag):
    p = ROOT / "results" / f"{name}_{tag}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    runs = [r for r in load("runs", args.tag) if r.get("ok")]
    fb = load("feedback", args.tag)
    gold = {g["run_id"]: g for g in load("gold", args.tag)}
    evruns = {e["run_id"]: e for e in load("evaluator_runs", args.tag)}
    direct = load("direct", args.tag)
    m = {"n_runs": len(runs)}

    # ---- agent run stats
    m["agent"] = {
        "cost_total_usd": sum(r["cost_usd"] for r in runs),
        "cost_per_run_usd": float(np.mean([r["cost_usd"] for r in runs])),
        "latency_p50_s": pct([r["latency_s"] for r in runs], 50),
        "latency_p95_s": pct([r["latency_s"] for r in runs], 95),
        "tool_calls_mean": float(np.mean([r["n_tool_calls"] for r in runs])),
    }

    # ---- feedback table: run_id x key -> score/value
    table = defaultdict(dict)
    lag = defaultdict(list)
    for f in fb:
        table[f["run_id"]][f["key"]] = f
        if f["lag_s"] is not None and f["key"] != "comment":
            lag[f["key"].split("_")[0]].append(f["lag_s"])

    # ---- coverage
    cov = {}
    for judge in ("jev", "llm"):
        full = sum(1 for r in runs if all(f"{judge}_{q}" in table[r["run_id"]] for q in QS))
        anyk = sum(1 for r in runs if any(f"{judge}_{q}" in table[r["run_id"]] for q in QS))
        cov[judge] = {"all_keys": full, "any_key": anyk, "share_all_keys": full / len(runs)}
    m["coverage"] = cov

    # ---- lag (run end -> feedback created)
    m["lag_s"] = {j: {"p50": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95), "max": float(max(v)), "n": len(v)}
                  for j, v in lag.items()}

    # ---- evaluator run stats (judge latency, tokens, cost as seen by LangSmith)
    ev = {}
    for judge in ("jev", "llm"):
        rows = [e for e in evruns.values() if e["judge"] == judge]
        if rows:
            lat = [e["latency_s"] for e in rows if e.get("latency_s") is not None]
            toks = [e.get("prompt_tokens") or 0 for e in rows]
            ev[judge] = {"n": len(rows), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
                         "prompt_tokens_mean": float(np.mean(toks)) if toks else None,
                         "completion_tokens_mean": float(np.mean([e.get("completion_tokens") or 0 for e in rows])),
                         "cost_mean_usd": float(np.mean([e.get("cost_usd") or 0 for e in rows]))}
    m["evaluator_runs"] = ev

    # ---- gold grading
    grades = Counter(g["grade"] for g in gold.values())
    m["gold"] = dict(grades)

    # ---- accuracy vs gold
    def score(rid, key):
        f = table[rid].get(key)
        return None if f is None else f["score"]

    def value(rid, key):
        f = table[rid].get(key)
        return None if f is None else f["value"]

    acc = {}
    for judge in ("jev", "llm"):
        # correct: gold CORRECT vs INCORRECT (NOT_ATTEMPTED excluded for AUROC), probability of correct
        ys, ps = [], []
        ys2, ps2 = [], []
        att_y, att_p = [], []
        for r in runs:
            rid = r["run_id"]
            g = gold.get(rid, {}).get("grade")
            p = score(rid, f"{judge}_correct")
            a = score(rid, f"{judge}_answered")
            if g in ("CORRECT", "INCORRECT") and p is not None:
                ys.append(1 if g == "CORRECT" else 0); ps.append(p)
            if g is not None and p is not None:
                ys2.append(1 if g == "CORRECT" else 0); ps2.append(p)
            if g is not None and a is not None:
                att_y.append(0 if g == "NOT_ATTEMPTED" else 1); att_p.append(a)
        d = {}
        if len(set(ys)) == 2:
            d["correct_auroc_correct_vs_incorrect"] = float(roc_auc_score(ys, ps))
            d["n_correct_vs_incorrect"] = len(ys)
            thr = 0.5
            tp = sum(1 for y, p in zip(ys, ps) if y == 1 and p >= thr)
            fp = sum(1 for y, p in zip(ys, ps) if y == 0 and p >= thr)
            fn = sum(1 for y, p in zip(ys, ps) if y == 1 and p < thr)
            tn = sum(1 for y, p in zip(ys, ps) if y == 0 and p < thr)
            d["correct_confusion_at_0.5"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            d["correct_accuracy_at_0.5"] = (tp + tn) / len(ys)
            d["flag_rate_of_wrong_answers_at_0.5"] = tn / max(1, tn + fp)
            d["false_alarm_rate_at_0.5"] = fn / max(1, tp + fn)
        if len(set(ys2)) == 2:
            d["correct_auroc_all"] = float(roc_auc_score(ys2, ps2))
        if len(set(att_y)) == 2:
            d["answered_auroc_vs_not_attempted"] = float(roc_auc_score(att_y, att_p))
            d["n_not_attempted"] = att_y.count(0)
        acc[judge] = d
    m["accuracy_vs_gold"] = acc

    # ---- agreement between judges
    agree = {}
    for q in QS:
        pairs = [(table[r["run_id"]].get(f"jev_{q}"), table[r["run_id"]].get(f"llm_{q}")) for r in runs]
        pairs = [(a, b) for a, b in pairs if a and b]
        if not pairs:
            continue
        if q == "outcome":
            agree[q] = {"n": len(pairs), "exact_agreement": sum(1 for a, b in pairs if a["value"] == b["value"]) / len(pairs)}
        elif q == "confidence":
            diffs = [abs((a["score"] or 0) - (b["score"] or 0)) for a, b in pairs]
            agree[q] = {"n": len(pairs), "within_0.5": sum(1 for d in diffs if d <= 0.5) / len(pairs),
                        "mean_abs_diff": float(np.mean(diffs))}
        else:
            xs = np.array([a["score"] for a, b in pairs], float); ysv = np.array([b["score"] for a, b in pairs], float)
            agree[q] = {"n": len(pairs), "binary_agreement_at_0.5": float(np.mean((xs >= .5) == (ysv >= .5))),
                        "pearson": float(np.corrcoef(xs, ysv)[0, 1]) if xs.std() > 0 and ysv.std() > 0 else None,
                        "jev_mean": float(xs.mean()), "llm_mean": float(ysv.mean())}
    m["agreement"] = agree

    # ---- direct calls: latency, cost, repeatability
    if direct:
        d0 = [d for d in direct if d["rep"] == 0]
        dd = {}
        for judge in ("jev", "llm"):
            lat = [d[judge]["latency_s"] for d in d0 if d[judge].get("latency_s") is not None and d[judge].get("answers")]
            cost = [d[judge]["cost_usd"] for d in d0]
            dd[judge] = {"n": len(lat), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
                         "latency_mean_s": float(np.mean(lat)) if lat else None, "cost_mean_usd": float(np.mean(cost)),
                         "failures": sum(1 for d in d0 if not d[judge].get("answers"))}
        # repeatability: per qid, std of 'correct' across reps
        reps = defaultdict(lambda: {"jev": [], "llm": []})
        for d in direct:
            for judge in ("jev", "llm"):
                a = d[judge].get("answers")
                if a:
                    v = a["jev_correct"]["noul"] if judge == "jev" else a["llm_correct"]
                    reps[d["qid"]][judge].append(v)
        for judge in ("jev", "llm"):
            stds = [np.std(v[judge]) for v in reps.values() if len(v[judge]) >= 3]
            flips = [1 if len({x >= .5 for x in v[judge]}) > 1 else 0 for v in reps.values() if len(v[judge]) >= 3]
            dd[judge]["repeat_correct_std_mean"] = float(np.mean(stds)) if stds else None
            dd[judge]["repeat_flip_rate"] = float(np.mean(flips)) if flips else None
            dd[judge]["repeat_n_states"] = len(stds)
        m["direct"] = dd

    # ---- misses: Jev confidently wrong vs gold
    misses = []
    for r in runs:
        rid = r["run_id"]; g = gold.get(rid, {}).get("grade"); p = score(rid, "jev_correct"); pl = score(rid, "llm_correct")
        if g == "INCORRECT" and p is not None and p >= 0.5:
            misses.append({"type": "jev_said_correct_but_wrong", "qid": r["qid"], "q": r["question"], "gold": r["gold"],
                           "answer": r["answer"][:200], "jev_correct": p, "llm_correct": pl, "run_id": rid})
        if g == "CORRECT" and p is not None and p < 0.5:
            misses.append({"type": "jev_said_wrong_but_correct", "qid": r["qid"], "q": r["question"], "gold": r["gold"],
                           "answer": r["answer"][:200], "jev_correct": p, "llm_correct": pl, "run_id": rid})
    m["jev_misses"] = misses
    m["spend_ledger"] = json.loads((ROOT / "results" / "ledger.json").read_text())

    (ROOT / "results" / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps({k: v for k, v in m.items() if k != "jev_misses"}, indent=1, default=str))
    print(f"jev misses: {len(misses)}")


if __name__ == "__main__":
    main()
