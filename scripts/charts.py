"""Two figures for the essay. Palette: dataviz reference slots 1 (blue) and 2 (orange)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = {"jev": "#2a78d6", "llm": "#eb6834", "pplx": "#1baf7a", "luna6": "#eda100"}
LABEL = {"jev": "Jev (jev-1.13.0)", "llm": "gpt-5.6-luna judge", "pplx": "Perplexity Decisions", "luna6": "gpt-6-luna judge"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6})


def load(name):
    return [json.loads(l) for l in (ROOT / "results" / name).read_text().splitlines() if l.strip()]


def ecdf(ax, xs, color, label):
    xs = np.sort(np.asarray(xs, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    (line,) = ax.step(xs, ys, where="post", color=color, lw=2, label=label)
    p50, p95 = np.percentile(xs, 50), np.percentile(xs, 95)
    ax.scatter([p50, p95], [0.5, 0.95], s=28, color=color, edgecolor=SURF, linewidth=1.5, zorder=3, label="_nolegend_")
    ecdf.last = line
    return p50, p95


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    fb = load(f"feedback_{tag}.jsonl")
    seen, lag = set(), {"jev": [], "llm": []}
    for f in fb:
        if f["key"] == "comment" or f["feedback_id"] in seen or f["lag_s"] is None:
            continue
        seen.add(f["feedback_id"]); lag[f["key"].split("_")[0]].append(f["lag_s"])
    direct = [d for d in load(f"direct_{tag}.jsonl") if d["rep"] == 0]
    dl = {j: [d[j]["latency_s"] for d in direct if d[j].get("answers") and d[j].get("latency_s")] for j in ("jev", "llm")}
    for j in ("pplx", "luna6"):
        p = ROOT / "results" / f"direct_{j}_{tag}.jsonl"
        if p.exists():
            dj = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            dl[j] = [d[j]["latency_s"] for d in dj if d["rep"] == 0 and d[j].get("answers") and d[j].get("latency_s")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
    ax = axes[0]
    notes = []
    for j in ("jev", "llm"):
        p50, p95 = ecdf(ax, lag[j], C[j], LABEL[j]); notes.append((j, p50, p95))
    ax.set_title("Run end to score on trace, via LangSmith evaluator", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("seconds"); ax.set_ylabel("share of traces scored"); ax.grid(axis="y"); ax.set_ylim(0, 1.02)
    for j, p50, p95 in notes:
        ax.annotate(f"p50 {p50:.0f}s · p95 {p95:.0f}s", xy=(p50, 0.5), xytext=(8, -4 if j == "llm" else -16),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax = axes[1]
    notes = []
    handles = []
    for j in ("jev", "pplx", "luna6", "llm"):
        if dl.get(j):
            p50, p95 = ecdf(ax, dl[j], C[j], LABEL[j]); notes.append((j, p50, p95)); handles.append(ecdf.last)
    ax.set_xscale("log"); ax.set_title("Judge call latency, same state sent directly", loc="left", fontsize=10, color=INK)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [0.3, 0.5, 1, 2, 3, 5]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("seconds (log scale)"); ax.grid(axis="y"); ax.set_ylim(0, 1.02)
    ax.legend(handles, [f"{LABEL[j]}: p50 {p50:.2f}s, p95 {p95:.2f}s" for j, p50, p95 in notes], frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "essay" / "img" / "latency.png"); plt.close(fig)

    # cost per evaluated trace
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    ev = m.get("direct", {})
    costs = {j: ev[j]["cost_mean_usd"] for j in ("jev", "llm") if j in ev}
    for j, e in (m.get("extra_judges") or {}).items():
        if e.get("cost_mean_usd") is not None:
            costs[j] = e["cost_mean_usd"]
    costs = dict(sorted(costs.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=160)
    names = [LABEL[j] for j in costs]; vals = [costs[j] * 1000 for j in costs]
    bars = ax.barh(names, vals, color=[C[j] for j in costs], height=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.015, b.get_y() + b.get_height() / 2, f"${v:.2f} per 1,000 traces", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, max(vals) * 1.45); ax.invert_yaxis(); ax.grid(axis="x"); ax.set_xlabel("USD per 1,000 evaluated traces (five questions each)")
    ax.set_title("Judge cost per evaluated trace, measured token usage", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(ROOT / "essay" / "img" / "cost.png"); plt.close(fig)
    print("wrote essay/img/latency.png and cost.png")


if __name__ == "__main__":
    main()
