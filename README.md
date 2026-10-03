# Jev as an online evaluator in LangSmith

A small, real test of TypeSafe's Jev decision model as a LangSmith **online evaluator**: a Deep Agents web-research agent answers SimpleQA questions, every trace is scored at 100% sampling by two LangSmith evaluators configured with the same five questions, one using Jev (`jev-1.13.0`) and one using gpt-5.6-luna as a conventional LLM judge. The essay is in `essay/`.

## Layout

- `src/jev_online_eval/agent.py` – Deep Agent (gpt-5.6-luna) with `web_search` (ddgs) and `fetch_page` tools.
- `src/jev_online_eval/ledger.py` – spend ledger; every script prints projected cost and aborts at 80% of the caps.
- `src/jev_online_eval/questions.py` – the five questions (wording adapted from Openlayer's jevals).
- `scripts/run_traffic.py` – sends SimpleQA questions through the agent, traced to LangSmith.
- `scripts/collect_feedback.py` – pulls the evaluators' feedback for each run and measures lag.
- `scripts/evaluator_runs.py` – pulls the evaluators' own runs (latency, tokens, cost).
- `scripts/grade_gold.py` – grades answers against SimpleQA gold with the SimpleQA grader scheme.
- `scripts/judge_latency.py` – calls Jev and gpt-5.6-luna directly with the exact state the online evaluators render, for latency, cost and repeatability.
- `scripts/judge_direct.py` – the same for extra judges: `--judge pplx` (Perplexity Decisions API, `pplx-decider-v1-27b`, key in `PERPLEXITY_API_KEY`) and `--judge luna6` (gpt-6-luna).
- `scripts/post_extra_feedback.py` – posts the extra judges' verdicts onto the traces as feedback (self-hosted evaluator path, `extend_trace_retention=False`).
- `src/jev_online_eval/pplx_client.py` – Perplexity Decisions client; same question and answer wire shapes as Jev.
- `scripts/analyze.py` – coverage, lag, cost, accuracy vs gold, agreement, misses → `results/metrics.json`.
- `results/jev-online_prompt.json`, `results/llm-online_prompt.json` – the two evaluator configurations as saved by LangSmith.

## Reproduce

```
cp .env.example .env   # fill in keys
uv sync
uv run scripts/run_traffic.py --n 10 --tag smoke
# create the two evaluators in the LangSmith UI (Evaluators tab); configs in results/*_prompt.json
uv run scripts/run_traffic.py --n 300 --offset 13 --concurrency 6 --tag main
uv run scripts/collect_feedback.py --tag main --wait 900
uv run scripts/evaluator_runs.py --tag main
uv run scripts/grade_gold.py --tag main
uv run scripts/judge_latency.py --tag main --repeats 5 --repeat-n 20
uv run scripts/judge_direct.py --judge pplx --tag main --repeats 5 --repeat-n 20
uv run scripts/judge_direct.py --judge luna6 --tag main --repeats 5 --repeat-n 20
uv run scripts/post_extra_feedback.py --tag main
uv run scripts/analyze.py --tag main
```
