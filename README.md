# Decision models for agentic online evaluation

Code, data and analysis behind the essay *Decision models for agentic online evaluation*.

The question: if a judge is cheap and fast enough, can you score **every** production trace of an agent instead of sampling a few percent, and are the scores worth having? The judges compared are TypeSafe's **Jev**, Perplexity's **Decisions API** and OpenAI's **Decisions API** (decision models that answer typed questions with probabilities) against **gpt-5.6-luna** and **gpt-6-luna** used as conventional LLM judges.

## What was run

1. A small web-research agent (Deep Agents, gpt-5.6-luna, `web_search` + `fetch_page`) answered 300 questions from OpenAI's SimpleQA set, traced to one LangSmith project. 287 finished; 13 hit the recursion limit.
2. Two **LangSmith online evaluators** scored every finished trace at 100% sampling with the same five questions: one on Jev (`jev-1.13.0`), one on gpt-5.6-luna with the questions in a prompt and a JSON reply.
3. Every final answer was graded against its SimpleQA gold answer (`CORRECT` / `INCORRECT` / `NOT_ATTEMPTED`), giving 268 / 17 / 2.
4. The exact rendered state of every trace was sent **directly** to all five judges to measure call latency, billed tokens and repeatability (20 states × 6 repeats). Perplexity Decisions, OpenAI Decisions and gpt-6-luna ran only on this offline path; Perplexity's and gpt-6-luna's verdicts were also posted back onto the traces as feedback.

## Headline numbers

| | Jev | Perplexity Decisions | OpenAI Decisions | gpt-6-luna | gpt-5.6-luna |
|---|---|---|---|---|---|
| Cost per 1K traces, five questions | $0.42 | $1.89 | $0.94 | $0.95 | $1.91 |
| Billed input tokens per trace | 10K | 47K | 9.4K | 9K | 9K |
| Direct call latency, p50 | 0.42 s | 0.67 s | 0.52 s | 2.2 s | 2.09 s |
| Run end to score on trace, p50 (LangSmith online evaluator) | 69 s | – | – | – | 80 s |
| Traces scored, of 287 | 286 | 287 | 287 | 287 | 287 |
| Mean `correct` score, agent right / wrong | 0.86 / 0.71 | 0.97 / 0.86 | 0.98 / 0.85 | 1.00 / 0.91 | 1.00 / 0.96 |
| AUROC, right vs wrong | 0.83 | 0.90 | 0.80 | 0.74 | 0.72 |
| Wrong answers caught at p < 0.5 | 2 of 17 | 1 of 17 | 1 of 17 | 1 of 17 | 0 of 17 |

OpenAI prices the Decisions API at $0.10 per million input tokens, with no output or cache charges ([pricing](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability)).

Three things worth knowing:

- **Perplexity bills the state once per question.** One state with 1, 2 and 5 questions billed 5.6K, 11.1K and 27.6K tokens; Jev billed 5.8K, 5.9K and 6.2K; OpenAI Decisions 5.1K, 5.3K and 5.8K. Near-identical list prices, 4.5× different cost per trace at five questions.
- **The LangSmith queue, not the judge, sets the lag.** Both online evaluators landed 70 to 110 s after the run ended; the judges themselves take 0.4 s and 2 s.
- **No judge reliably catches individual wrong answers without a reference.** At the default 0.5 cutoff the best caught 2 of 17. Of the 17 wrong answers, 11 repeat a fact from the agent's own tool results that disagrees with gold, and 3 state something no tool returned (for example "Pierre" Ledoux from a page that said "P. Ledoux"); no judge's `grounded` score flagged those 3. The decision models' scores do drop when the agent is wrong and the LLMs' barely move, so only the decision models give a usable drift signal.

Spend as billed by the providers: about $2.80 on OpenAI, $0.28 on Jev, $0.73 on Perplexity.

## Layout

```
src/jev_online_eval/
  agent.py          Deep Agent with web_search and fetch_page
  questions.py      the five questions (wording adapted from Openlayer's jevals)
  jev_client.py     POST https://api.typesafe.ai/v1/systemone, retries on 429/529
  pplx_client.py    POST https://api.perplexity.ai/v1/decisions, same wire shapes
  openai_decisions_client.py  POST https://api.openai.com/v1/decisions; converts to and from the Jev shapes
  llm_judge.py      gpt-5.6-luna with the same questions in a prompt, JSON schema reply
  state.py          compact judge state from a LangSmith run (used for smoke tests)
  ledger.py         spend ledger; every script prints projected cost and aborts at 80% of a cap
scripts/
  run_traffic.py        send SimpleQA questions through the agent, traced to LangSmith
  collect_feedback.py   pull the online evaluators' feedback per run; measure lag
  evaluator_runs.py     pull the evaluators' own runs (latency, tokens, cost)
  grade_gold.py         SimpleQA grading of each answer against gold
  judge_latency.py      direct calls to Jev and gpt-5.6-luna with the exact online state
  judge_direct.py       the same for extra judges: --judge pplx | luna6 | oai
  post_extra_feedback.py  post extra judges' verdicts onto the traces (extend_trace_retention=False)
  analyze.py            coverage, lag, cost, accuracy vs gold, agreement, misses -> results/metrics.json
  charts.py             the PNG charts in essay/img (the site uses SVG components instead)
results/
  runs_main.jsonl             one row per agent run: question, gold, answer, tokens, cost, latency
  feedback_main.jsonl         every feedback item on those runs, with lag from run end
  evaluator_runs_main.jsonl   the LangSmith evaluators' own runs
  gold_main.jsonl             SimpleQA grade per run
  direct_main.jsonl           direct calls: Jev and gpt-5.6-luna (rep 0 = first pass, 1..5 = repeats)
  direct_pplx_main.jsonl      direct calls: Perplexity Decisions
  direct_luna6_main.jsonl     direct calls: gpt-6-luna
  direct_oai_main.jsonl       direct calls: OpenAI Decisions
  jev-online_prompt.json      the Jev evaluator exactly as LangSmith saved it (questions, model)
  llm-online_prompt.json      the gpt-5.6-luna evaluator exactly as LangSmith saved it (prompt, schema)
  metrics.json                everything analyze.py computes
  ledger.json                 cumulative spend recorded by the scripts (excludes LangSmith-run calls)
essay/                        the essay draft and PNG charts
data/simple_qa_test_set.csv   OpenAI SimpleQA
```

## Reproduce

Keys go in `.env` (see `.env.example`): `OPENAI_API_KEY`, `TYPESAFE_AI_KEY`, `LANGSMITH_API_KEY`, `PERPLEXITY_API_KEY`. Nothing is read from anywhere else and nothing is logged.

```bash
uv sync
uv run scripts/run_traffic.py --n 10 --tag smoke                 # ~2 min, ~$0.04
# create the two online evaluators in the LangSmith UI on the project
# (Evaluators tab). The exact configs are results/jev-online_prompt.json
# and results/llm-online_prompt.json.
uv run scripts/run_traffic.py --n 300 --offset 13 --concurrency 6 --tag main   # ~22 min, ~$1.15
uv run scripts/collect_feedback.py --tag main --wait 900
uv run scripts/evaluator_runs.py --tag main
uv run scripts/grade_gold.py --tag main
uv run scripts/judge_latency.py --tag main --repeats 5 --repeat-n 20            # ~17 min
uv run scripts/judge_direct.py --judge pplx --tag main --repeats 5 --repeat-n 20
uv run scripts/judge_direct.py --judge luna6 --tag main --repeats 5 --repeat-n 20
uv run scripts/judge_direct.py --judge oai --tag main --repeats 5 --repeat-n 20
uv run scripts/post_extra_feedback.py --tag main
uv run scripts/analyze.py --tag main
```

Every script prints its projected cost and the cumulative spend before it starts. Caps: $10 OpenAI, $5 Jev, $5 Perplexity.

Note that calls LangSmith makes with your keys (the two online evaluators) are billed by the providers but do not pass through `ledger.json`.

## Models and prices used

- `gpt-5.6-luna`: $0.20 in / $1.20 out per million tokens, Responses API.
- `gpt-6-luna`: $0.10 / $0.50.
- `jev-1.13.0`: $0.042 per million input tokens, output free, 32K-token state limit.
- `pplx-decider-v1-27b`: $0.04 per million input tokens, output free, 262K-token limit, bills the state once per question.
- OpenAI Decisions (`gpt-6-luna`, public beta from 6 October 2026): $0.10 per million input tokens, no output or cache charges; bills the state once per request.
