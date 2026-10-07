# Decision models for agentic online evaluation

> A decision model makes it cheap and fast to score every trace an agent produces. Whether you can trust the score depends on the question you ask it.

## TL;DR

I ran TypeSafe's Jev and gpt-5.6-luna as LangSmith online evaluators on every trace of a live web-research agent, 300 runs at 100% sampling, and checked their verdicts against known answers. Perplexity's and OpenAI's Decisions APIs and gpt-6-luna scored the same traces offline.

Scoring every trace is cheap with any of them: $0.42 per 1K traces for Jev, $0.94 for OpenAI's Decisions API, up to $1.91 for gpt-5.6-luna. Perplexity lists nearly Jev's price but bills the trace once per question, so at five questions it costs 4.5× more. The bigger bills sit elsewhere: LangSmith's retention upgrade on every scored trace costs more than any of the judges, and its evaluator queue lands each score about a minute after the run, whichever judge you pick.

**No judge, decision model or LLM, can tell you a single answer is wrong when it has no reference to compare against.** Most of the agent's wrong answers faithfully quoted a web page that was itself wrong.

So use them for trends, not verdicts. When the agent was wrong, the decision models scored it lower, just rarely low enough to flag. Across a day of traffic those small drops add up, so a falling average shows the agent getting worse. The LLM judges score nearly everything as correct, so their average barely moves.

Code and data: [adimyth/decision-models-for-agentic-online-evaluation](https://github.com/adimyth/decision-models-for-agentic-online-evaluation).

## Why everyone samples

Online evaluation means a judge scores your production traces as they arrive, and the scores sit on the traces for dashboards and alerts.

Almost nobody runs it on every trace. [LangSmith's guide](https://docs.langchain.com/langsmith/online-evaluations) suggests applying the evaluator to 10% of traces to control costs, and [Langfuse](https://langfuse.com/blog/2026-09-23-catching-conversation-signals-in-langfuse) says the same: with LLM-as-a-judge at scale, costs "were primarily contained through sampling". I did the same on my own agents: sampled a few percent, and over time stopped checking even those.

TypeSafe released [Jev](https://typesafe.ai), a decision model that answers typed questions about a piece of context and returns probabilities instead of prose, at $0.042 per million input tokens. Three of the launch-week posts measured it as a judge:

- [LangChain](https://www.langchain.com/blog/jev-agent-evals-langsmith) judged five recorded agent runs 100 times each: Jev matched the human label on all 500, gpt-5.6-luna on 96.4%, Claude Sonnet on 80%.
- [Arize](https://arize.com/blog/jev-as-a-judge/) reports a threshold-tuned Jev matching Claude Opus 5 at 87% on a hallucination benchmark, at about 1/300 of the cost.
- [Langfuse](https://langfuse.com/blog/2026-09-22-running-evals-with-jev) quotes a Good Start Labs study: Jev agreed with Claude Fable 5.1 on 91.5% of 6,003 rubric checks.

Each of those measures Jev on recorded examples. I wanted to see how it holds up in practice: as an online evaluator on a real agent, scoring every trace as it arrives. That meant measuring what each trace costs, how long the score takes to land, and whether the verdicts can be trusted, with known answers to check them against.

## Setup

### The agent

A small web-research agent built with Deep Agents on `gpt-5.6-luna`, with two tools and a system prompt asking for a short cited answer:

- `web_search`: DuckDuckGo via the ddgs library.
- `fetch_page`: httpx plus trafilatura, pages cut to 4K tokens.

I sent it 300 questions from OpenAI's SimpleQA set: short factual questions written by people, each with a verified gold answer. For example:

> Who won the Eddington Medal in 1972?

Every run was traced to one LangSmith project. One trace holds the question, every search and page fetch with its result, and the final answer.

### The five questions

An online evaluator is a judge that reads each trace as it arrives and writes scores onto it. The judge does not write a review. It fills in a fixed checklist, and each item on the checklist becomes one feedback key on the trace. This experiment uses the same five-item checklist for every judge. The wording is adapted from Openlayer's jevals library.

<div className="wide-table">

| Key | Question | Answer type | What comes back |
|---|---|---|---|
| `answered` | Did the agent give a direct answer, rather than decline or say it could not find one? | yes/no | a probability, 0 to 1 |
| `grounded` | Does the answer use what the tools returned, without adding facts they do not contain? | yes/no | a probability, 0 to 1 |
| `correct` | Is the answer factually correct? Judged from the tool results and general knowledge, with no reference answer | yes/no | a probability, 0 to 1 |
| `confidence` | How definite is the answer? | score, 4 levels | 0 (no answer) to 3 (definite, with a source) |
| `outcome` | What happened overall? | choice | one of `answered`, `could_not_find`, `partial`, `refused` |

</div>

> `correct` is the question that matters most and the hardest to answer. An online judge never sees the gold answer, because production traffic has no answer key. It decides from the trace alone.

### The five judges

Two kinds of model answered the checklist.

- **Decision models** (Jev, Perplexity Decisions, OpenAI Decisions) take the five questions as structured input and return five structured answers.
- **Chat models** (gpt-5.6-luna, gpt-6-luna) only take text. The same five questions are written out in a prompt, the trace is pasted underneath, and the model is told to reply as a JSON object with one field per question.

The prompt route is the usual way to build an LLM-as-judge evaluator, and it is the baseline here.

<div className="wide-table">

| Judge | Kind | How it receives the questions | Where it ran |
|---|---|---|---|
| Jev `jev-1.13.0` | decision model | as typed question objects | LangSmith online evaluator, 100% sampling |
| gpt-5.6-luna | chat model | as text in a prompt, JSON reply | LangSmith online evaluator, 100% sampling |
| Perplexity Decisions `pplx-decider-v1-27b` | decision model | the same question objects as Jev | offline, on the same traces after the run |
| OpenAI Decisions `gpt-6-luna` | decision model | the same questions in OpenAI's question format | offline, on the same traces after the run |
| gpt-6-luna | chat model | the same prompt as gpt-5.6-luna | offline, on the same traces after the run |

</div>

The first two ran live as LangSmith online evaluators; the other three scored the same traces offline afterwards, so they have cost and accuracy numbers but no lag.

This is what three of the five questions look like as Jev receives them. The `state` is the trace; `{{input}}` and `{{output}}` are LangSmith variables holding the run's input and output.

```json
{
  "model": "jev-1.13.0",
  "state": "<input>{{input}}</input>\n<output>{{output}}</output>",
  "questions": {
    "jev_correct": {
      "type": "noul",
      "instructions": "Is the final assistant message in output factually correct as an answer to the user's question in input? Judge from the tool results in output and general knowledge. True: the stated answer is the right one. False: the stated answer is wrong, or no answer was given."
    },
    "jev_confidence": {
      "type": "score",
      "instructions": "How definite is the final assistant message in output about its answer?",
      "criteria": [
        "No answer or explicit inability to answer",
        "Hedged: offers a candidate but flags real doubt",
        "Fairly definite: states an answer with mild caveats",
        "Definite: states the answer plainly with a source"
      ]
    },
    "jev_outcome": {
      "type": "choice",
      "instructions": "How did the assistant's final message in output respond to the user's question in input?",
      "criteria": {
        "answered": "Provided a specific answer",
        "could_not_find": "Said it could not find reliable evidence and gave no answer",
        "partial": "Gave an answer to part of the question or a vague approximation",
        "refused": "Declined for policy or safety reasons"
      }
    }
  }
}
```

For the chat models the same questions are written out as prose in the prompt, and a JSON schema forces the reply into this shape:

```json
{
  "llm_answered": 1.0,
  "llm_grounded": 1.0,
  "llm_correct": 0.93,
  "llm_confidence": 3,
  "llm_outcome": "answered",
  "comment": "One sentence of reasoning."
}
```

### The reference and the measurements

After the run I graded every final answer against its SimpleQA gold answer using the SimpleQA grading scheme. Each trace is labelled `CORRECT`, `INCORRECT` or `NOT_ATTEMPTED`, and that label is what the judges' `correct` scores are checked against.

This is what one scored trace looks like, with all five judges' feedback keys on it. The agent's answer is wrong against gold, and every judge called it correct:

```json
{
  "question": "Who won the Eddington Medal in 1972?",
  "gold": "Paul Ledoux",
  "agent_answer": "Pierre Ledoux won the Eddington Medal in 1972.",
  "feedback": {
    "jev_answered": 0.99,
    "jev_grounded": 0.95,
    "jev_correct": 0.89,
    "jev_confidence": 3.0,
    "jev_outcome": "answered",
    "llm_answered": 1.0,
    "llm_grounded": 1.0,
    "llm_correct": 1.0,
    "llm_confidence": 3.0,
    "llm_outcome": "answered",
    "comment": "The assistant directly identified Pierre Ledoux, exactly matching the fetched source's 1972 Eddington Medal entry, and provided the source link.",
    "pplx_answered": 0.99,
    "pplx_grounded": 0.96,
    "pplx_correct": 0.91,
    "pplx_confidence": 2.97,
    "pplx_outcome": "answered",
    "luna6_answered": 1.0,
    "luna6_grounded": 1.0,
    "luna6_correct": 1.0,
    "luna6_confidence": 3,
    "luna6_outcome": "answered",
    "oai_answered": 1.0,
    "oai_grounded": 1.0,
    "oai_correct": 1.0,
    "oai_confidence": 3.0,
    "oai_outcome": "answered"
  }
}
```

`answered` 0.99 and `outcome` `answered` say the agent committed to a name, which it did. `correct` is the only key that could have caught the error, and all five judges put it at 0.89 or above, because the fetched Wikipedia page itself says Pierre Ledoux. The `comment` is the gpt-5.6-luna judge's reasoning; decision models produce none.

## What it cost

| Judge | Cost per 1K traces | Billed input tokens per trace |
|---|---|---|
| Jev | $0.42 | 10K |
| OpenAI Decisions | $0.94 | 9.4K |
| gpt-6-luna | $0.95 | 9K |
| Perplexity Decisions | $1.89 | 47K |
| gpt-5.6-luna | $1.91 | 9K |

Each figure uses the input tokens the vendor itself billed for the same rendered state, which carries fetched web pages, at its published rate. OpenAI [prices](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability) the Decisions API at $0.10 per million input tokens, with no output or cache charges.

![Judge cost per evaluated trace](img/cost.png)

The agent run itself cost $3.77 per 1K traces, so even the most expensive judge, gpt-5.6-luna, adds about half the cost of the run it scores.

Perplexity lists almost the same price per token as Jev but costs 4.5× more per trace because it bills the state once per question: the same state sent with one, two and five questions billed 5.6K, 11.1K and 27.6K tokens, where Jev billed 5.8K, 5.9K and 6.2K and OpenAI 5.1K, 5.3K and 5.8K. Compare judges on cost per trace at your question count, never on list price.

### The judge was not the expensive part

Jev is about 4.5× cheaper than gpt-5.6-luna and about half the price of gpt-6-luna. Against a cheap modern LLM, the judge was already affordable. The cost that dominates at scale is keeping the traces.

LangSmith charges for keeping traces. On the Developer plan the first 5K base traces a month are free, each one after that costs $5 per 1K, and a base trace is kept for 14 days. Upgrading a trace to extended retention, 180 days, adds $2.50 per 1K ([pricing](https://www.langchain.com/pricing)). Feedback itself is free. The catch is that any online evaluator run moves its trace from base to extended, so scoring every trace adds $2.50 per 1K traces, six times the Jev cost and more than any judge here, whichever judge you pick.

> Before choosing a judge, price what your platform charges to keep the traces it scores. On LangSmith that charge is more than the judge itself.

## How fast the scores arrived

![Lag](img/latency.png)

Dots mark the median. p95 is 98 s for Jev and 108 s for gpt-5.6-luna. The judge itself accounts for 0.4 s and 2 s of that; the rest is the evaluator queue.



Dots mark the median. p95 is 0.75 s for Jev, 0.99 s for OpenAI Decisions, 1.24 s for Perplexity, 3.5 s for gpt-6-luna and 3.4 s for gpt-5.6-luna. The two LLMs overlap.

The decision models answer in under a second and the LLMs in about two, but on the trace both scores land a minute or more after the run, because both wait in LangSmith's scheduling queue. If you need scores within seconds, the queue is your bottleneck and Jev does not fix it. If a couple of minutes is fine, for dashboards and alerts, both work.

## Do the scores mean anything

First the ground truth, which the judges never see. Of the 287 answers the research agent produced, the SimpleQA grader found 268 right, 17 wrong and 2 not attempted. A useful judge should score the 17 wrong answers lower than the 268 right ones. Seventeen is a thin basis, so treat what follows as indicative.

### The 268 right answers

(Chart: each judge's correct score on the 268 right answers. See metrics.json.)

Every judge called nearly all of them correct. The two LLM judges and OpenAI's Decisions API put almost every right answer at or near 1.0. Jev's scores spread lower, averaging 0.86, and it put three right answers below the line; OpenAI's Decisions API put one there.

### The 17 wrong answers

(Chart: each judge's correct score on the 17 wrong answers. See metrics.json.)

This is where a judge has to earn its place, and none does it per answer. gpt-5.6-luna scored every wrong answer 0.7 or higher and averaged 0.96: it was fooled every time. The decision models scored wrong answers lower, Jev most of all at 0.71, but almost every dot still sits on the "correct" side. Jev pushed two wrong answers below 0.5, and Perplexity, OpenAI's Decisions API and gpt-6-luna one each.

### Ranking right above wrong



AUROC ignores the threshold and asks only whether right answers outscore wrong ones. Perplexity leads because its scores are very consistent, so even a small step down sorts cleanly. All three decision models beat both LLM judges.

Read the charts in three lines.

1. **All five judges give the same yes or no verdict on 276 of 286 traces**, and none catches more than two of the 17 wrong answers. A reference-free judge cannot see that a faithfully quoted web page disagrees with the gold answer.
2. **They differ in how much the average drops when the agent is wrong.** Jev by 0.15, OpenAI's Decisions API by 0.13, Perplexity by 0.11, gpt-6-luna by 0.09, gpt-5.6-luna by 0.04.
3. **That drop is what makes a score usable on a drift chart.** If the agent started getting more answers wrong, the decision models' daily average would visibly fall, and gpt-5.6-luna's would barely move.

So, can you trust them? On a single trace, no judge here can be trusted to say whether the agent's answer is correct; alerting on one score would miss nearly every real error. Over a day of traces, the decision models can be trusted to show the agent getting worse, and the LLM judges cannot. On the structural questions the judges can be trusted per trace: every judge's `answered` score matched the gold "not attempted" label on all 286 traces, and all five picked the same outcome on 285 of them.

Repeatability did not separate them. Twenty states sent six times each produced no flipped verdicts from any judge; Perplexity and OpenAI's Decisions API returned identical probabilities every time, the LLMs moved by 0.003, Jev by 0.008.

## Where Jev was wrong

Fifteen wrong answers got a Jev `correct` probability above 0.5, and three correct answers got one below.

**Most of the fifteen faithfully repeat a source that disagrees with the gold answer.**

- 1941 instead of 1942 for a Columbia master's degree.
- 3:02:25 instead of 3:02:24 for a cycle race.
- "Pierre Ledoux" instead of "Paul Ledoux" for the 1972 Eddington Medal.
- 560 passengers instead of 583 at Tenerife, because the agent excluded crew.

A reference-free judge with the same web page in front of it cannot see these. None of the five did. Only a judge with the gold answer could.

Jev cannot explain itself, so every one of these took a human reading the trace. The LLM judge's comment was accurate about the evidence on all 17: on the cycle race it wrote that 3:02:25 "matches the fetched Wikipedia event table", and then scored the answer correct anyway.

> The explanation was right about the evidence and told me nothing about the verdict.

## What to take away

If you have been sampling a few percent of traces because the judge was too expensive, you can afford to score all of them. At 100% sampling, five questions per trace cost:

| Judge | Cost per trace |
|---|---|
| Jev | about four hundredths of a cent |
| OpenAI Decisions, gpt-6-luna | about a tenth of a cent |
| gpt-5.6-luna, Perplexity | a fifth of a cent |
| The agent run itself | four tenths of a cent |

Three things to check before you switch it on:

1. **The platform bill, not the judge bill.** On LangSmith, every evaluated trace moves to extended retention, which adds $2.50 per 1K traces, six times the Jev cost. Price the traces first.
2. **What you will do with a score that arrives a minute late.** Both online evaluators landed on the trace 70 to 110 seconds after the run ended, almost all of it LangSmith's queue. Fine for dashboards and daily alerts. Not fine for blocking or routing a live response; that needs a call from inside the agent, where Jev's 0.4 seconds does matter.
3. **Which traces it will skip.** Of my 300 runs, 13 failed on the agent's recursion limit and the evaluators never saw them, because they only fire on successful runs. One more was a 12-tool-call trace whose state passed Jev's 32K-token limit, and Jev's evaluator failed on it while the other four judges scored it. Decide what should happen to those before you trust the coverage number.

On what the scores are worth: the decision models give a usable drift signal on "is this answer correct" and the LLM judges do not, because the LLMs say 0.98 to almost everything. No judge catches an individual wrong answer, because most wrong answers here were faithful summaries of a web page that disagreed with the reference.

> Use a decision model online for aggregate quality tracking and for structural questions like "did the agent answer". Keep a reference-based check for per-answer correctness.
