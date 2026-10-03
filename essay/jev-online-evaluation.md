# Jev for online evaluation

Online evaluation means a judge scores your production traces as they arrive, and the scores sit on the traces for dashboards and alerts. Almost nobody runs it on every trace. [LangSmith's guide](https://docs.langchain.com/langsmith/online-evaluations) suggests applying the evaluator to 10% of traces to control costs, and [Langfuse](https://langfuse.com/blog/2026-09-23-catching-conversation-signals-in-langfuse) describes the same habit: with LLM-as-a-judge at scale, costs "were primarily contained through sampling". I did the same on my own agents, sampled a few percent, and then stopped looking at those too.

TypeSafe released [Jev](https://typesafe.ai), a decision model that answers typed questions about a piece of context and returns probabilities instead of prose, at $0.042 per million input tokens. Three of the launch-week posts measured it as a judge:

- [LangChain](https://www.langchain.com/blog/jev-agent-evals-langsmith) judged five recorded agent runs 100 times each: Jev matched the human label on all 500, gpt-5.6-luna on 96.4%, Claude Sonnet on 80%.
- [Arize](https://arize.com/blog/jev-as-a-judge/) reports a threshold-tuned Jev matching Claude Opus 5 at 87% on a hallucination benchmark, at about 1/300 of the cost.
- [Langfuse](https://langfuse.com/blog/2026-09-22-running-evals-with-jev) quotes a Good Start Labs study: Jev agreed with Claude Fable 5.1 on 91.5% of 6,003 rubric checks.

Each of those compares Jev with a human label or a stronger model on recorded examples. Several launch posts then argue that a judge this cheap and fast means you can stop sampling. What I could not find is the measurement behind that argument: Jev running on a live project at 100% sampling, with the lag, coverage and cost per trace that result, and gold answers to check the scores against. This essay reports that for one real agent.

## Setup

I built a small web-research agent with Deep Agents on gpt-5.6-luna. It has two tools, `web_search` (DuckDuckGo via the ddgs library) and `fetch_page` (httpx plus trafilatura, pages truncated to 4K tokens), and a system prompt asking for a short cited answer. I sent it 300 questions from OpenAI's SimpleQA set, which are short factual questions written by people, each with a verified gold answer, and traced everything to one LangSmith project.

Code and data: [adimyth/jev-online-eval](https://github.com/adimyth/jev-online-eval).

On that project I created two online evaluators, both on root runs at a sampling rate of 100%, both asking the same five questions about each trace:

- **jev-online** sends the questions to Jev (`jev-1.13.0`) as typed question objects. LangSmith's decision-model evaluator builds the state from the run input and output and posts one feedback key per question.
- **llm-online** is a conventional LLM-as-judge on gpt-5.6-luna. The same five questions are written out in a prompt, the same input and output are pasted in, and the model returns a JSON object with one field per question. This is how you would have built an online evaluator before Jev.

The five questions, with wording adapted from Openlayer's jevals library: did the agent answer (yes/no probability), is the answer grounded in what the tools returned (yes/no probability), is the answer factually correct, judged without a reference (yes/no probability), how definite is the answer (a four-level score), and what was the outcome (a choice among answered, could not find, partial and refused). Three of them as Jev receives them:

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

After the run I graded each final answer against the SimpleQA gold answer with the SimpleQA grading scheme, pulled every feedback item and every evaluator run back through the SDK, and separately sent the exact same rendered state to both judges directly, to measure call latency and repeatability without LangSmith's queue in the way. On that direct path I added two more judges for comparison: Perplexity's [Decisions API](https://docs.perplexity.ai/docs/decisions/quickstart) (`pplx-decider-v1-27b`), which takes the same typed questions as Jev at a near-identical list price of $0.04 per million input tokens, and gpt-6-luna with the same prompt as the gpt-5.6-luna evaluator. LangSmith has no built-in for either, so their verdicts were posted onto the traces as feedback through the SDK, which is the self-hosted way to run an online evaluator.

## What it cost

<div className="wide-table">

| | Agent run | Jev | gpt-6-luna | gpt-5.6-luna | Perplexity Decisions |
|---|---|---|---|---|---|
| Per trace, measured | $0.0038 | $0.00043 | $0.00095 | $0.0019 | $0.0019 |
| Per 1K traces | $3.77 | $0.43 | $0.95 | $1.91 | $1.89 |
| At 10K traces a day | $38 | $4.30 | $9.50 | $19 | $19 |
| Billed input tokens per trace | | 7.5K | 9K | 9K | 47K |

</div>

Each figure uses the input token count the vendor itself billed for the same rendered state, which carries fetched web pages. The last row is the surprise. Jev and Perplexity list almost the same price per million tokens, yet Perplexity costs 4.4× more per trace. The reason is how they count. I sent one 15K-character state with one, two and five questions to both. Jev billed 5.8K, 5.9K and 6.2K tokens: the state once, plus a little per question. Perplexity billed 5.6K, 11.1K and 27.6K: the state again for every question. With a single question the two cost the same. With the five questions this experiment asks per trace, Perplexity costs what gpt-5.6-luna does. Compare judges on cost per trace at your question count, never on list price. For the whole experiment, 300 agent runs plus 287 evaluations by each judge, Jev cost 12 cents, gpt-6-luna 27 cents, gpt-5.6-luna 55 cents and Perplexity 55 cents. LangSmith's evaluators tab shows the two native judges as a daily spend chart, so their running cost is visible without any of my scripts.

![Judge cost per evaluated trace](img/cost.png)

Jev is about 4.5× cheaper than gpt-5.6-luna and twice as cheap as gpt-6-luna configured the same way, not the 100× in the launch post, which compared against Claude Sonnet. Against a cheap modern LLM the judge was already affordable. The cost that dominates at scale is a different one: every online evaluator run upgrades the trace to extended data retention, which LangSmith bills at $5 per 1K traces. At 10K traces a day that is $50 a day, more than both judges combined, and it is the same whichever judge you pick. My project was on the long-lived tier from the start, so I did not pay it here. Budget for the traces before the judge.

## How fast the scores arrived

![Lag and latency](img/latency.png)

<div className="wide-table">

| | Jev | Perplexity Decisions | gpt-6-luna | gpt-5.6-luna |
|---|---|---|---|---|
| Run end to score on trace, p50 (LangSmith evaluator) | 69 s | | | 80 s |
| Run end to score on trace, p95 (LangSmith evaluator) | 98 s | | | 108 s |
| Direct call, same state, p50 | 0.42 s | 0.67 s | 2.2 s | 2.09 s |
| Direct call, same state, p95 | 0.75 s | 1.24 s | 3.5 s | 3.41 s |

</div>

The decision models are fast: Jev answers in 0.42 seconds at the median and Perplexity in 0.67, against about 2.1 seconds for either LLM. Jev the online evaluator is barely faster than the LLM one, because both wait in the same LangSmith scheduling queue for a minute or more before either model is called. If you need scores within seconds of a trace landing, the queue is your bottleneck and Jev does not fix it. If you need them within a couple of minutes for dashboards and alerts, both work.

## Did every trace get scored

Of 300 requests, 287 finished and 13 hit the agent's recursion limit while looping on searches. The evaluators only fire on successful root runs, so neither judge scored those 13. The traces most likely to be interesting are the ones online evaluation skips.

Of the 287 finished traces, the LLM evaluator scored all 287 with all five keys. Jev scored 286. The one it missed was a 12-tool-call trace whose rendered state came to about 31.7K tokens, and the evaluator failed with "the model's context limit was exceeded". Jev's limit is 32K tokens and LangSmith maps the whole run output in by default, so long agent traces fall off the edge. Map a narrower variable than the whole output, as the error message says. Perplexity's limit is 262K tokens and it scored that trace without complaint; the two LLM judges did too.

This is what one scored trace looks like, with both judges' feedback keys on it. The agent's answer here is wrong against gold, and both judges called it correct:

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
    "comment": "The assistant directly identified Pierre Ledoux, exactly matching the fetched source's 1972 Eddington Medal entry, and provided the source link."
  }
}
```

## Do the scores mean anything

The agent got 268 of 287 answers right against gold, 17 wrong, 2 not attempted. Seventeen negatives is a thin basis, so treat the accuracy numbers as indicative.

<div className="wide-table">

| Reference-free "is the answer correct" | Jev | Perplexity Decisions | gpt-6-luna | gpt-5.6-luna |
|---|---|---|---|---|
| Mean score when the agent was right | 0.87 | 0.97 | 1.00 | 0.99 |
| Mean score when the agent was wrong | 0.67 | 0.86 | 0.91 | 0.96 |
| Gap between the two | 0.20 | 0.11 | 0.09 | 0.03 |
| AUROC, right vs wrong | 0.83 | 0.90 | 0.74 | 0.72 |
| Wrong answers caught at p < 0.5 | 2 of 17 | 1 of 17 | 1 of 17 | 0 of 17 |
| Right answers wrongly flagged at p < 0.5 | 3 of 267 | 0 of 268 | 0 of 268 | 0 of 268 |

</div>

Read the table in three lines. All four judges give the same yes or no verdict on 98% of traces, and none of them catches more than two of the 17 wrong answers, because a reference-free judge cannot see that a faithfully quoted web page disagrees with the gold answer. The judges differ in how much their score moves when the agent is wrong: Jev drops by 0.20 on average, Perplexity by 0.11, gpt-6-luna by 0.09 and gpt-5.6-luna by 0.03. So if the agent started getting more answers wrong, Jev's daily average would visibly fall, Perplexity's would dip, and gpt-5.6-luna's would barely move. That is what makes a score usable on a drift chart, and it is the only quality difference between these judges that matters for online monitoring.

AUROC tells the same story with one twist: Perplexity ranks right and wrong answers best (0.90) because its scores are tightly consistent, while Jev's larger drop comes with more spread. Both decision models beat both LLMs on this measure.

Repeatability did not separate them. I sent 20 states six times each to every judge. None flipped a single binary verdict. Perplexity returned identical probabilities every time, gpt-6-luna and gpt-5.6-luna moved by 0.002 to 0.003, and Jev by 0.008. The variance advantage the launch post measured against sampling LLM judges does not show up against structured-output LLMs on this task.

## Where Jev was wrong

Fifteen wrong answers got a Jev "correct" probability above 0.5, and three correct answers got one below.

- Most of the fifteen are answers that faithfully repeat a source that disagrees with the gold answer: 1941 instead of 1942 for a Columbia master's degree, 3:02:25 instead of 3:02:24 for a cycle race, "Pierre Ledoux" instead of "Paul Ledoux" for the 1972 Eddington Medal, 560 passengers instead of 583 at Tenerife because the agent excluded crew. A reference-free judge with the same web page in front of it cannot see these. Neither judge did, and only a judge with the gold answer could.
- A few are genuine misses of the kind TypeSafe documents, arithmetic and dates: the Dark Souls patch dated 23 October against a gold of 22 October got 0.60, and the passenger count got 0.91 despite the answer itself saying the total "including crew" was different.
- The three false alarms, 0.45 to 0.49 on correct answers about Delhi's forest cover, Pavlov's psychic secretion and Oprah's 164 acres, look like Jev being unsure on numeric answers rather than wrong. The LLM gave all three 0.97 or more.
- Jev's "grounded" score correlated only 0.65 with the LLM's and ran lower on average, 0.94 against 0.99. I read the six lowest Jev scores expecting to find unsupported claims. Five of the six were correct answers that quoted their source, such as Pavlov for psychic secretion at 0.57 and a Terraria patch name at 0.67. Only the lowest, a forest-cover figure at 0.18, was one both judges doubted. So the lower Jev scores look like hesitation on short numeric or name answers, not detected hallucination, and I would not alert on them.

Jev cannot explain itself, so every one of these took a human reading the trace. The LLM judge's one-line comment was accurate about what it had looked at on all 17 wrong answers, saying for example that 3:02:25 "matches the fetched Wikipedia event table", and then scored the answer correct anyway. The explanation was right about the evidence and told me nothing about the verdict.

## What to take away

If you have been sampling a few percent of traces because the judge was too expensive, you can stop. Jev scores a trace with five questions for about four hundredths of a cent, gpt-6-luna for a tenth of a cent, and gpt-5.6-luna or Perplexity for a fifth of a cent. At 10K traces a day that is $4 to $19, all of it small next to the $38 the agent itself costs.

Three things to check before you switch it on:

1. **The platform bill, not the judge bill.** On LangSmith, every evaluated trace moves to extended retention at $5 per 1K traces. At 10K traces a day that is $50, ten times the Jev cost. Price the traces first.
2. **What you will do with a score that arrives a minute late.** Both judges landed on the trace 70 to 110 seconds after the run ended, almost all of it LangSmith's queue. Fine for dashboards and daily alerts. Not fine for blocking or routing a live response; that needs a call from inside the agent, where Jev's 0.4 seconds does matter.
3. **Which traces it will skip.** Failed runs were not evaluated at all, and one long trace overflowed Jev's 32K context. Decide what you want to happen to those before you trust the coverage number.

On what the scores are worth: for the reference-free question "is this answer correct", the decision models give a usable drift signal and the LLM judges do not, because the LLMs say 0.98 to almost everything. Jev's score moves most when the agent is wrong; Perplexity's is the most consistent but bills the state once per question, so at five questions it costs four times as much per trace. No judge catches an individual wrong answer, because most wrong answers here were faithful summaries of a web page that disagreed with the reference. Use a decision model online for aggregate quality tracking and for structural questions like "did the agent answer", and keep a reference-based check for per-answer correctness.

This experiment says nothing about whether Jev can judge multi-step agent behaviour such as wrong tool choices or policy violations. That needs an agent with known-correct trajectories, and it is the next test to run.

## Method notes

300 SimpleQA questions, fixed random subset, 6 concurrent agent runs, 21.7 minutes of wall clock. Agent and LLM judge on gpt-5.6-luna at $0.20 in and $1.20 out per million tokens, gpt-6-luna at $0.10 and $0.50, both through the Responses API. Jev pinned to `jev-1.13.0`, Perplexity `pplx-decider-v1-27b` at $0.04 per million input tokens. Jev and gpt-5.6-luna ran as LangSmith online evaluators; Perplexity and gpt-6-luna ran on the direct path only, with their verdicts posted to the traces afterwards. Total spend as billed by the providers: about $2.80 on OpenAI for the agent, grading, the two LangSmith-run gpt-5.6-luna evaluations per trace and the direct-call passes; $0.28 on Jev, of which $0.12 was the LangSmith evaluator and $0.17 the direct calls; $0.73 on Perplexity. Gold grading used the SimpleQA grader prompt with gpt-5.6-luna, so a few of the 17 "wrong" answers may be grader or gold errors; I did not adjudicate them by hand.
