# Jev for online evaluation

Online evaluation means a judge scores your production traces as they arrive, and the scores sit on the traces for dashboards and alerts. Almost nobody runs it on every trace. LangSmith's own guide suggests that "to control costs, you may want to set a filter to only apply the evaluator to 10% of traces" ([LangSmith docs](https://docs.langchain.com/langsmith/online-evaluations)). Langfuse describes the same habit: with LLM-as-a-judge at scale, "the caused costs were primarily contained through sampling" ([Langfuse](https://langfuse.com/blog/2026-09-23-catching-conversation-signals-in-langfuse)). I did the same on my own agents, sampled a few percent, and then stopped looking at those too.

In September 2026 TypeSafe released [Jev](https://typesafe.ai), a decision model that answers typed questions about a piece of context and returns probabilities instead of prose, at $0.042 per million input tokens. The launch-week posts from [LangChain](https://www.langchain.com/blog/jev-agent-evals-langsmith), [Openlayer](https://www.openlayer.com/blog/introducing-jevals-agent-evals-guardrails), [Arize](https://arize.com/blog/jev-as-a-judge/), [Langfuse](https://langfuse.com/blog/2026-09-22-running-evals-with-jev), [Datadog](https://datadoghq.com/blog/jev-evals-agent-observability) and [DeepEval](https://deepeval.com/integrations/models/typesafe-ai) show how to wire Jev in as a judge and measure its accuracy, cost and latency on recorded examples or benchmark sets. Several of them make the natural next argument: a judge this cheap and fast means you can stop sampling. What I could not find in any of them is the measurement that argument rests on. When Jev runs as an online evaluator on a live project at 100% sampling, how long after each trace does the score arrive, does every trace get one, what does it cost per trace, and are the scores worth having? This essay reports those four numbers for one real agent.

## Setup

I built a small web-research agent with Deep Agents on gpt-5.6-luna. It has two tools, `web_search` (DuckDuckGo via the ddgs library) and `fetch_page` (httpx plus trafilatura, pages truncated to 4K tokens), and a system prompt asking for a short cited answer. I sent it 300 questions from OpenAI's SimpleQA set, which are short factual questions written by people, each with a verified gold answer, and traced everything to one LangSmith project.

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

After the run I graded each final answer against the SimpleQA gold answer with the SimpleQA grading scheme, pulled every feedback item and every evaluator run back through the SDK, and separately sent the exact same rendered state to both judges directly, to measure call latency and repeatability without LangSmith's queue in the way. Both evaluator configurations as LangSmith saved them, every feedback item, the evaluator runs and the analysis are in the companion repository, [`adimyth/jev-online-eval`](https://github.com/adimyth/jev-online-eval).

## What it cost

<div className="wide-table">

| | Agent run | Jev evaluation | gpt-5.6-luna evaluation |
|---|---|---|---|
| Per trace, measured | $0.0038 | $0.00043 | $0.0019 |
| Per 1K traces | $3.77 | $0.43 | $1.91 |
| At 10K traces a day | $38 | $4.30 | $19 |

</div>

The Jev figure uses the input token count Jev reported when I sent it the same rendered state directly, about 7.5K tokens on average because the state carries fetched web pages, at list price. The LLM figure uses the token counts LangSmith recorded on the evaluator's own runs, about 9K input and 95 output tokens. For the whole experiment, 300 agent runs plus 287 evaluations by each judge, Jev cost 12 cents and the LLM judge 55 cents. LangSmith's evaluators tab shows both as a daily spend chart, so the running cost of each judge is visible without any of my scripts.

![Judge cost per evaluated trace](img/cost.png)

Jev is about 4.5× cheaper than gpt-5.6-luna configured the same way, not the 100× in the launch post, which compared against Claude Sonnet. Against a cheap modern LLM the judge was already affordable. The cost that dominates at scale is a different one: every online evaluator run upgrades the trace to extended data retention, which LangSmith bills at $5 per 1K traces. At 10K traces a day that is $50 a day, more than both judges combined, and it is the same whichever judge you pick. My project was on the long-lived tier from the start, so I did not pay it here. Budget for the traces before the judge.

## How fast the scores arrived

![Lag and latency](img/latency.png)

<div className="wide-table">

| | Jev | gpt-5.6-luna |
|---|---|---|
| Run end to score on trace, p50 | 69 s | 80 s |
| Run end to score on trace, p95 | 98 s | 108 s |
| Evaluator run inside LangSmith, p50 | 0.72 s | 3.7 s |
| Direct call, same state, p50 | 0.42 s | 2.09 s |
| Direct call, same state, p95 | 0.75 s | 3.41 s |

</div>

Jev the model is fast: 0.42 seconds at the median on 7.5K-token states, against 2.09 seconds for the LLM. Jev the online evaluator is barely faster than the LLM one, because both wait in the same LangSmith scheduling queue for a minute or more before either model is called. If you need scores within seconds of a trace landing, the queue is your bottleneck and Jev does not fix it. If you need them within a couple of minutes for dashboards and alerts, both work.

## Did every trace get scored

Of 300 requests, 287 finished and 13 hit the agent's recursion limit while looping on searches. The evaluators only fire on successful root runs, so neither judge scored those 13. The traces most likely to be interesting are the ones online evaluation skips.

Of the 287 finished traces, the LLM evaluator scored all 287 with all five keys. Jev scored 286. The one it missed was a 12-tool-call trace whose rendered state came to about 31.7K tokens, and the evaluator failed with "the model's context limit was exceeded". Jev's limit is 32K tokens and LangSmith maps the whole run output in by default, so long agent traces fall off the edge. Map a narrower variable than the whole output, as the error message says.

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

| Reference-free "is the answer correct" | Jev | gpt-5.6-luna |
|---|---|---|
| AUROC, correct vs incorrect | 0.83 | 0.72 |
| Wrong answers flagged at p < 0.5 | 2 of 17 | 0 of 17 |
| Correct answers flagged at p < 0.5 | 3 of 267 | 0 of 268 |
| Mean score on correct answers | 0.87 | 0.99 |
| Mean score on wrong answers | 0.67 | 0.96 |

</div>

The two judges agree on the binary verdict 98% of the time and on the outcome category every time, mostly on easy cases. The difference is in the numbers behind the verdicts. The LLM judge gave almost every answer a correctness probability of 0.96 or above, right or wrong, so its score tells you nothing you did not know. Jev gave wrong answers 0.6 to 0.9 and correct ones 0.95 or above. Neither gap is wide enough to flag a single bad trace, which is what the 0.5 threshold row shows. But averaged over a day of traffic, Jev's mean "correct" score would drop if the agent started getting more answers wrong, and the LLM judge's would not move. That is the difference between a score you can put on a drift chart and one you cannot.

Repeatability did not separate them. I sent 20 states six times each to both judges. Neither flipped a single binary verdict, and the LLM's "correct" probability moved less between repeats (standard deviation 0.003) than Jev's (0.008). The variance advantage the launch post measured against sampling LLM judges does not show up against gpt-5.6-luna with structured output on this task.

## Where Jev was wrong

Fifteen wrong answers got a Jev "correct" probability above 0.5, and three correct answers got one below.

- Most of the fifteen are answers that faithfully repeat a source that disagrees with the gold answer: 1941 instead of 1942 for a Columbia master's degree, 3:02:25 instead of 3:02:24 for a cycle race, "Pierre Ledoux" instead of "Paul Ledoux" for the 1972 Eddington Medal, 560 passengers instead of 583 at Tenerife because the agent excluded crew. A reference-free judge with the same web page in front of it cannot see these. Neither judge did, and only a judge with the gold answer could.
- A few are genuine misses of the kind TypeSafe documents, arithmetic and dates: the Dark Souls patch dated 23 October against a gold of 22 October got 0.60, and the passenger count got 0.91 despite the answer itself saying the total "including crew" was different.
- The three false alarms, 0.45 to 0.49 on correct answers about Delhi's forest cover, Pavlov's psychic secretion and Oprah's 164 acres, look like Jev being unsure on numeric answers rather than wrong. The LLM gave all three 0.97 or more.
- Jev's "grounded" score correlated only 0.65 with the LLM's and ran lower on average, 0.94 against 0.99. I read the six lowest Jev scores expecting to find unsupported claims. Five of the six were correct answers that quoted their source, such as Pavlov for psychic secretion at 0.57 and a Terraria patch name at 0.67. Only the lowest, a forest-cover figure at 0.18, was one both judges doubted. So the lower Jev scores look like hesitation on short numeric or name answers, not detected hallucination, and I would not alert on them.

Jev cannot explain itself, so every one of these took a human reading the trace. The LLM judge's one-line comment was accurate about what it had looked at on all 17 wrong answers, saying for example that 3:02:25 "matches the fetched Wikipedia event table", and then scored the answer correct anyway. The explanation was right about the evidence and told me nothing about the verdict.

## What to take away

If you have been sampling a few percent of traces because the judge was too expensive, you can stop. Jev scores a trace with five questions for about four hundredths of a cent, and even gpt-5.6-luna does it for a fifth of a cent. At 10K traces a day that is $4 or $19, either of which is small next to the $38 the agent itself costs.

Three things to check before you switch it on:

1. **The platform bill, not the judge bill.** On LangSmith, every evaluated trace moves to extended retention at $5 per 1K traces. At 10K traces a day that is $50, ten times the Jev cost. Price the traces first.
2. **What you will do with a score that arrives a minute late.** Both judges landed on the trace 70 to 110 seconds after the run ended, almost all of it LangSmith's queue. Fine for dashboards and daily alerts. Not fine for blocking or routing a live response; that needs a call from inside the agent, where Jev's 0.4 seconds does matter.
3. **Which traces it will skip.** Failed runs were not evaluated at all, and one long trace overflowed Jev's 32K context. Decide what you want to happen to those before you trust the coverage number.

On what the scores are worth: for the reference-free question "is this answer correct", Jev's probability is a usable drift signal and the LLM judge's is not, because the LLM says 0.98 to everything. Neither catches an individual wrong answer, because most wrong answers here were faithful summaries of a web page that disagreed with the reference. Use Jev online for aggregate quality tracking and for structural questions like "did the agent answer", and keep a reference-based check for per-answer correctness.

This experiment says nothing about whether Jev can judge multi-step agent behaviour such as wrong tool choices or policy violations. That needs an agent with known-correct trajectories, and it is the next test to run.

## Method notes

300 SimpleQA questions, fixed random subset, 6 concurrent agent runs, 21.7 minutes of wall clock. Agent and LLM judge on gpt-5.6-luna at $0.20 in and $1.20 out per million tokens, through the Responses API. Jev pinned to `jev-1.13.0`. Total spend: $1.89 on OpenAI including grading and the direct-call pass, $0.17 on Jev. Gold grading used the SimpleQA grader prompt with gpt-5.6-luna, so a few of the 17 "wrong" answers may be grader or gold errors; I did not adjudicate them by hand.
