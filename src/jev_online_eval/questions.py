"""The five questions asked of every trace, by both judges.

Wording adapted from Openlayer's jevals (MIT): UsedToolResult, Refusal, Quality.
`correct` is reference-free on purpose: an online evaluator has no gold answer.
State fields: request, tool_results (compact list of search/fetch results), final_answer.
"""

QUESTIONS = {
    "answered": {
        "type": "noul",
        "instructions": "Did final_answer give a direct answer to request, rather than declining, hedging without an answer, or saying it could not find one?",
        "criteria": {
            "true": "A specific answer to the question is stated",
            "false": "No specific answer is given: the assistant declined, said it could not find one, or only described what it tried",
        },
    },
    "grounded": {
        "type": "noul",
        "instructions": "Does final_answer use the information in tool_results? It should reflect what the tools returned, not ignore or contradict it.",
        "criteria": {
            "true": "The answer restates or follows directly from something in tool_results",
            "false": "The answer adds facts, names, numbers or dates that tool_results do not contain, or contradicts them",
        },
    },
    "correct": {
        "type": "noul",
        "instructions": "Is final_answer factually correct as an answer to request? Judge from tool_results and general knowledge.",
        "criteria": {
            "true": "The stated answer is the right one",
            "false": "The stated answer is wrong, or no answer was given",
        },
    },
    "confidence": {
        "type": "score",
        "instructions": "How definite is final_answer about its answer?",
        "criteria": [
            "No answer or explicit inability to answer",
            "Hedged: offers a candidate but flags real doubt",
            "Fairly definite: states an answer with mild caveats",
            "Definite: states the answer plainly with a source",
        ],
    },
    "outcome": {
        "type": "choice",
        "instructions": "How did the assistant respond to request?",
        "criteria": {
            "answered": "Provided a specific answer",
            "could_not_find": "Said it could not find reliable evidence and gave no answer",
            "partial": "Gave an answer to part of the question or a vague approximation",
            "refused": "Declined for policy or safety reasons",
        },
    },
}

# Plain-text rendering for the LLM judge prompt and for pasting into the LangSmith UI.
def render_for_prompt() -> str:
    lines = []
    for k, q in QUESTIONS.items():
        lines.append(f"{k} ({q['type']}): {q['instructions']}")
        c = q["criteria"]
        if isinstance(c, list):
            for i, lvl in enumerate(c):
                lines.append(f"  level {i}: {lvl}")
        else:
            for opt, desc in c.items():
                lines.append(f"  {opt}: {desc}")
    return "\n".join(lines)
