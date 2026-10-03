"""gpt-5.6-luna as a conventional LLM judge answering the same five questions with structured output.

Used by the direct-latency side script. The online version is configured in the LangSmith UI
with the same prompt text (see render_prompt()).
"""
import json
import os
import time

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .questions import QUESTIONS, render_for_prompt

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.6-luna")


class Verdict(BaseModel):
    answered: float = Field(description="probability 0-1 that the statement is true")
    grounded: float = Field(description="probability 0-1")
    correct: float = Field(description="probability 0-1")
    confidence: int = Field(description="level index 0-3")
    outcome: str = Field(description="one of answered, could_not_find, partial, refused")


def render_prompt() -> str:
    return (
        "You are evaluating one run of a research assistant. Answer every question below about the STATE.\n"
        "Return probabilities for noul questions, a level index for score questions, and an option name for choice questions.\n\n"
        "QUESTIONS\n" + render_for_prompt() + "\n\nSTATE\n{state}"
    )


_llm = None


def _model():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=JUDGE_MODEL, max_retries=3, timeout=60).with_structured_output(
            Verdict, include_raw=True)
    return _llm


def ask(state: dict) -> dict:
    prompt = render_prompt().replace("{state}", json.dumps(state, ensure_ascii=False))
    t0 = time.perf_counter()
    res = _model().invoke(prompt)
    lat = time.perf_counter() - t0
    raw = res["raw"]
    um = raw.usage_metadata or {}
    v = res["parsed"]
    answers = None if v is None else v.model_dump()
    return {"answers": answers, "usage": {"input_tokens": um.get("input_tokens", 0),
            "output_tokens": um.get("output_tokens", 0)}, "latency_s": lat,
            "error": None if v is not None else str(res.get("parsing_error"))[:300]}
