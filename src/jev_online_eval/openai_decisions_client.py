"""OpenAI Decisions API client: POST https://api.openai.com/v1/decisions (public beta, gpt-6-luna).

Request: {model, input, questions:[{type, name, instructions, choices|levels}]}; answers come back in
question order as a list. This module converts the Jev-style question dict used everywhere else in the
project (noul / choice / score with `criteria`) into OpenAI's shape, and converts answers back to the
Jev answer shape ({"type":"noul","noul":p}, {"type":"choice","choice":...}, {"type":"score","score":...})
so the analysis code can treat every decision model alike. Key from OPENAI_API_KEY only; never logged.
"""
import os
import random
import time

import httpx

MODEL = "gpt-6-luna"
URL = "https://api.openai.com/v1/decisions"


def to_openai_questions(questions: dict) -> list:
    out = []
    for name, q in questions.items():
        if q["type"] == "noul":
            instr = q.get("instructions", "")
            crit = q.get("criteria")
            if crit:
                instr = f"{instr} True: {crit.get('true', '')}. False: {crit.get('false', '')}."
            out.append({"type": "predicate", "name": name, "instructions": instr})
        elif q["type"] == "choice":
            out.append({"type": "choice", "name": name, "instructions": q["instructions"],
                        "choices": [{"value": k, "description": v} for k, v in q["criteria"].items()]})
        elif q["type"] == "score":
            out.append({"type": "score", "name": name, "instructions": q["instructions"],
                        "levels": [{"label": lvl} for lvl in q["criteria"]]})
    return out


def from_openai_answers(answers: list) -> dict:
    out = {}
    for a in answers:
        n = a.get("name")
        if a["type"] == "predicate":
            out[n] = {"type": "noul", "noul": a["probability"]}
        elif a["type"] == "choice":
            out[n] = {"type": "choice", "choice": a["choice"], "confidence": a.get("confidence"),
                      "probabilities": {p["value"]: p["probability"] for p in a.get("probabilities") or []}}
        elif a["type"] == "score":
            out[n] = {"type": "score", "score": a["score"], "confidence": a.get("confidence")}
        else:
            out[n] = {"type": a["type"]}
    return out


def ask(state, questions: dict, timeout: float = 60.0, max_attempts: int = 6) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = {"model": MODEL, "input": state if isinstance(state, str) else str(state), "questions": to_openai_questions(questions)}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(URL, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            last = repr(e); time.sleep(min(10, 0.5 * 2 ** attempt) + random.random() * 0.3); continue
        lat = time.perf_counter() - t0
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(10, 0.5 * 2 ** attempt) + random.random() * 0.3)
            continue
        if r.status_code >= 400:
            return {"answers": None, "usage": {}, "latency_s": lat, "attempts": attempt, "status": r.status_code, "error": r.text[:500]}
        data = r.json()
        return {"answers": from_openai_answers(data.get("answers") or []), "usage": data.get("usage") or {},
                "latency_s": lat, "attempts": attempt, "status": r.status_code}
    return {"answers": None, "usage": {}, "latency_s": None, "attempts": max_attempts, "status": None, "error": last}
