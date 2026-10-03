"""Direct Jev client: POST /v1/systemone with retries on 429/529, returns answers, usage and latency."""
import os
import random
import time

import httpx

JEV_MODEL = "jev-1.13.0"
URL = "https://api.typesafe.ai/v1/systemone"


def _key() -> str:
    key = os.environ.get("TYPESAFE_AI_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_AI_KEY not set")
    return key


def ask(state: dict, questions: dict, timeout: float = 20.0, max_attempts: int = 6) -> dict:
    """Return {"answers", "usage", "latency_s", "attempts", "status"}."""
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(URL, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            last = repr(e)
            time.sleep(min(8, 0.5 * 2 ** attempt) + random.random() * 0.3)
            continue
        lat = time.perf_counter() - t0
        if r.status_code in (429, 529) or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(min(8, 0.5 * 2 ** attempt) + random.random() * 0.3)
            continue
        if r.status_code >= 400:
            return {"answers": None, "usage": {}, "latency_s": lat, "attempts": attempt,
                    "status": r.status_code, "error": r.text[:500]}
        data = r.json()
        return {"answers": data.get("answers"), "usage": data.get("usage") or {}, "latency_s": lat,
                "attempts": attempt, "status": r.status_code, "raw": data}
    return {"answers": None, "usage": {}, "latency_s": None, "attempts": max_attempts, "status": None, "error": last}
