"""Perplexity Decisions API client: POST /v1/decisions. Same question and answer wire shapes as Jev.

Docs: https://docs.perplexity.ai/docs/decisions/quickstart and /api-reference/decisions-post.
Key from PERPLEXITY_API_KEY only; never logged.
"""
import os
import random
import time

import httpx

PPLX_MODEL = "pplx-decider-v1-27b"
URL = "https://api.perplexity.ai/v1/decisions"


def _key() -> str:
    key = os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY not set; create one in the Perplexity API Console and export it")
    return key


def ask(state, questions: dict, timeout: float = 60.0, max_attempts: int = 6) -> dict:
    """Return {"answers", "usage", "latency_s", "attempts", "status"}; retries 429/5xx honouring Retry-After."""
    body = {"model": PPLX_MODEL, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(URL, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            last = repr(e)
            time.sleep(min(10, 0.5 * 2 ** attempt) + random.random() * 0.3)
            continue
        lat = time.perf_counter() - t0
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(10, 0.5 * 2 ** attempt) + random.random() * 0.3)
            continue
        if r.status_code >= 400:
            return {"answers": None, "usage": {}, "latency_s": lat, "attempts": attempt, "status": r.status_code, "error": r.text[:500]}
        data = r.json()
        return {"answers": data.get("answers"), "usage": data.get("usage") or {}, "latency_s": lat, "attempts": attempt,
                "status": r.status_code}
    return {"answers": None, "usage": {}, "latency_s": None, "attempts": max_attempts, "status": None, "error": last}
