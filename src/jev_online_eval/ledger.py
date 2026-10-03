"""Cumulative spend ledger. Every script calls `check(service, projected)` before spending."""
import json
import os
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "results" / "ledger.json"
CAPS = {"openai": 10.0, "jev": 5.0, "perplexity": 5.0}
STOP_FRACTION = 0.8  # abort if cumulative + projected would pass 80% of the cap

PRICES = {  # USD per million tokens (input, output)
    "gpt-6-luna": (0.10, 0.50),  # not available on this key
    "gpt-5.6-luna": (0.20, 1.20),  # verified 2026-10-03 from public pricing trackers
    "jev-1.13.0": (0.042, 0.0),
    "pplx-decider-v1-27b": (0.04, 0.0),
}
JEV_OVERHEAD_TOKENS = 270


def _load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"openai": 0.0, "jev": 0.0, "perplexity": 0.0}


def spent(service: str) -> float:
    return _load().get(service, 0.0)


def add(service: str, usd: float) -> None:
    data = _load()
    data[service] = data.get(service, 0.0) + usd
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=2))


def check(service: str, projected: float, what: str) -> None:
    """Print projection and abort if it would cross the stop line."""
    cum = spent(service)
    cap = CAPS[service]
    print(f"[ledger] {what}: projected {service} spend ${projected:.4f}; "
          f"spent so far ${cum:.4f}; cap ${cap:.2f} (stop at {STOP_FRACTION:.0%})")
    if cum + projected > cap * STOP_FRACTION:
        print(f"[ledger] ABORT: ${cum + projected:.4f} would exceed stop line ${cap * STOP_FRACTION:.2f}")
        sys.exit(2)


def over_cap(service: str) -> bool:
    return spent(service) > CAPS[service] * STOP_FRACTION


def cost_openai(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def cost_pplx(input_tokens: int) -> float:
    return input_tokens * PRICES["pplx-decider-v1-27b"][0] / 1e6


def cost_jev(input_tokens: int) -> float:
    return (input_tokens + JEV_OVERHEAD_TOKENS) * PRICES["jev-1.13.0"][0] / 1e6
