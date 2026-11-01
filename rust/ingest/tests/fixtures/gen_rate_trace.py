#!/usr/bin/env python3
"""Generate rate_trace.json — a golden trace of the REAL
ingest/rate.py RateLimiter under a monkeypatched clock and sleep.

No real time ever passes: time.monotonic returns a script variable
and asyncio.sleep records its requested duration, pops the next
scripted clock advance, advances the fake clock and returns.
Scenarios deliberately include partial advances (forcing the acquire
loop to sleep more than once), a backwards clock, and a NaN rate.

The Rust test replays the same scripted advances against the pure
TokenBucket and must reproduce every wait and token level BIT-EXACTLY
(same float ops in the same order — see src/rate.rs).

FLOAT BLACK HOLE (found the hard way): when the advance script runs
dry, the fallback advances by max(requested_wait, ulp(t)). Advancing
by the bare requested wait can be a NO-OP — a wait of 2.22e-17 s added
to t=0.8 rounds away (below half an ulp of t), the clock never moves,
the token level never recovers, and the real RateLimiter loop spins
forever. A real clock always advances, so production Python never
sees this; a scripted clock must guarantee at least one representable
step. The WAITS recorded in the golden are unaffected — they are
computed by the real RateLimiter from the token level, never from the
advance rule. The Rust replay must use the SAME fallback.

A NaN rate is written as the string "nan" (JSON cannot carry NaN);
every recorded number is finite.

Run from the repo root with the wt3 venv:
    .venv/bin/python rust/ingest/tests/fixtures/gen_rate_trace.py
"""

import asyncio
import json
import math
import time
from pathlib import Path

import ingest.rate as rate_mod

OUT = Path(__file__).resolve().parent / "rate_trace.json"

# Scenario spec: (name, rate, burst, t0, acquire_count, advances).
# advances are seconds popped per sleep call; when dry, the exact
# requested duration is granted. Values chosen to exercise: burst
# drain, partial advances (multi-sleep acquires), one-step waits, a
# backwards clock (negative advance), and long gaps capped at
# capacity.
SCENARIOS = [
    ("steady-10rps", 10.0, 1, 1000.0, 6,
     [0.05, 0.03, 0.099, 0.2, 0.1, 0.1, 0.1, 0.1]),
    ("burst2-then-pace", 5.0, 2, 0.0, 6,
     [0.01, 0.01, 0.19, 0.2, 0.2]),
    ("partial-wakes", 4.0, 3, 42.0, 8,
     [0.02, 0.02, 0.02, 0.1, 0.1, 0.25, 0.249, 0.25, 0.25, 0.25]),
    ("backwards-clock", 8.0, 2, 10.0, 4, [-0.5, 0.3, 0.2]),
    ("long-gap-caps", 2.0, 2, 0.0, 5, [500.0, 500.0, 0.4, 0.5]),
    ("slow-frac-rate", 0.25, 1, 7.5, 4, [1.0, 1.5, 3.0, 3.9, 4.0]),
    ("nan-rate-unlimited", "nan", 2, 3.0, 10, []),
]


async def drive(name, rate, burst, t0, n, advances):
    state = {"t": float(t0)}
    script = list(advances)
    current = {"sleeps": []}

    real_monotonic = time.monotonic
    real_sleep = asyncio.sleep

    def fake_monotonic():
        return state["t"]

    async def fake_sleep(d):
        current["sleeps"].append(d)
        if script:
            state["t"] += script.pop(0)
        else:
            # At least one representable step — see the module docs.
            state["t"] += max(d, math.ulp(state["t"]))

    per_acquire = []
    time.monotonic = fake_monotonic
    asyncio.sleep = fake_sleep
    try:
        lim = rate_mod.RateLimiter(float(rate), int(burst))
        for _ in range(n):
            current["sleeps"] = []
            await lim.acquire()
            per_acquire.append({
                "at": state["t"],
                "tokens": lim._tokens,
                "sleeps": current["sleeps"],
            })
        return {"name": name, "rate": rate, "burst": burst, "t0": t0,
                "advances": list(advances), "acquires": per_acquire}
    finally:
        time.monotonic = real_monotonic
        asyncio.sleep = real_sleep


async def main():
    scenarios = []
    for name, rate, burst, t0, n, advances in SCENARIOS:
        scenarios.append(await drive(name, rate, burst, t0, n, advances))

    # Constructor validation — Python's exact ValueError message.
    ctor_errors = []
    for bad in (0, -1, -0.001):
        try:
            rate_mod.RateLimiter(bad)
        except ValueError as e:
            ctor_errors.append({"rate": bad, "message": str(e)})
        else:
            raise AssertionError(f"rate={bad} unexpectedly accepted")

    doc = {
        "source": "ingest/rate.py (real implementation, clock+sleep patched)",
        "scenarios": scenarios,
        "ctor_errors": ctor_errors,
    }
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    n_sleeps = sum(len(a["sleeps"]) for s in scenarios for a in s["acquires"])
    print(f"wrote {OUT.name}: {len(scenarios)} scenarios, "
          f"{sum(len(s['acquires']) for s in scenarios)} acquires, "
          f"{n_sleeps} sleeps")


if __name__ == "__main__":
    asyncio.run(main())
