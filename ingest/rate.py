"""Rate limiting — a real requests-per-second budget.

A semaphore caps CONCURRENCY, not RATE: 40 concurrent in-flight requests
against a fast API is a burst of hundreds per second. This token bucket
paces the caller to at most `rate` acquisitions per second, smoothly
(credit refills continuously, not in whole-second batches).
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token-bucket pacing: `acquire()` blocks until a token is available."""

    def __init__(self, rate_per_second: float, burst: int = 1) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self._rate = float(rate_per_second)
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(
            self._capacity, self._tokens + (now - self._updated) * self._rate
        )
        self._updated = now

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self._rate)
