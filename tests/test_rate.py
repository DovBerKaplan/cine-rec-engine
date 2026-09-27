"""Token-bucket rate limiter — actual pacing, not just concurrency."""

import time

import pytest

from ingest.rate import RateLimiter


async def test_no_throttle_under_budget():
    rl = RateLimiter(rate_per_second=1000)
    start = time.monotonic()
    for _ in range(10):
        await rl.acquire()
    assert time.monotonic() - start < 0.2


async def test_paces_to_budget():
    rl = RateLimiter(rate_per_second=50)  # 20ms between acquires
    start = time.monotonic()
    for _ in range(6):
        await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 5 * 0.02 - 0.01  # ~5 gaps of >=20ms (clock tolerance)


async def test_rejects_bad_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)
