"""Cache shim tests — the in-process fallback used when Redis is absent."""

import asyncio

from cine_rec_engine.cache import _TTLCache


async def test_set_get_roundtrip():
    c = _TTLCache()
    await c.set("k", [1, 2], ttl=60)
    assert await c.get("k") == [1, 2]


async def test_ttl_expiry():
    c = _TTLCache()
    await c.set("k", "v", ttl=0)
    await asyncio.sleep(0.01)
    assert await c.get("k") is None


async def test_set_nx_semantics():
    c = _TTLCache()
    assert await c.set_nx("lock", "1", ttl=60) is True   # first taker wins
    assert await c.set_nx("lock", "1", ttl=60) is False  # second blocked
    await c.delete("lock")
    assert await c.set_nx("lock", "1", ttl=60) is True   # released


async def test_exists_and_delete():
    c = _TTLCache()
    await c.set("k", {"a": 1}, ttl=60)
    assert await c.exists("k") is True
    await c.delete("k")
    assert await c.exists("k") is False


async def test_missing_key_is_none():
    c = _TTLCache()
    assert await c.get("nope") is None
