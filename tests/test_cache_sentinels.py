"""Cache sentinel semantics: an EMPTY cached result is a HIT.

Both directions have bitten: `if cached:` turned every user with no
history into a permanent cache miss for the watched/rated exclusion
sets (one Postgres query per request), and `if val:` on the page-row
write side never cached legitimately empty rows.
"""

from __future__ import annotations

import asyncio

from cine_rec_engine import watched


class FakeCache:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value


def _patch(monkeypatch, cache):
    async def _get_cache():
        return cache

    async def _no_db(*args, **kwargs):
        raise AssertionError("a cached empty result must not hit the DB")

    monkeypatch.setattr(watched, "get_cache", _get_cache)
    monkeypatch.setattr(watched.db, "fetch", _no_db)


def test_empty_watched_set_is_a_cache_hit(monkeypatch):
    cache = FakeCache()
    cache.store["watched:7"] = []  # user 7 has no history — valid, cached
    _patch(monkeypatch, cache)
    assert asyncio.run(watched.get_user_watched(object(), 7)) == set()


def test_empty_rated_set_is_a_cache_hit(monkeypatch):
    cache = FakeCache()
    cache.store["rated:7"] = []
    _patch(monkeypatch, cache)
    assert asyncio.run(watched.get_user_rated(object(), 7)) == set()
