"""Cache layer for the engine: Redis when available, in-process TTL otherwise.

The engine touches the cache through a tiny async surface —
``get / set / set_nx / exists / delete / is_connected`` — so any client
implementing it plugs in (we ship one below). If a Redis URL is
configured via ``CINE_REC_REDIS_URL`` we lazily connect with ``redis``
(asyncio); otherwise a per-process TTL dict keeps everything working
with zero infrastructure.

Every cache call in the engine is wrapped in ``try/except`` and degrades
on failure — a cache outage must never break a recommendation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

_client: Any = None
_client_checked = False
_fallback: Optional["_TTLCache"] = None


class _TTLCache:
    """Minimal async cache with TTL + NX semantics (single-process)."""

    def __init__(self) -> None:
        self._data: dict = {}
        self._lock = asyncio.Lock()

    def is_connected(self) -> bool:
        return True

    async def get(self, key: str) -> Any:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if expires is not None and expires < time.monotonic():
                del self._data[key]
                return None
            # JSON round-trip so callers see the same shape as Redis gives
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Any:
        async with self._lock:
            serialized = json.dumps(value, default=str)
            self._data[key] = (serialized, time.monotonic() + ttl if ttl is not None else None)
            return True

    async def set_nx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            item = self._data.get(key)
            if item is not None:
                expires = item[1]
                if expires is None or expires >= time.monotonic():
                    return False
                del self._data[key]
            self._data[key] = (json.dumps(value, default=str),
                               time.monotonic() + ttl if ttl is not None else None)
            return True

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)


async def get_cache() -> Any:
    """Return a cache client, or None if nothing is available.

    Priority: Redis (``CINE_REC_REDIS_URL``) → in-process TTL cache.
    Redis is probed once per process; afterwards the decision sticks
    unless a live connection errors out.
    """
    global _client, _client_checked, _fallback

    if not _client_checked:
        _client_checked = True
        url = os.getenv("CINE_REC_REDIS_URL")
        if url:
            try:
                import redis.asyncio as aioredis  # type: ignore

                _client = aioredis.from_url(
                    url, decode_responses=True, socket_timeout=2,
                    socket_connect_timeout=2,
                )
                await _client.ping()
            except Exception:
                _client = None

    if _client is not None:
        try:
            if await _client.ping():
                return _client
        except Exception:
            _client = None  # Redis died mid-flight — fall through to local
    elif _fallback is not None:
        # Redis was already ruled out this process; stay local without
        # re-probing on every call (a recommendation path must not ping).
        return _fallback

    if _fallback is None:
        _fallback = _TTLCache()
    return _fallback
