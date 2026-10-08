"""Sync TMDB behavioral recommendations into the local database.

TMDB's ``/{movie,tv}/{id}/recommendations`` endpoint returns titles derived
from the real viewing/rating behavior of millions of TMDB users — the same
signal that makes TMDB's own recommendation UX feel
spot-on. This module caches that signal locally (``tmdb_recommendations``
table) so candidate recall becomes a plain SQL JOIN with no runtime API
dependency.

Two sync paths:
- **On demand**: the first time the recommendation engine uses a seed, its
  top-20 TMDB recommendations are fetched (bounded timeout) and stored.
  Subsequent queries hit the local table only.
- **Daily bulk**: a Celery task backfills popular titles that were never
  queried, a batch per day.

Failures degrade silently — no local rows just means the other recall
channels (genre+popularity, pgvector KNN) still work.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import aiohttp
from loguru import logger
from . import db

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"
RECOMMENDATIONS_PAGE_SIZE = 20  # TMDB returns 20 results per page
FETCH_TIMEOUT_SECONDS = 4.0  # on-demand budget per seed — engine latency matters
FETCH_CONCURRENCY = 3  # parallel on-demand fetches

# --- Redis coordination (shared across bot + worker processes) ---
NEGATIVE_TTL = 24 * 3600  # don't retry an empty/failed seed for a day
LOCK_TTL = 15  # in-flight fetch lock: bounded, self-expiring
_NEG_PREFIX = "tmdbrecs:neg:"
_LOCK_PREFIX = "tmdbrecs:lock:"

# RAM fallback for when Redis is unavailable (fail-open, per process).
_negative_cache: Dict[Tuple[int, str], float] = {}


async def _shared_cache():
    """The app's Redis cache client, or None when Redis is down."""
    try:
        from .cache import get_cache

        cache = await get_cache()
        return cache if cache.is_connected() else None
    except Exception:
        return None


async def _is_negatively_cached(media_id: int, media_type: str) -> bool:
    """Whether this seed is known to yield no recommendations (or failed).

    Redis-first so every process shares one answer; the per-process dict is
    only a fallback for when Redis is unavailable.
    """
    cache = await _shared_cache()
    if cache is not None:
        try:
            return bool(await cache.exists(f"{_NEG_PREFIX}{media_id}:{media_type}"))
        except Exception:
            pass  # fall through to RAM
    ts = _negative_cache.get((media_id, media_type))
    return ts is not None and (time.monotonic() - ts) < NEGATIVE_TTL


async def _mark_negative(media_id: int, media_type: str) -> None:
    """Remember for NEGATIVE_TTL that this seed has no fetchable recs."""
    _negative_cache[(media_id, media_type)] = time.monotonic()
    cache = await _shared_cache()
    if cache is not None:
        try:
            await cache.set(f"{_NEG_PREFIX}{media_id}:{media_type}", "1", ttl=NEGATIVE_TTL)
        except Exception:
            pass


async def _acquire_fetch_lock(media_id: int, media_type: str) -> bool:
    """Try to become the one process fetching this seed right now.

    Returns True when Redis is unavailable (fail-open — the world then
    degrades to at worst one duplicate fetch per process).
    """
    cache = await _shared_cache()
    if cache is None:
        return True
    try:
        return bool(await cache.set_nx(f"{_LOCK_PREFIX}{media_id}:{media_type}", "1", ttl=LOCK_TTL))
    except Exception:
        return True


async def _release_fetch_lock(media_id: int, media_type: str) -> None:
    cache = await _shared_cache()
    if cache is None:
        return
    try:
        await cache.delete(f"{_LOCK_PREFIX}{media_id}:{media_type}")
    except Exception:
        pass


async def has_local_recommendations(pool, media_id: int, media_type: str) -> bool:
    """Whether this seed already has synced recommendation rows.

    Args:
        pool: asyncpg pool.
        media_id: TMDB id of the seed.
        media_type: 'movie' or 'tv'.

    Returns:
        True if at least one recommendation row exists locally.
    """
    async with pool.acquire() as conn:
        row = await db.fetchrow(
            conn,
            "SELECT 1 FROM {t_tmdb_recommendations} "
            "WHERE (media_id, media_type) = ($1, $2) LIMIT 1",
            media_id,
            media_type,
        )
        return row is not None


async def fetch_recommendations_from_tmdb(
    session: aiohttp.ClientSession,
    media_id: int,
    media_type: str,
    fetch_with_retry=None,
) -> Optional[List[dict]]:
    """Fetch page 1 of TMDB's recommendations for one seed.

    Args:
        session: Shared aiohttp session.
        media_id: TMDB id of the seed.
        media_type: 'movie' or 'tv'.
        fetch_with_retry: Injectable fetcher (tests); defaults to the
            shared retrying fetch from engine.tmdb_client.

    Returns:
        List of {id, popularity} dicts (rank = list order), or None on
        failure / empty result.
    """
    from .tmdb_client import fetch_with_retry as _default_fetch

    fetch = fetch_with_retry or _default_fetch
    endpoint = "movie" if media_type == "movie" else "tv"
    data = await fetch(
        session,
        f"{TMDB_BASE}/{endpoint}/{media_id}/recommendations",
        {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1},
    )
    if not data or "error" in data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return [
        {"id": int(r["id"]), "popularity": float(r.get("popularity") or 0.0)}
        for r in results[:RECOMMENDATIONS_PAGE_SIZE]
        if isinstance(r, dict) and r.get("id") is not None
    ]


async def store_recommendations(pool, media_id: int, media_type: str, recs: List[dict]) -> int:
    """Upsert one seed's recommendation rows (rank = list order).

    Args:
        pool: asyncpg pool.
        media_id: TMDB id of the seed.
        media_type: 'movie' or 'tv'.
        recs: Ordered [{id, popularity}] from TMDB.

    Returns:
        Number of rows stored.
    """
    if not recs:
        return 0
    rows = [
        (media_id, media_type, r["id"], media_type, i + 1, r.get("popularity"))
        for i, r in enumerate(recs)
    ]
    async with pool.acquire() as conn:
        await db.executemany(conn,
            """
            INSERT INTO {t_tmdb_recommendations} (
                media_id, media_type, rec_media_id, rec_media_type,
                rank, popularity
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (media_id, media_type, rec_media_id, rec_media_type)
            DO UPDATE SET rank = EXCLUDED.rank,
                          popularity = EXCLUDED.popularity,
                          synced_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
    return len(rows)


async def sync_seed_recommendations(
    pool, media_id: int, media_type: str, session: Optional[aiohttp.ClientSession] = None
) -> int:
    """Fetch + store one seed's recommendations (best-effort).

    Guarded by a short Redis in-flight lock (SET NX) so concurrent queries
    for the same cold seed across processes trigger exactly one API call —
    the losers skip and let the winner's rows serve everyone.

    Args:
        pool: asyncpg pool.
        media_id: TMDB id of the seed.
        media_type: 'movie' or 'tv'.
        session: Optional shared aiohttp session; a short-lived one is
            created when omitted.

    Returns:
        Number of rows stored (0 on failure/empty/duplicate-in-flight —
        failures and empties are cached negatively in Redis).
    """
    if not await _acquire_fetch_lock(media_id, media_type):
        logger.debug(f"tmdb_recs: fetch lock held, skipping {media_type}:{media_id}")
        return 0
    try:

        async def _run(s: aiohttp.ClientSession) -> int:
            recs = await asyncio.wait_for(
                fetch_recommendations_from_tmdb(s, media_id, media_type),
                timeout=FETCH_TIMEOUT_SECONDS,  # honor the documented budget (was +10)
            )
            if recs is None:
                await _mark_negative(media_id, media_type)
                return 0
            return await store_recommendations(pool, media_id, media_type, recs)

        if session is not None:
            return await _run(session)
        async with aiohttp.ClientSession() as fresh:
            return await _run(fresh)
    except Exception as e:
        logger.debug(f"tmdb_recs: sync failed for {media_type}:{media_id}: {e}")
        await _mark_negative(media_id, media_type)
        return 0
    finally:
        await _release_fetch_lock(media_id, media_type)


async def ensure_seeds_synced(
    pool, seeds: Iterable[Tuple[int, str]], session: Optional[aiohttp.ClientSession] = None
) -> int:
    """On-demand hook: sync seeds that lack local rows (rate-bounded).

    Skips seeds already negatively cached (shared Redis answer). Parallel
    fetches are capped at FETCH_CONCURRENCY; each seed is bounded by
    FETCH_TIMEOUT_SECONDS so the engine's latency budget survives a cold
    seed.

    Args:
        pool: asyncpg pool.
        seeds: (media_id, media_type) pairs to check.
        session: Optional shared aiohttp session.

    Returns:
        Number of seeds newly synced (rows written > 0).
    """
    if not TMDB_API_KEY:
        return 0  # no key configured — behavioral channel stays local-only

    async def _needs_sync(seed: Tuple[int, str]) -> Optional[Tuple[int, str]]:
        media_id, media_type = seed
        if await _is_negatively_cached(media_id, media_type):
            return None
        if not await has_local_recommendations(pool, media_id, media_type):
            return seed
        return None

    checked = await asyncio.gather(*[_needs_sync(s) for s in seeds])
    todo: List[Tuple[int, str]] = [s for s in checked if s is not None]
    if not todo:
        return 0

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()

    async def _one(seed: Tuple[int, str]) -> int:
        async with sem:
            return await sync_seed_recommendations(pool, seed[0], seed[1], session=session)

    try:
        results = await asyncio.gather(*[_one(s) for s in todo], return_exceptions=True)
        synced = sum(1 for r in results if isinstance(r, int) and r > 0)
        if synced:
            logger.info(f"tmdb_recs: on-demand synced {synced}/{len(todo)} seeds")
        return synced
    finally:
        if owns_session:
            await session.close()
