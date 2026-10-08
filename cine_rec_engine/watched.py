"""Per-user watch history for recommendation filtering.

The engine never recommends a title the requesting user already watched.
"Watched" = rows in YOUR user_watches table (or any table you point
USER_WATCHES_SQL at) — one row per (user, tmdb_id, media_type) watch event.

Exclusion keys are (tmdb_id, media_type) pairs, NOT bare ids: numeric ids
collide between movies and series in TMDB (e.g. 69557 is both "The
Pumpkin Eater" movie and "Fauda" tv), so a bare-id set would drop
innocent titles. The set is cached per user for 15 minutes — watch events
during a session don't need instant visibility.
"""

import logging
from typing import Optional, Set, Tuple

from .cache import get_cache
from . import db

logger = logging.getLogger(__name__)

CACHE_PREFIX = "watched:"
CACHE_TTL = 900  # 15 minutes

WatchedSet = Set[Tuple[int, str]]

# The engine is DB-agnostic: it reads whatever *your* database records as
# "this user watched this title". Map user_watches to your own events
# table (orders, plays, downloads, ratings — anything) via the table
# registry (CINE_REC_SCHEMA_MAP / CINE_REC_TABLE_USER_WATCHES) as long
# as it yields (tmdb_id bigint, media_type 'movie'|'tv') for $1 = user_id
# — column names are the contract, alias them in a thin view if yours
# differ. docs/user_data.sql ships the ready-to-use default.
USER_WATCHES_SQL = """
    SELECT DISTINCT tmdb_id::bigint, media_type
    FROM {t_user_watches}
    WHERE user_id = $1
"""


async def get_user_watched(pool, user_id: Optional[int]) -> WatchedSet:
    """(tmdb_id, media_type) pairs the user already watched. Empty on failure.

    Resolution: Redis cache (15 min) → PostgreSQL → cache write-back. A
    history lookup must never break a recommendation.
    """
    if user_id is None or pool is None:
        return set()

    cache_key = f"{CACHE_PREFIX}{user_id}"
    try:
        cache = await get_cache()
        cached = await cache.get(cache_key)
        if cached:
            # json lists load back as lists of [id, type] — tuples for use
            return {(int(i), str(t)) for i, t in cached}
    except Exception as e:
        logger.debug(f"watched cache read failed for {user_id}: {e}")

    watched: WatchedSet = set()
    try:
        rows = await db.fetch(pool, USER_WATCHES_SQL, user_id)
        watched = {(int(r["tmdb_id"]), str(r["media_type"] or "")) for r in rows}
    except Exception as e:
        logger.debug(f"watched DB read failed for {user_id}: {e}")
        return set()

    try:
        cache = await get_cache()
        await cache.set(cache_key, sorted(watched), ttl=CACHE_TTL)
    except Exception:
        pass
    return watched


_RATED_SQL = """
    SELECT tmdb_id::bigint AS tmdb_id, media_type
    FROM {t_title_ratings}
    WHERE user_id = $1
"""

_RATED_CACHE_PREFIX = "rated:"


async def get_user_rated(pool, user_id: Optional[int]) -> WatchedSet:
    """(tmdb_id, media_type) pairs the user rated (any rating) — 15 min cache."""
    if user_id is None or pool is None:
        return set()

    cache_key = f"{_RATED_CACHE_PREFIX}{user_id}"
    try:
        cache = await get_cache()
        cached = await cache.get(cache_key)
        if cached:
            return {(int(i), str(t)) for i, t in cached}
    except Exception as e:
        logger.debug(f"rated cache read failed for {user_id}: {e}")

    rated: WatchedSet = set()
    try:
        rows = await db.fetch(pool, _RATED_SQL, user_id)
        rated = {(int(r["tmdb_id"]), str(r["media_type"] or "")) for r in rows}
    except Exception as e:
        logger.debug(f"rated DB read failed for {user_id}: {e}")
        return set()

    try:
        cache = await get_cache()
        await cache.set(cache_key, sorted(rated), ttl=CACHE_TTL)
    except Exception:
        pass
    return rated


async def get_user_recommendation_exclusions(pool, user_id: Optional[int]) -> WatchedSet:
    """Titles hidden from a user's recommendations: watched OR rated.

    A rated title never returns in that user's recommendation lists, no
    matter which button was pressed — rating signals we already know the
    verdict, recommending it again is noise.
    """
    watched = await get_user_watched(pool, user_id)
    rated = await get_user_rated(pool, user_id)
    return watched | rated


_PROGRESS_MOVIE_SQL = """
    SELECT EXISTS (
        SELECT 1
        FROM {t_user_watches}
        WHERE user_id = $1 AND tmdb_id = $2::bigint AND media_type = 'movie'
    ) AS watched
"""


async def get_user_title_progress(pool, user_id, tmdb_id: int, media_type: str) -> dict:
    """The user's watch state for ONE title (card rendering).

    movie → {"watched": bool}
    tv → {"watched", "watched_all", "reached_last", "max_season", "max_episode"}:
      - watched_all: every catalog episode seen
      - reached_last: the catalog's last episode (highest season, then
        highest episode in it) is seen — even with gaps
      - max_season/max_episode: furthest progress (season DESC priority),
        for the "עד עונה X פרק X" label
    """
    out = {"watched": False}
    if user_id is None or pool is None:
        return out
    try:
        if media_type == "tv":
            row = await db.fetchrow(pool,
                """
                WITH seen AS (
                    SELECT uw.season, uw.episode
                    FROM {t_user_watches} uw
                    WHERE uw.user_id = $1
                      AND uw.tmdb_id = $2::bigint
                      AND uw.media_type = 'tv'
                      AND uw.season >= 1 AND uw.episode >= 1
                    GROUP BY uw.season, uw.episode
                ),
                catalog AS (
                    SELECT MAX(e.season) AS cat_s,
                           MAX(e.episode) AS cat_e,
                           COUNT(DISTINCT (e.season, e.episode)) AS cat_n
                    FROM {t_media_episodes} e
                    WHERE e.tmdb_id = $2::bigint
                      AND e.season >= 1 AND e.episode >= 1
                )
                SELECT
                    (SELECT season FROM seen ORDER BY season DESC, episode DESC LIMIT 1) AS max_s,
                    (SELECT episode FROM seen ORDER BY season DESC, episode DESC LIMIT 1) AS max_e,
                    (SELECT COUNT(*) FROM seen) AS seen_n,
                    catalog.cat_s, catalog.cat_e, catalog.cat_n
                FROM catalog
                """,
                user_id,
                str(tmdb_id),
            )
            if row and row["seen_n"] and row["seen_n"] > 0:
                max_s, max_e = row["max_s"], row["max_e"]
                reached_last = (max_s, max_e) >= (row["cat_s"], row["cat_e"])
                out = {
                    "watched": True,
                    "watched_all": row["seen_n"] >= row["cat_n"] and row["cat_n"] > 0,
                    "reached_last": reached_last,
                    "max_season": max_s,
                    "max_episode": max_e,
                }
            return out
        row = await db.fetchrow(pool, _PROGRESS_MOVIE_SQL, user_id, str(tmdb_id))
        if row:
            out["watched"] = bool(row["watched"])
        return out
    except Exception as e:
        logger.debug(f"progress read failed for {user_id}/{media_type}/{tmdb_id}: {e}")
        return out


async def get_title_rating(pool, user_id, tmdb_id: int, media_type: str) -> Optional[str]:
    """The user's current rating ('love'/'like'/'dislike') or None."""
    if user_id is None or pool is None:
        return None
    try:
        row = await db.fetchrow(pool,
            "SELECT rating FROM {t_title_ratings} "
            "WHERE user_id = $1 AND tmdb_id = $2 AND media_type = $3",
            user_id,
            tmdb_id,
            media_type,
        )
        return row["rating"] if row else None
    except Exception as e:
        logger.debug(f"rating read failed: {e}")
        return None

_DISLIKES_SQL = """
    SELECT tmdb_id::bigint, media_type FROM {t_user_feedback}
    WHERE user_id = $1 AND kind = 'dislike'
"""


async def get_user_dislikes(pool, user_id: Optional[int]) -> WatchedSet:
    """Explicit dislikes — hard-filtered from every recommendation (§F.2)."""
    if user_id is None or pool is None:
        return set()
    try:
        rows = await db.fetch(pool, _DISLIKES_SQL, user_id)
        return {(int(r["tmdb_id"]), str(r["media_type"])) for r in rows}
    except Exception as e:
        logger.debug(f"dislikes read failed for {user_id}: {e}")
        return set()
