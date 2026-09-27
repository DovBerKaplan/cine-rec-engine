"""User statistics layer (spec v0.2 §B.3, §C, §G).

Owns every write to user_title_stats / user_stats / user_genre_stats:
- record_event: raw append + same-transaction title-stats recompute and
  user_stats.last_event_at bump (acceptance rule §J).
- refresh_user_stats: full recompute of the per-user profile + genres.
- nightly_recompute: recency (hence w_i) refresh for active users, vector
  rebuild for stale ones (§G.2).

The raw events are the source of truth; every derived table here can be
rebuilt from them at any time.
"""

from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger

from . import user_weights as uw

EPISODE_THRESHOLD_RATIO = 0.50   # an episode counts as watched at ≥ 50%

_UPSERT_TITLE_STATS = """
    INSERT INTO user_title_stats (
        user_id, tmdb_id, media_type, sessions, total_watched_sec, max_ratio,
        pause_count_total, rewatch_count, last_watched_at, episodes_watched,
        last_season, last_episode, last_ep_watched_sec, last_ep_duration_sec,
        dropped, w_item, updated_at
    )
    SELECT
        $1, $2, $3,
        COUNT(*),
        COALESCE(SUM(e.watched_sec), 0),
        MAX(CASE WHEN COALESCE(e.duration_sec, 0) > 0
                 THEN e.watched_sec::float / e.duration_sec END),
        COALESCE(SUM(e.pause_count), 0),
        -- rewatch = a session of a unit already completed earlier
        -- (same season/episode for tv; the movie itself for movies) —
        -- continuing to the NEXT episode is progress, never a rewatch.
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM user_watch_events p
            WHERE p.user_id = e.user_id AND p.tmdb_id = e.tmdb_id
              AND p.media_type = e.media_type AND p.completed
              AND p.watched_at < e.watched_at
              AND p.season IS NOT DISTINCT FROM e.season
              AND p.episode IS NOT DISTINCT FROM e.episode)),
        MAX(e.watched_at),
        COUNT(DISTINCT CASE WHEN e.completed
              OR (COALESCE(e.duration_sec,0) > 0
                  AND e.watched_sec::float / e.duration_sec >= $4)
              THEN (e.season, e.episode) END),
        last_ev.season,
        last_ev.episode,
        last_ev.watched_sec,
        last_ev.duration_sec,
        FALSE, 0, now()
    FROM user_watch_events e
    LEFT JOIN LATERAL (
        SELECT h.season, h.episode, h.watched_sec, h.duration_sec
        FROM user_watch_events h
        WHERE h.user_id = e.user_id AND h.tmdb_id = e.tmdb_id
          AND h.media_type = e.media_type
        ORDER BY h.watched_at DESC LIMIT 1
    ) last_ev ON true
    WHERE e.user_id = $1 AND e.tmdb_id = $2 AND e.media_type = $3
    GROUP BY last_ev.season, last_ev.episode,
             last_ev.watched_sec, last_ev.duration_sec
    ON CONFLICT (user_id, tmdb_id, media_type) DO UPDATE SET
        sessions = EXCLUDED.sessions,
        total_watched_sec = EXCLUDED.total_watched_sec,
        max_ratio = EXCLUDED.max_ratio,
        pause_count_total = EXCLUDED.pause_count_total,
        rewatch_count = EXCLUDED.rewatch_count,
        last_watched_at = EXCLUDED.last_watched_at,
        episodes_watched = EXCLUDED.episodes_watched,
        last_season = EXCLUDED.last_season,
        last_episode = EXCLUDED.last_episode,
        last_ep_watched_sec = EXCLUDED.last_ep_watched_sec,
        last_ep_duration_sec = EXCLUDED.last_ep_duration_sec,
        dropped = EXCLUDED.dropped,
        w_item = EXCLUDED.w_item,
        updated_at = now()
"""




async def record_event(pool: asyncpg.Pool, event: dict) -> None:
    """Append one watch event and update everything it touches (§G.1).

    Single transaction: raw event + user_title_stats recompute for that
    title + w_i recompute + user_stats.last_event_at. Vector rebuild is
    async (the caller queues it) — never on the player's hot path.
    """
    required = ("user_id", "tmdb_id", "media_type", "watched_at")
    missing = [k for k in required if event.get(k) is None]
    if missing:
        raise ValueError(f"watch event missing fields: {missing}")

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO user_watch_events (
                    user_id, tmdb_id, media_type, watched_at, watched_sec,
                    duration_sec, pause_count, completed, season, episode,
                    last_position_sec
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                event["user_id"], event["tmdb_id"], event["media_type"],
                event["watched_at"], event.get("watched_sec") or 0,
                event.get("duration_sec"), event.get("pause_count") or 0,
                bool(event.get("completed")), event.get("season"),
                event.get("episode"), event.get("last_position_sec"),
            )
            row = await _recompute_title_stats(
                conn, event["user_id"], event["tmdb_id"], event["media_type"]
            )
            await conn.execute(
                """
                INSERT INTO user_stats (user_id, last_event_at, updated_at)
                VALUES ($1, $2, now())
                ON CONFLICT (user_id) DO UPDATE SET
                    last_event_at = EXCLUDED.last_event_at, updated_at = now()
                """,
                event["user_id"], event["watched_at"],
            )
    if row:
        logger.debug(
            f"user {event['user_id']} title {event['tmdb_id']}: "
            f"w_item={row['w_item']:.3f} dropped={row['dropped']}"
        )


async def _recompute_title_stats(
    conn, user_id: int, tmdb_id: int, media_type: str
) -> Optional[asyncpg.Record]:
    """Rebuild one user_title_stats row from the raw events + feedback."""
    await conn.execute(
        _UPSERT_TITLE_STATS, user_id, tmdb_id, media_type, EPISODE_THRESHOLD_RATIO
    )
    row = await conn.fetchrow(
        """SELECT uts.*, COALESCE(
               (SELECT TRUE FROM user_feedback f
                WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                  AND f.media_type = uts.media_type AND f.kind = 'favorite'), FALSE)
           AS favorite,
           EXISTS (SELECT 1 FROM user_feedback f
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                     AND f.media_type = uts.media_type AND f.kind = 'dislike')
           AS disliked
           FROM user_title_stats uts
           WHERE uts.user_id = $1 AND uts.tmdb_id = $2 AND uts.media_type = $3""",
        user_id, tmdb_id, media_type,
    )
    if row is None:
        return None

    total_eps = await _catalog_episode_count(
        conn, tmdb_id, media_type
    ) if media_type == "tv" else 0

    series = uw.SeriesSignals(
        episodes_watched=row["episodes_watched"],
        total_episodes=total_eps,
        last_ep_ratio=(
            (row["last_ep_watched_sec"] or 0) / (row["last_ep_duration_sec"] or 1)
            if (row["last_ep_watched_sec"] or 0) > 0
            and (row["last_ep_duration_sec"] or 0) > 0 else None
        ),
        last_ep_watched_sec=row["last_ep_watched_sec"] or 0,
        total_watched_sec=row["total_watched_sec"],
    )
    w = uw.w_item(
        media_type,
        max_ratio=row["max_ratio"],
        pause_count_total=row["pause_count_total"],
        series=series,
        last_watched_at=row["last_watched_at"],
        rewatch_count=row["rewatch_count"],
        favorite=row["favorite"],
        disliked=row["disliked"],
    )
    # §D.1 only — an explicit dislike zeroes w_item but is NOT a drop-off
    # (movies_dropped must count genuine quit-early behavior, §C.1)
    dropped = uw.is_dropped(
        media_type, row["max_ratio"],
        series.episodes_watched, series.last_ep_ratio,
    )
    await conn.execute(
        """UPDATE user_title_stats
           SET w_item = $4, dropped = $5, updated_at = now()
           WHERE user_id = $1 AND tmdb_id = $2 AND media_type = $3""",
        user_id, tmdb_id, media_type, w, dropped,
    )
    return {"w_item": w, "dropped": dropped}


async def _catalog_episode_count(conn, tmdb_id: int, media_type: str) -> int:
    """Total episodes for a series from the catalog (0 when unknown)."""
    if media_type != "tv":
        return 0
    try:
        n = await conn.fetchval(
            "SELECT number_of_episodes FROM tmdb_tv WHERE id = $1", tmdb_id
        )
        return int(n or 0)
    except Exception:
        return 0


async def refresh_user_stats(pool: asyncpg.Pool, user_id: int) -> None:
    """§C — full recompute of user_stats + user_genre_stats for one user."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO user_stats (
                    user_id, titles_touched, titles_weighted,
                    movies_completed, movies_dropped,
                    series_started, series_hooked, series_abandoned_mid_ep,
                    rewatch_titles, explicit_dislikes, explicit_favorites,
                    watchlist_open,
                    total_watch_sec_30d, total_watch_sec_all,
                    movie_share_30d, tv_share_30d,
                    median_session_sec, avg_pauses_per_unfinished_movie,
                    updated_at
                )
                SELECT
                    $1,
                    COUNT(*),
                    COUNT(*) FILTER (WHERE w_item > 0),
                    COUNT(*) FILTER (WHERE media_type='movie'
                                     AND max_ratio >= 0.75),
                    COUNT(*) FILTER (WHERE media_type='movie' AND dropped),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND episodes_watched >= 1),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND episodes_watched >= 3),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND last_ep_watched_sec > 0
                                     AND COALESCE(
                                         last_ep_watched_sec::float
                                         / NULLIF(last_ep_duration_sec,0), 1)
                                         < 0.20),
                    COUNT(*) FILTER (WHERE rewatch_count >= 1),
                    (SELECT COUNT(*) FROM user_feedback f
                     WHERE f.user_id = $1 AND f.kind = 'dislike'),
                    (SELECT COUNT(*) FROM user_feedback f
                     WHERE f.user_id = $1 AND f.kind = 'favorite'),
                    (SELECT COUNT(*) FROM user_feedback f
                     WHERE f.user_id = $1 AND f.kind = 'watchlist'
                       AND NOT EXISTS (
                           SELECT 1 FROM user_title_stats uts
                           WHERE uts.user_id = $1 AND uts.tmdb_id = f.tmdb_id
                             AND uts.media_type = f.media_type)),
                    COALESCE(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0),
                    COALESCE(SUM(total_watched_sec), 0),
                    COALESCE((
                        SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'
                            AND media_type='movie')
                        ::float / NULLIF(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0)), 0),
                    COALESCE((
                        SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'
                            AND media_type='tv')
                        ::float / NULLIF(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0)), 0),
                    (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY watched_sec)
                     FROM user_watch_events WHERE user_id = $1)::int,
                    COALESCE((
                        SUM(uts.pause_count_total) FILTER (WHERE uts.media_type='movie' AND uts.max_ratio < 0.75)
                        ::float / NULLIF(COUNT(*) FILTER (WHERE uts.media_type='movie' AND uts.max_ratio < 0.75), 0)), 0),
                    now()
                FROM user_title_stats uts
                WHERE uts.user_id = $1
                ON CONFLICT (user_id) DO UPDATE SET
                    titles_touched = EXCLUDED.titles_touched,
                    titles_weighted = EXCLUDED.titles_weighted,
                    movies_completed = EXCLUDED.movies_completed,
                    movies_dropped = EXCLUDED.movies_dropped,
                    series_started = EXCLUDED.series_started,
                    series_hooked = EXCLUDED.series_hooked,
                    series_abandoned_mid_ep = EXCLUDED.series_abandoned_mid_ep,
                    rewatch_titles = EXCLUDED.rewatch_titles,
                    explicit_dislikes = EXCLUDED.explicit_dislikes,
                    explicit_favorites = EXCLUDED.explicit_favorites,
                    watchlist_open = EXCLUDED.watchlist_open,
                    total_watch_sec_30d = EXCLUDED.total_watch_sec_30d,
                    total_watch_sec_all = EXCLUDED.total_watch_sec_all,
                    movie_share_30d = EXCLUDED.movie_share_30d,
                    tv_share_30d = EXCLUDED.tv_share_30d,
                    median_session_sec = EXCLUDED.median_session_sec,
                    avg_pauses_per_unfinished_movie =
                        EXCLUDED.avg_pauses_per_unfinished_movie,
                    updated_at = now()
                """,
                user_id,
            )

            # genre distribution from the weighted catalog join
            await conn.execute("DELETE FROM user_genre_stats WHERE user_id = $1", user_id)
            await conn.execute(
                """
                INSERT INTO user_genre_stats (
                    user_id, media_type, genre_id, watch_sec, weighted_sum,
                    title_count, last_watched_at
                )
                SELECT $1, gm.media_type, gm.genre_id,
                       SUM(uts.total_watched_sec),
                       SUM(uts.w_item),
                       COUNT(*),
                       MAX(uts.last_watched_at)
                FROM user_title_stats uts
                JOIN tmdb_media_genres gm
                  ON gm.media_id = uts.tmdb_id AND gm.media_type = uts.media_type
                WHERE uts.user_id = $1
                GROUP BY gm.media_type, gm.genre_id
                """,
                user_id,
            )


async def recompute_weights(pool: asyncpg.Pool, user_id: int) -> None:
    """Recency moves → recompute w_i for every title of one user (§G.2)."""
    rows = await pool.fetch(
        """SELECT uts.*, COALESCE((SELECT TRUE FROM user_feedback f
               WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                 AND f.media_type = uts.media_type AND f.kind='favorite'),
               FALSE) AS favorite,
           EXISTS (SELECT 1 FROM user_feedback f
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                     AND f.media_type = uts.media_type AND f.kind='dislike')
           AS disliked
           FROM user_title_stats uts WHERE uts.user_id = $1""",
        user_id,
    )
    tv_ids = [r["tmdb_id"] for r in rows if r["media_type"] == "tv"]
    ep_counts = {}
    if tv_ids:
        cat = await pool.fetch(
            "SELECT id, number_of_episodes FROM tmdb_tv WHERE id = ANY($1::bigint[])",
            tv_ids,
        )
        ep_counts = {r["id"]: int(r["number_of_episodes"] or 0) for r in cat}

    async with pool.acquire() as conn:
        for r in rows:
            series = uw.SeriesSignals(
                episodes_watched=r["episodes_watched"],
                total_episodes=ep_counts.get(r["tmdb_id"], 0),
                last_ep_ratio=(
                    (r["last_ep_watched_sec"] or 0) / (r["last_ep_duration_sec"] or 1)
                    if (r["last_ep_watched_sec"] or 0) > 0
                    and (r["last_ep_duration_sec"] or 0) > 0 else None),
                last_ep_watched_sec=r["last_ep_watched_sec"] or 0,
                total_watched_sec=r["total_watched_sec"],
            )
            w = uw.w_item(
                r["media_type"],
                max_ratio=r["max_ratio"],
                pause_count_total=r["pause_count_total"],
                series=series,
                last_watched_at=r["last_watched_at"],
                rewatch_count=r["rewatch_count"],
                favorite=r["favorite"],
                disliked=r["disliked"],
            )
            await conn.execute(
                """UPDATE user_title_stats SET w_item = $4, updated_at = now()
                   WHERE user_id=$1 AND tmdb_id=$2 AND media_type=$3""",
                r["user_id"], r["tmdb_id"], r["media_type"], w,
            )


async def nightly_recompute(
    pool: asyncpg.Pool,
    active_days: int = 90,
    vector_max_age_hours: int = 24,
    vector_builder=None,
) -> dict:
    """§G.2 — the nightly batch: weights for active users, stats, vectors.

    vector_builder: async callable(pool, user_id); defaults to
    cine_rec_engine.user_vector.build_user_vector when left None.
    Dead users (>active_days without events) are skipped entirely.
    """
    if vector_builder is None:
        from .user_vector import build_user_vector as vector_builder

    users = await pool.fetch(
        """SELECT user_id FROM user_stats
           WHERE last_event_at > now() - ($1 || ' days')::interval""",
        str(active_days),
    )
    rebuilt_vectors = 0
    for u in users:
        user_id = u["user_id"]
        await recompute_weights(pool, user_id)
        await refresh_user_stats(pool, user_id)
        stale = await pool.fetchval(
            """SELECT vector_updated_at IS NULL
                    OR vector_updated_at < now() - ($2 || ' hours')::interval
               FROM user_stats WHERE user_id = $1""",
            user_id, str(vector_max_age_hours),
        )
        if stale:
            await vector_builder(pool, user_id)
            rebuilt_vectors += 1
    logger.info(
        f"nightly: {len(users)} active users, {rebuilt_vectors} vectors rebuilt"
    )
    return {"users": len(users), "vectors_rebuilt": rebuilt_vectors}

async def record_feedback(
    pool: asyncpg.Pool, user_id: int, tmdb_id: int, media_type: str, kind: str,
) -> None:
    """§B.2 + §G.1 — upsert an explicit signal and refresh what it touches.

    dislike/favorite change w_i (and the vector later); watchlist-only
    titles get their intent weight on the next recompute.
    """
    if kind not in ("dislike", "favorite", "watchlist"):
        raise ValueError(f"kind must be dislike|favorite|watchlist, got {kind!r}")
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO user_feedback (user_id, tmdb_id, media_type, kind)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (user_id, tmdb_id, media_type, kind) DO NOTHING""",
            user_id, tmdb_id, media_type, kind,
        )
    has_stats = await pool.fetchval(
        """SELECT 1 FROM user_title_stats
           WHERE user_id=$1 AND tmdb_id=$2 AND media_type=$3""",
        user_id, tmdb_id, media_type,
    )
    if has_stats:
        async with pool.acquire() as conn:
            await _recompute_title_stats(conn, user_id, tmdb_id, media_type)

