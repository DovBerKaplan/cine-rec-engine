"""Live-DB remap proof: alias views + a table map must reproduce the
canonical results end to end (recall, scoring, the user path).

Only the table names change — the alias views are plain SELECT * over
the canonical tables, so any result divergence means a query escaped
the registry.
"""

from __future__ import annotations

import pytest

from cine_rec_engine import tables

pytestmark = [pytest.mark.integration]

# everything the find_similar + user paths touch that exists in the
# demo DB; missing optional tables are skipped when creating aliases
ALIAS_CANDIDATES = [
    "tmdb_media", "tmdb_genres", "tmdb_media_genres", "tmdb_media_keywords",
    "tmdb_keywords", "tmdb_media_companies", "tmdb_production_companies",
    "tmdb_media_networks", "tmdb_networks", "tmdb_cast", "tmdb_crew",
    "tmdb_people", "tmdb_movies", "tmdb_tv", "tmdb_movie_genres_map",
    "tmdb_tv_genres_map", "tmdb_recommendations",
    "user_watch_events", "user_title_stats", "user_stats",
    "user_genre_stats", "user_vectors", "user_watches", "user_feedback",
    "title_ratings",
]


async def _create_aliases(conn) -> list:
    import asyncpg

    created = []
    for logical in ALIAS_CANDIDATES:
        physical = tables.name(logical)
        try:
            await conn.execute(
                f"CREATE OR REPLACE VIEW alias_{logical} AS SELECT * FROM {physical}"
            )
            created.append(logical)
        except asyncpg.UndefinedTableError:
            pass  # optional channel not present in this DB
    return created


async def _drop_aliases(conn, created) -> None:
    for logical in created:
        await conn.execute(f"DROP VIEW IF EXISTS alias_{logical}")


async def test_remap_reproduces_find_similar(integration_env):
    pool = integration_env["pool"]
    service = integration_env["service"]

    baseline = await service.find_similar(155, limit=5)

    async with pool.acquire() as conn:
        created = await _create_aliases(conn)
    tables.set_table_map({n: f"alias_{n}" for n in created})
    try:
        # different limit → different cache key → the remapped recall
        # path actually executes instead of serving the cached page
        remapped = await service.find_similar(155, limit=6)
        assert [r["tmdb_id"] for r in remapped[:5]] == \
            [r["tmdb_id"] for r in baseline]
    finally:
        tables.set_table_map()
        async with pool.acquire() as conn:
            await _drop_aliases(conn, created)


async def test_remap_user_path_writes_through_views(integration_env):
    import datetime

    from cine_rec_engine.user_stats import record_event

    pool = integration_env["pool"]
    service = integration_env["service"]
    user_id = 990001  # disposable

    async with pool.acquire() as conn:
        created = await _create_aliases(conn)
        await conn.execute(
            "DELETE FROM user_watch_events WHERE user_id = $1", user_id)
        await conn.execute(
            "DELETE FROM user_title_stats WHERE user_id = $1", user_id)
        await conn.execute(
            "DELETE FROM user_stats WHERE user_id = $1", user_id)
    tables.set_table_map({n: f"alias_{n}" for n in created})
    try:
        # the one-transaction event write must land through the mapped
        # names (auto-updatable views) and be visible to the mapped reads
        await record_event(pool, {
            "user_id": user_id, "tmdb_id": 155, "media_type": "movie",
            "watched_sec": 8000, "duration_sec": 8000, "completed": True,
            "watched_at": datetime.datetime.now(datetime.timezone.utc),
        })
        async with pool.acquire() as conn:
            n = await conn.fetchval(
                "SELECT count(*) FROM alias_user_title_stats WHERE user_id = $1",
                user_id)
        assert n == 1
        got = await service.recommend_for_user(user_id, limit=5)
        assert got and got.get("results") is not None
    finally:
        tables.set_table_map()
        async with pool.acquire() as conn:
            await _drop_aliases(conn, created)
            await conn.execute(
                "DELETE FROM user_watch_events WHERE user_id = $1", user_id)
            await conn.execute(
                "DELETE FROM user_title_stats WHERE user_id = $1", user_id)
            await conn.execute(
                "DELETE FROM user_stats WHERE user_id = $1", user_id)
