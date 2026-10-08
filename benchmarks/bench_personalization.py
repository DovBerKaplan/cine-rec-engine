"""End-to-end recommend_for_user latency benchmark with phase breakdown.

Builds on bench_e2e's synthetic 25k catalog, adds a populated embedding
column (+ HNSW index), one user with real w_i history (via record_event),
then measures cold / warm / per-phase latency of recommend_for_user —
the plug-and-play path (inline ensure_user_vector).

    python benchmarks/bench_personalization.py --dsn postgresql://user:pw@host/db
"""

import asyncio
import os
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DSN = os.environ.get("BENCH_DSN") or os.environ.get(
    "DATABASE_URL", "postgresql://localhost/postgres"
)

SCHEMA = (Path(__file__).parent.parent / "docs" / "schema.sql").read_text()
USER_SQL = (Path(__file__).parent.parent / "docs" / "user_data.sql").read_text()

DIM = 384  # matches the default "original" space (column `embedding`)

PHASES: dict = {}


def _wrap_async(obj, name, key=None):
    orig = getattr(obj, name)
    key = key or name

    async def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return await orig(*a, **kw)
        finally:
            PHASES[key] = PHASES.get(key, 0.0) + (time.perf_counter() - t0) * 1000
            PHASES.setdefault(key + "#calls", 0)
            PHASES[key + "#calls"] += 1

    setattr(obj, name, timed)


async def seed_catalog(pool):
    rng = random.Random(7)
    genres_m = [(28, "Action"), (35, "Comedy"), (18, "Drama"), (80, "Crime"),
                (878, "Science Fiction"), (27, "Horror"), (10749, "Romance"),
                (16, "Animation")]
    genres_t = [(18, "Drama"), (35, "Comedy"), (10759, "Action & Adventure"),
                (16, "Animation"), (9648, "Mystery")]
    kw = [(i, f"kw{i}") for i in range(400)]
    people = [(i, f"Person {i}") for i in range(4000)]
    async with pool.acquire() as c:
        await c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await c.execute(SCHEMA)
        await c.execute(USER_SQL)
        await c.executemany("INSERT INTO tmdb_movie_genres VALUES ($1,$2) ON CONFLICT DO NOTHING", genres_m)
        await c.executemany("INSERT INTO tmdb_tv_genres VALUES ($1,$2) ON CONFLICT DO NOTHING", genres_t)
        await c.executemany("INSERT INTO tmdb_keywords VALUES ($1,$2) ON CONFLICT DO NOTHING", kw)
        await c.executemany("INSERT INTO tmdb_people VALUES ($1,$2) ON CONFLICT DO NOTHING", people)
        rows = []
        for i in range(1, 20001):
            rows.append((i, f"Movie {i}", f"overview {i} action drama", "en",
                         date(1950 + i % 76, 1, 1), 120, "Released", False,
                         5.5 + (i % 40) / 10, 100 + i % 9000, 1.0 + i % 90,
                         f"/p{i}.jpg", f"/b{i}.jpg", 1000 + i % 500, None, 0, 0))
        await c.executemany(
            """INSERT INTO tmdb_movies (id,title,overview,original_language,release_date,
               runtime,status,adult,vote_average,vote_count,popularity,poster_path,
               backdrop_path,collection_id,imdb_id,budget,revenue)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)""", rows)
        rows = []
        for i in range(1, 5001):
            rows.append((i, f"Show {i}", f"overview show {i} drama", "en",
                         date(1990 + i % 36, 1, 1), "Ended", False, 3,
                         6.0 + (i % 35) / 10, 50 + i % 8000,
                         1.0 + i % 80, f"/p{i}.jpg", f"/b{i}.jpg", None))
        await c.executemany(
            """INSERT INTO tmdb_tv (id,name,overview,original_language,first_air_date,
               status,in_production,number_of_seasons,vote_average,
               vote_count,popularity,poster_path,backdrop_path,imdb_id)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""", rows)
        gm, km, cm, cw = [], [], [], []
        for i in range(1, 20001):
            gm.append((i, rng.choice(genres_m)[0]))
            for k in rng.sample(kw, 5):
                km.append((i, k[0]))
            for p in rng.sample(people, 5):
                cm.append((i, p[0], f"char{p[0]}", rng.randint(0, 4)))
            cw.append((i, rng.choice(people)[0], "Director", "Directing"))
        await c.executemany("INSERT INTO tmdb_movie_genres_map VALUES ($1,$2) ON CONFLICT DO NOTHING", gm)
        await c.executemany("INSERT INTO tmdb_movie_keywords VALUES ($1,$2) ON CONFLICT DO NOTHING", km)
        await c.executemany("INSERT INTO tmdb_movie_cast VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING", cm)
        await c.executemany("INSERT INTO tmdb_movie_crew VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING", cw)
        recs = []
        for i in range(1, 20001):
            for r, rid in enumerate(rng.sample(range(1, 20001), 10), start=1):
                recs.append((i, rid, r, 5.0))
        await c.executemany(
            "INSERT INTO tmdb_movie_recommendations (movie_id, rec_movie_id, rank, popularity)"
            " VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING", recs)
        # embeddings for the default space — KNN + user-vector channels ON
        await c.execute(f"ALTER TABLE tmdb_movies ADD COLUMN embedding vector({DIM})")
        await c.execute(f"ALTER TABLE tmdb_tv ADD COLUMN embedding vector({DIM})")
        from pgvector.asyncpg import register_vector
        await register_vector(c)
        emb_m = []
        for i in range(1, 20001):
            v = [rng.uniform(-1, 1) for _ in range(DIM)]
            n = sum(x * x for x in v) ** 0.5
            emb_m.append((i, [x / n for x in v]))
        await c.executemany("UPDATE tmdb_movies SET embedding = $2 WHERE id = $1", emb_m)
        emb_t = []
        for i in range(1, 5001):
            v = [rng.uniform(-1, 1) for _ in range(DIM)]
            n = sum(x * x for x in v) ** 0.5
            emb_t.append((i, [x / n for x in v]))
        await c.executemany("UPDATE tmdb_tv SET embedding = $2 WHERE id = $1", emb_t)
        await c.execute(
            "CREATE INDEX IF NOT EXISTS hnsw_m ON tmdb_movies USING hnsw (embedding vector_cosine_ops)")
        await c.execute(
            "CREATE INDEX IF NOT EXISTS hnsw_t ON tmdb_tv USING hnsw (embedding vector_cosine_ops)")
        await c.execute("ANALYZE")


async def seed_user(pool, user_id, n_movies=15, n_tv=3, n_watchlist=2, seed=1):
    """A realistic weighted history through the real event pipeline."""
    from cine_rec_engine import user_stats as us

    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    for i in range(1, n_movies + 1):
        await us.record_event(pool, dict(
            user_id=user_id, tmdb_id=i, media_type="movie",
            watched_at=now - timedelta(days=rng.randint(1, 60)),
            watched_sec=7200, duration_sec=7200, completed=True))
    for i in range(1, n_tv + 1):
        for ep in range(1, 4):
            await us.record_event(pool, dict(
                user_id=user_id, tmdb_id=i, media_type="tv",
                watched_at=now - timedelta(days=rng.randint(1, 40), hours=ep),
                watched_sec=2700, duration_sec=2700, completed=True,
                season=1, episode=ep))
    for j in range(n_watchlist):
        await us.record_feedback(
            pool, user_id, 10000 + j, "movie", "watchlist")


def _install_probes():
    import cine_rec_engine.service as S
    import cine_rec_engine.queries as Q
    import cine_rec_engine.user_vector as UV
    import cine_rec_engine.watched as W
    import cine_rec_engine.tmdb_recs as T

    import cine_rec_engine.scoring as SC
    for n in ("generate_candidates", "generate_knn_candidates",
              "generate_tmdb_rec_candidates",
              "generate_director_candidates", "generate_user_vector_candidates",
              "enrich_candidates_batch", "get_movie_info_batch"):
        if hasattr(Q, n):
            _wrap_async(Q, n, key="q:" + n)
    _wrap_async(SC, "semantic_overview_similarity", key="q:semantic_overview_similarity")
    if hasattr(SC, "semantic_overview_similarity_batch"):
        _wrap_async(SC, "semantic_overview_similarity_batch",
                    key="q:semantic_overview_similarity_batch")
    # service.py imported some names before our patch — rebind there too
    for n in ("generate_candidates", "generate_knn_candidates",
              "generate_tmdb_rec_candidates",
              "generate_director_candidates", "enrich_candidates_batch",
              "get_movie_info_batch", "semantic_overview_similarity",
              "semantic_overview_similarity_batch",
              "generate_user_vector_candidates"):
        if hasattr(S, n):
            setattr(S, n, getattr(Q if hasattr(Q, n) else SC, n))
    for n in ("ensure_user_vector", "build_user_vector", "load_user_vector",
              "top_weighted_seeds"):
        _wrap_async(UV, n, key="uv:" + n)
    for n in ("ensure_seeds_synced", "has_local_recommendations"):
        if hasattr(T, n):
            _wrap_async(T, n, key="t:" + n)
    for n in ("get_user_recommendation_exclusions", "get_user_dislikes"):
        _wrap_async(W, n, key="w:" + n)
    from cine_rec_engine.service import RecommendationService as RS
    for n in ("_apply_user_filters", "_merged_exclusions"):
        _wrap_async(RS, n, key="s:" + n)
    # pure scoring loop — count pairs + cpu time
    orig = S._score_pair_fast
    stats = {"calls": 0, "ms": 0.0}

    def timed_pair(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            stats["calls"] += 1
            stats["ms"] += (time.perf_counter() - t0) * 1000

    S._score_pair_fast = timed_pair
    return stats


def _report(label, wall_ms, pair_stats, phases_snapshot):
    print(f"\n=== {label}: wall={wall_ms:.0f}ms "
          f"pairs={pair_stats.get('calls', 0)} "
          f"score_cpu={pair_stats.get('ms', 0.0):.0f}ms ===")
    rows = sorted(
        ((k, v) for k, v in phases_snapshot.items() if not k.endswith("#calls")),
        key=lambda x: -x[1])
    for k, v in rows:
        if v >= 0.5:
            n = phases_snapshot.get(k + "#calls", 1)
            print(f"  {k:55s} sum={v:8.1f}ms x{n} avg={v/max(n,1):7.1f}ms")
    acq = phases_snapshot.get("pool:acquire_wait")
    if acq is not None:
        print(f"  {'pool:acquire_wait (sum across coroutines)':55s} "
              f"{acq:8.1f}ms  x{phases_snapshot.get('pool:acquire_wait#calls', '?')}")


class _TimedAcquire:
    """Wrap pool.acquire() to expose queue wait (async-with and await)."""

    def __init__(self, ctx):
        self._ctx = ctx

    async def __aenter__(self):
        t0 = time.perf_counter()
        try:
            return await self._ctx.__aenter__()
        finally:
            PHASES["pool:acquire_wait"] = PHASES.get("pool:acquire_wait", 0.0) + (
                time.perf_counter() - t0) * 1000
            PHASES["pool:acquire_wait#calls"] = (
                PHASES.get("pool:acquire_wait#calls", 0) + 1)

    async def __aexit__(self, *exc):
        return await self._ctx.__aexit__(*exc)

    def __await__(self):
        async def _run():
            t0 = time.perf_counter()
            try:
                conn = await self._ctx
                PHASES["pool:acquire_wait#calls"] = (
                    PHASES.get("pool:acquire_wait#calls", 0) + 1)
                return conn
            finally:
                PHASES["pool:acquire_wait"] = PHASES.get("pool:acquire_wait", 0.0) + (
                    time.perf_counter() - t0) * 1000
        return _run().__await__()


def _instrument_pool(pool):
    import asyncpg.pool

    orig = asyncpg.pool.Pool.acquire

    def timed_acquire(self, *a, **kw):
        return _TimedAcquire(orig(self, *a, **kw))

    asyncpg.pool.Pool.acquire = timed_acquire


class _ProbeCache:
    def __init__(self, inner):
        self._inner = inner

    async def get(self, key):
        t0 = time.perf_counter()
        try:
            v = await self._inner.get(key)
        finally:
            PHASES["cache:get"] = PHASES.get("cache:get", 0.0) + (
                time.perf_counter() - t0) * 1000
        k = "cache:hit" if v is not None else "cache:miss"
        PHASES[k] = PHASES.get(k, 0) + 1
        return v

    async def set(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return await self._inner.set(*a, **kw)
        finally:
            PHASES["cache:set"] = PHASES.get("cache:set", 0.0) + (
                time.perf_counter() - t0) * 1000
            PHASES["cache:set#calls"] = PHASES.get("cache:set#calls", 0) + 1

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _instrument_cache():
    import cine_rec_engine.cache as C

    orig_get_cache = C.get_cache

    async def probed_get_cache():
        inner = await orig_get_cache()
        return _ProbeCache(inner) if inner is not None else None

    C.get_cache = probed_get_cache


async def main():
    import asyncpg
    from cine_rec_engine import RecommendationService

    pool = await asyncpg.create_pool(DSN)  # app defaults: min 0 / max 10
    _instrument_pool(pool)
    _instrument_cache()
    print("seeding 25k catalog + embeddings + HNSW ...")
    if not os.environ.get("BENCH_SKIP_SEED"):
        await seed_catalog(pool)
    pair_stats = _install_probes()
    print("seeding user histories (record_event) ...")
    await seed_user(pool, 42)
    for u in (100, 101, 102, 103, 104):
        await seed_user(pool, u, n_movies=8 + u % 10, seed=u)

    rec = RecommendationService()
    await rec.initialize(pool)
    # wipe any stored vector so call #1 exercises the inline rebuild
    await pool.execute("DELETE FROM user_vectors WHERE user_id = 42")

    def snap():
        return dict(PHASES)

    # 1) FIRST request for the user: vector rebuild + everything cold
    PHASES.clear(); pair_stats["calls"] = 0; pair_stats["ms"] = 0.0
    t0 = time.perf_counter()
    r1 = await rec.recommend_for_user(42, limit=30)
    wall = (time.perf_counter() - t0) * 1000
    _report("user42 FIRST request (stale vector, inline rebuild, cold cache)",
            wall, pair_stats, snap())

    # 2) SECOND request, same user: find_similar cache should hit
    PHASES.clear(); pair_stats["calls"] = 0; pair_stats["ms"] = 0.0
    t0 = time.perf_counter()
    r2 = await rec.recommend_for_user(42, limit=30)
    wall = (time.perf_counter() - t0) * 1000
    _report("user42 SECOND request (warm cache, fresh vector)",
            wall, pair_stats, snap())

    # 3) distinct users, cold caches — steady-state cold latency
    times = []
    for u in (100, 101, 102, 103, 104):
        PHASES.clear(); pair_stats["calls"] = 0; pair_stats["ms"] = 0.0
        t0 = time.perf_counter()
        await rec.recommend_for_user(u, limit=30)
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times) // 2]
    _report(f"distinct users cold median={med:.0f}ms all={[f'{t:.0f}' for t in times]}",
            med, pair_stats, snap())

    print(f"\nresults: first={len(r1['results'])} second={len(r2['results'])} "
          f"vector_used={r1['vector_used']}")
    await pool.close()


asyncio.run(main())
