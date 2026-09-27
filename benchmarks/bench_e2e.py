"""End-to-end find_similar benchmark on a synthetic 25k-title catalog.

Creates (and later drops) the full schema from docs/schema.sql inside the
target database, seeds 20k movies + 5k tv with bridges and behavioral
recs, then measures cold/warm/multi-seed latency. Leaves nothing behind.

    python benchmarks/bench_e2e.py --dsn postgresql://user:pw@host/db

Numbers below are from a 2-core container against PostgreSQL 16 (2026-09):
cold single-seed median ≈ 45ms · warm (cache) 0.4ms · 5-seed 322ms
"""

import asyncio, os, random, sys, time
from datetime import date
from pathlib import Path


DSN = os.environ.get("BENCH_DSN") or os.environ.get("DATABASE_URL", "postgresql://localhost/postgres")

SCHEMA = (Path(__file__).parent.parent / "docs" / "schema.sql").read_text()

async def seed(pool):
    rng = random.Random(7)
    genres_m = [(28,"Action"),(35,"Comedy"),(18,"Drama"),(80,"Crime"),(878,"Science Fiction"),(27,"Horror"),(10749,"Romance"),(16,"Animation")]
    genres_t = [(18,"Drama"),(35,"Comedy"),(10759,"Action & Adventure"),(16,"Animation"),(9648,"Mystery")]
    kw = [(i, f"kw{i}") for i in range(400)]
    people = [(i, f"Person {i}") for i in range(4000)]
    async with pool.acquire() as c:
        await c.execute(SCHEMA)
        await c.executemany("INSERT INTO tmdb_movie_genres VALUES ($1,$2) ON CONFLICT DO NOTHING", genres_m)
        await c.executemany("INSERT INTO tmdb_tv_genres VALUES ($1,$2) ON CONFLICT DO NOTHING", genres_t)
        await c.executemany("INSERT INTO tmdb_keywords VALUES ($1,$2) ON CONFLICT DO NOTHING", kw)
        await c.executemany("INSERT INTO tmdb_people VALUES ($1,$2) ON CONFLICT DO NOTHING", people)
        # 20k movies
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
        # 5k tv
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
        # bridges for movies: genres, keywords, cast, crew, recs
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
            "INSERT INTO tmdb_movie_recommendations (movie_id, rec_movie_id, rank, popularity) VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING", recs)

PHASES = {}
def _wrap(mod, name):
    orig = getattr(mod, name)
    async def timed(*a, **kw):
        import time as _t
        t0 = _t.perf_counter()
        try:
            return await orig(*a, **kw)
        finally:
            PHASES[name] = PHASES.get(name, 0.0) + (_t.perf_counter() - t0) * 1000
    setattr(mod, name, timed)

async def analyze(pool):
    async with pool.acquire() as c:
        await c.execute("ANALYZE")


async def main():
    import asyncpg
    import cine_rec_engine.service as S
    import cine_rec_engine.queries as Q
    from cine_rec_engine import RecommendationService
    del Q
    for n in ("generate_candidates", "generate_tmdb_rec_candidates",
              "generate_director_candidates", "generate_knn_candidates",
              "generate_knn_finetuned_candidates", "enrich_candidates_batch",
              "get_movie_info_batch", "semantic_overview_similarity"):
        if hasattr(S, n): _wrap(S, n)
    # class methods (self-bound) — patch on the class, not the module
    from cine_rec_engine.service import RecommendationService as RS
    for n in ("_apply_user_filters",):
        orig = getattr(RS, n)
        async def timed_method(self, *a, **kw):
            import time as _t
            t0 = _t.perf_counter()
            try:
                return await orig(self, *a, **kw)
            finally:
                PHASES[n] = PHASES.get(n, 0.0) + (_t.perf_counter() - t0) * 1000
        setattr(RS, n, timed_method)
    for n in ("_mmr_order", "_dedup_same_story", "_balance_by_media_type"):
        origf = getattr(S, n)
        def timed_sync(*a, _orig=origf, _n=n, **kw):
            import time as _t
            PHASES[_n+"_calls"] = PHASES.get(_n+"_calls", 0) + 1
            if a and isinstance(a[0], list):
                PHASES[_n+"_maxlen"] = max(PHASES.get(_n+"_maxlen", 0), len(a[0]))
            t0 = _t.perf_counter()
            try:
                return _orig(*a, **kw)
            finally:
                PHASES[n] = PHASES.get(n, 0.0) + (_t.perf_counter() - t0) * 1000
        setattr(S, n, timed_sync)
    pool = await asyncpg.create_pool(DSN, min_size=2, max_size=4)
    await seed(pool)
    await analyze(pool)
    rec = RecommendationService()
    await rec.initialize(pool)
    # warm-up: statement prep + planner cache
    await rec.find_similar(999001, limit=30)

    # isolate _balance_by_media_type: capture its args on the next call
    import cine_rec_engine.service as _S
    _cap = {}
    _orig_b = _S._balance_by_media_type
    def cap(*a, **kw):
        _cap.setdefault("args", (list(a[0]), a[1], a[2]))
        return _orig_b(*a, **kw)
    _S._balance_by_media_type = cap
    await rec.find_similar([2001, 2002, 2003], limit=60)
    _S._balance_by_media_type = _orig_b
    if _cap.get("args"):
        import time as _t2
        rs, mr, lm = _cap["args"]
        _t2.sleep(0.05)
        t0 = _t2.perf_counter()
        for _ in range(5):
            _orig_b(rs, mr, lm)
        print(f"ISOLATED balance: n={len(rs)} ratio={mr} limit={lm} "
              f"per_call={(_t2.perf_counter()-t0)/5*1000:.2f}ms "
              f"types_movie={sum(1 for r in rs if r['media_type']=='movie')}")
    # 5 fresh single-seed cold calls (distinct seeds, cache misses)
    times = []
    for sid in (155, 27205, 1396, 12, 762):
        t0 = time.perf_counter()
        await rec.find_similar(sid, limit=30)
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times)//2]
    # warm (cache path)
    t0 = time.perf_counter()
    r2 = await rec.find_similar(155, limit=30)
    warm = (time.perf_counter() - t0) * 1000
    # multi-seed with REAL ids (fresh combo → cache miss)
    t0 = time.perf_counter()
    r3 = await rec.find_similar([1001, 1002, 1003, 1004, 1005], limit=60)
    multi = (time.perf_counter() - t0) * 1000
    print(f"RESULTS: cold_median={med:.0f}ms cold_all={[f'{t:.0f}' for t in times]} "
          f"warm={warm:.1f}ms multi_seed_5={multi:.0f}ms n={len(r2)}/{len(r3)}")
    print("PHASES:", {k: f"{v:.0f}ms" for k, v in sorted(PHASES.items(), key=lambda x: -x[1])})
    await pool.close()

asyncio.run(main())
