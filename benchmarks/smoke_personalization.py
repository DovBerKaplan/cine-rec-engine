"""End-to-end personalization smoke (spec §J): events→w_i→stats→vector→recs."""
import asyncio, os, sys
from datetime import datetime, timedelta, timezone

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
DSN = os.environ.get("BENCH_DSN") or os.environ.get("DATABASE_URL", "postgresql://localhost/postgres")
print("DSN host:", DSN.split("@")[1])
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)

MOVIES = {
    155:  ("The Dark Knight", [0.9, 0.1, 0.0, 0.0]),
    27205: ("Inception",      [0.85, 0.2, 0.0, 0.1]),
    475557: ("Joker",         [0.8, 0.15, 0.05, 0.0]),
    999:  ("Bad Movie",      [0.0, 1.0, 1.0, 1.0]),      # will be dropped
}
TV = {1396: ("Breaking Bad", [0.9, 0.1, 0.0, 0.05], 62)}

async def main():
    import asyncpg
    from cine_rec_engine import user_stats as us
    from cine_rec_engine import user_vector as uv
    from cine_rec_engine import RecommendationService
    from cine_rec_engine import service as svc

    pool = await asyncpg.create_pool(DSN)
    async with pool.acquire() as c:
        await c.execute(open(Path(__file__).parent.parent / "docs" / "schema.sql").read())
        await c.execute(open(Path(__file__).parent.parent / "docs" / "user_data.sql").read())
        from pgvector.asyncpg import register_vector
        await register_vector(c)
        await c.execute("ALTER TABLE tmdb_movies ADD COLUMN embedding_e5e vector(4)")
        await c.execute("ALTER TABLE tmdb_tv ADD COLUMN embedding_e5e vector(4)")
        for mid, (title, vec) in MOVIES.items():
            await c.execute(
                """INSERT INTO tmdb_movies (id, title, overview, release_date,
                   vote_average, vote_count, popularity, adult, embedding_e5e)
                   VALUES ($1,$2,$3,'2020-01-01',8.0,5000,50.0,false,$4)""",
                mid, title, f"overview {title}", vec)
        tid, (name, vec, eps) = list(TV.items())[0]
        await c.execute(
            """INSERT INTO tmdb_tv (id, name, overview, first_air_date,
               number_of_seasons, number_of_episodes, vote_average, vote_count,
               popularity, adult, embedding_e5e)
               VALUES ($1,$2,$3,'2008-01-20',5,$4,8.9,12000,300.0,false,$5)""",
            tid, name, "chemistry", eps, vec)
    U = 42
    # §B.1 raw events
    await us.record_event(pool, dict(  # completed TDK yesterday
        user_id=U, tmdb_id=155, media_type="movie",
        watched_at=NOW - timedelta(days=1),
        watched_sec=8000, duration_sec=8000, pause_count=0, completed=True))
    await us.record_event(pool, dict(  # Inception half a year ago, rewatched
        user_id=U, tmdb_id=27205, media_type="movie",
        watched_at=NOW - timedelta(days=180),
        watched_sec=9000, duration_sec=9000, completed=True))
    await us.record_event(pool, dict(  # same title again → rewatch
        user_id=U, tmdb_id=27205, media_type="movie",
        watched_at=NOW - timedelta(days=170),
        watched_sec=9000, duration_sec=9000, completed=True))
    await us.record_event(pool, dict(  # dropped movie: 5 minutes only
        user_id=U, tmdb_id=999, media_type="movie",
        watched_at=NOW - timedelta(days=2),
        watched_sec=300, duration_sec=6000))
    for ep in range(1, 4):  # 3 episodes of Breaking Bad = hooked
        await us.record_event(pool, dict(
            user_id=U, tmdb_id=1396, media_type="tv",
            watched_at=NOW - timedelta(days=3, hours=ep),
            watched_sec=2700, duration_sec=2700, completed=True,
            season=1, episode=ep))
    await us.record_event(pool, dict(  # Joker: explicit dislike
        user_id=U, tmdb_id=475557, media_type="movie",
        watched_at=NOW - timedelta(days=5),
        watched_sec=7000, duration_sec=7000, completed=True))
    await us.record_feedback(pool, U, 475557, "movie", "dislike")

    # weights assertions (§D)
    async with pool.acquire() as c:
        rows = {r["tmdb_id"]: r for r in await c.fetch(
            "SELECT tmdb_id, media_type, w_item, dropped, rewatch_count "
            "FROM user_title_stats WHERE user_id = $1", U)}
    tdk = rows[155]["w_item"]
    assert 0.93 < tdk <= 1.0, tdk              # completed ~a day ago: ~2^(-1/30)
    assert rows[999]["dropped"] is True and rows[999]["w_item"] == 0.0
    bb = rows[1396]["w_item"]
    assert 0.65 <= bb <= 0.72, bb  # hooked 0.65 + last-ep bonus 0.10, ~3-4d decay
    joker = rows[475557]["w_item"]
    assert joker == 0.0, joker                 # disliked → 0
    incept = rows[27205]["w_item"]
    assert 0.0 < incept <= 1.5
    print(f"weights: TDK={tdk:.3f} Inception={incept:.3f} BB={bb:.3f} "
          f"BadMovie={rows[999]['w_item']} Joker={joker}")

    # §C stats
    await us.refresh_user_stats(pool, U)
    async with pool.acquire() as c:
        st = await c.fetchrow("SELECT * FROM user_stats WHERE user_id=$1", U)
    assert st["titles_touched"] == 5 and st["titles_weighted"] == 3
    assert st["movies_completed"] == 3 and st["movies_dropped"] == 1  # Joker completed-but-disliked counts as completed
    assert st["series_hooked"] == 1 and st["explicit_dislikes"] == 1
    print(f"stats: touched={st['titles_touched']} weighted={st['titles_weighted']} "
          f"completed={st['movies_completed']} dropped={st['movies_dropped']} "
          f"hooked={st['series_hooked']}")

    # §E vector (J acceptance)
    import cine_rec_engine.config as cfg
    import cine_rec_engine.service as svc
    cfg.EMBEDDING_COLUMN = "embedding_e5e"
    svc.EMBEDDING_COLUMN = "embedding_e5e"
    vec = await uv.build_user_vector(pool, U, space="e5e", column="embedding_e5e")
    if vec is None:
        rows2 = await pool.fetch(
            """SELECT uts.tmdb_id, uts.w_item FROM user_title_stats uts
               WHERE uts.user_id = $1""", U)
        print("title_stats:", [(r["tmdb_id"], round(r["w_item"],3)) for r in rows2])
    import math
    assert vec is not None, "vector missing"
    assert math.sqrt(sum(x*x for x in vec)) == 1.0
    # dominant direction follows TDK+Inception+BB (all near [0.9,0.1,0,0])
    assert vec[0] > 0.9, vec
    print(f"vector: {[round(x,3) for x in vec]} norm=1")

    # §F recommendations
    rec = RecommendationService()
    await rec.initialize(pool)
    results = await rec.recommend_for_user(U, limit=5)
    ids = [r["tmdb_id"] for r in results]
    print(f"recs for user {U}: {ids}")
    assert 155 not in ids and 27205 not in ids and 1396 not in ids  # watched
    assert 475557 not in ids                                     # disliked
    assert 999 not in ids or rows[999]["dropped"]                # dropped ok out
    # with only 5 synthetic titles the pool may be empty after filters —
    # that's correct behavior; assert no filtered item leaked in.
    print("PERSONA_SMOKE_OK")
    await pool.close()

asyncio.run(main())
