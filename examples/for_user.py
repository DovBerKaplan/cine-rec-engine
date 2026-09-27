"""The one closed user path (spec §A): user_id in → list with why out.

Copy-paste against the demo catalog:

    cd demo && docker compose up -d db
    psql -h localhost -p 54329 -U demo -d demo \
        -f ../docs/schema.sql -f ../docs/user_data.sql   # first run only
    python examples/for_user.py --dsn postgresql://demo:demo@localhost:54329/demo

Saga rule demonstrated live: a completed title never returns, and its
saga ADVANCES (watch the collection feature in the why).
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

NOW = datetime.now(timezone.utc)

# two demo personas over the bundled catalog — swap freely
PROFILE_NOLAN_FAN = [
    dict(tmdb_id=155, media_type="movie", watched_sec=8000, duration_sec=8000, completed=True),
    dict(tmdb_id=77, media_type="movie", watched_sec=7000, duration_sec=7000, completed=True),
    dict(tmdb_id=157336, media_type="movie", watched_sec=8000, duration_sec=8000, completed=True),
]
PROFILE_CRIME_TV = [
    dict(tmdb_id=1396, media_type="tv", watched_sec=2700, duration_sec=2700,
         completed=True, season=1, episode=1),
    dict(tmdb_id=1396, media_type="tv", watched_sec=2700, duration_sec=2700,
         completed=True, season=1, episode=2),
    dict(tmdb_id=1396, media_type="tv", watched_sec=2700, duration_sec=2700,
         completed=True, season=1, episode=3),
    dict(tmdb_id=1438, media_type="tv", watched_sec=2700, duration_sec=2700,
         completed=True, season=1, episode=1),
]


async def run(dsn: str, user_id: int, profile: list) -> None:
    import asyncpg

    from cine_rec_engine import RecommendationService, user_stats, user_vector

    pool = await asyncpg.create_pool(dsn)

    # 1. events (hot path: exactly this, nothing more)
    for ev in profile:
        await user_stats.record_event(
            pool, {**ev, "user_id": user_id,
                   "watched_at": NOW - timedelta(days=1)})
    # 2. profile + vector (async/nightly in production; here for the demo)
    await user_stats.refresh_user_stats(pool, user_id)
    await user_vector.build_user_vector(pool, user_id)

    # 3. recommendations — user_id in, why out
    rec = RecommendationService()
    await rec.initialize(pool)
    out = await rec.recommend_for_user(user_id, limit=8, include_why=True)

    print(f"user {user_id}: reason={out['reason']} vector={out['vector_used']}")
    print(f"  seeds: {[(i, m, w) for i, m, w in out['seeds']]}")
    for r in out["results"]:
        title = r.get("title_en") or r.get("title") or r["tmdb_id"]
        why = " · ".join(out["why"].get(str(r["tmdb_id"]), []))
        print(f"  → {title} ({r['score']:.1f})  why: {why or '—'}")

    # 4. the invariants, asserted where they're cheap:
    seen = {(e["tmdb_id"], e["media_type"]) for e in profile}
    leaked = [r["tmdb_id"] for r in out["results"]
              if (r["tmdb_id"], r["media_type"]) in seen]
    assert not leaked, f"watched titles leaked: {leaked}"
    await pool.close()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://demo:demo@localhost:54329/demo")
    p.add_argument("--user", type=int, default=None,
                   help="existing user (skips the demo events)")
    args = p.parse_args()

    if args.user is not None:
        await run(args.dsn, args.user, [])
        return
    print("== persona A: Nolan films ==")
    await run(args.dsn, 1, PROFILE_NOLAN_FAN)
    print("\n== persona B: crime TV ==")
    await run(args.dsn, 2, PROFILE_CRIME_TV)


asyncio.run(main())
