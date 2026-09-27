"""Print recommendations with the WHY — each result's top feature drivers.

Run:  python demo/demo.py [tmdb_id]      (default: 155, The Dark Knight)
"""
import asyncio
import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()  # clean demo output

sys.path.insert(0, str(Path(__file__).parent.parent))

DSN = os.environ.get("DATABASE_URL", "postgresql://demo:demo@localhost:54329/demo")


async def main():
    seed_id = int(sys.argv[1]) if len(sys.argv) > 1 else 155

    import asyncpg
    from cine_rec_engine import RecommendationService
    from cine_rec_engine.queries import enrich_candidates_batch, get_movie_info_batch
    from cine_rec_engine.service import (
        ACTIVE_WEIGHTS,
        explain_features,
        feature_vector,
    )

    pool = await asyncpg.create_pool(DSN)
    rec = RecommendationService()
    await rec.initialize(pool)
    results = await rec.find_similar(seed_id, limit=8)

    seed_map = await get_movie_info_batch(pool, [seed_id])
    if seed_map:
        seed = dict(seed_map[seed_id])
        print(f"\nBecause you watched  {seed.get('title_en') or seed.get('title')}"
              f"  ({seed.get('release_year')})\n" + "─" * 62)
    else:
        print(f"\nSeed {seed_id}\n" + "─" * 62)

    enriched = await enrich_candidates_batch(pool, [r["tmdb_id"] for r in results])

    for r in results:
        cand = dict(enriched.get(r["tmdb_id"], {}))
        cand.setdefault("id", r["tmdb_id"])
        for k, v in r.items():
            cand.setdefault(k, v)
        vec = feature_vector(cand, cand, cand.get("genres", []), seed)
        why = " · ".join(explain_features(vec, ACTIVE_WEIGHTS)) or "—"

        title = r.get("title_en") or r.get("title") or r["tmdb_id"]
        kind = "series" if r["media_type"] == "tv" else "film"
        year = f" ({r.get('release_year')})" if r.get("release_year") else ""
        print(f"  {title}{year}  [{kind}]  score {r['score']:.1f}")
        print(f"      why: {why}")

    print("─" * 62 + "\n")
    await pool.close()


asyncio.run(main())
