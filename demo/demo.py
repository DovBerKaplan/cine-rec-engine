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

PRETTY = {
    "cosine_sim": "plot similarity",
    "keyword_sim": "shared keywords",
    "cast_sim": "shared cast",
    "director_match": "same director",
    "director_channel": "director recall",
    "writer_match": "same writer",
    "composer_match": "same composer",
    "dp_match": "same cinematographer",
    "tmdb_rec_decay": "TMDB behavior graph",
    "shared_collection": "same saga",
    "shared_network": "same network",
    "style_match": "same style tags",
    "company_sim": "same studio",
    "genre_priority_sum": "genre overlap",
    "tone_compatibility": "same tone",
    "medium_mismatch": "cross-medium penalty",
    "narrative_match": "same narrative structure",
}


async def main():
    seed_id = int(sys.argv[1]) if len(sys.argv) > 1 else 155

    import asyncpg
    from cine_rec_engine import RecommendationService
    from cine_rec_engine.queries import enrich_candidates_batch, get_movie_info_batch
    from cine_rec_engine.service import (
        ACTIVE_WEIGHTS,
        FEATURE_NAMES,
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
        # skip near-constant features in the WHY (they score everything):
        # audience/tone fire on ~all candidates — informative in the score,
        # noise in an explanation.
        BORING = {"audience_compatibility", "tone_compatibility",
                  "rating_bonus", "votes_gt15k", "low_votes_high_rating"}
        contributions = sorted(
            ((ACTIVE_WEIGHTS[name] * val, name)
             for name, val in zip(FEATURE_NAMES, vec)
             if name not in BORING),
            reverse=True,
        )
        why = " · ".join(
            f"{PRETTY.get(name, name)}" for score, name in contributions[:3] if score > 0
        )
        title = r.get("title_en") or r.get("title") or r["tmdb_id"]
        kind = "series" if r["media_type"] == "tv" else "film"
        year = f" ({r.get('release_year')})" if r.get("release_year") else ""
        print(f"  {title}{year}  [{kind}]  score {r['score']:.1f}")
        print(f"      why: {why}")

    print("─" * 62 + "\n")
    await pool.close()


asyncio.run(main())
