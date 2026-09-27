"""Quickstart — full flow against your own PostgreSQL.

Prereqs:
    psql -d yourdb -f docs/schema.sql     # create the tables
    # load a TMDB mirror (docs/data.md), then:

    python examples/quickstart.py postgresql://user:pw@localhost/yourdb 155
"""

import asyncio
import sys

import asyncpg

from cine_rec_engine import RecommendationService


async def main(dsn: str, seed: int) -> None:
    pool = await asyncpg.create_pool(dsn)
    rec = RecommendationService()
    await rec.initialize(pool)

    results = await rec.find_similar(seed, limit=10)
    print(f"Similar to tmdb {seed}:")
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("title_en") or r["tmdb_id"]
        print(f"{i:2}. {title} ({r.get('media_type')})  score={r['score']:.2f}")

    await pool.close()


if __name__ == "__main__":
    dsn = sys.argv[1] if len(sys.argv) > 1 else "postgresql://localhost/postgres"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 155  # The Dark Knight
    asyncio.run(main(dsn, seed))
