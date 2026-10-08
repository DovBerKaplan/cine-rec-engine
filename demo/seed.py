"""Load the bundled catalog dump (400 real TMDB titles) into the schema.

Uses the repo's own ingest upsert path — the demo data is exactly what
`ingest` writes in production. Data: TMDB, en-US, attribution in README.
"""
import asyncio
import gzip
import json
import os
import sys
from pathlib import Path
from cine_rec_engine import db
from cine_rec_engine import tables

sys.path.insert(0, str(Path(__file__).parent.parent))

DSN = os.environ.get("DATABASE_URL", "postgresql://demo:demo@localhost:54329/demo")
ROOT = Path(__file__).parent.parent


async def main():
    import asyncpg
    from ingest.loader import TmdbIngest

    pool = await asyncpg.create_pool(DSN)
    async with pool.acquire() as c:
        await db.execute(c, "CREATE EXTENSION IF NOT EXISTS vector")
        await db.execute(c, (ROOT / "docs" / "schema.sql").read_text())
        await db.execute(c,
            "ALTER TABLE {t_tmdb_movies} ADD COLUMN IF NOT EXISTS embedding_minilm vector(384)")
        await db.execute(c,
            "ALTER TABLE {t_tmdb_tv} ADD COLUMN IF NOT EXISTS embedding_minilm vector(384)")
    # offline loader: no fetches happen, so a placeholder key is fine
    ingest = TmdbIngest(pool, api_key="offline-demo")
    n = 0
    with gzip.open(ROOT / "demo" / "data" / "titles.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            await ingest._upsert_title(rec["payload"], rec["medium"])
            n += 1

    # vector writes need the pgvector codec on their connection
    from pgvector.asyncpg import register_vector

    async with pool.acquire() as c:
        await register_vector(c)
        with gzip.open(ROOT / "demo" / "data" / "titles.jsonl.gz", "rt") as f:
            for line in f:
                rec = json.loads(line)
                vec = rec.get("embedding_minilm")
                if vec:
                    table = tables.name(
                        "tmdb_movies" if rec["medium"] == "movie" else "tmdb_tv")
                    await db.execute(c,
                        f"UPDATE {table} SET embedding_minilm = $1 WHERE id = $2",
                        vec, rec["payload"]["id"])
    print(f"seeded {n} titles (+embeddings) into {DSN.split('@')[-1]}")
    await pool.close()


asyncio.run(main())
