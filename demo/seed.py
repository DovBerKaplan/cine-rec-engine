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

sys.path.insert(0, str(Path(__file__).parent.parent))

DSN = os.environ.get("DATABASE_URL", "postgresql://demo:demo@localhost:54329/demo")
ROOT = Path(__file__).parent.parent


async def main():
    import asyncpg
    from ingest.loader import TmdbIngest

    pool = await asyncpg.create_pool(DSN)
    async with pool.acquire() as c:
        await c.execute((ROOT / "docs" / "schema.sql").read_text())
    # offline loader: no fetches happen, so a placeholder key is fine
    ingest = TmdbIngest(pool, api_key="offline-demo")

    n = 0
    with gzip.open(ROOT / "demo" / "data" / "titles.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            await ingest._upsert_title(rec["payload"], rec["medium"])
            n += 1
    print(f"seeded {n} titles into {DSN.split('@')[-1]}")
    await pool.close()


asyncio.run(main())
