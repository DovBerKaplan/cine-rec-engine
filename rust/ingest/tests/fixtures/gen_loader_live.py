#!/usr/bin/env python3
"""LIVE loader parity, python side: run the REAL TmdbIngest against a
dedicated schema (`pyload`) and record the stats + the exact plan.

Usage:
    .venv/bin/python gen_loader_live.py --dsn postgresql://... --out loader_live_stats.json

What it does (all inside schema `pyload`):
  * CREATE SCHEMA pyload; applies docs/schema.sql with search_path=pyload
  * TmdbIngest with _fetch_title PATCHED per case (no HTTP): each plan
    step fetches its payload from loader_payloads.json
  * runs ingest_one per plan step — the real gate, the real upserts,
    the real bridge mirroring, the real stats accounting
  * writes {plan: [...], stats: {...}, schema: "pyload"} as JSON

The Rust live test replays the SAME plan against schema `rsload` on
the same database and compares table states (symmetric EXCEPT) and
the stats counters. The plan deliberately includes failure paths
(bind failures, gate rejections) — both sides must end with the SAME
partial state and counters.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # fixtures → tests → ingest → rust → repo root

sys.path.insert(0, str(ROOT))

import ingest.loader as L  # noqa: E402

# (case name, medium, min_popularity, include_adult) — the exact
# sequence the Rust side replays. movie-adult runs twice: once for the
# adult gate, once for the popularity gate.
PLAN = [
    ("movie-full", "movie", 0.0, False),
    ("movie-shrink", "movie", 0.0, False),   # mirror: shrunk bridges
    ("tv-full", "tv", 0.0, False),           # id 155 again, tv side
    ("movie-minimal", "movie", 0.0, False),
    ("movie-adult", "movie", 0.0, False),    # → filtered_adult
    ("movie-adult", "movie", 5.0, True),     # → filtered_popularity
    ("movie-int-votes", "movie", 0.0, False),
    ("movie-recs-empty", "movie", 0.0, False),
    ("tv-external-str", "tv", 0.0, False),
    ("movie-pop-str", "movie", 0.0, False),  # bind failure → failed
    ("movie-success-false", "movie", 0.0, False),  # gate rejection
    ("movie-float-id", "movie", 0.0, False),
    ("movie-bad-date", "movie", 0.0, False),     # builder ValueError
    ("movie-missing-id", "movie", 0.0, False),   # KeyError
    ("movie-rec-missing-id", "movie", 0.0, False),  # partial state
    ("movie-credited-noid", "movie", 0.0, False),   # people names KeyError
]

SCHEMA = "pyload"


def schema_sql():
    # The whole file, ONE execute — asyncpg's simple query protocol
    # takes multi-statement text (exactly what init_db.py does).
    return (ROOT / "docs" / "schema.sql").read_text()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payloads = {
        c["name"]: c
        for c in json.loads(
            (HERE / "loader_payloads.json").read_text())["payloads"]
    }

    # 1. fresh schema, applied once (search_path at apply time).
    admin = await asyncpg.connect(args.dsn)
    try:
        await admin.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await admin.execute(f"CREATE SCHEMA {SCHEMA}")
        await admin.execute(f"SET search_path TO {SCHEMA}")
        await admin.execute(schema_sql())
    finally:
        await admin.close()

    # 2. the ingest runs on a real POOL (the loader calls
    # pool.acquire()); every pooled connection gets the schema via
    # server_settings.
    pool = await asyncpg.create_pool(
        args.dsn, server_settings={"search_path": SCHEMA})
    try:
        ingest = L.TmdbIngest(pool, api_key="fixture-key")

        async def _return_payload(payload):
            return payload

        for name, medium, min_pop, include_adult in PLAN:
            payload = payloads[name]["payload"]
            # Per-case patched fetch: no HTTP, no retry — the payload
            # itself is exactly what the fixture holds.
            ingest._fetch_title = (
                lambda session, mid, m, _p=payload: _return_payload(_p)
            )
            await ingest.ingest_one(
                None, payload.get("id", 0), medium,
                min_popularity=min_pop, include_adult=include_adult,
            )
    finally:
        await pool.close()

    stats = {
        "attempted": ingest.stats.attempted,
        "ingested": ingest.stats.ingested,
        "filtered_adult": ingest.stats.filtered_adult,
        "filtered_popularity": ingest.stats.filtered_popularity,
        "failed": ingest.stats.failed,
        "complete": ingest.stats.complete,
    }
    out = {
        "schema": SCHEMA,
        "plan": [
            {"name": n, "medium": m, "min_popularity": mp, "include_adult": ia}
            for n, m, mp, ia in PLAN
        ],
        "stats": stats,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"pyload ingested; stats={stats}")


if __name__ == "__main__":
    asyncio.run(main())
