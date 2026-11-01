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

# (case name, medium, min_popularity, include_adult, fetch_none) —
# the exact sequence the Rust side replays. Notes on accounting:
# * movie-adult (5.0, True): Python's ingest_one checks
#   `payload.get("adult")` FIRST, so an adult title rejected on
#   popularity still counts filtered_ADULT.
# * movie-pop-str ("23.5") INGESTS — asyncpg decodes str→Decimal for
#   numeric (never was a bind failure).
# * fetch_none steps simulate a fetch error (the patched fetch
#   returns None / Rust returns None) → failed (review L-5).
PLAN = [
    ("movie-full", "movie", 0.0, False, False),
    ("movie-shrink", "movie", 0.0, False, False),   # mirror: shrunk bridges
    ("tv-full", "tv", 0.0, False, False),           # id 155 again, tv side
    ("movie-155-again", "movie", 0.0, False, False),  # H-1: tv networks must survive
    ("movie-minimal", "movie", 0.0, False, False),
    ("movie-adult", "movie", 0.0, False, False),    # → filtered_adult
    ("movie-adult", "movie", 5.0, True, False),     # → filtered_adult (adult checked first)
    ("movie-int-votes", "movie", 0.0, False, False),
    ("movie-recs-empty", "movie", 0.0, False, False),
    ("tv-external-str", "tv", 0.0, False, False),
    ("movie-pop-str", "movie", 0.0, False, False),  # INGESTS: str→Decimal numeric
    ("movie-success-false", "movie", 0.0, False, False),  # gate rejection
    ("movie-float-id", "movie", 0.0, False, False),
    ("movie-bad-date", "movie", 0.0, False, False),     # builder ValueError
    ("movie-missing-id", "movie", 0.0, False, False),   # KeyError
    ("movie-rec-missing-id", "movie", 0.0, False, False),  # partial state
    ("movie-credited-noid", "movie", 0.0, False, False),  # people names KeyError
    # ── review round 1 ────────────────────────────────────────────
    ("movie-runtime-str", "movie", 0.0, False, False),   # M-1: str int → bind DataError
    ("movie-genre-id-str", "movie", 0.0, False, False),  # M-1: same, at the genres dim
    ("movie-voteavg-true", "movie", 0.0, False, False),  # M-1: bool numeric → 1, ingests
    ("movie-pop-str-bad", "movie", 0.0, False, False),   # M-3: float("abc") → failed
    ("movie-collection-false", "movie", 0.0, False, False),  # M-3: ingests, collection NULL
    ("movie-genres-null", "movie", 0.0, False, False),   # M-3: TypeError → fact row only
    ("movie-kw-midfail", "movie", 0.0, False, False),    # M-2: fails after people+genres
    ("movie-co-midfail", "movie", 0.0, False, False),    # M-2: fails after keywords
    ("movie-order-huge", "movie", 0.0, False, False),    # M-2: INT4 bind after cast DELETE
    ("movie-crew-unhashable-job", "movie", 0.0, False, False),  # L-1: TypeError in keep_crew
    ("movie-recs-list", "movie", 0.0, False, False),     # L-1: AttributeError at recs
    ("movie-fetch-error", "movie", 0.0, False, True),    # L-5: fetch → None → failed
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

        for name, medium, min_pop, include_adult, fetch_none in PLAN:
            # fetch_none steps are synthetic — no payload needed.
            payload = payloads.get(name, {}).get("payload")
            # Per-case patched fetch: no HTTP, no retry (the patch
            # therefore also bypasses the semaphore/limiter and the
            # "error" in data → None conversion — the fetch-error path
            # itself is exercised by the fetch_none steps, which patch
            # the fetch to return None like a failed fetch would).
            if fetch_none:
                async def _no_payload():
                    return None
                ingest._fetch_title = (
                    lambda session, mid, m: _no_payload()
                )
            else:
                ingest._fetch_title = (
                    lambda session, mid, m, _p=payload: _return_payload(_p)
                )
            await ingest.ingest_one(
                None, (payload or {}).get("id", 0), medium,
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
            {
                "name": n,
                "medium": m,
                "min_popularity": mp,
                "include_adult": ia,
                "fetch_none": fn,
            }
            for n, m, mp, ia, fn in PLAN
        ],
        "stats": stats,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"pyload ingested; stats={stats}")


if __name__ == "__main__":
    asyncio.run(main())
