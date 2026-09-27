"""CLI: build and refresh the TMDB mirror.

    python -m ingest.cli bootstrap --dsn postgresql://... --popularity-min 1.0
    python -m ingest.cli refresh   --dsn postgresql://...
    python -m ingest.cli genres    --dsn postgresql://...
"""

from __future__ import annotations

import argparse
import asyncio
import os

import aiohttp
import asyncpg

from .exports import latest_export_url
from .loader import API_BASE, TmdbIngest
from cine_rec_engine.tmdb_client import fetch_with_retry


async def sync_genres(pool: asyncpg.Pool, api_key: str) -> None:
    """/genre/movie/list + /genre/tv/list — once per run (id lists rarely change)."""
    async with aiohttp.ClientSession() as session:
        for medium, table in (("movie", "tmdb_movie_genres"), ("tv", "tmdb_tv_genres")):
            data = await fetch_with_retry(
                session, f"{API_BASE}/genre/{medium}/list",
                {"api_key": api_key, "language": "en-US"},
            )
            rows = [(g["id"], g["name"]) for g in data.get("genres", [])]
            if rows:
                await pool.executemany(
                    f"INSERT INTO {table} (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name",
                    rows,
                )
            print(f"genres[{medium}]: {len(rows)}")


async def cmd_bootstrap(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(args.dsn)
    if not args.skip_genres:
        await sync_genres(pool, args.api_key)

    ingest = TmdbIngest(pool, api_key=args.api_key, rate_per_second=args.rate)

    async def stream():
        from .exports import iter_export_ids

        async with aiohttp.ClientSession() as session:
            for medium in ("movie", "tv"):
                url = latest_export_url(medium)
                print(f"discovery: {url}")
                count = 0
                async for entry in iter_export_ids(
                    medium, session, url,
                    min_popularity=args.popularity_min,
                    include_adult=args.include_adult,
                ):
                    count += 1
                    if count > args.limit:
                        break
                    yield entry.id, medium

    await ingest.bootstrap(stream(), workers=args.workers,
                           skip_existing=not args.force)
    print(ingest.stats.summary())
    await pool.close()


async def cmd_refresh(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(args.dsn)
    ingest = TmdbIngest(pool, api_key=args.api_key, rate_per_second=args.rate)
    await ingest.refresh(
        min_popularity=args.popularity_min,
        include_adult=args.include_adult,
        changes_days=args.days,
    )
    print(ingest.stats.summary())
    await pool.close()


async def cmd_genres(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(args.dsn)
    await sync_genres(pool, args.api_key)
    await pool.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="ingest", description="TMDB mirror loader")
    p.add_argument("--dsn", default=os.getenv("DATABASE_URL", ""))
    p.add_argument("--api-key", default=os.getenv("TMDB_API_KEY", ""))
    p.add_argument("--rate", type=int, default=40, help="max requests/second")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="full load from daily exports")
    b.add_argument("--popularity-min", type=float, default=0.0)
    b.add_argument("--include-adult", action="store_true")
    b.add_argument("--limit", type=int, default=10**9)
    b.add_argument("--skip-genres", action="store_true",
                   help="skip the genre-list sync (already synced)")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--force", action="store_true",
                   help="re-fetch titles already in the DB")
    b.set_defaults(fn=cmd_bootstrap)

    r = sub.add_parser("refresh", help="full daily loop: export diff + changes")
    r.add_argument("--days", type=int, default=1)
    r.add_argument("--popularity-min", type=float, default=0.0)
    r.add_argument("--include-adult", action="store_true")
    r.set_defaults(fn=cmd_refresh)

    g = sub.add_parser("genres", help="sync the two genre lists")
    g.set_defaults(fn=cmd_genres)

    args = p.parse_args()
    if not args.dsn:
        p.error("--dsn or DATABASE_URL required")
    if not args.api_key:
        p.error("--api-key or TMDB_API_KEY required")
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
