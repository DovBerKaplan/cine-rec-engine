"""cine-rec — deployment helper: init the schema, check the setup, serve.

    cine-rec init    [--dsn ...] [--catalog-only]   apply the schema (idempotent)
    cine-rec check   [--dsn ...]                    verify DSN, schema, table map
    cine-rec serve   [--host H] [--port P]          run the HTTP API (uvicorn)

Every subcommand takes --dsn, defaulting to DATABASE_URL then
CINE_REC_DATABASE_URL — same resolution order as the serving layer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from . import init_db, tables


def _dsn(args: argparse.Namespace) -> Optional[str]:
    dsn = args.dsn or os.getenv("DATABASE_URL") or os.getenv("CINE_REC_DATABASE_URL")
    if not dsn:
        print("error: no database — pass --dsn or set DATABASE_URL",
              file=sys.stderr)
        sys.exit(2)
    return dsn


async def cmd_init(args: argparse.Namespace) -> int:
    import asyncpg

    dsn = _dsn(args)
    conn = await asyncpg.connect(dsn)
    try:
        result = await init_db.apply_schema(conn, full=not args.catalog_only)
    finally:
        await conn.close()
    print(f"catalog schema: applied ({dsn.split('@')[-1]})")
    if result["user_layer"]:
        print("user layer: applied (events, stats, vectors)")
    elif result["user_layer_error"]:
        print(f"user layer: SKIPPED — {result['user_layer_error']}\n"
              "  (needs the pgvector extension; find_similar still works)")
    return 0


async def cmd_check(args: argparse.Namespace) -> int:
    import asyncpg

    dsn = _dsn(args)
    if err := tables.load_error():
        print(f"schema map: ERROR — {err}", file=sys.stderr)
        return 2
    conn = await asyncpg.connect(dsn)
    try:
        await conn.fetchval("SELECT 1")
        print(f"connect: ok ({dsn.split('@')[-1]})")
        mapped = tables.active_map()
        print(f"table map: {len(mapped)} remapped"
              + (f" — {', '.join(f'{k}->{v}' for k, v in sorted(mapped.items()))}"
                 if mapped else ""))
        problems = await init_db.check_schema(conn)
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        for w in await init_db.core_report(conn):
            print(f"warning: {w}")
        if problems:
            return 1
        print("schema: ok")
        return 0
    finally:
        await conn.close()


def cmd_serve(args: argparse.Namespace) -> int:
    if args.dsn:
        os.environ["DATABASE_URL"] = args.dsn
    try:
        import uvicorn
    except ImportError:
        print("error: serving needs the serve extra — "
              'pip install "cine-rec-engine[serve]"', file=sys.stderr)
        return 2
    uvicorn.run("cine_rec_engine.serve:app",
                host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cine-rec",
        description="Deploy the recommendation engine: init, check, serve.",
    )
    p.add_argument("--dsn", default=None,
                   help="PostgreSQL DSN (default: DATABASE_URL env)")
    # same flag accepted AFTER the subcommand (SUPPRESS keeps the root's
    # value when the subcommand form isn't used)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dsn", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", parents=[common],
                        help="apply the schema (idempotent)")
    pi.add_argument("--catalog-only", action="store_true",
                    help="skip the user layer (events/stats/vectors)")
    pi.set_defaults(fn=cmd_init)

    pc = sub.add_parser("check", parents=[common],
                        help="verify DSN, schema, and table map")
    pc.set_defaults(fn=cmd_check)

    ps = sub.add_parser("serve", parents=[common],
                        help="run the HTTP API (uvicorn)")
    ps.add_argument("--host", default=os.getenv("CINE_REC_HOST", "0.0.0.0"))
    ps.add_argument("--port", type=int,
                    default=int(os.getenv("CINE_REC_PORT", "8000")))
    ps.add_argument("--log-level", default="info")
    ps.set_defaults(fn=cmd_serve)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if asyncio.iscoroutinefunction(args.fn):
            return asyncio.run(args.fn(args))
        return args.fn(args)
    except tables.TableMapError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
