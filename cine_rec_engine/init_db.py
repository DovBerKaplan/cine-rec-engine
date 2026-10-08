"""One-shot schema bring-up and startup verification for deployments.

- apply_schema(): idempotently applies the bundled schema files (mirrors
  of docs/schema.sql and docs/user_data.sql — a test enforces the sync)
  over asyncpg's simple query protocol, one execute() per file.
- check_schema(): verifies an explicit table map against the database.

The failure policy follows the engine's invariant: an explicitly remapped
table that does not exist is a configuration error (loud), while a missing
default-named table only degrades its channel (reported, not fatal).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from . import tables

_SQL_DIR = Path(__file__).parent / "sql"

# The backbone of every recall path — worth reporting at startup even
# though each missing table degrades its channel rather than failing.
_CORE_PROBES = ("tmdb_media", "tmdb_movies", "tmdb_tv", "user_watch_events")


def _read_sql(filename: str) -> str:
    path = _SQL_DIR / filename
    if not path.exists():
        # source checkout without the copies, or an sdist without package-data
        alt = Path(__file__).resolve().parent.parent / "docs" / filename
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"{filename} not found (looked in {_SQL_DIR} and {alt})"
            )
    return path.read_text(encoding="utf-8")


async def apply_schema(conn, full: bool = True) -> dict:
    """Apply the catalog schema (and, with full=True, the user layer).

    Returns {"catalog": bool, "user_layer": bool, "user_layer_error": str?}.
    The catalog part is the hard requirement; the user layer needs the
    pgvector extension and degrades with a loud warning when absent —
    find_similar keeps working, personalization does not."""
    out = {"catalog": False, "user_layer": False, "user_layer_error": None}
    await conn.execute(_read_sql("schema.sql"))
    out["catalog"] = True
    if full:
        try:
            await conn.execute(_read_sql("user_data.sql"))
            out["user_layer"] = True
        except Exception as e:  # extension missing / no privileges
            out["user_layer_error"] = str(e).strip()
            logger.warning(
                "user layer NOT applied (pgvector extension unavailable?): "
                f"{out['user_layer_error']} — user recommendations disabled, "
                "find_similar unaffected"
            )
    return out


async def check_schema(pool) -> List[str]:
    """Hard problems only: explicitly-mapped tables that don't exist.

    Every entry in the active map was named by the operator, so a missing
    relation is a typo or a wrong database — exactly the kind of thing the
    degrade-never-fail invariant would otherwise hide behind empty results."""
    problems: List[str] = []
    for logical in sorted(tables.active_map()):
        physical = tables.name(logical)
        oid = await pool.fetchval("SELECT to_regclass($1)", physical)
        if oid is None:
            problems.append(
                f"mapped table {logical} -> {physical}: not found in the database"
            )
    return problems


async def core_report(pool) -> List[str]:
    """Soft warnings: core default channels that are missing (degraded)."""
    warnings: List[str] = []
    for logical in _CORE_PROBES:
        physical = tables.name(logical)
        oid = await pool.fetchval("SELECT to_regclass($1)", physical)
        if oid is None:
            warnings.append(f"core table {logical} (as {physical}): missing"
                            " — its channel is degraded")
    return warnings


async def startup_verify(pool) -> Optional[List[str]]:
    """Raise nothing; return the hard problems (empty = fine to serve).

    Called from the serving lifespan when a table map is active, so a bad
    map stops the container instead of serving empty recommendations."""
    if err := tables.load_error():
        return [f"schema map failed to load: {err}"]
    return await check_schema(pool)
