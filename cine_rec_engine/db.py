"""SQL execution helpers that resolve logical table names.

Thin wrappers around asyncpg's fetch/fetchrow/fetchval/execute: the SQL
passed in may contain {t_<logical>} placeholders (see tables.py), which
are substituted with the configured physical names before execution.
SQL without placeholders passes through unchanged, so existing call
sites work whether or not a table map is active.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import tables

__all__ = ["execute", "executemany", "fetch", "fetchrow", "fetchval"]


async def executemany(db: Any, sql: str, *args: Any) -> str:
    """await db.executemany(...) with logical-name resolution."""
    return await db.executemany(tables.resolve(sql), *args)


async def fetch(db: Any, sql: str, *args: Any) -> list:
    """await db.fetch(...) with logical-name resolution."""
    return await db.fetch(tables.resolve(sql), *args)


async def fetchrow(db: Any, sql: str, *args: Any) -> Any:
    """await db.fetchrow(...) with logical-name resolution."""
    return await db.fetchrow(tables.resolve(sql), *args)


async def fetchval(db: Any, sql: str, *args: Any) -> Any:
    """await db.fetchval(...) with logical-name resolution."""
    return await db.fetchval(tables.resolve(sql), *args)


async def execute(db: Any, sql: str, *args: Sequence[Any]) -> str:
    """await db.execute(...) with logical-name resolution."""
    return await db.execute(tables.resolve(sql), *args)
