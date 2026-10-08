"""Integration harness fixture (RFC M3): one suite, two entry points.

    pytest -m integration                       # uses CINE_REC_INTEGRATION_DSN if set
    make test-integration                       # spins a throwaway pgvector container

Skips cleanly (not fails) when neither a DSN nor the auto-spin env is
configured — the offline suite stays the default dev loop. The suite is
the CI demo-eval class of tests: recall smoke, the eval regression gate,
personalization end-to-end, and the user tilt — exactly the bugs the
offline suite cannot see (SQL + enrichment + channel wiring).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DSN_ENV = "CINE_REC_INTEGRATION_DSN"
AUTO_ENV = "CINE_REC_INTEGRATION_AUTO"
RESEED_ENV = "CINE_REC_INTEGRATION_RESEED"


def _psql_ready(dsn: str) -> bool:
    import asyncpg

    async def probe():
        # one asyncio.run: an asyncpg connection is bound to the loop it
        # was created on — probing across separate runs always fails
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()

    try:
        asyncio_run(probe())
        return True
    except Exception:
        return False


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def _seed_empty(dsn: str) -> bool:
    import asyncpg

    async def check():
        conn = await asyncpg.connect(dsn)
        try:
            n = await conn.fetchval("SELECT count(*) FROM tmdb_media")
            return (n or 0) == 0
        except asyncpg.UndefinedTableError:
            return True  # fresh database — no schema yet
        finally:
            await conn.close()

    return asyncio_run(check())


def _apply_schema(dsn: str) -> None:
    import asyncpg

    async def apply():
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(open(os.path.join(REPO, "docs", "schema.sql")).read())
            await conn.execute(open(os.path.join(REPO, "docs", "user_data.sql")).read())
        finally:
            await conn.close()

    asyncio_run(apply())


def _seed_demo(dsn: str) -> None:
    env = dict(os.environ, DATABASE_URL=dsn)
    subprocess.run(
        [os.path.join(REPO, ".venv", "bin", "python")
         if os.path.exists(os.path.join(REPO, ".venv", "bin", "python"))
         else "python3",
         os.path.join(REPO, "demo", "seed.py")],
        env=env, check=True, cwd=REPO,
    )


def _spin_container() -> tuple[str, str]:
    """Throwaway pgvector on a free port; returns (dsn, container_id)."""
    if not shutil.which("docker"):
        pytest.skip("integration: docker not available and no DSN configured")
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    name = f"cine-rec-it-{int(time.time())}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{port}:5432",
         "-e", "POSTGRES_USER=demo", "-e", "POSTGRES_PASSWORD=demo",
         "-e", "POSTGRES_DB=demo", "pgvector/pgvector:pg16"],
        check=True, capture_output=True)
    dsn = f"postgresql://demo:demo@127.0.0.1:{port}/demo"
    for _ in range(60):  # ~30s startup budget
        if _psql_ready(dsn):
            return dsn, name
        time.sleep(0.5)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    raise RuntimeError("throwaway postgres did not become ready in 30s")


@pytest.fixture(scope="session")
def integration_dsn():
    dsn = os.environ.get(DSN_ENV)
    container = None
    if dsn:
        if not _psql_ready(dsn):
            pytest.fail(f"integration: {DSN_ENV} set but not reachable: {dsn}")
    elif os.environ.get(AUTO_ENV) == "1":
        dsn, container = _spin_container()
    else:
        pytest.skip(
            f"integration: set {DSN_ENV} to a seeded demo DB, or run "
            f"`make test-integration` ({AUTO_ENV}=1 spins a throwaway container)")
    try:
        if os.environ.get(RESEED_ENV) == "1" or _seed_empty(dsn):
            _apply_schema(dsn)
            _seed_demo(dsn)
        yield dsn
    finally:
        if container:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)


@pytest.fixture()
async def integration_env(integration_dsn):
    """Pool + service on the TEST'S event loop (asyncpg pools are
    loop-bound, so this is function-scoped; the DSN/seeding above is
    the expensive session-level part) with the demo embedding space
    selected — the demo column is embedding_minilm, and the
    process-global config may already have imported a different
    default, so the switch mirrors benchmarks/smoke_personalization.py.
    """
    import asyncpg

    import cine_rec_engine.config as cfg
    import cine_rec_engine.service as svc

    cfg.EMBEDDING_COLUMN = "embedding_minilm"
    svc.EMBEDDING_COLUMN = "embedding_minilm"

    pool = await asyncpg.create_pool(integration_dsn, min_size=2, max_size=10)
    service = svc.RecommendationService()
    await service.initialize(pool)
    try:
        yield {"pool": pool, "service": service, "dsn": integration_dsn}
    finally:
        await pool.close()
