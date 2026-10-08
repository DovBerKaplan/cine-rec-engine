"""Deployment surface: bundled SQL sync, apply_schema, check_schema,
the cine-rec CLI, and the serving lifespan's new knobs."""

from __future__ import annotations

from pathlib import Path

import pytest

from cine_rec_engine import init_db, tables

REPO = Path(__file__).resolve().parent.parent
PKG_SQL = Path(init_db.__file__).parent / "sql"


class FakeConn:
    def __init__(self, execute_fails_on=None, regclass_missing=()):
        self.executed: list[str] = []
        self._fail_on = execute_fails_on or ()
        self._missing = set(regclass_missing)

    async def execute(self, sql, *args):
        self.executed.append(sql)
        for frag in self._fail_on:
            if frag in sql:
                raise RuntimeError(f"boom: {frag}")
        return "OK"

    async def fetchval(self, sql, *args):
        if "to_regclass" in sql:
            return None if args and args[0] in self._missing else "oid"
        return None

    async def close(self):
        pass


class FakePool:
    def __init__(self, conn):
        self._conn = conn
        self._closed = False

    def acquire(self):
        return self

    def get_max_size(self):
        return 20

    async def fetchval(self, sql, *args):
        return await self._conn.fetchval(sql, *args)

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False

    async def close(self):
        self._closed = True


class TestSqlPackageSync:
    def test_bundled_sql_matches_docs(self):
        for name in ("schema.sql", "user_data.sql"):
            bundled = (PKG_SQL / name).read_text(encoding="utf-8")
            docs = (REPO / "docs" / name).read_text(encoding="utf-8")
            assert bundled == docs, (
                f"{name}: cine_rec_engine/sql/ is stale — re-copy from docs/ "
                "(the wheel ships the package copy)"
            )


class TestApplySchema:
    async def test_full_apply_runs_both_files(self):
        conn = FakeConn()
        result = await init_db.apply_schema(conn, full=True)
        assert result["catalog"] is True
        assert result["user_layer"] is True
        assert len(conn.executed) == 2
        assert "tmdb_movies" in conn.executed[0]
        assert "user_watch_events" in conn.executed[1]

    async def test_user_layer_failure_degrades(self):
        conn = FakeConn(execute_fails_on=("user_watch_events",))
        result = await init_db.apply_schema(conn, full=True)
        assert result["catalog"] is True
        assert result["user_layer"] is False
        assert result["user_layer_error"]

    async def test_catalog_only(self):
        conn = FakeConn()
        result = await init_db.apply_schema(conn, full=False)
        assert result["catalog"] is True
        assert result["user_layer"] is False
        assert len(conn.executed) == 1


class TestCheckSchema:
    async def test_missing_mapped_table_is_a_problem(self):
        tables.set_table_map({"tmdb_media": "app_media"})
        try:
            pool = FakePool(FakeConn(regclass_missing=("app_media",)))
            problems = await init_db.check_schema(pool)
            assert problems == [
                "mapped table tmdb_media -> app_media: not found in the database"
            ]
        finally:
            tables.set_table_map()

    async def test_present_mapped_table_is_clean(self):
        tables.set_table_map({"tmdb_media": "app_media"})
        try:
            pool = FakePool(FakeConn())
            assert await init_db.check_schema(pool) == []
        finally:
            tables.set_table_map()

    async def test_no_map_no_queries(self):
        pool = FakePool(FakeConn())
        assert await init_db.check_schema(pool) == []

    async def test_core_report_warns_on_missing_defaults(self):
        pool = FakePool(FakeConn(regclass_missing=("tmdb_media",)))
        warnings = await init_db.core_report(pool)
        assert any("tmdb_media" in w for w in warnings)


class TestCli:
    def test_parser_shape(self):
        from cine_rec_engine.cli import build_parser

        p = build_parser()
        args = p.parse_args(["init", "--dsn", "postgresql://x"])
        assert args.fn.__name__ == "cmd_init"
        assert args.catalog_only is False
        args = p.parse_args(["check"])
        assert args.fn.__name__ == "cmd_check"
        # --dsn accepted after the subcommand too
        args = p.parse_args(["check", "--dsn", "postgresql://x"])
        assert args.dsn == "postgresql://x"
        args = p.parse_args(["serve", "--port", "9000"])
        assert args.port == 9000

    def test_init_applies_and_reports(self, monkeypatch, capsys):
        import asyncpg

        from cine_rec_engine import cli

        conn = FakeConn()

        async def fake_connect(dsn):
            return conn

        monkeypatch.setattr(asyncpg, "connect", fake_connect)
        rc = cli.main(["--dsn", "postgresql://u@h/db", "init"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "catalog schema: applied" in out
        assert "user layer: applied" in out

    def test_check_flags_missing_mapped_table(self, monkeypatch, capsys):
        import asyncpg

        from cine_rec_engine import cli

        tables.set_table_map({"tmdb_media": "app_media"})
        conn = FakeConn(regclass_missing=("app_media",))

        async def fake_connect(dsn):
            return conn

        monkeypatch.setattr(asyncpg, "connect", fake_connect)
        try:
            rc = cli.main(["--dsn", "postgresql://u@h/db", "check"])
            err = capsys.readouterr().err
            assert rc == 1
            assert "tmdb_media -> app_media" in err
        finally:
            tables.set_table_map()

    def test_check_requires_dsn(self, monkeypatch, capsys):
        from cine_rec_engine import cli

        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("CINE_REC_DATABASE_URL", raising=False)
        with pytest.raises(SystemExit):
            cli.main(["check"])


class TestServeLifespan:
    def _app(self):
        from cine_rec_engine.serve import create_app

        return create_app()

    def test_pool_size_env_is_honored(self, monkeypatch):
        import asyncpg

        from fastapi.testclient import TestClient

        captured = {}

        async def fake_create_pool(url, **kwargs):
            captured.update(kwargs)

            class P(FakePool):
                pass

            return P(FakeConn())

        monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
        monkeypatch.setenv("CINE_REC_POOL_SIZE", "7")
        with TestClient(self._app()) as client:
            assert client.get("/health").status_code == 200
        assert captured["max_size"] == 7

    def test_auto_init_applies_schema(self, monkeypatch):
        import asyncpg

        from fastapi.testclient import TestClient

        conn = FakeConn()

        async def fake_create_pool(url, **kwargs):
            return FakePool(conn)

        monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
        monkeypatch.setenv("CINE_REC_AUTO_INIT", "1")
        with TestClient(self._app()) as client:
            assert client.get("/health").status_code == 200
        assert len(conn.executed) == 2  # schema.sql + user_data.sql
        assert any("tmdb_movies" in s for s in conn.executed)

    def test_bad_map_fails_boot_loudly(self, monkeypatch):
        import asyncpg

        from fastapi.testclient import TestClient

        async def fake_create_pool(url, **kwargs):
            return FakePool(FakeConn(regclass_missing=("app_media",)))

        monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
        tables.set_table_map({"tmdb_media": "app_media"})
        try:
            with pytest.raises(RuntimeError, match="app_media"):
                with TestClient(self._app()):
                    pass
        finally:
            tables.set_table_map()
