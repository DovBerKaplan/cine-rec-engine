"""Offline tests for the logical table-name registry (cine_rec_engine.tables)."""

import json

import pytest

from cine_rec_engine import tables


@pytest.fixture(autouse=True)
def _clean_map(monkeypatch):
    monkeypatch.delenv("CINE_REC_SCHEMA_MAP", raising=False)
    for var in list(__import__("os").environ):
        if var.startswith("CINE_REC_TABLE_"):
            monkeypatch.delenv(var, raising=False)
    tables.reset()
    yield
    tables.reset()


class TestDefaults:
    def test_identity_when_nothing_mapped(self):
        assert tables.name("tmdb_media") == "tmdb_media"
        assert tables.name("user_watches") == "user_watches"

    def test_active_map_empty(self):
        assert tables.active_map() == {}

    def test_resolve_leaves_plain_sql_alone(self):
        sql = "SELECT id FROM tmdb_media WHERE id = $1"
        assert tables.resolve(sql) == sql

    def test_resolve_substitutes_placeholders(self):
        sql = "SELECT id FROM {t_tmdb_media} WHERE id = $1"
        assert tables.resolve(sql) == "SELECT id FROM tmdb_media WHERE id = $1"

    def test_unknown_logical_name_raises(self):
        with pytest.raises(tables.TableMapError, match="unknown logical table"):
            tables.name("not_a_table")


class TestFileMap:
    def test_json_map_is_loaded(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text(json.dumps({"tmdb_media": "app_media"}))
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        tables.reset()
        assert tables.name("tmdb_media") == "app_media"
        assert tables.name("tmdb_tv") == "tmdb_tv"  # untouched → default
        assert tables.active_map() == {"tmdb_media": "app_media"}

    def test_schema_qualified_names_allowed(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text(json.dumps({"user_watches": "app.user_watches"}))
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        tables.reset()
        assert tables.name("user_watches") == "app.user_watches"

    def test_missing_file_is_loud(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(tmp_path / "nope.json"))
        tables.reset()
        with pytest.raises(tables.TableMapError, match="cannot load schema map"):
            tables.name("tmdb_media")
        assert "cannot load schema map" in tables.load_error()

    def test_broken_json_is_loud(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text("{not json")
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        tables.reset()
        with pytest.raises(tables.TableMapError):
            tables.name("tmdb_media")

    def test_unknown_key_lists_valid_keys(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text(json.dumps({"tmdb_medai": "app_media"}))
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        tables.reset()
        with pytest.raises(tables.TableMapError, match="tmdb_media"):
            tables.name("tmdb_media")

    @pytest.mark.parametrize("bad", [
        "App_Media",           # mixed case — engine interpolates unquoted
        "app media",           # space
        "app-media",           # dash
        'app"; DROP TABLE x',  # injection
        "app..media",          # double dot
        "a.b.c",               # too many parts
        "",                    # empty
    ])
    def test_invalid_names_rejected(self, bad):
        with pytest.raises(tables.TableMapError):
            tables.set_table_map({"tmdb_media": bad})

    def test_non_string_name_rejected_by_set(self):
        with pytest.raises(tables.TableMapError):
            tables.set_table_map({"tmdb_media": None})


class TestEnvOverrides:
    def test_env_var_per_table(self, monkeypatch):
        monkeypatch.setenv("CINE_REC_TABLE_TMDB_MEDIA", "app_media")
        tables.reset()
        assert tables.name("tmdb_media") == "app_media"

    def test_env_wins_over_file(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text(json.dumps({"tmdb_media": "from_file",
                                    "tmdb_tv": "tv_from_file"}))
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        monkeypatch.setenv("CINE_REC_TABLE_TMDB_MEDIA", "from_env")
        tables.reset()
        assert tables.name("tmdb_media") == "from_env"
        assert tables.name("tmdb_tv") == "tv_from_file"

    def test_empty_env_value_ignored(self, tmp_path, monkeypatch):
        path = tmp_path / "map.json"
        path.write_text(json.dumps({"tmdb_media": "from_file"}))
        monkeypatch.setenv("CINE_REC_SCHEMA_MAP", str(path))
        monkeypatch.setenv("CINE_REC_TABLE_TMDB_MEDIA", "   ")
        tables.reset()
        assert tables.name("tmdb_media") == "from_file"


class TestProgrammaticMap:
    def test_set_and_reset(self):
        tables.set_table_map({"tmdb_media": "app_media"})
        assert tables.name("tmdb_media") == "app_media"
        tables.set_table_map()
        assert tables.name("tmdb_media") == "tmdb_media"

    def test_resolve_cache_invalidated_on_new_map(self):
        sql = "SELECT 1 FROM {t_tmdb_media}"
        tables.set_table_map({"tmdb_media": "app_media"})
        assert tables.resolve(sql) == "SELECT 1 FROM app_media"
        tables.set_table_map({"tmdb_media": "other_media"})
        assert tables.resolve(sql) == "SELECT 1 FROM other_media"

    def test_resolve_unknown_placeholder_raises(self):
        with pytest.raises(tables.TableMapError, match="unknown logical table"):
            tables.resolve("SELECT * FROM {t_bogus}")

    def test_every_logical_name_resolves_by_default(self):
        for logical in tables.LOGICAL_TABLES:
            assert tables.name(logical) == logical
