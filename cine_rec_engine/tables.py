"""Logical table-name registry — the single source for every SQL name.

The engine and the ingest loader reference tables by LOGICAL name (the
identifiers in docs/schema.sql and docs/user_data.sql). The physical name
defaults to the logical one. Integrators remap without touching code via:

- a JSON file:   CINE_REC_SCHEMA_MAP=/path/map.json
                 {"tmdb_media": "app_media", "user_watches": "app.user_watches"}
- per-table env: CINE_REC_TABLE_TMDB_MEDIA=app_media (wins over the file)

A remapped name is explicit intent: it is validated at load and verified
against the database at startup (see init_db.check_schema) so a typo fails
loudly instead of silently degrading a channel to empty results.

Valid physical names are lower-case identifiers, optionally schema-qualified
("schema.table", one dot) — the engine interpolates them unquoted, so mixed
case is out of scope by construction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, FrozenSet, Optional

__all__ = [
    "LOGICAL_TABLES",
    "TableMapError",
    "active_map",
    "load_error",
    "name",
    "resolve",
    "set_table_map",
    "reset",
]

# Every table/view the engine or the ingest loader can touch. Grouped by
# the schema files that define them; names here must match those files.
LOGICAL_TABLES: FrozenSet[str] = frozenset({
    # catalog compatibility views (docs/schema.sql §6) — the read path
    "tmdb_media", "tmdb_genres", "tmdb_media_genres", "tmdb_media_keywords",
    "tmdb_media_companies", "tmdb_media_networks", "tmdb_cast", "tmdb_crew",
    # catalog facts (§1) — KNN/embedding paths read these directly
    "tmdb_movies", "tmdb_tv",
    # dimensions (§2) and genre lists (§3)
    "tmdb_people", "tmdb_keywords", "tmdb_production_companies",
    "tmdb_networks", "tmdb_movie_genres", "tmdb_tv_genres",
    # per-medium bridges (§4)
    "tmdb_movie_genres_map", "tmdb_tv_genres_map", "tmdb_movie_keywords",
    "tmdb_tv_keywords", "tmdb_movie_companies", "tmdb_tv_companies",
    "tmdb_movie_cast", "tmdb_tv_cast", "tmdb_movie_crew", "tmdb_tv_crew",
    "tmdb_tv_networks_map",
    # behavioral graph (§5) — tmdb_recommendations is physical (UPSERTed)
    "tmdb_recommendations", "tmdb_movie_recommendations",
    "tmdb_tv_recommendations",
    # optional enrichment — not in the shipped schema, degrade if absent
    "tmdb_cinematic",
    # user layer (docs/user_data.sql); user_watches is a derived view
    "user_watch_events", "user_feedback", "user_searches", "title_ratings",
    "user_title_stats", "user_stats", "user_genre_stats", "user_vectors",
    "user_watches",
    # integrator-supplied episode detail (progress SQL); not shipped
    "media_episodes",
})

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_PLACEHOLDER = re.compile(r"\{t_([a-z_]+)\}")


class TableMapError(ValueError):
    """Invalid table map or unknown logical name — always a loud failure."""


_TABLES: Dict[str, str] = {}
_LOAD_ERROR: Optional[str] = None
_LOADED = False
_RESOLVED: Dict[str, str] = {}  # template -> rendered SQL, per active map


def _validate_physical(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 2 or not all(_IDENT.match(p) for p in parts):
        raise TableMapError(
            f"invalid table name {name!r}: expected a lower-case identifier,"
            " optionally schema-qualified (schema.table)"
        )
    return name


def _env_overrides() -> Dict[str, str]:
    prefix = "CINE_REC_TABLE_"
    return {
        var[len(prefix):].lower(): val
        for var, val in os.environ.items()
        if var.startswith(prefix) and val.strip()
    }


def load_table_map() -> Dict[str, str]:
    """Merge the JSON map (CINE_REC_SCHEMA_MAP) with per-table env overrides.

    Env wins over the file. Raises TableMapError on any bad key or name."""
    file_map: Dict[str, str] = {}
    path = os.getenv("CINE_REC_SCHEMA_MAP", "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                file_map = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            raise TableMapError(f"cannot load schema map {path!r}: {e}") from e
        if not isinstance(file_map, dict):
            raise TableMapError(f"schema map {path!r} must be a JSON object")

    merged = {**file_map, **_env_overrides()}
    for logical, physical in merged.items():
        if logical not in LOGICAL_TABLES:
            raise TableMapError(
                f"unknown table key {logical!r} in schema map; valid keys:"
                f" {', '.join(sorted(LOGICAL_TABLES))}"
            )
        if not isinstance(physical, str) or not physical.strip():
            raise TableMapError(f"table name for {logical!r} must be a"
                                " non-empty string")
        _validate_physical(physical.strip())
        merged[logical] = physical.strip()
    return merged


def _ensure_loaded() -> None:
    global _TABLES, _LOAD_ERROR, _LOADED
    if _LOADED:
        return
    try:
        _TABLES = load_table_map()
    except TableMapError as e:
        _LOAD_ERROR = str(e)
        raise
    finally:
        _LOADED = True


def load_error() -> Optional[str]:
    """The map-load failure, if any (without triggering a load)."""
    return _LOAD_ERROR


def name(logical: str) -> str:
    """Physical name for a logical table. Identity when nothing is mapped."""
    _ensure_loaded()
    if logical not in LOGICAL_TABLES:
        raise TableMapError(
            f"unknown logical table {logical!r}; valid names are defined in"
            " cine_rec_engine.tables.LOGICAL_TABLES"
        )
    return _TABLES.get(logical, logical)


def active_map() -> Dict[str, str]:
    """Only the non-default mappings (for startup verification/logging)."""
    _ensure_loaded()
    return dict(_TABLES)


def resolve(template: str) -> str:
    """Substitute {t_<logical>} placeholders with physical names.

    Cached per template; the cache is cleared when a new map is installed."""
    rendered = _RESOLVED.get(template)
    if rendered is not None:
        return rendered
    _ensure_loaded()
    rendered = _PLACEHOLDER.sub(lambda m: name(m.group(1)), template)
    _RESOLVED[template] = rendered
    return rendered


def set_table_map(mapping: Optional[Dict[str, str]] = None) -> None:
    """Install a map programmatically (tests, tooling).

    An empty dict / no argument restores all defaults; a dict is validated
    like a file map. Subsequent name() calls use it until the next
    set_table_map()/reset()."""
    global _TABLES, _LOAD_ERROR, _LOADED
    if mapping is None:
        mapping = {}
    for logical, physical in mapping.items():
        if logical not in LOGICAL_TABLES:
            raise TableMapError(
                f"unknown table key {logical!r}; valid keys:"
                f" {', '.join(sorted(LOGICAL_TABLES))}"
            )
        if not isinstance(physical, str) or not physical.strip():
            raise TableMapError(f"table name for {logical!r} must be a"
                                " non-empty string")
        _validate_physical(physical.strip())
    _TABLES = {k: v.strip() for k, v in mapping.items()}
    _LOAD_ERROR = None
    _LOADED = True
    _RESOLVED.clear()


def reset() -> None:
    """Forget state; the next access reloads from env/file (test teardown)."""
    global _TABLES, _LOAD_ERROR, _LOADED
    _TABLES = {}
    _LOAD_ERROR = None
    _LOADED = False
    _RESOLVED.clear()
