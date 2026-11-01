#!/usr/bin/env python3
"""Generate tables_cases.json — parity cases from the REAL
cine_rec_engine/tables.py.

Every case is executed against the real module:
  * install:"set"   → tables.set_table_map(pairs)
  * install:"load"  → CINE_REC_SCHEMA_MAP + CINE_REC_TABLE_* env, then
                      tables.reset() + active_map() (triggers the lazy
                      load; the map file is really written to /tmp)
  * install:"identity" → set_table_map() (all defaults)

Recorded: the exact error message on failure, else the active map
(sorted), `name()` probes and `resolve()` probes with their exact
outputs/messages. Env pairs are recorded in insertion order — the
order os.environ iterates them — and the Rust side replays that order.

The env is scrubbed of CINE_REC_TABLE_* / CINE_REC_SCHEMA_MAP before
each case and tables.reset() runs after; nothing leaks between cases.

io_edge cases (a missing map file) are Python-only: reading files is
the runtime edge wave D owns; the Rust side skips them and the
from_file_and_env API takes the already-parsed value instead.

Run from the repo root with the wt3 venv:
    .venv/bin/python rust/ingest/tests/fixtures/gen_tables_cases.py
"""

import json
import os
from pathlib import Path

import cine_rec_engine.tables as tables

HERE = Path(__file__).resolve().parent
MAP_PATH = "/tmp/cine_rec_tables_fixture_map.json"

PREFIXES = ("CINE_REC_TABLE_", "CINE_REC_SCHEMA_MAP")


def scrub_env():
    for var in list(os.environ):
        if var.startswith(PREFIXES):
            del os.environ[var]


NAME_PROBES = [
    "tmdb_media", "tmdb_movies", "user_watches", "tmdb_cinematic",
    "media_episodes", "nope", "TMDB_MEDIA",
]

RESOLVE_PROBES = [
    "SELECT * FROM {t_tmdb_media} JOIN {t_user_watches} ON x",
    "{t_tmdb_cast}",
    "{t_bogus}",
    "{T_tmdb_media} {t_} {t_ab2c} {} t_x {t_tmdb_media",
    "no placeholders here",
    "{t_tmdb_media} and again {t_tmdb_media}",
    "SELECT 'אור' FROM {t_tmdb_tv}",
]

CASES = [
    # ── set_table_map semantics ────────────────────────────────────
    dict(name="identity-defaults", install="identity", pairs=[]),
    dict(name="simple-remap", install="set",
         pairs=[("tmdb_media", "app_media")]),
    dict(name="schema-qualified", install="set",
         pairs=[("user_watches", "app.uw"), ("tmdb_media", "m")]),
    dict(name="value-stripped", install="set",
         pairs=[("tmdb_media", "  app_media  ")]),
    dict(name="set-unknown-key", install="set", pairs=[("nope", "x")]),
    dict(name="set-empty-value", install="set", pairs=[("tmdb_media", "")]),
    dict(name="set-whitespace-value", install="set", pairs=[("tmdb_media", "   ")]),
    dict(name="set-mixed-case", install="set", pairs=[("tmdb_media", "AppMedia")]),
    dict(name="set-three-dots", install="set", pairs=[("tmdb_media", "a.b.c")]),
    dict(name="set-empty-name", install="set", pairs=[("tmdb_media", "")]),
    dict(name="set-leading-digit", install="set", pairs=[("tmdb_media", "1x")]),
    dict(name="set-trailing-dot", install="set", pairs=[("tmdb_media", "app.")]),
    # ── load_table_map semantics (file + env, env wins) ────────────
    dict(name="load-env-only", install="load", file=None,
         env=[("CINE_REC_TABLE_TMDB_MEDIA", "env_media")]),
    dict(name="load-file-only", install="load",
         file={"tmdb_tv": "file_tv"}, env=[]),
    dict(name="load-env-wins", install="load",
         file={"tmdb_media": "file_media", "tmdb_tv": "file_tv"},
         env=[("CINE_REC_TABLE_TMDB_MEDIA", "env_media")]),
    dict(name="load-blank-env-ignored", install="load",
         file={"tmdb_movies": "file_movies"},
         env=[("CINE_REC_TABLE_TMDB_MOVIES", "   "),
              ("CINE_REC_TABLE_TMDB_TV", "  \t ")]),
    dict(name="load-lowercase-suffix-env", install="load", file=None,
         env=[("CINE_REC_TABLE_tmdb_tv", "env_tv")]),
    dict(name="load-unrelated-env-ignored", install="load", file=None,
         env=[("OTHER_VAR", "x"), ("CINE_REC_TABLEISH", "y")]),
    dict(name="load-env-value-stripped", install="load", file=None,
         env=[("CINE_REC_TABLE_TMDB_MEDIA", "  env_media  ")]),
    dict(name="load-unknown-env-key", install="load", file=None,
         env=[("CINE_REC_TABLE_BOGUS", "x")]),
    dict(name="load-file-unknown-key", install="load",
         file={"bogus_key": "x"}, env=[]),
    dict(name="load-file-empty-value", install="load",
         file={"tmdb_media": ""}, env=[]),
    dict(name="load-file-whitespace-value", install="load",
         file={"tmdb_media": "  "}, env=[]),
    dict(name="load-file-non-string-value", install="load",
         file={"tmdb_media": 5}, env=[]),
    dict(name="load-file-not-object", install="load", file=[1, 2], env=[]),
    dict(name="load-file-missing", install="load", file=None, env=[],
         missing_file=True, io_edge=True),
    dict(name="load-env-invalid-value", install="load",
         file={"tmdb_tv": "file_tv"},
         env=[("CINE_REC_TABLE_TMDB_MEDIA", "Bad Name")]),
]


def run_probe(fn, *args):
    try:
        out = fn(*args)
    except tables.TableMapError as e:
        return {"error": str(e)}
    return out


def run_case(case):
    scrub_env()
    tables.reset()
    try:
        if case["install"] == "set":
            try:
                tables.set_table_map(dict(case["pairs"]))
            except tables.TableMapError as e:
                return {"error": str(e)}
        elif case["install"] == "identity":
            tables.set_table_map()
        else:  # load
            if case.get("missing_file"):
                os.environ["CINE_REC_SCHEMA_MAP"] = MAP_PATH + ".missing"
            elif case.get("file") is not None:
                Path(MAP_PATH).write_text(json.dumps(case["file"]))
                os.environ["CINE_REC_SCHEMA_MAP"] = MAP_PATH
            for var, val in case["env"]:
                os.environ[var] = val
            try:
                tables.active_map()  # triggers _ensure_loaded
            except tables.TableMapError as e:
                return {"error": str(e)}
        return {
            "active": dict(sorted(tables.active_map().items())),
            "names": [[p, run_probe(tables.name, p)] for p in NAME_PROBES],
            "resolves": [[t, run_probe(tables.resolve, t)] for t in RESOLVE_PROBES],
        }
    finally:
        tables.reset()
        scrub_env()


def main():
    out = []
    for case in CASES:
        rec = {"name": case["name"], "install": case["install"],
               "io_edge": bool(case.get("io_edge"))}
        if case["install"] == "set":
            rec["pairs"] = case["pairs"]
        if case["install"] == "load":
            rec["file"] = case.get("file")
            rec["env"] = case["env"]
            if case.get("missing_file"):
                rec["missing_file"] = True
            elif case.get("file") is not None:
                rec["file_path"] = MAP_PATH
        rec["expect"] = run_case(case)
        out.append(rec)

    doc = {"source": "cine_rec_engine/tables.py (real implementation)",
           "cases": out}
    path = HERE / "tables_cases.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    errors = sum(1 for c in out if "error" in c["expect"])
    print(f"wrote {path.name}: {len(out)} cases ({errors} error cases)")


if __name__ == "__main__":
    main()
