"""Generate schema_map_cases.json — Python-computed parity cases for
the Rust SchemaMap (resolve/name semantics, including the tricky
placeholder grammar and the dotted-receiver escape).

Run from the repo root (worktree):

    .venv/bin/python rust/db/tests/fixtures/gen_schema_map_cases.py

Regenerate when templates.py-adjacent semantics change; the Rust test
replays every case and must agree bit-for-bit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from cine_rec_engine import tables  # noqa: E402

TEMPLATES = [
    # the real SQL shapes
    "SELECT * FROM {t_tmdb_media} WHERE id = $1",
    "SELECT 1 FROM {t_tmdb_media_genres} mg JOIN {t_tmdb_genres} g ON g.id = mg.genre_id",
    "UPDATE {t_user_watch_events} SET x = 1",
    # placeholder grammar edges (regex: \{t_([a-z_]+)\})
    "{t_}",                       # empty group — no match
    "{t_tmdb_media",              # unclosed — no match
    "{t_1abc}",                   # starts with digit — no match
    "{t_tmdb_media2}",            # digit inside — no match
    "{t_Tmdb_Media}",             # upper case — no match
    "{t_tmdb_media_extra}",       # unknown logical — ERROR via name()
    "x{t_tmdb_media}y{t_tmdb_tv}z",  # multiple
    "'{t_tmdb_media}'",           # inside quotes — STILL substituted
    "{T_tmdb_media}",             # capital T prefix — no match
    "{t_}",                       # no match
    "FROM {t_user_watches} , {t_tmdb_cast}",  # two tables
    "SELECT '{t_tmdb_media}' || {t_tmdb_crew}.media_id",  # quoted + dotted use
    "",                           # empty template
    "no placeholders at all",
]

MAPS = [
    {},
    {"tmdb_media": "app_media"},
    {"tmdb_media": "app.media", "tmdb_tv": "app_tv"},
    {"tmdb_media_genres": "g", "tmdb_genres": "g2", "user_watches": "u.w"},
    {"tmdb_movie_genres_map": "mgm", "tmdb_tv_genres_map": "tgm",
     "tmdb_movies": "m", "tmdb_tv": "t"},
]


def main() -> None:
    out = {
        # set parity: the exact LOGICAL_TABLES contents from Python
        "logical_tables_sorted": sorted(tables.LOGICAL_TABLES),
        "cases": [],
    }
    for mapping in MAPS:
        tables.set_table_map(mapping)
        for template in TEMPLATES:
            try:
                resolved = tables.resolve(template)
                out["cases"].append(
                    {"map": mapping, "template": template, "ok": True,
                     "resolved": resolved}
                )
            except tables.TableMapError as e:
                out["cases"].append(
                    {"map": mapping, "template": template, "ok": False,
                     "error_contains": str(e)[:60]}
                )
    tables.reset()

    # name() behavior incl. unknown names
    tables.set_table_map({})
    names = []
    for logical in ["tmdb_media", "unknown_table", "TMDB_MEDIA"]:
        try:
            names.append({"logical": logical, "ok": True,
                          "name": tables.name(logical)})
        except tables.TableMapError:
            names.append({"logical": logical, "ok": False})
    out["names"] = names
    tables.reset()

    dest = Path(__file__).parent / "schema_map_cases.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"wrote {dest}: {len(out['cases'])} resolve cases, "
          f"{len(out['names'])} name cases, "
          f"{len(out['logical_tables_sorted'])} logical tables")


if __name__ == "__main__":
    main()
