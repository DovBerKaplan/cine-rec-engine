#!/usr/bin/env python3
"""Generate the loader parity fixtures by running the REAL
ingest/loader.py pure functions.

Outputs (next to this script):
    loader_payloads.json    — the deterministic payload corpus (the
                              single source for BOTH the offline
                              builder parity and the live DB parity)
    loader_parity_cases.json — expected outputs of the real pure
                              functions per payload

What is pinned per payload:
  * movie_row / tv_row   — title_row_from_payload / tv_row_from_payload
    (raw values as Python builds them; dates serialized ISO)
  * cast / crew          — keep_cast / keep_crew
  * recs                 — rec_rows_from_payload
  * complete             — is_complete
  * gate                 — should_ingest over (min_popularity,
                           include_adult) grid
Plus append_to_response and a few builder ERRORS (exact class+message
where deterministic).

The corpus deliberately includes: a full movie with every table
touched, the SAME id re-ingested with SHRUNK bridges (the mirror
case), movie 155 + tv 155 together (ids collide across media), empty
title (`or ""`), a string popularity ("23.5" — builds a row and LANDS
as numeric 23.5: asyncpg decodes str→Decimal), a `success: false`
rejection, adult gate, float-int ids, cast order ties, dropped
cast/crew rows, keywords via both the movie and tv shapes, a non-dict
external_ids — plus the review-round-1 families: asyncpg-encoder
cases (string runtime/genre-id rejected at the bind, bool
vote_average stored as 1), gate errors (float("abc")), falsy
belongs_to_collection, null genres, mid-batch conversion failures,
INT4-busting cast order, float character repr, unhashable crew job
and a non-dict recommendations container.

Run from the repo root with the wt3 venv:
    .venv/bin/python rust/ingest/tests/fixtures/gen_loader_parity.py
"""

import asyncio
import json
from datetime import date
from pathlib import Path

import ingest.loader as L

HERE = Path(__file__).resolve().parent

# ── the corpus ────────────────────────────────────────────────────
# (name, medium, payload) — every payload a full appended TMDB shape.

MOVIE_FULL = {
    "id": 155, "title": "The North Star",
    "original_title": "Nordstern", "original_language": "en",
    "overview": "A movie overview.", "release_date": "2018-05-04",
    "runtime": 109, "status": "Released", "adult": False,
    "vote_average": 7.834, "vote_count": 1200, "popularity": 15.237,
    "poster_path": "/x.jpg", "backdrop_path": "/y.jpg",
    "belongs_to_collection": {"id": 448290, "name": "Collection"},
    "imdb_id": "tt1234567", "budget": 15000000, "revenue": 40120000,
    "genres": [
        {"id": 28, "name": "Action"},
        {"id": 12, "name": "Adventure"},
        {"id": 878, "name": "Science Fiction"},
    ],
    "keywords": {"keywords": [
        {"id": 818, "name": "based on novel"},
        {"id": 9840, "name": "space"},
        {"id": 10183, "name": "quiet"},
    ]},
    "credits": {
        "cast": [
            {"id": 11, "name": "Al Cast", "character": "Hero", "order": 0},
            {"id": 12, "name": "Bea Cast", "character": None, "order": 1},
            {"id": 13, "name": "Cee Cast", "order": 2},
            {"id": 14, "name": "Dee Cast", "character": "D", "order": 3},
            {"id": 15, "name": "Eee Cast", "character": "E", "order": 3},   # tie
            {"id": 16, "name": "Eff Cast", "character": "F", "order": 4},
            {"id": 17, "name": "Gee Cast", "character": "G", "order": 5},   # cut (top-5)
            {"id": 18, "name": "Aitch Cast", "character": "H"},            # no order → filtered
        ],
        "crew": [
            {"id": 21, "name": "Al Director", "job": "Director", "department": "Directing"},
            {"id": 22, "name": "Be Writer", "job": "Writer", "department": "Writing"},
            {"id": 22, "name": "Be Writer", "job": "Writer", "department": "Story"},  # dup (job,pid)
            {"id": 23, "name": "See Music", "job": "Original Music Composer"},
            {"id": 24, "name": "Dee Cam", "job": "Director of Photography", "department": "Camera"},
            {"id": 25, "name": "Eee Creator", "job": "Creator"},           # whitelisted even on movies
            {"id": 26, "name": "Eff Sound", "job": "Sound Re-recording Mixer"},  # dropped
            {"id": 27, "name": "Gee NoJob", "department": "Directing"},    # no job → dropped
        ],
    },
    "production_companies": [
        {"id": 9993, "name": "Night Owl Media"},
        {"id": 882, "name": "Film Four"},
    ],
    "recommendations": {"results": [
        {"id": 603, "popularity": 44.5},
        {"id": 680, "popularity": "51.3"},   # numeric string — float() takes it
        {"id": 13, "popularity": 0.0},
    ]},
}

# The MIRROR case: same movie 155 re-ingested with shrunk bridges.
MOVIE_SHRINK = {
    "id": 155, "title": "The North Star",
    "original_language": "en", "overview": "Recut overview.",
    "release_date": "2018-05-11", "runtime": 112, "status": "Released",
    "adult": False, "vote_average": 7.9, "vote_count": 1400,
    "popularity": 9.5, "imdb_id": "tt1234567",
    "genres": [{"id": 28, "name": "Action"}],          # 3 → 1 (stale deleted)
    "keywords": {"keywords": [
        {"id": 818, "name": "based on novel"},
        {"id": 9840, "name": "space"},
    ]},                                                 # 3 → 2
    "credits": {
        "cast": [                                       # full recast
            {"id": 40, "name": "New Lead", "character": "Hero", "order": 0},
            {"id": 41, "name": "New Second", "character": "Pal", "order": 1},
        ],
        "crew": [
            {"id": 21, "name": "Al Director", "job": "Director", "department": "Directing"},
        ],                                              # 5 kept rows → 1
    },
    "production_companies": [],                         # all stale → deleted
    "recommendations": {"results": [
        {"id": 603, "popularity": 45.0},
        {"id": 695, "popularity": 2.0},
    ]},                                                 # 3 → 2, rank 3 deleted
}

TV_FULL = {
    "id": 155, "name": "Echoes",                         # SAME id as the movie
    "original_name": "הדים", "original_language": "he",
    "overview": "A series overview.", "first_air_date": "2021-06-02",
    "last_air_date": "2022-04-08", "status": "Ended",
    "in_production": False, "number_of_seasons": 2,
    "number_of_episodes": 16, "adult": False,
    "vote_average": 8.1, "vote_count": 340, "popularity": 22.75,
    "poster_path": "/tv.jpg", "backdrop_path": "/tvb.jpg",
    "external_ids": {"imdb_id": "tt9801234"},
    "genres": [
        {"id": 18, "name": "Drama"},
        {"id": 9648, "name": "Mystery"},
    ],
    "keywords": {"results": [                            # TV shape
        {"id": 1305, "name": "memory"},
        {"id": 2085, "name": "small town"},
    ]},
    "credits": {
        "cast": [
            {"id": 51, "name": "Tee Vee", "character": "Lead", "order": 0},
            {"id": 52, "name": "Yu Vee", "character": "", "order": 1},
        ],
        "crew": [
            {"id": 53, "name": "Show Creator", "job": "Creator", "department": "Production"},
            {"id": 54, "name": "Tv Writer", "job": "Writer", "department": None},
        ],
    },
    "production_companies": [{"id": 882, "name": "Film Four"}],
    "networks": [
        {"id": 49, "name": "HBO"},
        {"id": 2552, "name": "Kan 11"},
    ],
    "recommendations": {"results": [
        {"id": 94602, "popularity": 30.1},
        {"id": 82856, "popularity": 12.2},
        {"id": 71912, "popularity": 8.8},
        {"id": 60622, "popularity": 5.5},
    ]},
}

CASES = [
    ("movie-full", "movie", MOVIE_FULL),
    ("movie-shrink", "movie", MOVIE_SHRINK),
    ("tv-full", "tv", TV_FULL),
    ("movie-minimal", "movie", {"id": 900, "title": ""}),
    ("movie-adult", "movie", {
        "id": 901, "title": "Adults Only", "adult": True,
        "belongs_to_collection": None, "release_date": "",
        "overview": None, "popularity": 3.0,
    }),
    ("movie-int-votes", "movie", {
        "id": 902, "title": "Ints", "vote_average": 7, "popularity": 0,
        "budget": 15000, "revenue": 0, "vote_count": 99, "runtime": 0,
    }),
    ("movie-recs-empty", "movie", {
        "id": 904, "title": "No Recs", "popularity": 1.0,
        "recommendations": {"results": []},
    }),
    ("tv-external-str", "tv", {
        "id": 903, "name": "Ext Str", "external_ids": "tt0000000",
        "popularity": 1.0,
    }),
    ("movie-pop-str", "movie", {
        "id": 905, "title": "String Pop", "popularity": "23.5",
        "genres": [{"id": 35, "name": "Comedy"}],
    }),
    ("movie-success-false", "movie", {
        "id": 906, "title": "Rejected", "success": False, "popularity": 5.0,
    }),
    ("movie-float-id", "movie", {
        "id": 907.0, "title": "Float Id", "popularity": 2.0,
    }),
    ("movie-bad-date", "movie", {
        "id": 908, "title": "Bad Date", "release_date": "2024-13-45",
    }),
    ("movie-missing-id", "movie", {"title": "No Id"}),
    ("movie-rec-missing-id", "movie", {
        "id": 909, "title": "Broken Rec",
        "recommendations": {"results": [{"popularity": 1.0}]},
    }),
    # A CREDITED person without an id: keep_cast/keep_crew filter it
    # out, but the PEOPLE upsert builds names from the RAW credit
    # lists — `int(c["id"])` raises and the whole title FAILS (with
    # its fact row already committed; no rollback). Pinned live.
    ("movie-credited-noid", "movie", {
        "id": 910, "title": "Broken Credits", "popularity": 2.0,
        "credits": {
            "cast": [
                {"id": 31, "name": "Ok Lead", "character": "L", "order": 0},
                {"name": "Ghost Extra", "character": "G", "order": 1},  # no id
            ],
            "crew": [],
        },
    }),
    # ── review round 1: encoder/truthiness/order cases ──────────────
    # M-1: asyncpg rejects STRING ints at the bind (int("120") would
    # succeed at builder sites); bools bind as 0/1 for numeric.
    ("movie-runtime-str", "movie", {
        "id": 920, "title": "Runtime Str", "runtime": "120", "popularity": 2.0,
    }),
    ("movie-genre-id-str", "movie", {
        "id": 921, "title": "Genre Id Str", "popularity": 2.0,
        "genres": [{"id": "18", "name": "Drama"}],
    }),
    ("movie-voteavg-true", "movie", {
        "id": 922, "title": "Voteavg True", "vote_average": True, "popularity": 2.0,
    }),
    # M-3: an unconvertible popularity RAISES in should_ingest
    # (float("abc")) — failed, never silently filtered.
    ("movie-pop-str-bad", "movie", {
        "id": 923, "title": "Pop Abc", "popularity": "abc",
    }),
    # M-3: a FALSY belongs_to_collection reads as {} → None (ingests).
    ("movie-collection-false", "movie", {
        "id": 924, "title": "Collection False", "belongs_to_collection": False,
        "popularity": 2.0,
    }),
    # M-3: genres PRESENT but null — iterating None is a TypeError
    # (pinned live: fact row only, no people).
    ("movie-genres-null", "movie", {
        "id": 925, "title": "Genres Null", "genres": None, "popularity": 2.0,
    }),
    # M-2: keyword id present, name missing — conversion fails INSIDE
    # the batch, after people+genres were written (pinned live).
    ("movie-kw-midfail", "movie", {
        "id": 926, "title": "Kw Midfail", "popularity": 2.0,
        "genres": [{"id": 28, "name": "Action"}],
        "credits": {"cast": [{"id": 61, "name": "Mid Lead", "character": "M", "order": 0}], "crew": []},
        "keywords": {"keywords": [{"id": 5001}]},
    }),
    # M-2: company without id — same mid-batch failure point, one
    # statement later.
    ("movie-co-midfail", "movie", {
        "id": 927, "title": "Co Midfail", "popularity": 2.0,
        "genres": [{"id": 28, "name": "Action"}],
        "credits": {"cast": [{"id": 61, "name": "Mid Lead", "character": "M", "order": 0}], "crew": []},
        "keywords": {"keywords": [{"id": 5002, "name": "ok"}]},
        "production_companies": [{"name": "NoId Co"}],
    }),
    # H-1: any movie 155 re-ingested AFTER the tv side — the tv
    # networks of 155 must survive (pinned live with an explicit
    # count assertion).
    ("movie-155-again", "movie", {
        "id": 155, "title": "The North Star", "popularity": 3.0,
    }),
    # M-2: a cast order beyond INT4 — int() succeeds at the builder
    # (bignum), the INT4 bind fails AFTER the cast DELETE (live).
    ("movie-order-huge", "movie", {
        "id": 928, "title": "Order Huge", "popularity": 2.0,
        "credits": {"cast": [
            {"id": 71, "name": "Huge Order", "character": "H", "order": 3000000000},
        ], "crew": []},
    }),
    # L-1: character as a float — Python str(5.0) is "5.0".
    ("movie-character-float", "movie", {
        "id": 929, "title": "Character Float", "popularity": 2.0,
        "credits": {"cast": [
            {"id": 72, "name": "Floaty", "character": 5.0, "order": 0},
        ], "crew": []},
    }),
    # L-1: an UNHASHABLE job — Python's set membership raises
    # TypeError (title fails); keep_crew records the error.
    ("movie-crew-unhashable-job", "movie", {
        "id": 930, "title": "Crew Unhashable", "popularity": 2.0,
        "credits": {"cast": [], "crew": [{"id": 73, "job": ["Director"]}]},
    }),
    # L-1: a truthy non-dict recommendations — AttributeError in
    # Python at the .get (title fails at the recs statement, live).
    ("movie-recs-list", "movie", {
        "id": 931, "title": "Recs List", "popularity": 2.0,
        "recommendations": [1],
    }),
]

GATE_GRID = [(0.0, False), (20.0, False), (0.0, True), (10.0, True)]


def ser(v):
    """Serialize a builder output value: dates → ISO, tuples → lists,
    everything else (raw json values) as-is."""
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [ser(x) for x in v]
    return v


def run_or_error(fn, *args):
    try:
        return {"ok": ser(fn(*args))}
    except Exception as e:  # noqa: BLE001 — record exactly what Python raises
        return {"error": f"{type(e).__name__}: {e}"}


def gate_result(payload, mp, ia):
    """should_ingest may RAISE (float("abc")) — record the exact
    class+message instead of a bool (review M-3)."""
    try:
        return L.should_ingest(payload, mp, ia)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    payloads = [{"name": n, "medium": m, "payload": p} for n, m, p in CASES]
    (HERE / "loader_payloads.json").write_text(
        json.dumps({"payloads": payloads}, indent=1) + "\n")

    out = []
    for name, medium, payload in CASES:
        movie = medium == "movie"
        rec = {
            "name": name,
            "medium": medium,
            "movie_row" if movie else "tv_row": run_or_error(
                L.title_row_from_payload if movie else L.tv_row_from_payload,
                payload,
            ),
            "cast": run_or_error(
                L.keep_cast, (payload.get("credits") or {}).get("cast") or []),
            "crew": run_or_error(
                L.keep_crew, (payload.get("credits") or {}).get("crew") or []),
            "recs": run_or_error(L.rec_rows_from_payload, payload),
            "complete": L.is_complete(payload),
            "gate": {
                f"{mp}/{ia}": gate_result(payload, mp, ia)
                for mp, ia in GATE_GRID
            },
        }
        out.append(rec)

    doc = {
        "source": "ingest/loader.py (real pure functions)",
        "append_to_response": {
            m: L.append_to_response(m) for m in ("movie", "tv", "other")
        },
        "cases": out,
    }
    (HERE / "loader_parity_cases.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote loader_parity_cases.json: {len(out)} cases; "
          f"payloads -> loader_payloads.json")


if __name__ == "__main__":
    assert asyncio  # the loader module imports asyncio
    main()
