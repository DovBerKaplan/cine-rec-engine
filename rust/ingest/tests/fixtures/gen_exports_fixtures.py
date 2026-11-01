#!/usr/bin/env python3
"""Generate the exports parity fixtures — by running the REAL
ingest/exports.py.

Outputs (next to this script):
    exports_cases.json  — the manifest (expected values live ONLY here)
    exports_<name>.gz   — the corpora the manifest refers to

What is pinned, and how:
  * lines   — `_parse_line(raw)` on crafted raw BYTES: skip / entry /
    loud TypeError (with Python's exact message). The swallow-vs-
    propagate split is the Python `except (KeyError, ValueError)`.
  * files   — `parse_export(gz_bytes)` on whole corpora (happy path,
    multi-member gzip, no trailing newline, corruption families).
    Error classes are recorded with a message_parity flag: "exact"
    where the port must match the text (KeyError/TypeError conversion
    errors), "kind" where texts are implementation-specific (JSON
    syntax, utf-8 positions, gzip internals).
  * urls    — `export_url_for` / `latest_export_url(medium, today=…)`
    across leap/month/year boundaries + error texts.
  * streams — the REAL `iter_export_ids` driven end-to-end through a
    duck-typed mock session (no HTTP, no aiohttp): raw gz bytes fed
    in 13-byte chunks. This pins the streaming semantics INCLUDING
    the faithful quirk that a trailing unterminated line is silently
    DROPPED (the loop only ever splits on b"\\n" — no final flush).

Known divergences are EXCLUDED from the corpus and pinned by Rust
unit tests instead (see src/exports.rs docs): Python-json-only NaN /
Infinity literals (serde_json is strict JSON) and out-of-i64 integer
ids (Python builds bignums).

Run from the repo root with the wt3 venv:
    .venv/bin/python rust/ingest/tests/fixtures/gen_exports_fixtures.py
"""

import asyncio
import gzip
import json
import zlib
from datetime import date, timedelta
from pathlib import Path

import ingest.exports as iex

HERE = Path(__file__).resolve().parent


def gz_of(lines, trailing_newline=True):
    payload = ("\n".join(lines) + ("\n" if trailing_newline else "")).encode("utf-8")
    return gzip.compress(payload, mtime=0)


def write(name, data: bytes) -> str:
    path = HERE / f"exports_{name}.gz"
    path.write_bytes(data)
    return path.name


# ── line corpus (fed to the REAL _parse_line) ─────────────────────
LINE_CASES = [
    ("normal", b'{"id": 1, "adult": false, "popularity": 1.25}'),
    ("unicode-title", '{"id": 2, "original_title": "אור: סרט", "popularity": 0}'.encode()),
    ("missing-id", b'{"adult": false}'),
    ("adult-missing", b'{"id": 3}'),
    ("adult-false", b'{"id": 3, "adult": false}'),
    ("adult-true", b'{"id": 3, "adult": true}'),
    ("adult-zero", b'{"id": 3, "adult": 0}'),
    ("adult-one", b'{"id": 3, "adult": 1}'),
    ("adult-empty-str", b'{"id": 3, "adult": ""}'),
    ("adult-str-yes", b'{"id": 3, "adult": "yes"}'),
    ("adult-str-false-is-truthy", b'{"id": 3, "adult": "false"}'),
    ("adult-null", b'{"id": 3, "adult": null}'),
    ("adult-list", b'{"id": 3, "adult": []}'),
    ("pop-null", b'{"id": 4, "popularity": null}'),
    ("pop-zero", b'{"id": 4, "popularity": 0}'),
    ("pop-neg-zero", b'{"id": 4, "popularity": -0.0}'),
    ("pop-empty-str", b'{"id": 4, "popularity": ""}'),
    ("pop-false", b'{"id": 4, "popularity": false}'),
    ("pop-str", b'{"id": 4, "popularity": "1.5"}'),
    ("pop-str-spaced", b'{"id": 4, "popularity": " 2.5 "}'),
    ("pop-str-underscore", b'{"id": 4, "popularity": "1_0.5"}'),
    ("pop-str-exp", b'{"id": 4, "popularity": "1e3"}'),
    ("pop-str-neg-inf", b'{"id": 4, "popularity": "-Infinity"}'),
    ("pop-str-abc-skip", b'{"id": 4, "popularity": "abc"}'),
    ("pop-int", b'{"id": 4, "popularity": 123}'),
    ("pop-big", b'{"id": 4, "popularity": 1e300}'),
    ("pop-empty-list-typeerror", b'{"id": 4, "popularity": []}'),
    ("pop-list-typeerror", b'{"id": 4, "popularity": [1]}'),
    ("pop-obj-typeerror", b'{"id": 4, "popularity": {}}'),
    ("id-float-integral", b'{"id": 42.0}'),
    ("id-float-truncates", b'{"id": 42.7}'),
    ("id-float-neg-truncates", b'{"id": -42.9}'),
    ("id-float-1e18", b'{"id": 1e18}'),
    ("id-str", b'{"id": " 42 "}'),
    ("id-str-underscore", b'{"id": "1_0"}'),
    ("id-str-signed", b'{"id": "+7"}'),
    ("id-bool", b'{"id": true}'),
    ("id-abc-skip", b'{"id": "abc"}'),
    ("id-null-typeerror", b'{"id": null}'),
    ("id-list-typeerror", b'{"id": [1]}'),
    ("json-int-line-typeerror", b"42"),
    ("json-null-line-typeerror", b"null"),
    ("json-true-line-typeerror", b"true"),
    ("json-float-line-typeerror", b"1.5"),
    ("json-str-line-typeerror", b'"hello"'),
    ("json-list-line-typeerror", b"[1, 2]"),
    ("bad-json-skip", b"not json at all"),
    ("bad-json-brace-skip", b'{"id": 5'),
    ("empty-skip", b""),
    ("whitespace-skip", b"   \t  "),
    ("crlf", b'{"id": 6}\r'),
    ("c0-control-stripped", b'{"id": 8}\x1c'),
    ("invalid-utf8-skip", b'{"id": 9, "x": "\xff\xff"}'),
    ("invalid-utf8-replaced-still-parses", b'{"id": 7, "x": "\xff"}'),
]


def run_line_cases():
    out = []
    for name, raw in LINE_CASES:
        try:
            got = iex._parse_line(raw)
        except TypeError as e:
            expect = {"type_error": str(e)}
        else:
            expect = {"skip": True} if got is None else {
                "entry": [got.id, got.adult, got.popularity]}
        out.append({"name": name, "line_hex": raw.hex(), "expect": expect})
    return out


# ── whole-file corpus (fed to the REAL parse_export) ──────────────
def build_file_cases():
    cases = []

    def run(name, gz, notes=""):
        try:
            entries = list(iex.parse_export(gz))
        except Exception as e:  # noqa: BLE001 — record whatever Python raises
            cls = type(e).__name__
            if isinstance(e, (KeyError, TypeError)):
                parity = "exact"
            else:
                parity = "kind"
            expect = {"error": {"class": cls, "message": str(e),
                                "message_parity": parity}}
        else:
            expect = {"entries": [[e.id, e.adult, e.popularity] for e in entries]}
        if notes:
            expect["notes"] = notes
        cases.append({"name": name, "file": write(name, gz), "expect": expect})

    run("happy", gz_of([
        '{"id": 1, "adult": false, "popularity": 1.25}',
        '',
        '{"id": 2, "original_title": "אור: סרט", "popularity": 0}',
        '{"adult": true, "id": 3}',
        '{"id": 4, "popularity": "1.5"}',
    ]), "blank line skipped; unicode; missing popularity; string popularity")

    run("no_trailing_newline", gzip.compress(b'{"id": 1}\n{"id": 9}', mtime=0),
        "parse_export INCLUDES the final unterminated line (for raw in f)")

    run("multi_member",
        gzip.compress(b'{"id": 1}\n', mtime=0) + gzip.compress(b'{"id": 2}\n', mtime=0),
        "two concatenated gzip members — Python reads BOTH")

    run("crlf_file", gzip.compress(b'{"id": 4}\r\n', mtime=0))

    run("missing_id_file", gz_of(['{"adult": false}']))

    run("type_error_file", gz_of(["42"]),
        "TypeError propagates out of parse_export (never swallowed)")

    bad_utf8 = gzip.compress(b'{"id": 1, "x": "\xff"}\n', mtime=0)
    run("invalid_utf8_strict", bad_utf8,
        "UnicodeDecodeError (a ValueError subclass) — strict decode")

    run("empty", b"", "empty input")

    run("not_gzip", b"hello, not a gzip file")

    normal = gz_of(['{"id": 1, "popularity": 5.0}'])
    run("truncated", normal[: len(normal) // 2])

    crc_flip = bytearray(normal)
    crc_flip[-5] ^= 0xFF  # inside the CRC32 trailer field
    run("bad_crc", bytes(crc_flip))

    garbage = bytearray(normal)
    garbage[15] ^= 0xFF  # mid-payload: garbage line fails JSON before CRC
    run("garbage_mid", bytes(garbage))

    return cases


# ── URL matrix ─────────────────────────────────────────────────────
URL_DATES = [
    (2024, 2, 29), (2023, 3, 1), (1900, 3, 1), (2000, 3, 1),
    (2024, 1, 1), (2026, 11, 1), (9, 1, 5), (9999, 12, 31), (2100, 3, 1),
    (2024, 2, 28), (1, 1, 2),
]

LATEST_TODAYS = [
    (2024, 3, 1), (2024, 1, 1), (2000, 3, 1), (2023, 3, 1),
    (2026, 11, 1), (1, 1, 2), (1, 1, 1),  # the last one overflows
]


def build_urls():
    url_for, latest = [], []
    for (y, m, d) in URL_DATES:
        for medium in ("movie", "tv"):
            url_for.append({"date": [y, m, d], "medium": medium,
                            "expect": {"url": iex.export_url_for(date(y, m, d), medium)}})
    for medium in ("film", "", "Movie", "tv "):
        try:
            iex.export_url_for(date(2024, 1, 1), medium)
        except ValueError as e:
            url_for.append({"date": [2024, 1, 1], "medium": medium,
                            "expect": {"error": str(e)}})
        else:
            raise AssertionError(f"medium={medium!r} unexpectedly accepted")
    for (y, m, d) in LATEST_TODAYS:
        try:
            got = iex.latest_export_url("movie", today=date(y, m, d))
        except OverflowError as e:
            latest.append({"today": [y, m, d], "medium": "movie",
                           "expect": {"error_class": "OverflowError",
                                      "message": str(e)}})
        else:
            latest.append({"today": [y, m, d], "medium": "movie",
                           "expect": {"url": got}})
    return {"export_url_for": url_for, "latest": latest}


# ── streaming corpus (REAL iter_export_ids via a mock session) ─────
class MockContent:
    def __init__(self, gz, chunk):
        self.gz, self.chunk = gz, chunk

    async def iter_chunked(self, _n):
        for i in range(0, len(self.gz), self.chunk):
            yield self.gz[i:i + self.chunk]


class MockResp:
    def __init__(self, gz, chunk):
        self.content = MockContent(gz, chunk)

    def raise_for_status(self):
        pass


class MockSession:
    """Duck-typed aiohttp stand-in: just enough for iter_export_ids."""

    def __init__(self, gz, chunk=13):
        self.gz, self.chunk = gz, chunk

    def get(self, _url):
        resp = MockResp(self.gz, self.chunk)
        class _CM:
            async def __aenter__(self):
                return resp
            async def __aexit__(self, *exc):
                return False
        return _CM()


def run_stream(name, gz, min_popularity, include_adult, notes=""):
    async def collect():
        out = []
        session = MockSession(gz)
        async for entry in iex.iter_export_ids(
            "movie", session, url="http://mock",
            min_popularity=min_popularity, include_adult=include_adult,
        ):
            out.append([entry.id, entry.adult, entry.popularity])
        return out

    entries = asyncio.run(collect())
    assert entries, f"stream {name} yielded nothing — pointless fixture"
    return {
        "name": name,
        "file": write(f"stream_{name}", gz),
        "min_popularity": min_popularity,
        "include_adult": include_adult,
        "chunk": 13,
        "expect": {"entries": entries, **({"notes": notes} if notes else {})},
    }


def build_streams():
    mixed = gz_of([
        '{"id": 1, "adult": false, "popularity": 0.0}',
        '{"id": 2, "adult": false, "popularity": 0.3}',
        '{"id": 3, "adult": false, "popularity": 0.5}',
        '{"id": 4, "adult": false, "popularity": 0.9}',
        '{"id": 5, "adult": true, "popularity": 5.0}',
        'not json skipped',
        '{"adult": false}',
    ])
    trailing = gzip.compress(
        b'{"id": 10, "popularity": 1.0}\n{"id": 11, "popularity": 1.0}',
        mtime=0,
    )
    return [
        run_stream("floor", mixed, 0.5, False,
                   "popularity floor + adult filter + lenient skips"),
        run_stream("adult_ok", mixed, 0.0, True,
                   "include_adult=True keeps id 5"),
        run_stream("all_default", mixed, 0.0, False),
        run_stream("trailing_dropped", trailing, 0.0, False,
                   "FAITHFUL QUIRK: the final unterminated line is dropped"),
    ]


def sanitize(obj):
    """Replace non-finite floats with string markers — JSON cannot
    carry them, and the consumer (serde_json) is strict. Entries CAN
    legitimately hold ±inf (e.g. popularity "-Infinity" as a string);
    the markers keep them representable."""
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


def main():
    doc = sanitize({
        "source": "ingest/exports.py (real implementation; streams via a "
                  "duck-typed mock session — no HTTP)",
        "lines": run_line_cases(),
        "files": build_file_cases(),
        "urls": build_urls(),
        "streams": build_streams(),
    })
    out = HERE / "exports_cases.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out.name}: {len(doc['lines'])} lines, {len(doc['files'])} files, "
          f"{len(doc['urls']['export_url_for']) + len(doc['urls']['latest'])} urls, "
          f"{len(doc['streams'])} streams")


if __name__ == "__main__":
    assert zlib  # the streaming path under test uses zlib internally
    assert timedelta  # date arithmetic under test
    main()
