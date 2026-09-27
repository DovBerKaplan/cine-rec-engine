"""Ingest pure-logic tests: export parsing, cast/crew filters, row builders,
the acceptance rule. No network, no DB."""

import gzip
import io
import json
from datetime import date

from ingest.exports import ExportEntry, export_url_for, latest_export_url, parse_export
from ingest.loader import (
    is_complete,
    keep_cast,
    keep_crew,
    rec_rows_from_payload,
    title_row_from_payload,
    tv_row_from_payload,
)


def _gz(lines):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write("\n".join(json.dumps(item) for item in lines).encode())
    return buf.getvalue()


class TestExports:
    def test_url_format(self):
        assert export_url_for(date(2026, 9, 27), "movie") == (
            "http://files.tmdb.org/p/exports/movie_ids_09_27_2026.json.gz"
        )
        assert "tv_series_ids" in export_url_for(date(2026, 9, 27), "tv")

    def test_latest_is_yesterday(self):
        url = latest_export_url("movie", today=date(2026, 9, 27))
        assert "09_26_2026" in url

    def test_parse_and_fields(self):
        blob = _gz([
            {"id": 1, "adult": False, "popularity": 5.5, "original_title": "A"},
            {"id": 2, "adult": True, "popularity": 9.9},
            {"id": 3, "adult": False, "popularity": 0.0},
        ])
        entries = list(parse_export(blob))
        assert entries == [
            ExportEntry(1, False, 5.5),
            ExportEntry(2, True, 9.9),
            ExportEntry(3, False, 0.0),
        ]


class TestCastCrewFilters:
    def test_cast_top5_by_order(self):
        cast = [{"id": i, "character": f"c{i}", "order": 9 - i} for i in range(8)]
        kept = keep_cast(cast)
        assert len(kept) == 5
        assert kept[0] == (7, "c7", 2)   # lowest `order` first (fixture: order=9-i)
        assert kept[-1][2] == 6          # 5th lowest

    def test_cast_skips_missing_order(self):
        assert keep_cast([{"id": 1, "character": "x"}]) == []

    def test_crew_whitelist(self):
        crew = [
            {"id": 1, "job": "Director", "department": "Directing"},
            {"id": 2, "job": "Original Music Composer", "department": "Sound"},
            {"id": 3, "job": "Casting", "department": "Production"},  # dropped
            {"id": 4, "job": "Editor", "department": "Editing"},      # dropped
            {"id": 5, "job": "Creator", "department": "Production"},
        ]
        kept = keep_crew(crew)
        assert {j for _, j, _ in kept} == {"Director", "Original Music Composer", "Creator"}

    def test_crew_keeps_department(self):
        (pid, job, dept) = keep_crew([{"id": 1, "job": "Writer", "department": "Writing"}])[0]
        assert (pid, job, dept) == (1, "Writer", "Writing")


class TestRowBuilders:
    MOVIE = {
        "id": 155, "title": "The Dark Knight", "original_title": "The Dark Knight",
        "original_language": "en", "overview": "Batman.",
        "release_date": "2008-07-16", "runtime": 152, "status": "Released",
        "adult": False, "vote_average": 8.5, "vote_count": 30000,
        "popularity": 90.1, "poster_path": "/p.jpg", "backdrop_path": "/b.jpg",
        "belongs_to_collection": {"id": 9735}, "imdb_id": "tt0468569",
        "budget": 185000000, "revenue": 1004558444,
    }

    def test_movie_row_column_order(self):
        row = title_row_from_payload(self.MOVIE)
        assert row[0] == 155
        assert row[1] == "The Dark Knight"
        assert row[14] == 9735          # collection_id from belongs_to_collection
        assert row[15] == "tt0468569"

    def test_movie_row_nullable_collection(self):
        row = title_row_from_payload({**self.MOVIE, "belongs_to_collection": None})
        assert row[14] is None

    def test_tv_row_uses_name(self):
        payload = {
            "id": 1396, "name": "Breaking Bad", "original_name": "Breaking Bad",
            "original_language": "en", "overview": "Chemistry teacher.",
            "first_air_date": "2008-01-20", "last_air_date": "2013-09-29",
            "status": "Ended", "in_production": False,
            "number_of_seasons": 5, "number_of_episodes": 62, "adult": False,
            "vote_average": 8.9, "vote_count": 12000, "popularity": 300.0,
            "poster_path": "/p.jpg", "backdrop_path": "/b.jpg",
            "external_ids": {"imdb_id": "tt0903747"},
        }
        row = tv_row_from_payload(payload)
        assert row[1] == "Breaking Bad"   # name lands in the name column
        assert row[17] == "tt0903747"     # imdb from external_ids

    def test_recs_within_medium_and_ranked(self):
        payload = {
            "id": 155, "title": "The Dark Knight",
            "recommendations": {"results": [
                {"id": 155, "popularity": 1},   # self-edges allowed by API? still ranked
                {"id": 24428, "popularity": 50.0},
            ]},
        }
        rows = rec_rows_from_payload(payload)
        assert all(r[1] == "movie" and r[3] == "movie" for r in rows)
        assert [r[4] for r in rows] == [1, 2]

    def test_recs_tv_detection(self):
        payload = {"id": 1396, "name": "Breaking Bad",
                   "recommendations": {"results": [{"id": 1, "popularity": 2}]}}
        assert rec_rows_from_payload(payload)[0][1] == "tv"


class TestAcceptanceRule:
    def test_complete_when_all_sections_present(self):
        payload = {"id": 1, "title": "X", "genres": [], "credits": {},
                   "keywords": {}, "recommendations": {"results": []}}
        assert is_complete(payload) is True  # empty sections still count as present

    def test_incomplete_when_section_missing(self):
        payload = {"id": 1, "title": "X", "genres": [], "credits": {}}
        assert is_complete(payload) is False


class TestAppendToResponse:
    def test_movie_list(self):
        from ingest.loader import append_to_response

        assert append_to_response("movie") == "credits,keywords,recommendations"

    def test_tv_includes_external_ids_with_separator(self):
        from ingest.loader import append_to_response

        value = append_to_response("tv")
        assert value == "credits,keywords,recommendations,external_ids"
        assert "recommendationsexternal" not in value  # the regression


class TestIngestGate:
    def test_rejects_adult_by_default(self):
        from ingest.loader import should_ingest

        assert should_ingest({"adult": True, "popularity": 9.0}) is False
        assert should_ingest({"adult": True, "popularity": 9.0}, include_adult=True)

    def test_applies_popularity_floor(self):
        from ingest.loader import should_ingest

        assert should_ingest({"adult": False, "popularity": 0.5},
                             min_popularity=1.0) is False
        assert should_ingest({"adult": False, "popularity": 1.2},
                             min_popularity=1.0) is True

    def test_rejects_error_payload(self):
        from ingest.loader import should_ingest

        assert should_ingest(None) is False
        assert should_ingest({"success": False}) is False


class TestBridgeMirroring:
    """The generated sync SQL must delete stale rows and keep current ones."""

    def _fake_conn(self):
        class Conn:
            def __init__(self):
                self.executed = []
                self.rows = []

            async def execute(self, sql, *args):
                self.executed.append((sql, args))

            async def executemany(self, sql, rows):
                self.executed.append((sql, rows))

        return Conn()

    async def test_delete_stale_keeps_current(self):
        from ingest.loader import _sync_bridge

        conn = self._fake_conn()
        await _sync_bridge(
            conn, "tmdb_movie_cast", "movie_id", 155, "person_id",
            [1, 2, 3], "INSERT INTO x VALUES ($1,$2)", [(155, 1)],
        )
        delete_sql = conn.executed[0][0]
        assert "DELETE FROM tmdb_movie_cast" in delete_sql
        assert "<> ALL($2::bigint[])" in delete_sql

    async def test_empty_payload_wipes_all_rows(self):
        from ingest.loader import _sync_bridge

        conn = self._fake_conn()
        await _sync_bridge(conn, "tmdb_movie_cast", "movie_id", 155,
                           "person_id", [], None, None)
        sql, args = conn.executed[0]
        assert "DELETE FROM tmdb_movie_cast WHERE movie_id = $1" in sql
        assert args == (155,)

    def test_table_name_helpers(self):
        from ingest.loader import cast_table, crew_table, gmap_table

        assert cast_table(True) == "tmdb_movie_cast"
        assert crew_table(False) == "tmdb_tv_crew"
        assert gmap_table(False) == "tmdb_tv_genres_map"


class TestDateCoercion:
    def test_iso_string_becomes_date(self):
        from datetime import date as d

        from ingest.loader import _as_date

        assert _as_date("2008-07-16") == d(2008, 7, 16)
        assert _as_date(None) is None
        assert _as_date("") is None
        assert _as_date(d(2020, 1, 2)) == d(2020, 1, 2)


class TestBootstrapResumeAndWorkers:
    async def test_skips_existing_and_ingests_rest_concurrently(self):
        import asyncio

        from ingest.loader import TmdbIngest

        ingest = TmdbIngest.__new__(TmdbIngest)  # skip __init__ (needs key)
        ingest.stats = __import__("ingest.loader", fromlist=["IngestStats"]).IngestStats()

        async def fake_existing():
            return {"movie": {1, 2}, "tv": set()}

        async def fake_ingest_one(session, media_id, medium, **kw):
            await asyncio.sleep(0)  # yield like a real await
            ingest.stats.attempted += 1
            return True

        ingest.existing_ids = fake_existing
        ingest.ingest_one = fake_ingest_one

        async def entries():
            for pair in [(1, "movie"), (2, "movie"), (3, "movie"), (10, "tv")]:
                yield pair

        await ingest.bootstrap(entries(), workers=3)
        # ids 1 and 2 skipped (already in DB); 3 and 10 attempted
        assert ingest.stats.attempted == 2
