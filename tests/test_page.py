"""Page composition, session filters, and impression tokens (RFC M1).

Pure/offline: DB-backed row generators are monkeypatched; the composition
semantics (concurrency shape, first-row-wins dedup, degrade-to-omitted,
token issuance) are what these pin.
"""

import pytest

from cine_rec_engine.impressions import issue, verify
from cine_rec_engine.page import ROW_LIMIT_CAP, compose_page, parse_row_specs
from cine_rec_engine.queries import filters_fingerprint, normalize_filters


class TestParseRowSpecs:
    def test_full_grammar(self):
        specs = parse_row_specs(
            "top_picks,because:155:movie,because:1396,hidden_gems,trending:80")
        assert [s.kind for s in specs] == [
            "top_picks", "because", "because", "hidden_gems", "trending"]
        assert specs[1].seed == (155, "movie")
        assert specs[2].seed == (1396, None)
        assert specs[4].genre_id == 80

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown row kind"):
            parse_row_specs("mystery_row")

    def test_bad_because_seed_rejected(self):
        with pytest.raises(ValueError, match="because"):
            parse_row_specs("because:abc")
        with pytest.raises(ValueError, match="media_type"):
            parse_row_specs("because:155:film")

    def test_bad_trending_rejected(self):
        with pytest.raises(ValueError, match="trending"):
            parse_row_specs("trending:crime")

    def test_top_picks_takes_no_argument(self):
        with pytest.raises(ValueError, match="no argument"):
            parse_row_specs("top_picks:155")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_row_specs(" , ")


class TestNormalizeFilters:
    def test_none_and_empty_pass_through(self):
        assert normalize_filters(None) is None
        assert normalize_filters({}) is None

    def test_string_genre_ids_normalized(self):
        f = normalize_filters({"genre_ids": "80, 18", "year_min": "1995"})
        assert f == {"year_min": 1995, "year_max": None,
                     "genre_ids": [18, 80], "exclude_genre_ids": None,
                     "max_runtime": None}

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown filter keys"):
            normalize_filters({"yaer_min": 1995})  # the classic typo

    def test_year_order_rejected(self):
        with pytest.raises(ValueError, match="year_min"):
            normalize_filters({"year_min": 2000, "year_max": 1990})

    def test_pin_and_exclude_overlap_rejected(self):
        with pytest.raises(ValueError, match="both pinned and excluded"):
            normalize_filters({"genre_ids": [80], "exclude_genre_ids": "80"})

    def test_runtime_bounds(self):
        assert normalize_filters({"max_runtime": 120})["max_runtime"] == 120
        with pytest.raises(ValueError, match="max_runtime"):
            normalize_filters({"max_runtime": 0})


class TestFiltersFingerprint:
    def test_none_is_none(self):
        assert filters_fingerprint(None) == "none"
        assert filters_fingerprint({}) == "none"

    def test_order_independent_and_stable(self):
        a = filters_fingerprint(
            {"year_min": 1980, "genre_ids": [80, 18], "max_runtime": 120})
        b = filters_fingerprint(
            {"max_runtime": 120, "genre_ids": [18, 80], "year_min": 1980})
        assert a == b == "ym=1980|g=18,80|rt=120"

    def test_different_filters_differ(self):
        assert filters_fingerprint({"year_min": 1980}) != \
            filters_fingerprint({"year_min": 1990})


class TestImpressions:
    def test_roundtrip(self):
        tok = issue({"u": 7, "r": "because", "i": [1, 2]}, now=1000.0)
        assert verify(tok, now=1000.0) == {
            "u": 7, "r": "because", "i": [1, 2], "exp": 1000 + 3600}

    def test_tampered_payload_rejected(self):
        tok = issue({"u": 7}, now=1000.0)
        b64, _, sig = tok.rpartition(".")
        assert verify(f"{b64}x.{sig}", now=1000.0) is None
        assert verify(f"{b64}.{'0' * 32}", now=1000.0) is None

    def test_expired_rejected(self):
        tok = issue({"u": 7}, ttl=10, now=1000.0)
        assert verify(tok, now=2000.0) is None

    def test_garbage_never_raises(self):
        assert verify(None) is None
        assert verify("not-a-token") is None
        assert verify("....") is None


class _PagePool:
    async def fetch(self, *a, **k):
        return []


class FakePageService:
    def __init__(self):
        self.pool = _PagePool()
        self.similar_calls = []
        self.user_calls = []

    async def recommend_for_user(self, user_id, **kwargs):
        self.user_calls.append((user_id, kwargs))
        return {"results": [
            {"tmdb_id": 1, "media_type": "movie", "score": 9.0, "title": "A"},
            {"tmdb_id": 2, "media_type": "tv", "score": 8.0, "title": "B"},
        ], "reason": "personalized", "seeds": [], "vector_used": False,
            "why": {}}

    async def find_similar(self, seeds, **kwargs):
        self.similar_calls.append((seeds, kwargs))
        return [
            {"tmdb_id": 2, "media_type": "tv", "score": 7.0, "title": "B"},
            {"tmdb_id": 3, "media_type": "movie", "score": 6.0, "title": "C"},
        ]


@pytest.fixture
def patched_rows(monkeypatch):
    import cine_rec_engine.queries as q

    calls = {}

    async def fake_gems(pool, limit=12, genre_ids=None, filters=None):
        calls["gems"] = (genre_ids, filters)
        return [{"tmdb_id": 9, "media_type": "movie", "rating": 8.1,
                 "title": "G"}]

    async def fake_trending(pool, genre_id, limit=12, filters=None):
        calls["trending"] = (genre_id, filters)
        return [{"tmdb_id": 5, "media_type": "movie", "popularity": 99.0,
                 "title": "T"}]

    async def fake_top_genres(pool, user_id, limit=3):
        calls["top_genres"] = user_id
        return [80]

    monkeypatch.setattr(q, "generate_hidden_gems", fake_gems)
    monkeypatch.setattr(q, "generate_trending_genre", fake_trending)
    monkeypatch.setattr(q, "top_user_genre_ids", fake_top_genres)
    return calls


class TestComposePage:
    async def test_rows_dedup_first_row_wins(self, patched_rows):
        svc = FakePageService()
        page = await compose_page(
            svc, user_id=42, rows="top_picks,because:155", row_limit=6)

        assert page["user_id"] == 42
        by_kind = {r["kind"]: r for r in page["rows"]}
        assert [t["tmdb_id"] for t in by_kind["top_picks"]["titles"]] == [1, 2]
        # title 2 already used by top_picks → dropped from the because row
        assert [t["tmdb_id"] for t in by_kind["because"]["titles"]] == [3]

    async def test_every_row_carries_a_verifiable_impression(self, patched_rows):
        page = await compose_page(
            FakePageService(), user_id=42, rows="top_picks,hidden_gems")
        for row in page["rows"]:
            payload = verify(row["impression"])
            assert payload is not None
            assert payload["u"] == 42
            # composite keys — the click must attribute the right medium
            assert payload["i"] == [[t["tmdb_id"], t["media_type"]]
                                    for t in row["titles"]]
        assert verify(page["page_token"])["page"] == ["top_picks", "hidden_gems"]

    async def test_failing_row_omitted_page_survives(self, patched_rows):
        import cine_rec_engine.queries as q

        async def exploding(pool, genre_id, limit=12, filters=None):
            raise RuntimeError("genre table gone")

        orig = q.generate_trending_genre
        q.generate_trending_genre = exploding
        try:
            page = await compose_page(
                FakePageService(), user_id=42,
                rows="top_picks,trending:80,hidden_gems")
        finally:
            q.generate_trending_genre = orig
        assert [r["kind"] for r in page["rows"]] == ["top_picks", "hidden_gems"]

    async def test_top_picks_requires_user(self, patched_rows):
        with pytest.raises(ValueError, match="require user_id"):
            await compose_page(FakePageService(), user_id=None,
                               rows="top_picks")

    async def test_filters_thread_into_because_and_gem_rows(self, patched_rows):
        svc = FakePageService()
        flt = {"year_min": 1995, "year_max": 2010, "genre_ids": None,
               "exclude_genre_ids": None, "max_runtime": None}
        await compose_page(svc, user_id=42, rows="because:155,hidden_gems",
                           filters=flt)
        _, kwargs = svc.similar_calls[-1]
        assert kwargs["filters"] is flt
        assert patched_rows["gems"] == ([80], flt)  # user's top genre scope

    async def test_row_limit_capped(self, patched_rows):
        page = await compose_page(FakePageService(), user_id=42,
                                  rows="hidden_gems", row_limit=999)
        assert page["row_limit"] == ROW_LIMIT_CAP
