"""The HTTP layer (cine_rec_engine.serve) — contract tests over an
injected fake engine, no database, no network.

Pinned behaviors:
- seed grammar: bare ids OR id:media_type composites, never mixed
- user_id on /similar arms the per-user pass (exclude is not None)
- unknown model names are 400s, not silent defaults
- /health reports DB readiness without crashing on a broken pool
"""

import pytest
from fastapi.testclient import TestClient

from cine_rec_engine.serve import create_app


class FakePool:
    def __init__(self, fail=False):
        self._fail = fail

    async def fetchval(self, *args, **kwargs):
        if self._fail:
            raise RuntimeError("db gone")
        return 1


class FakeService:
    def __init__(self, pool=None):
        self.pool = pool or FakePool()
        self.similar_calls = []
        self.user_calls = []

    async def find_similar(self, seeds, **kwargs):
        self.similar_calls.append((seeds, kwargs))
        return [{"tmdb_id": 111, "score": 1.0, "title": "A",
                 "title_en": "A", "media_type": "movie", "rating": 8.0,
                 "genres": ["Drama"], "poster_path": None},
                {"tmdb_id": 112, "score": 0.9, "title": "A2",
                 "title_en": "A2", "media_type": "movie", "rating": 7.5,
                 "genres": ["Crime"], "poster_path": None}]

    async def recommend_for_user(self, user_id, **kwargs):
        self.user_calls.append((user_id, kwargs))
        return {"results": [
            {"tmdb_id": 111, "score": 1.0, "title": "A",
             "title_en": "A", "media_type": "movie", "rating": 8.0,
             "genres": ["Drama"], "poster_path": None}],
            "reason": "cold_start", "seeds": [],
            "vector_used": False, "why": {}}


@pytest.fixture
def client():
    svc = FakeService()
    app = create_app(service=svc)
    with TestClient(app) as c:  # context manager runs the lifespan
        c.service = svc
        yield c


class TestSimilar:
    def test_single_bare_seed(self, client):
        r = client.get("/similar", params={"seed": "155"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == len(body["results"]) == 2
        assert body["results"][0]["tmdb_id"] == 111
        seeds, kwargs = client.service.similar_calls[-1]
        assert seeds == [155]
        assert kwargs["exclude"] is None

    def test_composite_seeds(self, client):
        r = client.get("/similar", params={"seed": "155:movie,1396:tv"})
        assert r.status_code == 200
        seeds, _ = client.service.similar_calls[-1]
        assert seeds == [(155, "movie"), (1396, "tv")]

    def test_mixed_seed_forms_rejected(self, client):
        r = client.get("/similar", params={"seed": "155,1396:tv"})
        assert r.status_code == 400
        assert "mixed" in r.json()["detail"]

    def test_bad_id_rejected(self, client):
        assert client.get("/similar", params={"seed": "abc"}).status_code == 400
        assert client.get("/similar", params={"seed": "-3"}).status_code == 400

    def test_bad_media_type_rejected(self, client):
        r = client.get("/similar", params={"seed": "155:film"})
        assert r.status_code == 400

    def test_user_id_arms_the_per_user_pass(self, client):
        r = client.get("/similar", params={"seed": "155", "user_id": 42})
        assert r.status_code == 200
        _, kwargs = client.service.similar_calls[-1]
        assert kwargs["user_id"] == 42
        assert kwargs["exclude"] is not None  # None skips watched filtering

    def test_model_passthrough_and_validation(self, client):
        r = client.get("/similar", params={"seed": "155", "model": "e5e"})
        assert r.status_code == 200
        _, kwargs = client.service.similar_calls[-1]
        assert kwargs["rec_model"] == "e5e"

        r = client.get("/similar", params={"seed": "155", "model": "nope"})
        assert r.status_code == 400
        assert "unknown model" in r.json()["detail"]

    def test_limit_bounds(self, client):
        assert client.get(
            "/similar", params={"seed": "155", "limit": 0}).status_code == 422
        assert client.get(
            "/similar", params={"seed": "155", "limit": 500}).status_code == 422

    def test_process_time_header(self, client):
        r = client.get("/similar", params={"seed": "155"})
        assert "x-process-time-ms" in r.headers


class TestForUser:
    def test_basic_call(self, client):
        r = client.get("/for-user/42")
        assert r.status_code == 200
        body = r.json()
        assert body["reason"] == "cold_start"
        user_id, kwargs = client.service.user_calls[-1]
        assert user_id == 42
        assert kwargs["vector_space"] is None
        assert kwargs["include_why"] is False

    def test_model_maps_to_vector_space(self, client):
        r = client.get("/for-user/42", params={"model": "e5e",
                                               "include_why": "true"})
        assert r.status_code == 200
        _, kwargs = client.service.user_calls[-1]
        assert kwargs["vector_space"] == "e5e"
        assert kwargs["include_why"] is True

    def test_unknown_model_rejected(self, client):
        assert client.get("/for-user/42", params={"model": "x"}
                          ).status_code == 400


class TestSessionFilters:
    def test_filters_passed_to_engine_normalized(self, client):
        r = client.get("/similar", params={
            "seed": "155", "year_min": 1995, "genres": "80,18",
            "max_runtime": 120})
        assert r.status_code == 200
        _, kwargs = client.service.similar_calls[-1]
        assert kwargs["filters"] == {
            "year_min": 1995, "year_max": None, "genre_ids": [18, 80],
            "exclude_genre_ids": None, "max_runtime": 120}

    def test_no_filter_params_pass_none(self, client):
        client.get("/similar", params={"seed": "155"})
        _, kwargs = client.service.similar_calls[-1]
        assert kwargs["filters"] is None

    def test_contradictory_genres_400(self, client):
        r = client.get("/similar", params={
            "seed": "155", "genres": "80", "exclude_genres": "80"})
        assert r.status_code == 400
        assert "both pinned and excluded" in r.json()["detail"]

    def test_year_bounds_422(self, client):
        assert client.get("/similar", params={
            "seed": "155", "year_min": 1850}).status_code == 422


class TestPage:
    @pytest.fixture
    def page_client(self, monkeypatch):
        import cine_rec_engine.queries as q

        async def fake_gems(pool, limit=12, genre_ids=None, filters=None):
            return [{"tmdb_id": 9, "media_type": "movie", "rating": 8.1,
                     "title": "G", "title_en": "G"}]

        async def fake_trending(pool, genre_id, limit=12, filters=None):
            return [{"tmdb_id": 5, "media_type": "movie", "title": "T",
                     "title_en": "T"}]

        async def fake_top_genres(pool, user_id, limit=3):
            return [80]

        monkeypatch.setattr(q, "generate_hidden_gems", fake_gems)
        monkeypatch.setattr(q, "generate_trending_genre", fake_trending)
        monkeypatch.setattr(q, "top_user_genre_ids", fake_top_genres)
        svc = FakeService()
        app = create_app(service=svc)
        with TestClient(app) as c:
            c.service = svc
            yield c

    def test_page_composes_and_dedups(self, page_client):
        r = page_client.get("/page", params={
            "user_id": 42, "rows": "top_picks,because:155,hidden_gems",
            "row_limit": 5})
        assert r.status_code == 200
        page = r.json()
        kinds = [row["kind"] for row in page["rows"]]
        assert kinds == ["top_picks", "because", "hidden_gems"]
        seen = set()
        for row in page["rows"]:
            assert row["impression"]
            for t in row["titles"]:
                key = (t["tmdb_id"], t["media_type"])
                assert key not in seen  # cross-row dedup holds
                seen.add(key)

    def test_page_bad_rows_400(self, page_client):
        r = page_client.get("/page", params={"rows": "nonsense"})
        assert r.status_code == 400
        assert "unknown row kind" in r.json()["detail"]

    def test_page_top_picks_without_user_400(self, page_client):
        assert page_client.get("/page", params={"rows": "top_picks"}
                               ).status_code == 400

    def test_page_threads_filters(self, page_client):
        page_client.get("/page", params={
            "user_id": 42, "rows": "because:155", "year_min": 2000})
        _, kwargs = page_client.service.similar_calls[-1]
        assert kwargs["filters"]["year_min"] == 2000


class TestHealth:
    def test_ok_with_live_pool(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "db": True}

    def test_degraded_reports_broken_pool(self):
        svc = FakeService(pool=FakePool(fail=True))
        app = create_app(service=svc)
        with TestClient(app) as c:
            r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "degraded", "db": False}
