"""The HTTP layer (cine_rec_engine.serve) — contract tests over an
injected fake engine, no database, no network.

Pinned behaviors:
- seed grammar: bare ids OR id:media_type composites, never mixed
- user_id on /similar arms the per-user pass (exclude is not None)
- unknown model names are 400s, not silent defaults
- /health reports DB readiness without crashing on a broken pool
"""

import pytest
from fastapi import Request
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
        self.explain_calls = []

    async def explain_results(self, results, seed):
        self.explain_calls.append((results, seed))
        return {"why": {"111": ["same director"]},
                "why_scores": {"111": {"director_match": 3.0}}}

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

    def test_degraded_reports_broken_pool_as_503(self):
        svc = FakeService(pool=FakePool(fail=True))
        app = create_app(service=svc)
        with TestClient(app) as c:
            r = c.get("/health")
        assert r.status_code == 503
        assert r.json() == {"status": "degraded", "db": False}


class TestCors:
    """With CINE_REC_CORS_ORIGINS set, a browser preflight for POST
    /feedback must pass — the middleware defaults (GET-only, no headers)
    reject it with 400 Disallowed CORS method/headers."""

    def test_preflight_post_feedback_is_allowed(self, monkeypatch):
        monkeypatch.setenv("CINE_REC_CORS_ORIGINS", "https://app.example.com")
        app = create_app(service=FakeService())
        with TestClient(app) as c:
            r = c.options("/feedback", headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })
        assert r.status_code == 200
        assert "POST" in r.headers.get("access-control-allow-methods", "").upper()
        assert "content-type" in r.headers.get(
            "access-control-allow-headers", "").lower()
        assert r.headers.get("access-control-allow-origin") \
            == "https://app.example.com"

    def test_preflight_from_foreign_origin_gets_no_cors_grant(self, monkeypatch):
        monkeypatch.setenv("CINE_REC_CORS_ORIGINS", "https://app.example.com")
        app = create_app(service=FakeService())
        with TestClient(app) as c:
            r = c.options("/feedback", headers={
                "Origin": "https://evil.example.net",
                "Access-Control-Request-Method": "POST",
            })
        assert "access-control-allow-origin" not in r.headers


class TestForUserValidation:
    def test_nonpositive_user_id_is_422(self, client):
        assert client.get("/for-user/0").status_code == 422
        assert client.get("/for-user/-5").status_code == 422


class TestSimilarWhy:
    def test_why_param_attaches_labels_and_scores(self, client):
        r = client.get("/similar", params={"seed": "155:movie", "why": "1"})
        assert r.status_code == 200
        body = r.json()
        assert body["why"] == {"111": ["same director"]}
        assert body["why_scores"] == {"111": {"director_match": 3.0}}
        results, seed = client.service.explain_calls[-1]
        assert seed == (155, "movie") and len(results) == 2

    def test_bare_seed_key_takes_media_type_from_results(self, client):
        client.get("/similar", params={"seed": "155", "why": "1"})
        _, seed = client.service.explain_calls[-1]
        assert seed == (155, "movie")

    def test_why_off_by_default(self, client):
        body = client.get("/similar", params={"seed": "155"}).json()
        assert "why" not in body and "why_scores" not in body

    def test_explain_failure_never_fails_the_request(self):
        class Boom(FakeService):
            async def explain_results(self, results, seed):
                raise RuntimeError("no enrichment for you")

        app = create_app(service=Boom())
        with TestClient(app) as c:
            r = c.get("/similar", params={"seed": "155:movie", "why": "1"})
        assert r.status_code == 200
        assert "why" not in r.json()


class TestV1Aliases:
    """The five data routes serve both /v1/* (canonical) and their bare
    legacy paths (deprecated aliases until 1.0 — docs/stability.md)."""

    def test_v1_similar_works(self, client):
        r = client.get("/v1/similar", params={"seed": "155"})
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_v1_for_user_works(self, client):
        r = client.get("/v1/for-user/42")
        assert r.status_code == 200
        assert r.json()["reason"] == "cold_start"

    def test_legacy_paths_carry_deprecation_headers(self, client):
        r = client.get("/similar", params={"seed": "155"})
        assert r.headers.get("deprecation") == "true"
        assert r.headers.get("x-deprecated-removal") == "1.0"

    def test_v1_paths_have_no_deprecation_headers(self, client):
        r = client.get("/v1/similar", params={"seed": "155"})
        assert "deprecation" not in r.headers

    def test_openapi_lists_both_paths_legacy_deprecated(self, client):
        spec = client.get("/openapi.json").json()
        for bare, v1 in (("/similar", "/v1/similar"),
                         ("/feedback", "/v1/feedback"),
                         ("/for-user/{user_id}", "/v1/for-user/{user_id}")):
            assert bare in spec["paths"] and v1 in spec["paths"]
            method = "post" if "feedback" in bare else "get"
            assert spec["paths"][bare][method]["deprecated"] is True
            assert "deprecated" not in spec["paths"][v1][method]


class TestAuth:
    """CINE_REC_API_KEYS (or an injected auth_dependency) gates every
    data route on both paths; /health stays open for healthchecks; with
    neither configured the API stays open (degrade never fail)."""

    def _app(self, monkeypatch, keys="k1,k2"):
        monkeypatch.setenv("CINE_REC_API_KEYS", keys)
        return create_app(service=FakeService())

    def test_missing_key_is_401_on_both_paths(self, monkeypatch):
        with TestClient(self._app(monkeypatch)) as c:
            assert c.get("/similar", params={"seed": "155"}).status_code == 401
            assert c.get("/v1/similar",
                         params={"seed": "155"}).status_code == 401

    def test_x_api_key_accepted(self, monkeypatch):
        with TestClient(self._app(monkeypatch)) as c:
            r = c.get("/similar", params={"seed": "155"},
                      headers={"X-API-Key": "k2"})
            assert r.status_code == 200

    def test_bearer_accepted_and_wrong_key_401(self, monkeypatch):
        with TestClient(self._app(monkeypatch)) as c:
            ok = c.get("/v1/similar", params={"seed": "155"},
                       headers={"Authorization": "Bearer k1"})
            assert ok.status_code == 200
            bad = c.get("/similar", params={"seed": "155"},
                        headers={"Authorization": "Bearer nope"})
            assert bad.status_code == 401

    def test_health_and_docs_stay_open(self, monkeypatch):
        with TestClient(self._app(monkeypatch)) as c:
            assert c.get("/health").status_code in (200, 503)

    def test_unset_keys_mean_open_service(self, client):
        assert client.get("/similar",
                          params={"seed": "155"}).status_code == 200

    def test_injected_auth_hook_applies_to_both_paths(self):
        from fastapi import HTTPException

        # NB: a hook parameter MUST be annotated with the real class —
        # a bare `request` would be parsed as a query param and 422
        # before the hook ever runs.
        async def _deny(request: Request) -> None:
            raise HTTPException(status_code=403, detail="hook says no")

        app = create_app(service=FakeService(), auth_dependency=_deny)
        with TestClient(app) as c:
            assert c.get("/similar",
                         params={"seed": "155"}).status_code == 403
            assert c.get("/v1/for-user/42").status_code == 403
