"""M2 (RFC): exploration budget, outcome feedback, NL cold start.

Pure semantics + API contracts; DB-backed pieces are monkeypatched.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from cine_rec_engine.service import _exploration_slice, _insert_exploration


def _row(tid, score, genres, media_type="movie", aff=None):
    return {"tmdb_id": tid, "media_type": media_type, "score": score,
            "genres": genres, "aff": aff}


class TestExplorationSlice:
    def _affinities(self, rows):
        return {(r["tmdb_id"], r["media_type"]): r["aff"] for r in rows
                if r["aff"] is not None}

    def test_novel_and_relevant_pick_is_reserved(self):
        rows = [_row(1, 10.0, ["Crime"], aff=0.9),
                _row(2, 9.0, ["Crime"], aff=0.8),
                _row(3, 8.0, ["Documentary"], aff=0.85),  # novel + relevant
                _row(4, 7.0, ["Romance"], aff=0.1)]       # novel, irrelevant
        main, picks = _exploration_slice(
            rows, 10, {"Crime"}, self._affinities(rows), share=0.2)
        assert [r["tmdb_id"] for r in picks] == [3]
        assert 4 not in [r["tmdb_id"] for r in picks]  # below median bar
        assert all(r["tmdb_id"] != 3 for r in main)

    def test_genre_overlap_is_not_novel(self):
        rows = [_row(1, 10.0, ["Crime"], aff=0.9),
                _row(2, 9.0, ["Crime", "Drama"], aff=0.85)]
        main, picks = _exploration_slice(
            rows, 10, {"Crime"}, self._affinities(rows), share=0.3)
        assert picks == []  # nothing novel → no exploration, not noise

    def test_no_affinity_signal_no_exploration(self):
        rows = [_row(1, 10.0, ["Crime"]), _row(2, 9.0, ["Romance"])]
        main, picks = _exploration_slice(
            rows, 10, {"Crime"}, {}, share=0.3)
        assert picks == [] and main is rows

    def test_share_zero_disables(self):
        rows = [_row(1, 10.0, ["Crime"], aff=0.9)]
        main, picks = _exploration_slice(
            rows, 10, set(), self._affinities(rows), share=0.0)
        assert picks == []

    def test_top_three_never_explored_into(self):
        # limit 4 → n = min(ceil(0.5*4), 4-3) = 1 slot only
        rows = [_row(i, 10.0 - i, ["Crime" if i == 1 else "X"], aff=0.9)
                for i in range(1, 6)]
        main, picks = _exploration_slice(
            rows, 4, {"Crime"}, self._affinities(rows), share=0.5)
        assert len(picks) == 1

    def test_insert_pins_top3_and_is_deterministic(self):
        main = [_row(i, 10 - i, ["A"]) for i in range(1, 8)]
        picks = [_row(99, 5.0, ["Z"]), _row(98, 4.5, ["Z"])]
        out = _insert_exploration(main, picks, 9)
        assert [r["tmdb_id"] for r in out[:3]] == [1, 2, 3]  # pinned
        assert out == _insert_exploration(main, picks, 9)    # deterministic
        assert len(out) == 9


# ---------------------------------------------------------------------------
# /feedback — outcome ingestion over verified impression tokens
# ---------------------------------------------------------------------------

from cine_rec_engine.impressions import issue  # noqa: E402
from cine_rec_engine.serve import create_app  # noqa: E402


class FakeFeedbackPool:
    async def execute(self, *a, **k):
        return "UPDATE 1"


class FakeFeedbackService:
    def __init__(self):
        self.pool = FakeFeedbackPool()


@pytest.fixture
def fb_client(monkeypatch):
    import cine_rec_engine.user_stats as us

    calls = []

    async def fake_record_event(pool, ev):
        calls.append(("event", ev))

    async def fake_record_feedback(pool, user_id, tmdb_id, media_type, kind):
        calls.append(("feedback", (user_id, tmdb_id, media_type, kind)))

    async def fake_skip_decay(pool, user_id, tmdb_id, media_type,
                              factor=None):
        calls.append(("decay", (user_id, tmdb_id, media_type)))
        return True

    monkeypatch.setattr(us, "record_event", fake_record_event)
    monkeypatch.setattr(us, "record_feedback", fake_record_feedback)
    monkeypatch.setattr(us, "apply_skip_decay", fake_skip_decay)

    app = create_app(service=FakeFeedbackService())
    with TestClient(app) as c:
        c.calls = calls
        yield c


def _token(user_id=7, ids=((111, "movie"), (222, "tv"))):
    return issue({"u": user_id, "r": "because",
                  "i": [[i, m] for i, m in ids]})


class TestFeedbackEndpoint:
    def test_click_recorded(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": _token(), "outcome": "click", "tmdb_id": 111})
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"
        assert fb_client.calls == [("feedback", (7, 111, "movie", "click"))]

    def test_watch_becomes_a_completed_event(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": _token(), "outcome": "watch", "tmdb_id": 222})
        assert r.status_code == 200
        kind, ev = fb_client.calls[0]
        assert kind == "event"
        assert ev["user_id"] == 7 and ev["tmdb_id"] == 222
        assert ev["media_type"] == "tv" and ev["completed"] is True

    def test_skip_decays_boundedly(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": _token(), "outcome": "skip", "tmdb_id": 111})
        assert r.status_code == 200
        kinds = [c[0] for c in fb_client.calls]
        assert kinds == ["feedback", "decay"]  # recorded AND nudged

    def test_invalid_token_400(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": "garbage.token", "outcome": "click", "tmdb_id": 111})
        assert r.status_code == 400

    def test_title_outside_impression_400(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": _token(), "outcome": "click", "tmdb_id": 999})
        assert r.status_code == 400
        assert "not part of this impression" in r.json()["detail"]

    def test_bad_outcome_422(self, fb_client):
        r = fb_client.post("/feedback", json={
            "token": _token(), "outcome": "love", "tmdb_id": 111})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /by-text — NL cold start
# ---------------------------------------------------------------------------

from cine_rec_engine import text_query  # noqa: E402


class TestByText:
    async def test_falls_back_to_popularity_without_encoder(self, monkeypatch):
        import cine_rec_engine.queries as q

        monkeypatch.delenv("CINE_REC_ENCODER", raising=False)
        text_query._encoder_cache.clear()

        captured = {}

        class FakeSvc:
            pool = object()

            async def find_similar(self, seeds, **kwargs):
                captured.update(kwargs)
                assert seeds == [text_query.VIRTUAL_SEED_ID]
                return []

        async def fake_popular(pool, limit=120, filters=None):
            return [{"tmdb_id": 1, "via": "text"}]

        monkeypatch.setattr(q, "generate_popular_candidates", fake_popular)
        out = await text_query.by_text(FakeSvc(), "like Inception but darker")
        assert out["vector_used"] is False
        assert captured["extra_candidates"] == [{"tmdb_id": 1, "via": "text"}]
        assert captured["seed_info_override"]["overview"] == \
            "like Inception but darker"
        assert "seed_vectors" not in captured or \
            captured["seed_vectors"] is None

    async def test_encoder_anchors_the_ann_channel(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        import cine_rec_engine.queries as q

        monkeypatch.setenv("CINE_REC_ENCODER", "fake_enc:encode")
        text_query._encoder_cache.clear()

        async def fake_ann(pool, uvec, limit=60, emb_column=None):
            return [{"tmdb_id": 5, "via": "uvec", "knn_similarity": 0.8}]

        monkeypatch.setattr(q, "generate_user_vector_candidates", fake_ann)

        captured = {}

        class FakeSvc:
            pool = object()

            async def find_similar(self, seeds, **kwargs):
                captured.update(kwargs)
                return []

        def encode(text):
            return [0.1, 0.2]

        monkeypatch.setitem(sys.modules, "fake_enc",
                            SimpleNamespace(encode=encode))
        out = await text_query.by_text(FakeSvc(), "q")
        assert out["vector_used"] is True
        from cine_rec_engine.config import EMBEDDING_COLUMN

        assert captured["seed_vectors"] == {
            text_query.VIRTUAL_SEED_ID: {EMBEDDING_COLUMN: [0.1, 0.2]}}

    async def test_empty_query_rejected(self):
        class FakeSvc:
            pool = object()

            async def find_similar(self, *a, **k):
                return []

        with pytest.raises(ValueError, match="must not be empty"):
            await text_query.by_text(FakeSvc(), "   ")

    def test_serve_contract(self, monkeypatch):
        import cine_rec_engine.serve as serve_mod

        async def fake_by_text(svc, q, **kwargs):
            return {"query": q, "results": [], "count": 0,
                    "vector_used": False}

        monkeypatch.setattr(serve_mod, "by_text", fake_by_text)
        app = create_app(service=FakeFeedbackService())
        with TestClient(app) as c:
            r = c.get("/by-text", params={"q": "like Inception but darker"})
            assert r.status_code == 200
            assert r.json()["query"] == "like Inception but darker"
            assert c.get("/by-text").status_code == 422  # q required


# ---------------------------------------------------------------------------
# discover row + explore threading on the page
# ---------------------------------------------------------------------------

class TestDiscoverRow:
    async def test_discover_row_present_with_vector(self, monkeypatch):
        import cine_rec_engine.queries as q
        import cine_rec_engine.user_vector as uv
        from cine_rec_engine.page import compose_page
        from tests.test_page import FakePageService

        async def fake_ensure(pool, user_id, space=None, column=None,
                              max_age_hours=None):
            return [0.1, 0.2]

        async def fake_discovery(pool, uvec, exclude_ids, limit=60,
                                 emb_column=None):
            assert exclude_ids == [80]
            return [{"tmdb_id": 777, "media_type": "movie",
                     "title": "Far Out", "knn_similarity": 0.7}]

        async def fake_top_genres(pool, user_id, limit=3):
            return [80]

        monkeypatch.setattr(uv, "ensure_user_vector", fake_ensure)
        monkeypatch.setattr(q, "generate_discovery_candidates", fake_discovery)
        monkeypatch.setattr(q, "top_user_genre_ids", fake_top_genres)

        page = await compose_page(
            FakePageService(), user_id=42, rows="discover", row_limit=6)
        assert [r["kind"] for r in page["rows"]] == ["discover"]
        assert page["rows"][0]["titles"][0]["tmdb_id"] == 777

    async def test_discover_omitted_without_vector(self, monkeypatch):
        import cine_rec_engine.queries as q
        import cine_rec_engine.user_vector as uv
        from cine_rec_engine.page import compose_page
        from tests.test_page import FakePageService

        async def no_vector(pool, user_id, space=None, column=None,
                            max_age_hours=None):
            return None

        async def fake_top_genres(pool, user_id, limit=3):
            return [80]

        monkeypatch.setattr(uv, "ensure_user_vector", no_vector)
        monkeypatch.setattr(q, "top_user_genre_ids", fake_top_genres)

        page = await compose_page(
            FakePageService(), user_id=42, rows="discover")
        assert page["rows"] == []  # degraded to omitted, page survives

    async def test_explore_threads_into_rows(self, monkeypatch):
        import cine_rec_engine.queries as q
        import cine_rec_engine.user_vector as uv
        from cine_rec_engine.page import compose_page
        from tests.test_page import FakePageService

        captured = {}

        async def no_vector(pool, user_id, space=None, column=None,
                            max_age_hours=None):
            return None

        monkeypatch.setattr(uv, "ensure_user_vector", no_vector)
        monkeypatch.setattr(q, "top_user_genre_ids",
                            lambda *a, **k: asyncio.sleep(0, result=[]))

        svc = FakePageService()
        orig_similar = svc.find_similar

        async def spy_similar(seeds, **kwargs):
            captured.update(kwargs)
            return await orig_similar(seeds, **kwargs)

        svc.find_similar = spy_similar
        await compose_page(svc, user_id=42, rows="because:155", explore=0.2)
        assert captured["explore"] == 0.2
        assert captured["user_context"]["exclusions"] == set()
