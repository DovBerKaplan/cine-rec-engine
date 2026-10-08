"""User-recommendation product rules (spec v0.3 §A4/§B/§C4).

Pure unit tests: the plan (cold start), the why (explain_features),
the dislike hard-filter, taste ordering (recent >> ancient), and the
Letterboxd importer's pure parts.
"""

from datetime import datetime, timedelta, timezone

import pytest


from cine_rec_engine.letterboxd import (
    LetterboxdRow,
    parse_letterboxd_csv,
    rating_to_kind,
)
from cine_rec_engine.service import (
    FEATURE_NAMES,
    explain_features,
    recommendation_plan,
)
from cine_rec_engine.user_weights import w_item

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class TestRecommendationPlan:
    def test_personalized_with_enough_history(self):
        seeds = [(i, "movie", 1.0) for i in range(3)]
        mode, used = recommendation_plan(seeds, [])
        assert mode == "personalized"
        assert used == seeds

    def test_partial_history_blends_watchlist_not_persona(self):
        seeds = [(1, "movie", 0.9)]
        wl = [(7, "tv", 1.0), (1, "movie", 1.0)]  # 1 already weighted
        mode, used = recommendation_plan(seeds, wl)
        assert mode == "personalized"
        assert [s[0] for s in used] == [1, 7]  # no duplicate seed

    def test_watchlist_only(self):
        mode, used = recommendation_plan([], [(9, "tv", 1.0)])
        assert mode == "watchlist"
        assert used == [(9, "tv", 1.0)]

    def test_cold_start_is_explicit_never_silent(self):
        mode, used = recommendation_plan([], [])
        assert mode == "cold_start" and used == []
        # the API contract: cold_start ⇒ empty results + the reason


class TestExplainFeatures:
    # explicit weights: the heuristic baseline zeroes the DNA features by
    # design (learned-scorer inputs), so explainability is tested against
    # a weights table where they matter.
    W = {"composer_match": 5.5, "audience_compatibility": 1.0,
         "tmdb_rec_decay": 13.7, "shared_collection": 2.5,
         "keyword_sim": 4.3}

    def _vec_with(self, name, value=1.0):
        v = [0.0] * len(FEATURE_NAMES)
        v[FEATURE_NAMES.index(name)] = value
        return v

    def test_top_feature_named(self):
        why = explain_features(self._vec_with("composer_match"), self.W)
        assert why == ["same composer"]

    def test_boring_features_hidden_by_default(self):
        why = explain_features(self._vec_with("audience_compatibility"), self.W)
        assert why == []  # fires on everything → hidden; empty is honest

    def test_zero_contribution_excluded(self):
        assert explain_features([0.0] * len(FEATURE_NAMES), self.W) == []

    def test_ordering_by_contribution(self):
        vec = self._vec_with("tmdb_rec_decay")
        vec[FEATURE_NAMES.index("shared_collection")] = 1.0
        why = explain_features(vec, self.W)
        assert why == ["TMDB behavior graph", "same saga"]  # 13.7 > 2.5


class TestDislikeHardFilter:
    async def test_dislike_merged_into_exclusions(self, monkeypatch):
        from cine_rec_engine import service as svc
        from cine_rec_engine import watched as watched_mod

        async def fake_excl(pool, user_id):
            return {(155, "movie")}

        async def fake_dislikes(pool, user_id):
            return {(475557, "movie")}

        monkeypatch.setattr(watched_mod, "get_user_recommendation_exclusions", fake_excl)
        monkeypatch.setattr(watched_mod, "get_user_dislikes", fake_dislikes)

        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = object()
        merged = await rec._merged_exclusions({(999, "tv")}, 42)
        assert (475557, "movie") in merged   # dislike hard-filtered
        assert (155, "movie") in merged      # watched kept
        assert (999, "tv") in merged         # explicit exclude kept


class TestRecommendForUserUserContext:
    async def test_find_similar_receives_exclude_so_user_filters_run(self, monkeypatch):
        """The per-user pass inside find_similar only runs when exclude is
        not None — recommend_for_user must always arm it, or watched and
        disliked titles outside the seed set leak into the list."""
        from cine_rec_engine import service as svc
        from cine_rec_engine import user_vector as uv

        async def fake_seeds(pool, user_id, limit=20):
            return [(155, "movie", 1.0), (27205, "movie", 0.8),
                    (1396, "tv", 0.5)]

        async def no_vector(pool, user_id, space=None, column=None,
                            max_age_hours=None):
            return None

        class FakePool:
            async def fetch(self, *args, **kwargs):
                return []  # empty watchlist

        captured = {}

        async def fake_find_similar(self, tmdb_id, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(uv, "top_weighted_seeds", fake_seeds)
        monkeypatch.setattr(uv, "ensure_user_vector", no_vector)
        monkeypatch.setattr(svc.RecommendationService, "find_similar",
                            fake_find_similar)

        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = FakePool()
        out = await rec.recommend_for_user(42, limit=8)

        assert out["reason"] == "personalized"
        assert out["vector_used"] is False
        assert captured["user_id"] == 42
        assert captured["exclude"] is not None  # None disarms the user pass

    async def test_vector_channel_runs_when_a_vector_exists(self, monkeypatch):
        from cine_rec_engine import service as svc
        from cine_rec_engine import user_vector as uv

        async def fake_seeds(pool, user_id, limit=20):
            return [(155, "movie", 1.0), (27205, "movie", 0.8),
                    (1396, "tv", 0.5)]

        async def has_vector(pool, user_id, space=None, column=None,
                             max_age_hours=None):
            return [0.1, 0.2, 0.3]

        class FakePool:
            async def fetch(self, *args, **kwargs):
                return []

        async def fake_find_similar(self, tmdb_id, **kwargs):
            # the lazy channel: invoke the factory the way the real
            # find_similar would on a cache miss
            if kwargs.get("extra_factory"):
                assert await kwargs["extra_factory"]() == [{"via": "user_vector"}]
            return []

        async def fake_ann(pool, uvec, limit=60, emb_column=None):
            return [{"via": "user_vector"}]

        monkeypatch.setattr(uv, "top_weighted_seeds", fake_seeds)
        monkeypatch.setattr(uv, "ensure_user_vector", has_vector)
        # module-level import in service.py — patch the service's reference
        monkeypatch.setattr(svc, "generate_user_vector_candidates", fake_ann)
        monkeypatch.setattr(svc.RecommendationService, "find_similar",
                            fake_find_similar)

        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = FakePool()
        out = await rec.recommend_for_user(42, limit=8)

        assert out["vector_used"] is True

    async def test_solo_model_space_threads_one_column_end_to_end(self, monkeypatch):
        """vector_space='e5e' must build/ensure, ANN-recall, and score in
        the e5e column — a vector from one space searched against another
        is a dimension mismatch."""
        from cine_rec_engine import service as svc
        from cine_rec_engine import user_vector as uv

        async def fake_seeds(pool, user_id, limit=20):
            return [(155, "movie", 1.0)]

        ensured = {}
        ann = {}

        async def fake_ensure(pool, user_id, space=None, column=None,
                              max_age_hours=None):
            ensured["space"] = space
            return [0.1]

        async def fake_ann(pool, uvec, limit=60, emb_column=None):
            ann["emb_column"] = emb_column
            return [{"via": "user_vector"}]

        captured = {}

        async def fake_find_similar(self, tmdb_id, **kwargs):
            captured.update(kwargs)
            if kwargs.get("extra_factory"):
                await kwargs["extra_factory"]()  # runs the ANN capture
            return []

        class FakePool:
            async def fetch(self, *args, **kwargs):
                return []

        monkeypatch.setattr(uv, "top_weighted_seeds", fake_seeds)
        monkeypatch.setattr(uv, "ensure_user_vector", fake_ensure)
        monkeypatch.setattr(svc, "generate_user_vector_candidates", fake_ann)
        monkeypatch.setattr(svc.RecommendationService, "find_similar",
                            fake_find_similar)

        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = FakePool()
        await rec.recommend_for_user(42, vector_space="e5e")

        assert ensured["space"] == "e5e"
        assert ann["emb_column"] == "embedding_e5e"
        assert captured["rec_model"] == "e5e"


class _FakeConn:
    def __init__(self, row=None, fail=False):
        self._row = row
        self._fail = fail

    async def register_vector(self):
        if self._fail:
            raise RuntimeError("no vector infrastructure")

    async def fetchrow(self, *args, **kwargs):
        if self._fail:
            raise RuntimeError("probe blew up")
        return self._row


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakeVectorPool:
    def __init__(self, conn):
        self._conn = conn
        self.conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


class TestEnsureUserVector:
    @staticmethod
    def _noop_register(monkeypatch):
        # ensure_user_vector imports register_vector lazily per call, so
        # patching the pgvector module attribute reaches it. The real one
        # introspects a live asyncpg connection — fatal to a fake conn.
        import pgvector.asyncpg as pga

        async def register_vector(conn):
            return None

        monkeypatch.setattr(pga, "register_vector", register_vector)

    def _pool(self, row=None, fail=False):
        return _FakeVectorPool(_FakeConn(row=row, fail=fail))

    async def test_fresh_vector_returned_without_rebuild(self, monkeypatch):
        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)

        async def forbidden(*args, **kwargs):
            raise AssertionError("rebuild must not run for a fresh vector")

        monkeypatch.setattr(uv, "build_user_vector", forbidden)
        vec = await uv.ensure_user_vector(
            self._pool(row={"embedding": [0.1, 0.2]}), 7)
        assert vec == [0.1, 0.2]

    async def test_fresh_vector_as_pgvector_object_is_coerced(self, monkeypatch):
        """The asyncpg codec hands back pgvector.Vector, which is NOT
        iterable — `list(v)` raises TypeError. That error inside the probe
        used to read as 'vector layer broken', silently disabling the ANN
        channel and busting the cache key on every second request."""
        from pgvector import Vector

        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)

        async def forbidden(*args, **kwargs):
            raise AssertionError("rebuild must not run for a fresh vector")

        monkeypatch.setattr(uv, "build_user_vector", forbidden)
        vec = await uv.ensure_user_vector(
            self._pool(row={"embedding": Vector([0.1, 0.2])}), 7)
        # Vector stores float32 — approx, not exact equality
        assert vec == pytest.approx([0.1, 0.2])

    async def test_load_user_vector_coerces_pgvector_object(self, monkeypatch):
        """Same coercion on the plain load path — it raises uncaught on a
        Vector value, killing the whole channel for the caller."""
        from pgvector import Vector

        from cine_rec_engine import user_vector as uv

        import pgvector.asyncpg as pga

        async def register_vector(conn):
            return None

        monkeypatch.setattr(pga, "register_vector", register_vector)
        vec = await uv.load_user_vector(
            _FakeVectorPool(_FakeConn(row={"embedding": Vector([0.3, 0.4])})), 7)
        assert vec == pytest.approx([0.3, 0.4])

    async def test_missing_vector_triggers_rebuild(self, monkeypatch):
        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)
        calls = []

        async def fake_build(pool, user_id, space, column):
            calls.append((user_id, space, column))
            return [0.9]

        monkeypatch.setattr(uv, "build_user_vector", fake_build)
        vec = await uv.ensure_user_vector(self._pool(row=None), 7)
        assert vec == [0.9]
        assert calls == [(7, None, None)]  # default space, default column

    async def test_solo_space_rebuilds_from_its_own_column(self, monkeypatch):
        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)
        calls = []

        async def fake_build(pool, user_id, space, column):
            calls.append((space, column))
            return [0.5]

        monkeypatch.setattr(uv, "build_user_vector", fake_build)
        await uv.ensure_user_vector(self._pool(row=None), 7, space="e5e")
        assert calls == [("e5e", "embedding_e5e")]  # never the default column

    async def test_probe_failure_degrades_to_none(self, monkeypatch):
        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)

        async def forbidden(*args, **kwargs):
            raise AssertionError("no rebuild after a failed probe")

        monkeypatch.setattr(uv, "build_user_vector", forbidden)
        assert await uv.ensure_user_vector(self._pool(fail=True), 7) is None

    async def test_rebuild_failure_degrades_to_none(self, monkeypatch):
        from cine_rec_engine import user_vector as uv

        self._noop_register(monkeypatch)

        async def exploding(*args, **kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(uv, "build_user_vector", exploding)
        assert await uv.ensure_user_vector(self._pool(row=None), 7) is None


class TestUserTilt:
    """Controlled personalization: boost-only, capped, seed stays primary."""

    def _row(self, tid, score, media_type="movie"):
        return {"tmdb_id": tid, "media_type": media_type, "score": score,
                "title": f"t{tid}"}

    def test_near_tie_flips_toward_user_affinity(self):
        from cine_rec_engine.service import _tilt_scores

        results = [self._row(1, 10.0), self._row(2, 9.6)]
        out = _tilt_scores(results, {(2, "movie"): 0.8}, alpha=0.15)
        assert [r["tmdb_id"] for r in out] == [2, 1]  # 9.6·1.12 > 10.0
        assert out[0]["user_affinity"] == 0.8

    def test_clear_gap_never_flips_seed_relevance_primary(self):
        from cine_rec_engine.service import _tilt_scores

        results = [self._row(1, 10.0), self._row(2, 5.0)]
        out = _tilt_scores(results, {(2, "movie"): 1.0}, alpha=0.15)
        assert [r["tmdb_id"] for r in out] == [1, 2]  # 5.75 < 10, no flip

    def test_negative_affinity_never_buries(self):
        from cine_rec_engine.service import _tilt_scores

        results = [self._row(1, 8.0)]
        out = _tilt_scores(results, {(1, "movie"): -0.9}, alpha=0.15)
        assert out[0]["score"] == 8.0  # clamped to zero boost
        assert out[0]["user_affinity"] == 0.0

    def test_no_affinity_passes_through_and_cache_not_mutated(self):
        from cine_rec_engine.service import _tilt_scores

        results = [self._row(1, 7.0), self._row(2, 7.5)]
        out = _tilt_scores(results, {}, alpha=0.15)
        assert [r["tmdb_id"] for r in out] == [2, 1]
        assert "user_affinity" not in out[0]
        assert out[0] is results[1]  # same dict — payload never re-packed

    def test_cosine_handles_unnormalized_vectors(self):
        from cine_rec_engine.service import _cosine

        assert _cosine([2.0, 0.0], [5.0, 0.0]) == 1.0
        assert _cosine([1.0, 0.0], [0.0, 3.0]) == 0.0
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector safe

    async def test_apply_user_tilt_end_to_end(self, monkeypatch):
        from cine_rec_engine import queries as queries_mod
        from cine_rec_engine import service as svc
        from cine_rec_engine import user_vector as uv

        async def has_vector(pool, user_id, space=None, column=None,
                             max_age_hours=None):
            return [1.0, 0.0]

        async def fake_embeddings(pool, pairs, emb_column=None):
            assert emb_column is None  # default space
            return {(111, "movie"): [1.0, 0.0],   # aligned with the user
                    (222, "movie"): [0.0, 1.0]}   # orthogonal

        monkeypatch.setattr(uv, "ensure_user_vector", has_vector)
        monkeypatch.setattr(queries_mod, "fetch_embeddings_batch",
                            fake_embeddings)

        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = object()
        results = [self._row(222, 9.5), self._row(111, 9.0)]
        out = await rec._apply_user_tilt(results, 7, None)

        assert [r["tmdb_id"] for r in out] == [111, 222]  # 9.0·1.15 > 9.5
        assert out[0]["user_affinity"] == 1.0
        assert out[1]["user_affinity"] == 0.0
        assert results[0]["score"] == 9.5  # input list untouched

    async def test_apply_user_tilt_degrades_without_vector(self, monkeypatch):
        from cine_rec_engine import service as svc
        from cine_rec_engine import user_vector as uv

        async def no_vector(pool, user_id, space=None, column=None,
                            max_age_hours=None):
            return None

        monkeypatch.setattr(uv, "ensure_user_vector", no_vector)
        rec = svc.RecommendationService.__new__(svc.RecommendationService)
        rec.pool = object()
        results = [self._row(1, 5.0)]
        assert await rec._apply_user_tilt(results, 7, None) is results


class TestTasteOrdering:
    def test_recent_watch_dominates_ancient(self):
        recent = w_item("movie", max_ratio=0.9,
                        last_watched_at=NOW - timedelta(days=1), now=NOW)
        ancient = w_item("movie", max_ratio=0.9,
                         last_watched_at=NOW - timedelta(days=180), now=NOW)
        assert recent > ancient * 5  # 2^-1/30 vs 2^-180/30 ≈ 5.3×

    def test_dropped_movie_is_never_a_seed(self):
        assert w_item("movie", max_ratio=0.10, last_watched_at=NOW, now=NOW) == 0.0

    def test_hooked_series_is_a_seed(self):
        from cine_rec_engine.user_weights import SeriesSignals

        w = w_item("tv", series=SeriesSignals(
            episodes_watched=3, total_episodes=10,
            last_ep_ratio=1.0, last_ep_watched_sec=2400,
            total_watched_sec=3 * 2700),
            last_watched_at=NOW, now=NOW)
        assert w > 0


class TestLetterboxdParser:
    CSV = (chr(0xFEFF) + "Date,Name,Year,Letterboxd URI,Rating\n"
           "2026-09-01,The Dark Knight,2008,https://x,5.0\n"
           "2026-09-02,Memento,2000,https://x,3.5\n"
           "2026-09-03,Catwoman,2004,https://x,1.0\n"
           "2026-09-04,No Year,,https://x,4.0\n"
           "2026-09-05,,,,\n")

    def test_parses_rows_with_bom_and_blanks(self):
        rows = parse_letterboxd_csv(self.CSV)
        assert len(rows) == 4
        assert rows[0] == LetterboxdRow("The Dark Knight", 2008, 5.0)
        assert rows[3].year is None

    def test_rating_thresholds(self):
        assert rating_to_kind(5.0) == "favorite"
        assert rating_to_kind(3.5) == "favorite"
        assert rating_to_kind(3.0) is None      # unremarkable → no signal
        assert rating_to_kind(2.0) == "dislike"
        assert rating_to_kind(0.0) == "dislike"  # unrated in a ratings file

    def test_real_export_shape(self):
        # actual header from Letterboxd: Date,Name,Year,Letterboxd URI,Rating
        text = "Date,Name,Year,Letterboxd URI,Rating\n2026-01-01,Se7en,1995,u,4.5\n"
        rows = parse_letterboxd_csv(text)
        assert rows == [LetterboxdRow("Se7en", 1995, 4.5)]


class TestExplainWithWeights:
    W = {"composer_match": 5.5, "genre_priority_sum": 0.5}

    def test_returns_label_contribution_tuples(self):
        from cine_rec_engine.service import FEATURE_NAMES, explain_features

        vec = [0.0] * len(FEATURE_NAMES)
        vec[FEATURE_NAMES.index("composer_match")] = 1.0
        out = explain_features(vec, self.W, with_weights=True)
        assert out == [("same composer", 5.5)]

    def test_default_unchanged(self):
        from cine_rec_engine.service import FEATURE_NAMES, explain_features

        vec = [0.0] * len(FEATURE_NAMES)
        vec[FEATURE_NAMES.index("composer_match")] = 1.0
        assert explain_features(vec, self.W) == ["same composer"]
