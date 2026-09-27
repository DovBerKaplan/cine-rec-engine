"""User-recommendation product rules (spec v0.3 §A4/§B/§C4).

Pure unit tests: the plan (cold start), the why (explain_features),
the dislike hard-filter, taste ordering (recent >> ancient), and the
Letterboxd importer's pure parts.
"""

from datetime import datetime, timedelta, timezone


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
