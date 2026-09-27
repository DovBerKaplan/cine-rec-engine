"""Offline tests for the personalization layer (spec §D, §E.1, §J).

Pure Python only — no DB. The SQL plumbing is exercised by the live
Postgres smoke (benchmarks + server run in the repo workflow).
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from cine_rec_engine.user_vector import aggregate
from cine_rec_engine.user_weights import (
    SeriesSignals,
    is_dropped,
    recency_decay,
    s_movie,
    s_series,
    w_item,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


class TestSMovie:
    def test_dropped_below_15pct(self):
        assert s_movie(0.10) == 0.0
        assert s_movie(0.0) == 0.0
        assert s_movie(None) == 0.0

    def test_linear_between_15_and_75(self):
        assert s_movie(0.15) == pytest.approx(0.15)
        assert s_movie(0.50) == pytest.approx(0.50)
        assert s_movie(0.749) == pytest.approx(0.749)

    def test_completed_is_flat_1(self):
        assert s_movie(0.75) == 1.0
        assert s_movie(0.99) == 1.0

    def test_pause_penalty_only_when_unfinished(self):
        assert s_movie(0.50, pause_count_total=5) == pytest.approx(0.50 * 0.85)
        assert s_movie(0.90, pause_count_total=9) == 1.0  # completed: ignored
        assert s_movie(0.50, pause_count_total=2) == pytest.approx(0.50)  # below min


class TestSSeries:
    def test_first_episode_only_half_watched(self):
        sig = SeriesSignals(episodes_watched=0, last_ep_ratio=0.55,
                            last_ep_watched_sec=1200)
        assert s_series(sig) == pytest.approx(0.30)

    def test_depth_ladder(self):
        # 1 of 10 episodes → max(0.30, 0.10) = 0.30
        assert s_series(SeriesSignals(episodes_watched=1, total_episodes=10)) == \
            pytest.approx(0.30)
        # 2 of 10 → max(0.45, 0.20) = 0.45
        assert s_series(SeriesSignals(episodes_watched=2, total_episodes=10)) == \
            pytest.approx(0.45)
        # 3 of 10 → max(0.65, 0.30) = 0.65 (the 3-episode rule)
        assert s_series(SeriesSignals(episodes_watched=3, total_episodes=10)) == \
            pytest.approx(0.65)
        # 10 of 10 → max(0.65, 1.0) = 1.0 (ratio only lifts)
        assert s_series(SeriesSignals(episodes_watched=10, total_episodes=10)) == \
            pytest.approx(1.0)

    def test_long_watch_floor(self):
        sig = SeriesSignals(episodes_watched=1, total_episodes=100,
                            total_watched_sec=90 * 60)
        assert s_series(sig) == pytest.approx(0.65)

    def test_last_episode_exit_bonus(self):
        sig = SeriesSignals(episodes_watched=2, total_episodes=10,
                            last_ep_ratio=0.85, last_ep_watched_sec=2000)
        assert s_series(sig) == pytest.approx(min(1.0, 0.45 + 0.10))

    def test_last_episode_abandon_penalty(self):
        sig = SeriesSignals(episodes_watched=2, total_episodes=10,
                            last_ep_ratio=0.10, last_ep_watched_sec=120)
        assert s_series(sig) == pytest.approx(0.45 * 0.90)

    def test_natural_stop_no_penalty(self):
        # finished ep 2 fully, never started ep 3 → untouched
        sig = SeriesSignals(episodes_watched=2, total_episodes=10,
                            last_ep_ratio=1.0, last_ep_watched_sec=2400)
        assert s_series(sig) == pytest.approx(0.55)  # 0.45 + exit bonus


class TestDropped:
    def test_movie_drop(self):
        assert is_dropped("movie", 0.14, 0, None) is True
        assert is_dropped("movie", 0.15, 0, None) is False

    def test_series_first_ep_drop(self):
        assert is_dropped("tv", None, 0, 0.49) is True
        assert is_dropped("tv", None, 0, 0.50) is False
        assert is_dropped("tv", None, 1, 0.10) is False  # a full ep saves it


class TestRecency:
    def test_half_life_30_days(self):
        then = NOW - timedelta(days=30)
        assert recency_decay(then, now=NOW) == pytest.approx(0.5)
        assert recency_decay(NOW, now=NOW) == pytest.approx(1.0)
        assert recency_decay(None) == 0.0

    def test_configurable_half_life(self):
        then = NOW - timedelta(days=10)
        assert recency_decay(then, now=NOW, half_life_days=10) == \
            pytest.approx(0.5)


class TestWItem:
    def test_full_formula_clips_at_1_5(self):
        w = w_item(
            "movie", max_ratio=1.0,
            last_watched_at=NOW, now=NOW, rewatch_count=2,
        )
        assert w == pytest.approx(1.5)  # 1.0 * 1.0 * 1.5 clipped

    def test_recent_completed_movie(self):
        w = w_item("movie", max_ratio=0.80, last_watched_at=NOW, now=NOW)
        assert w == pytest.approx(1.0)

    def test_decayed_series(self):
        watched = NOW - timedelta(days=60)  # two half-lives
        w = w_item(
            "tv", series=SeriesSignals(episodes_watched=3, total_episodes=10),
            last_watched_at=watched, now=NOW,
        )
        assert w == pytest.approx(0.65 * 0.25)

    def test_dislike_is_zero(self):
        assert w_item("movie", max_ratio=1.0, disliked=True) == 0.0

    def test_dropped_is_zero(self):
        assert w_item("movie", max_ratio=0.10, last_watched_at=NOW) == 0.0
        assert w_item("tv", series=SeriesSignals(
            episodes_watched=0, last_ep_ratio=0.3, last_ep_watched_sec=300),
            last_watched_at=NOW) == 0.0

    def test_favorite_without_completion(self):
        w = w_item("movie", max_ratio=0.40, last_watched_at=NOW, now=NOW,
                   favorite=True)
        assert w == pytest.approx(min(0.40 * 1.5, 1.5))

    def test_watchlist_only(self):
        w = w_item("movie", watchlist_only=True)
        assert w == pytest.approx(0.7 * 1.2)


class TestAggregate:
    def test_weighted_sum_and_l2(self):
        v = aggregate([[1.0, 0.0], [0.0, 1.0]], [2.0, 1.0])
        n = math.sqrt((2 / math.sqrt(5)) ** 2 + (1 / math.sqrt(5)) ** 2)
        assert v[0] == pytest.approx(2 / math.sqrt(5))
        assert v[1] == pytest.approx(1 / math.sqrt(5))
        assert n == pytest.approx(1.0)

    def test_zero_weights_excluded(self):
        v = aggregate([[1.0, 0.0], [9.0, 9.0]], [0.0, 0.0])
        assert v is None

    def test_no_items_is_none_not_zero_vector(self):
        assert aggregate([], []) is None

    def test_opposite_vectors_cancel_to_none(self):
        # sum is the zero vector — not a fake search direction (§E.1)
        assert aggregate([[1.0, 0.0], [-1.0, 0.0]], [1.0, 1.0]) is None

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            aggregate([[1.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 1.0])


class TestAcceptanceRuleJ:
    """§J — a user vector is valid iff: ≥1 weighted item with embedding,
    u normalized, dislikes/drop-offs absent from the sum."""

    def test_valid_vector_normalized_excludes_dislikes(self):
        # weighted: two liked; a disliked one carries w=0 → excluded upstream
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        weights = [1.0, 1.0]          # dislike already zeroed by w_item
        u = aggregate(vectors, weights)
        assert u is not None
        assert math.sqrt(sum(x * x for x in u)) == pytest.approx(1.0)

    def test_all_disliked_user_has_no_vector(self):
        assert aggregate([[1.0, 0.0]], [0.0]) is None
