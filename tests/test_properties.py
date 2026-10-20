"""Property-based tests (hypothesis) for the pure scoring core.

The golden suite pins FIXED vectors; these properties must hold for ANY
input the generators can build: scores are finite and non-negative,
partial input (empty keyword/cast lists) never produces NaN, results do
not depend on dict/list ordering, and the fast path stays equal to the
full path on random pairs — not just on the captured goldens.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cine_rec_engine.service import (
    FEATURE_NAMES,
    HEURISTIC_WEIGHTS,
    _entity_profile,
    _score_candidate_vs_seed,
    _score_pair_fast,
    explain_contributions,
    feature_vector,
)

W_LIST = list(HEURISTIC_WEIGHTS.values())

_GENRES = ["Action", "Drama", "Comedy", "Thriller", "Sci-Fi", "Crime",
           "Animation", "Horror", "Romance", "Adventure"]
_KEYWORDS = [f"kw{i}" for i in range(60)]
_COMPANIES = [f"co{i}" for i in range(20)]
_NETWORKS = [f"nt{i}" for i in range(8)]
_VIA = ["knn", "tmdb", "director", "writer", ""]


@st.composite
def _entity(draw):
    return {
        "id": draw(st.integers(1, 10_000)),
        "media_type": draw(st.sampled_from(["movie", "tv"])),
        "title": "t",
        "title_en": "t",
        "genres": draw(st.lists(st.sampled_from(_GENRES), max_size=4,
                                unique=True)),
        "keywords": draw(st.lists(st.sampled_from(_KEYWORDS), max_size=12,
                                  unique=True)),
        "cast_ids": draw(st.lists(st.integers(1, 500), max_size=5,
                                  unique=True)),
        "director_ids": draw(st.lists(st.integers(1, 200), max_size=2,
                                      unique=True)),
        "writer_ids": draw(st.lists(st.integers(1, 200), max_size=2,
                                    unique=True)),
        "composer_ids": draw(st.lists(st.integers(1, 200), max_size=2,
                                      unique=True)),
        "dp_ids": draw(st.lists(st.integers(1, 200), max_size=2,
                                unique=True)),
        "companies": draw(st.lists(st.sampled_from(_COMPANIES), max_size=3,
                                   unique=True)),
        "networks": draw(st.lists(st.sampled_from(_NETWORKS), max_size=2,
                                  unique=True)),
        "original_language": draw(st.sampled_from(["en", "ko", "ja", "he"])),
        "overview": draw(st.text(min_size=0, max_size=80)),
        "overview_en": None,
        "release_year": draw(st.integers(1900, 2030)),
        "vote_average": draw(st.floats(0.0, 10.0)),
        "vote_count": draw(st.integers(0, 100_000)),
        "collection_id": draw(st.one_of(st.none(), st.integers(1, 500))),
        "adult": False,
        "via": draw(st.one_of(st.none(), st.sampled_from(_VIA))),
    }


_common = settings(max_examples=100, deadline=None)


@_common
@given(cand=_entity(), seed=_entity())
def test_scores_finite_and_non_negative(cand, seed):
    score = _score_candidate_vs_seed(
        cand, cand, cand["genres"], seed, weights=HEURISTIC_WEIGHTS)
    assert math.isfinite(score)
    assert score >= 0.0


@_common
@given(cand=_entity(), seed=_entity())
def test_feature_vector_finite_correct_length_no_nan_on_partial_input(cand, seed):
    vec = feature_vector(cand, cand, cand["genres"], seed)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(math.isfinite(v) for v in vec)
    assert not any(math.isnan(v) for v in vec)


@_common
@given(cand=_entity(), seed=_entity())
def test_result_independent_of_dict_and_list_order(cand, seed):
    reordered = {k: cand[k] for k in reversed(list(cand.keys()))}
    reordered["genres"] = list(reversed(reordered["genres"]))
    reordered["keywords"] = list(reversed(reordered["keywords"]))
    v1 = feature_vector(cand, cand, cand["genres"], seed)
    v2 = feature_vector(reordered, reordered, reordered["genres"], seed)
    assert v1 == pytest.approx(v2, rel=1e-12, abs=1e-12)


@_common
@given(cand=_entity(), seed=_entity())
def test_fast_path_equals_full_path_on_random_pairs(cand, seed):
    sp = _entity_profile(seed, None)
    cp = _entity_profile(cand, cand, genres_override=cand["genres"])
    fast = _score_pair_fast(
        sp, cp, cand, cand, set(cand["genres"]) or None, 0.0, W_LIST)
    full = _score_candidate_vs_seed(
        cand, cand, cand["genres"], seed, weights=HEURISTIC_WEIGHTS)
    assert fast == pytest.approx(full, rel=1e-12, abs=1e-12)


@_common
@given(cand=_entity(), seed=_entity())
def test_contributions_bounded_and_named(cand, seed):
    vec = feature_vector(cand, cand, cand["genres"], seed)
    scores = explain_contributions(vec, HEURISTIC_WEIGHTS)
    assert len(scores) <= 3
    assert set(scores) <= set(FEATURE_NAMES)
    assert all(math.isfinite(v) and v > 0 for v in scores.values())
