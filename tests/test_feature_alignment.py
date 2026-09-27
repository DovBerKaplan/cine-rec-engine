"""Regression: the feature vector must align 1:1 with FEATURE_NAMES.

narrative_match was computed but never added to the vec, shifting every
mood feature (index 22+) onto its neighbor's weight — tone_compat was
priced at narrative's 1.5 instead of its own 5.0.
"""

from cine_rec_engine.service import (
    FEATURE_NAMES,
    HEURISTIC_WEIGHTS,
    feature_vector,
)


def _seed():
    return {
        "id": 155, "media_type": "movie", "genres": ["Action", "Crime"],
        "director_ids": [1], "keywords": ["batman"],
        "collection_id": 9735, "title": "The Dark Knight",
    }


def _cand():
    return {
        "id": 146233, "media_type": "movie", "genres": ["Action", "Thriller"],
        "director_ids": [1], "keywords": ["batman", "gotham"],
        "collection_id": 9735, "title": "The Batman",
        "vote_average": 7.8, "vote_count": 9000, "popularity": 50.0,
        "via": None,
    }


def test_vec_length_matches_feature_names():
    vec = feature_vector(_cand(), _cand(), ["Action", "Thriller"], _seed())
    assert len(vec) == len(FEATURE_NAMES) == 28


def test_named_weights_equal_zip_score():
    vec = feature_vector(_cand(), _cand(), ["Action", "Thriller"], _seed())
    named = sum(HEURISTIC_WEIGHTS[n] * v for n, v in zip(FEATURE_NAMES, vec))
    zipped = sum(w * v for w, v in zip(HEURISTIC_WEIGHTS.values(), vec))
    assert named == zipped  # any silent shift breaks this equality
