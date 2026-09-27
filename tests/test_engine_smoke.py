"""Offline smoke tests: import surface, weights artifact, model registry."""

import json
from pathlib import Path

import pytest


def test_public_api_exports():
    from cine_rec_engine import RecommendationService, __version__

    assert __version__
    assert callable(RecommendationService)


def test_weights_artifact_is_valid():
    pkg = Path(__file__).parent.parent / "cine_rec_engine" / "weights.json"
    data = json.loads(pkg.read_text(encoding="utf-8"))
    assert data["model"] == "pairwise-logistic-sgd"
    names = set(data["feature_names"])
    assert set(data["weights"]) == names
    assert len(names) == 22
    # spot-check a few published weights (values frozen in 0.1.0)
    assert data["weights"]["cosine_sim"] == pytest.approx(5.13288871749017)
    assert data["weights"]["tmdb_rec_decay"] == pytest.approx(13.744720743566049)


def test_model_spaces_registry():
    from cine_rec_engine.model_spaces import REC_MODELS, is_solo, normalize_model

    assert normalize_model("ENSEMBLE") == "ensemble"
    assert normalize_model(None) is None
    assert normalize_model("bogus") is None
    assert is_solo("ensemble") is False
    solo = next(k for k, v in REC_MODELS.items() if v["column"])
    assert is_solo(solo) is True


def test_find_similar_signature():
    import inspect

    from cine_rec_engine import RecommendationService

    sig = inspect.signature(RecommendationService.find_similar)
    expected = (
        "tmdb_id", "limit", "randomness", "allow_cross_media",
        "seed_weights", "rec_model", "user_id",
    )
    for expected in expected:
        assert expected in sig.parameters
