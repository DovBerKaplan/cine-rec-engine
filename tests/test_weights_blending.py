"""Learned-weights loading: opt-in gating + partial application."""

import json

import importlib

import pytest

from cine_rec_engine import service as svc


@pytest.fixture(autouse=True)
def _restore_default_scorer(monkeypatch):
    """Module reloads mutate process-global weights — always end clean."""
    yield
    monkeypatch.delenv("CINE_REC_SCORER", raising=False)
    monkeypatch.delenv("REC_SCORER", raising=False)
    importlib.reload(svc)


def _reload(monkeypatch, scorer):
    monkeypatch.setenv("CINE_REC_SCORER", scorer)
    return importlib.reload(svc)


def test_default_is_heuristic(monkeypatch):
    monkeypatch.delenv("CINE_REC_SCORER", raising=False)
    monkeypatch.delenv("REC_SCORER", raising=False)
    mod = _reload(monkeypatch, "heuristic")
    assert mod.LEARNED_WEIGHTS is None
    assert mod.ACTIVE_WEIGHTS is mod.HEURISTIC_WEIGHTS


def test_partial_artifact_blends_with_heuristic(monkeypatch):
    """The published weights.json covers 22/28 features — the 6 mood
    features must keep their heuristic coefficients instead of rejecting
    the whole artifact."""
    import pathlib

    mod = _reload(monkeypatch, "learned")
    assert mod.LEARNED_WEIGHTS is not None
    data = json.loads(
        (pathlib.Path(mod.__file__).parent / "weights.json").read_text()
    )
    # learned values win where present
    assert mod.LEARNED_WEIGHTS["cosine_sim"] == data["weights"]["cosine_sim"]
    # heuristic fills the features the artifact lacks
    assert mod.LEARNED_WEIGHTS["tone_compatibility"] == \
        mod.HEURISTIC_WEIGHTS["tone_compatibility"]
    assert len(mod.LEARNED_WEIGHTS) == len(mod.FEATURE_NAMES) == 28


def test_env_aliases(monkeypatch):
    monkeypatch.delenv("CINE_REC_SCORER", raising=False)
    monkeypatch.setenv("REC_SCORER", "learned")
    mod = importlib.reload(svc)
    assert mod.LEARNED_WEIGHTS is not None
