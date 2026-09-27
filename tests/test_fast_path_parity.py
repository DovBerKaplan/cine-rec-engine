"""Fast scoring path parity — locked to the golden vectors.

The hot loop (_score_pair_fast + _entity_profile) must reproduce
_score_candidate_vs_seed's outputs to float ULP tolerance (summation
order differs; max observed relative delta ~2e-16). The golden file was
captured from the pre-optimization implementation.
"""

import json
from pathlib import Path

import pytest

from cine_rec_engine.service import (
    HEURISTIC_WEIGHTS,
    _entity_profile,
    _score_candidate_vs_seed,
    _score_pair_fast,
    feature_vector,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden_feature_vectors.json").read_text()
)

W_LIST = list(HEURISTIC_WEIGHTS.values())


def _entities():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
    from bench_scoring import build

    return build(120, 4)


def test_fast_scores_match_golden_within_ulp():
    cands, seeds = _entities()
    checked = 0
    for i, (cand, seed) in enumerate((c, s) for c in cands for s in seeds):
        cp = _entity_profile(cand, cand, genres_override=cand["genres"])
        sp = _entity_profile(seed, None)
        fast = _score_pair_fast(
            sp, cp, cand, cand, set(cand["genres"]) or None, 0.0, W_LIST
        )
        assert fast == pytest.approx(GOLDEN[i]["score"], rel=1e-12)
        checked += 1
    assert checked == len(GOLDEN) == 480


def test_feature_vectors_match_golden_within_ulp():
    """feature_vector matches the pre-optimization vectors to float ULPs.

    Set-iteration order (hash-seeded per process) reorders float sums, so
    exact bit equality is not portable; sums are deterministic (sorted)
    and equality holds to ≤1 ULP everywhere.
    """
    cands, seeds = _entities()
    for i, (cand, seed) in enumerate((c, s) for c in cands for s in seeds):
        vec = feature_vector(cand, cand, cand["genres"], seed)
        assert vec == pytest.approx(GOLDEN[i]["vec"], rel=1e-12, abs=1e-12)


def test_gate_exemptions_match():
    """via-carrying candidates bypass the genre gate on both paths."""
    cands, seeds = _entities()
    seed = seeds[0]
    sp = _entity_profile(seed, None)
    for via in ("knn", "tmdb", "director", "writer", ""):
        cand = dict(cands[0])
        cand["via"] = via
        # ensure a genre clash so only the exemption can save the score
        cand["genres"] = [g for g in cand["genres"] if g not in seed["genres"]] or ["War"]
        ref = _score_candidate_vs_seed(
            cand, cand, cand["genres"], seed, weights=HEURISTIC_WEIGHTS
        )
        cp = _entity_profile(cand, cand, genres_override=cand["genres"])
        fast = _score_pair_fast(
            sp, cp, cand, cand, set(cand["genres"]) or None, 0.0, W_LIST
        )
        assert fast == pytest.approx(ref, rel=1e-12), f"via={via!r}"


def test_hot_loop_shape_is_fast():
    """Loose perf smoke: the pair loop stays in the tens-of-µs regime."""
    import time

    cands, seeds = _entities()
    seed_profiles = [_entity_profile(s, None) for s in seeds]
    t0 = time.perf_counter()
    for cand in cands[:50]:
        cp = _entity_profile(cand, cand, genres_override=cand["genres"])
        cgs = set(cand["genres"]) or None
        for sp in seed_profiles:
            _score_pair_fast(sp, cp, cand, cand, cgs, 0.0, W_LIST)
    per_pair_us = (time.perf_counter() - t0) / (50 * len(seeds)) * 1e6
    # generous bound — CI boxes vary; catches 10x regressions only
    assert per_pair_us < 250, f"{per_pair_us:.1f} µs/pair regressed"
