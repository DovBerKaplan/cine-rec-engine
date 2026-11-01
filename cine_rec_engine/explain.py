"""WHY rendering — human/machine-readable reasons for one score.

Split from service.py (phase 5a, call-flow split): everything that
turns a feature vector into explanations. Pure published-weights
arithmetic only — exposes nothing beyond weights.json.

`explain_results` re-derives the vector post-hoc against the FIRST
seed only — an approximation by design (same as the /for-user path).
"""

from __future__ import annotations

from loguru import logger

from .queries import enrich_candidates_batch, get_movie_info_batch

FEATURE_LABELS = {
    "cosine_sim": "plot similarity",
    "keyword_sim": "shared keywords",
    "cast_sim": "shared cast",
    "director_match": "same director",
    "director_channel": "director recall",
    "writer_match": "same writer",
    "composer_match": "same composer",
    "dp_match": "same cinematographer",
    "tmdb_rec_decay": "TMDB behavior graph",
    "shared_collection": "same saga",
    "shared_network": "same network",
    "style_match": "same style tags",
    "company_sim": "same studio",
    "genre_priority_sum": "genre overlap",
    "tone_compatibility": "same tone",
    "narrative_match": "same narrative structure",
    "emotional_arc_match": "same emotional arc",
    "pacing_match": "same pacing",
    "audience_compatibility": "same audience",
}
# near-constant features: they fire on ~everything, so they explain a
# score but not a CHOICE — hidden from the why by default
_BORING_FEATURES = frozenset({
    "audience_compatibility", "tone_compatibility", "rating_bonus",
    "votes_gt15k", "low_votes_high_rating", "year_le5", "year_le10",
    "heuristic_score",
})


def explain_features(vec, weights, top=3, with_weights=False):
    """Human-readable WHY for one (seed, candidate) score (pure).

    Maps the feature vector to its top contributing features, skipping
    near-constant ones by default (they score everything equally).
    with_weights=True returns (label, contribution) tuples — the demo
    shows the weight×feature number next to each reason.
    """
    # FEATURE_NAMES/feature_vector/ACTIVE_WEIGHTS live in service.py and
    # are REBINDABLE (the Rust scoring ext swaps them post-install), so
    # they are read through the module at CALL time — a module-level
    # import here would both cycle (service imports this module) and
    # freeze the pre-install pure-Python binding.
    from . import service as _svc
    contrib = sorted(
        ((weights.get(name, 0.0) * val, name)
         for name, val in zip(_svc.FEATURE_NAMES, vec)),
        reverse=True,
    )
    # Only distinctive features. An empty answer is honest — "nothing
    # set this apart" — never a fallback to features that fire on all
    # candidates anyway.
    picked = [
        (FEATURE_LABELS.get(name, name), score)
        for score, name in contrib
        if score > 0 and name not in _BORING_FEATURES
    ][:top]
    if with_weights:
        return picked
    return [label for label, _score in picked]


def explain_contributions(vec, weights, top=3):
    """Machine-readable WHY: {feature name: weight×value} for the same
    top non-boring picks explain_features labels — the numbers the
    label-only path computes and drops. Pure published-weights
    arithmetic: exposes nothing beyond weights.json."""
    from . import service as _svc
    contrib = sorted(
        ((weights.get(name, 0.0) * val, name)
         for name, val in zip(_svc.FEATURE_NAMES, vec)),
        reverse=True,
    )
    picked = [(name, score) for score, name in contrib
              if score > 0 and name not in _BORING_FEATURES][:top]
    return {name: round(score, 4) for name, score in picked}


async def seed_info_for(pool, seed: tuple):
    """Full info dict for one (id, media_type) seed (why rendering)."""
    try:
        info = await get_movie_info_batch(pool, [seed[0]])
        if info:
            row = dict(info[seed[0]])
            row.setdefault("media_type", seed[1])
            return row
    except Exception as e:
        logger.debug(f"seed info fetch failed for {seed}: {e}")
    return None


async def explain_results(pool, results: list, seed: tuple) -> dict:
    """{"why": labels, "why_scores": {feature: contribution}} for an
    already-computed result list against ONE (id, media_type[, w])
    seed."""
    empty: dict = {"why": {}, "why_scores": {}}
    if not results:
        return empty
    try:
        enriched = await enrich_candidates_batch(
            pool, [r["tmdb_id"] for r in results])
        top_seed = await seed_info_for(pool, seed)
        if not top_seed:
            return empty
    except Exception as e:
        logger.debug(f"explain enrichment failed for seed {seed}: {e}")
        return empty
    why: dict = {}
    why_scores: dict = {}
    # see explain_features: read the REBINDABLE service attributes at
    # call time (ACTIVE_WEIGHTS reload-sensitive, feature_vector may be
    # the Rust ext)
    from . import service as _svc
    for r in results:
        cand = dict(enriched.get(r["tmdb_id"], {}))
        cand.setdefault("id", r["tmdb_id"])
        for k, v in r.items():
            cand.setdefault(k, v)
        vec = _svc.feature_vector(cand, cand, cand.get("genres", []), top_seed)
        key = str(r["tmdb_id"])
        why[key] = explain_features(vec, _svc.ACTIVE_WEIGHTS)
        why_scores[key] = explain_contributions(vec, _svc.ACTIVE_WEIGHTS)
    return {"why": why, "why_scores": why_scores}
