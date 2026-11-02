"""§F — recommend_for_user: the personalized entry point.

The cold/warm
planning (recommendation_plan) and the user pipeline that delegates
to find_similar. Receives the service instance (`svc`) so pool and
find_similar stay single-sourced.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from loguru import logger

from . import db
from .config import USER_SEEDS

# --- user-recommendation plan (pure, spec: cold start rule) ---------------

COLD_START_MIN_TITLES = 3


def recommendation_plan(weighted_seeds, watchlist_seeds, min_titles=COLD_START_MIN_TITLES):
    """Decide HOW to recommend for a user (pure; spec §A4/§C4).

    Returns (mode, seeds):
    - "personalized": ≥ min_titles weighted titles — full persona path.
    - "watchlist":    no weighted history but explicit intent — seed by
                      watchlist (documented as weaker taste evidence).
    - "cold_start":   nothing usable — the CALLER must say so explicitly
                      (reason=cold_start), never silently serve a list.
    """
    if len(weighted_seeds) >= min_titles:
        return "personalized", weighted_seeds
    if weighted_seeds:
        # some history but not enough to claim a persona: blend what
        # exists with the watchlist, and say we're below the bar
        return "personalized", weighted_seeds + [
            w for w in watchlist_seeds
            if w[0] not in {s[0] for s in weighted_seeds}
        ]
    if watchlist_seeds:
        return "watchlist", watchlist_seeds
    return "cold_start", []




async def recommend_for_user(
    svc,
    user_id: int,
    limit: int = 30,
    randomness: float = 0.0,
    vector_space: Optional[str] = None,
    include_why: bool = False,
    explore: Optional[float] = None,
    user_context: Optional[dict] = None,
) -> dict:
    """§F end-to-end: user_id in, personalized list out.

    Returns a dict — the caller never assembles the pipeline:
      results:      ranked titles (score, media_type, ...)
      reason:       "personalized" | "watchlist" | "cold_start"
                    (cold_start ⇒ results is EMPTY by design —
                    never a silent blockbuster list)
      seeds:        the top-10 (tmdb_id, media_type, w_i) used
      vector_used:  whether the user-vector ANN channel is active —
                    a usable vector existed (a missing/stale one is
                    rebuilt inline first, bounded by
                    USER_VECTOR_MAX_AGE_HOURS, so the channel works
                    with no nightly cron). The ANN itself runs lazily:
                    cached lists skip the query, the flag still
                    reflects that the channel shaped the payload.
      why:          per-result top contributing features
                    (include_why=True; costs one extra pass)

    Saga rule (documented): a seed never returns its own saga —
    watched Rocky I blocks I AND recommends II via advancement.

    vector_space accepts a model key from model_spaces.REC_MODELS
    (e.g. "e5e"): the vector, the ANN recall, and the scorer's
    cosine all run in that space's embedding column for the whole
    request; unknown/absent → the default space.
    """
    # generate_user_vector_candidates is patched by tests on the SERVICE
    # namespace — resolve it through the module at call time
    from . import service as _svc
    from .model_spaces import column_for, normalize_model
    from .user_vector import ensure_user_vector, top_weighted_seeds

    # One model space for the whole request: the user vector is
    # built/ensured, ANN-recalled, and cosine-scored in the SAME
    # embedding column — a vector from one space searched against
    # another is a dimension mismatch, not a soft error.
    model_key = normalize_model(vector_space)

    # The three request inputs (seeds, watchlist, user vector) are
    # independent small reads — fetch them concurrently; wall time
    # is the max of the three instead of their sum. The vector's
    # failures stay local (None), matching the previous degrade.
    async def _uvec_or_none():
        try:
            return await ensure_user_vector(svc.pool, user_id, model_key)
        except Exception as e:
            logger.debug(f"user-vector recall skipped for {user_id}: {e}")
            return None

    weighted, watchlist_rows, uvec = await asyncio.gather(
        top_weighted_seeds(svc.pool, user_id, limit=USER_SEEDS),
        db.fetch(svc.pool,
            """SELECT f.tmdb_id, f.media_type FROM {t_user_feedback} f
               WHERE f.user_id = $1 AND f.kind = 'watchlist'
               ORDER BY f.created_at DESC LIMIT 10""",
            user_id,
        ),
        _uvec_or_none(),
    )
    watchlist = [(r["tmdb_id"], r["media_type"], 1.0) for r in watchlist_rows]
    mode, seeds = recommendation_plan(weighted, watchlist)

    if mode == "cold_start":
        logger.debug(f"recommend user={user_id}: cold_start (no signal)")
        return {
            "results": [], "reason": "cold_start",
            "seeds": [], "vector_used": False, "why": {},
        }

    extra_factory = None
    vector_used = False
    if uvec:
        # The ANN query runs lazily INSIDE find_similar: a cache
        # hit never needs the rows, so it is skipped entirely —
        # but the cache key still records that the channel was
        # armed (xc=f60), keeping hits stable across requests.
        uvec_column = column_for(model_key)

        async def _uvec_ann() -> List[dict]:
            return await _svc.generate_user_vector_candidates(
                svc.pool, uvec, limit=60,
                emb_column=uvec_column,
            )

        extra_factory = _uvec_ann
        vector_used = True

    # Thread the vector through the context: the tilt pass inside
    # find_similar reuses it instead of re-probing (a page does the
    # same via context["uvec"] across its rows).
    if user_context is None:
        user_context = {}
    if uvec:
        user_context["uvec"] = uvec

    results = await svc.find_similar(
        [(tmdb_id, mt) for tmdb_id, mt, _w in seeds],  # composite key!
        # (bare ids would mix movie 155 with tv 155 — the engine's
        # own invariant; see AGENTS.md)
        limit=limit,
        randomness=randomness,
        # An EMPTY set, not None: None skips the per-user pass inside
        # find_similar entirely, so watched/disliked titles outside
        # the seed set would surface in the list.
        exclude=set(),
        user_id=user_id,
        rec_model=model_key,
        explore=explore,
        user_context=user_context,
        extra_factory=extra_factory,
        extra_limit=60,
    )

    why: dict = {}
    why_scores: dict = {}
    if include_why and results:
        explained = await svc.explain_results(results, seeds[0])
        why = explained["why"]
        why_scores = explained["why_scores"]

    logger.debug(
        f"recommend user={user_id}: mode={mode} seeds="
        f"{[(i, round(w, 2)) for i, _m, w in seeds[:10]]} "
        f"excluded=watched|rated|disliked vector={vector_used} "
        f"results={len(results)}"
    )
    return {
        "results": results,
        "reason": mode,
        "seeds": [(i, m, round(w, 3)) for i, m, w in seeds[:10]],
        "vector_used": vector_used,
        "why": why,
        "why_scores": why_scores,
    }
