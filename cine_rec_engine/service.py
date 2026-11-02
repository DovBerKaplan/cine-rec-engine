"""Content-based movie & series recommendation engine.

All data comes from YOUR PostgreSQL database (schema: docs/schema.sql) —
no external API calls on the hot path. Optional Redis cache
(CINE_REC_REDIS_URL), with an in-process TTL cache fallback.

Usage:
    import asyncpg
    from cine_rec_engine import RecommendationService

    pool = await asyncpg.create_pool(dsn="postgresql://user:pw@localhost/mydb")
    rec = RecommendationService()
    await rec.initialize(pool)

    # Similar to one title:
    results = await rec.find_similar(119051, limit=10)

    # Multi-seed (movies + series the user liked):
    results = await rec.find_similar(
        tmdb_id=[155, 27205, 1396],  # Dark Knight, Inception, Breaking Bad
        limit=60,
    )
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

import asyncpg
from loguru import logger

from . import config as rec_config_module
from .config import (
    COLLECTION_MATCH_BONUS,
    COMPANY_SIMILARITY_WEIGHT,
    DIRECTOR_CHANNEL_BONUS,
    DIRECTOR_MATCH_BONUS,
    EMBEDDING_COLUMN,
    GENRE_PRIORITY,
    NETWORK_MATCH_BONUS,
    POP_ACTION_PENALTY,
    POP_ACTION_SIM_FLOOR,
    TMDB_REC_BONUS,
    WRITER_BONUS,
)
from .queries import (
    generate_user_vector_candidates,  # noqa: F401 — patch anchor: recommend.py resolves this through the service namespace
    )
from .scoring import (
    audience_score,
    canonical_genres,
    detect_style,
    bigram_set,
    keyword_set,
    tone_score,
)



# --- WHY rendering: moved to explain.py (5a call-flow split) — re-exported
# here so every pre-split import path keeps working (tests, demo, eval).
from .explain import (  # noqa: E402
    FEATURE_LABELS as FEATURE_LABELS,
    _BORING_FEATURES as _BORING_FEATURES,
    explain_contributions as explain_contributions,
    explain_features as explain_features,
)
from . import explain as _explain  # noqa: E402
from . import find_similar as _find_similar  # noqa: E402
from .find_similar import (  # noqa: E402,F401 — used by the py-scoring layer, page.py, bench_e2e and tests via service
    _get_cache as _get_cache,
    _modality as _modality,
    _story_key as _story_key,
    _dedup_same_story as _dedup_same_story,
    _mmr_order as _mmr_order,
    _balance_by_media_type as _balance_by_media_type,
    _apply_noise as _apply_noise,
    _pair_similarity as _pair_similarity,
)
from . import recommend as _recommend  # noqa: E402
from .recommend import (  # noqa: E402,F401 — re-export contract (tests import these from service)
    COLD_START_MIN_TITLES as COLD_START_MIN_TITLES,
    recommendation_plan as recommendation_plan,
)
from . import user_context as _user_context  # noqa: E402
from .user_context import (  # noqa: E402,F401 — re-export contract: tests import the OLD names from service
    _cosine as _cosine,
    exploration_slice as _exploration_slice,
    insert_exploration as _insert_exploration,
    tilt_scores as _tilt_scores,
)


class RecommendationService:
    """Generates movie/series recommendations from local PostgreSQL data.

    Follows the same singleton + lazy-init pattern as PostgresTMDBService
    in app/interactive_search/db_services.py.
    """

    CACHE_PREFIX = "rec:"
    CACHE_TTL = 3600  # 1 hour

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()

    async def initialize(self, pool: asyncpg.Pool) -> None:
        """Initialize with an asyncpg connection pool (lazy, thread-safe).

        Args:
            pool: asyncpg connection pool (reuse the same pool as
                  PostgresTMDBService to avoid extra connections).
        """
        if self.pool is not None:
            return

        async with self._init_lock:
            if self.pool is not None:
                return
            self.pool = pool
            try:
                max_size = pool.get_max_size()
            except Exception:
                max_size = None
            if max_size is not None and max_size < 20:
                # Recall fans out ~5 queries per seed concurrently; small
                # pools queue them and that queueing dominates cold-path
                # latency.
                logger.info(
                    "RecommendationService initialized with shared pool "
                    f"(max_size={max_size}; >=20 recommended for cold-path latency)"
                )
            else:
                logger.info("RecommendationService initialized with shared pool")

    async def _ensure_initialized(self) -> None:
        """Ensure the connection pool is set (thread-safe)."""
        if self.pool is None:
            raise RuntimeError(
                "RecommendationService not initialized. "
                "Call await recommendation_service.initialize(pool) first."
            )

    async def _seed_saga_blocked(self, seed_ids: set) -> Optional[set]:
        """The seeds plus their whole sagas (5a split): delegates to
        user_context.seed_saga_blocked."""
        return await _user_context.seed_saga_blocked(self.pool, seed_ids)

    async def _apply_user_filters(
        self,
        results: List[dict],
        exclude: Optional[set],
        seed_ids: Optional[set] = None,
        blocked: Optional[set] = None,
    ) -> List[dict]:
        """Watched filtering + saga advancement + seed guard (5a
        split): delegates to user_context.apply_user_filters."""
        return await _user_context.apply_user_filters(
            self.pool, results, exclude, seed_ids, blocked)

    async def _apply_user_tilt(
        self, results: List[dict], user_id: int, model_key: Optional[str],
        context: Optional[dict] = None,
    ) -> List[dict]:
        """Gentle taste tilt, boost-only (5a split): delegates to
        user_context.apply_user_tilt."""
        return await _user_context.apply_user_tilt(
            self.pool, results, user_id, model_key, context)

    async def _user_top_genres(self, user_id: int) -> set:
        """Top genre cluster names (5a split): delegates to
        user_context.user_top_genres."""
        return await _user_context.user_top_genres(self.pool, user_id)

    async def _apply_exploration(
        self, results: List[dict], user_id: int, movie_ratio: float,
        limit: int, share: float, tilt_ctx: dict,
    ) -> List[dict]:
        """Deterministic discovery slice (5a split): delegates to
        user_context.apply_exploration."""
        return await _user_context.apply_exploration(
            self.pool, results, user_id, movie_ratio, limit, share, tilt_ctx)

    async def _merged_exclusions(
        self, exclude: Optional[set], user_id: Optional[int]
    ) -> Optional[set]:
        """Explicit exclude ∪ user's watched/rated (5a split): delegates
        to user_context.merged_exclusions."""
        return await _user_context.merged_exclusions(self.pool, exclude, user_id)

    async def recommend_for_user(
        self,
        user_id: int,
        limit: int = 30,
        randomness: float = 0.0,
        vector_space: Optional[str] = None,
        include_why: bool = False,
        explore: Optional[float] = None,
        user_context: Optional[dict] = None,
    ) -> dict:
        """§F end-to-end: user_id in, personalized list out (5a split):
        delegates to recommend.recommend_for_user."""
        return await _recommend.recommend_for_user(
            self, user_id, limit=limit, randomness=randomness,
            vector_space=vector_space, include_why=include_why,
            explore=explore, user_context=user_context)

    async def _seed_info_for(self, seed: tuple) -> Optional[dict]:
        """Full info dict for one (id, media_type) seed (why rendering).
        Moved to explain.py in the 5a split; thin delegate."""
        return await _explain.seed_info_for(self.pool, seed)

    async def explain_results(self, results: list, seed: tuple) -> dict:
        """{"why": labels, "why_scores": {feature: contribution}} for an
        already-computed result list against ONE (id, media_type[, w])
        seed. Moved to explain.py in the 5a split; thin delegate."""
        return await _explain.explain_results(self.pool, results, seed)

    async def find_similar(
        self,
        tmdb_id,
        limit=None,
        randomness=0.0,
        allow_cross_media=False,
        seed_weights=None,
        media_type=None,
        rec_model=None,
        exclude=None,
        seed_vectors=None,
        seed_info_override=None,
        user_id=None,
        extra_candidates=None,
        extra_factory=None,
        extra_limit=60,
        filters=None,
        explore=None,
        user_context=None,
    ):
        """Find similar movies/series, ranked by score (5a split): the
        full signature and docstring live in find_similar.py; thin
        delegate."""
        return await _find_similar.find_similar(
            self, tmdb_id, limit=limit, randomness=randomness,
            allow_cross_media=allow_cross_media, seed_weights=seed_weights,
            media_type=media_type, rec_model=rec_model, exclude=exclude,
            seed_vectors=seed_vectors, seed_info_override=seed_info_override,
            user_id=user_id, extra_candidates=extra_candidates,
            extra_factory=extra_factory, extra_limit=extra_limit,
            filters=filters, explore=explore, user_context=user_context)


# ---------------------------------------------------------------------------
# Learning-to-Rank feature space
#
# `_score_candidate_vs_seed` is a weighted dot product over `feature_vector`:
# every signal the engine uses is one coordinate with a RAW value (no baked
# weights), and HEURISTIC_WEIGHTS holds the hand-tuned coefficients so the
# heuristic path scores bit-identically to the pre-refactor engine. A learned
# linear model (scripts/eval/train_ltr.py) trains on difference vectors of
# the SAME function and lands in learned_weights.json — REC_SCORER=learned
# swaps the coefficients with zero train/serve skew.
# ---------------------------------------------------------------------------


FEATURE_NAMES = [
    "cosine_sim",  # pgvector overview cosine (legacy bigram fallback)
    "keyword_sim",  # Jaccard over keywords
    "cast_sim",  # Dice over top-5 cast person_ids
    "genre_priority_sum",  # sum of GENRE_PRIORITY over common canonical genres
    "director_match",  # person-id (or name-string fallback) director match
    "director_channel",  # gated auteur-channel value: quality fade x franchise mod (0..1)
    "writer_match",  # shared screenwriter person_ids
    "tmdb_rec_decay",  # behavioral rank-decay curve (tmdb@1=1.0 .. floor 0.25)
    "shared_collection",  # same saga/collection id
    "shared_network",  # same TV network
    "style_match",  # shared detected style (noir, cyberpunk, ...)
    "company_sim",  # Jaccard over production companies
    "year_le5",  # released within 5 years of the seed
    "year_le10",  # within 10 years
    "year_gt25",  # generations apart
    "votes_gt15k",  # mega-hit connectivity penalty flag
    "low_votes_high_rating",  # little-seen gem flag
    "rating_bonus",  # min(max(rating-6,0),2)*0.5
    "pop_action_leak",  # votes>10k AND semantic sim below floor
    "composer_match",  # shared Original Music Composer person_id
    "dp_match",  # shared Director of Photography person_id
    "medium_mismatch",  # seed and candidate live in DIFFERENT mediums
    # ---- reserved: cinematic-tag features (see weights.json note) ----
    # These six are NOT covered by the published weights.json — the
    # heuristic baseline serves them. Their tagger inputs come from the
    # private tmdb_cinematic enrichment: narrative/pacing/emotional_arc
    # are always 0 in public builds (no tag data); tone/audience
    # degrade to genre-only behavior; heuristic_score is a residual
    # tree input with self-weight 0. Kept (not pruned): the names are
    # zero-cost and removing them breaks golden-vector parity.
    "narrative_match",  # same narrative structure from tmdb_cinematic
    # Mood/tone features (a teen musical
    # next to a dark fantasy: keywords overlap while the vibe inverts).
    "tone_compatibility",  # 1.0 same register (light↔light), 0.0 opposite
    "emotional_arc_match",  # Jaccard over tmdb_cinematic emotional_arc tags
    "pacing_match",  # 1.0 same pacing (contemplative/propulsive/...)
    # Audience: youth seed (Kids/Family/Soap/Animation) vs adult candidate.
    "audience_compatibility",  # 1.0 same audience, 0.0 youth↔adult clash
    # The heuristic's OWN score over the features above — residual
    # learning input for the LambdaRank trees (they learn corrections on
    # top of the proven baseline instead of starting from zero). Its
    # HEURISTIC_WEIGHTS entry is 0.0 so the engine's dot product is
    # mathematically unchanged.
    "heuristic_score",
]

HEURISTIC_WEIGHTS = {
    # Space-aware: compressed embedding spaces need a larger cosine voice
    # in the blend to contribute equally.
    "cosine_sim": 26.0 if EMBEDDING_COLUMN == "embedding_v4" else 10.0,
    "keyword_sim": 3.0,
    "cast_sim": 4.0,
    "genre_priority_sum": 0.5,
    "director_match": DIRECTOR_MATCH_BONUS,
    "director_channel": DIRECTOR_CHANNEL_BONUS,
    "writer_match": WRITER_BONUS,
    "tmdb_rec_decay": TMDB_REC_BONUS,
    "shared_collection": COLLECTION_MATCH_BONUS * 2,
    "shared_network": NETWORK_MATCH_BONUS,
    "style_match": 3.0,
    "company_sim": COMPANY_SIMILARITY_WEIGHT,
    "year_le5": 1.5,
    "year_le10": 0.5,
    "year_gt25": -1.0,
    "votes_gt15k": -1.0,
    "low_votes_high_rating": 1.5,
    "rating_bonus": 1.0,
    "pop_action_leak": POP_ACTION_PENALTY,
    # Cinematic-DNA features: LTR inputs, weight 0.0 in the heuristic
    # baseline.
    "composer_match": 0.0,
    "dp_match": 0.0,
    # The medium signal must be LEARNED, not imposed: taste genuinely
    # crosses media, so a global medium rule can contradict real
    # preferences. The feature stays for the LTR to learn WHEN medium
    # matters.
    "medium_mismatch": -2.0,
    # Narrative-structure agreement (linear vs anthology): +1 same
    # structure, -1 clash.
    "narrative_match": 1.5,
    # Mood/tone: prevents light↔dark mismatches.
    "tone_compatibility": 5.0,
    # Emotional arc tag overlap (feel-good, tense, slow-burn...).
    # Only fires when both titles have tmdb_cinematic enrichment.
    "emotional_arc_match": 4.0,
    # Same pacing (contemplative vs propulsive vs frenetic).
    "pacing_match": 2.0,
    # Youth audience match: prevents teen telenovela ← adult prestige
    # drama.
    "audience_compatibility": 4.0,
    # Self-weight 0.0 keeps the engine's dot product identical to the
    # pre-refactor scoring (the feature is INPUT for trees, not signal).
    "heuristic_score": 0.0,
}


# Same-story dedup: keep one representative per collection / normalized
# title so a remake or re-series can waste only one list slot.


def _load_ensemble_config() -> None:
    """Apply a saved ensemble profile when REC_ENSEMBLE_CONFIG points at
    one (per-model weights + heuristic knobs).

    Pattern: a production-decision config file; sets the multi-column
    cosine blend (config.COSINE_BLEND) and the heuristic weights. Any
    failure leaves the defaults untouched — a corrupt artifact must
    never break the engine.
    """
    path = os.getenv("REC_ENSEMBLE_CONFIG", "")
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        w = cfg["weights"]
        HEURISTIC_WEIGHTS["cosine_sim"] = float(w["total_cosine"])
        HEURISTIC_WEIGHTS["tmdb_rec_decay"] = float(w["tmdb_rec_bonus"])
        HEURISTIC_WEIGHTS["shared_collection"] = float(w["collection_bonus"]) * 2
        HEURISTIC_WEIGHTS["company_sim"] = float(w["company_weight"])
        HEURISTIC_WEIGHTS["pop_action_leak"] = float(w["pop_action_penalty"])
        rec_config_module.COSINE_BLEND = {
            "embedding_mpnetae": float(w.get("w_mpnetae", 1.0)),
            "embedding_e5e": float(w.get("w_e5e", 0.0)),
            "embedding_mpnetan": float(w.get("w_mpnetan", 0.0)),
        }
        logger.info(
            f"Ensemble config applied from {path}: blend={rec_config_module.COSINE_BLEND} "
            f"cosine={HEURISTIC_WEIGHTS['cosine_sim']}"
        )
    except Exception as e:  # noqa: BLE001 — fail-safe by contract
        logger.warning(f"REC_ENSEMBLE_CONFIG load failed ({e}); using defaults")


_load_ensemble_config()


def _load_learned_weights() -> Optional[dict]:
    """Load learned LTR coefficients when CINE_REC_SCORER=learned.

    Artifact selection: CINE_REC_WEIGHTS=/path/artifact.json loads a
    user-fitted artifact (same schema, e.g. from eval/tune_weights.py);
    unset loads the bundled weights.json.

    Partial application: the artifact may cover a subset of FEATURE_NAMES
    (the published weights.json predates the mood features). Learned
    values override the heuristic baseline feature-by-feature; features
    the artifact lacks keep their heuristic coefficients. An artifact
    covering nothing (or a corrupt file) returns None — the engine then
    stays fully on the heuristic weights (fail-safe: a bad artifact must
    never break recommendations; a custom path fails with a LOUD warning
    so an owner-fitted deployment doesn't silently run heuristic)."""
    global SCORER_SOURCE
    if os.getenv("CINE_REC_SCORER", os.getenv("REC_SCORER", "heuristic")) != "learned":
        return None
    custom = os.getenv("CINE_REC_WEIGHTS", "").strip()
    try:
        import json
        from pathlib import Path

        if custom:
            data = json.loads(Path(custom).read_text())
        else:
            data = json.loads(
                (Path(__file__).parent / "weights.json").read_text())
        weights = {k: float(v) for k, v in data["weights"].items()
                   if k in FEATURE_NAMES}
        if not weights:
            logger.warning("learned weights cover no known features — using heuristic")
            return None
        blended = dict(HEURISTIC_WEIGHTS)
        blended.update(weights)
        missing = [n for n in FEATURE_NAMES if n not in weights]
        if missing:
            logger.info(
                f"learned weights cover {len(weights)}/{len(FEATURE_NAMES)} "
                f"features; heuristic fills: {missing}"
            )
        blend = data.get("cosine_blend")
        if custom and isinstance(blend, dict) and blend:
            cleaned = {str(k): float(v) for k, v in blend.items()
                       if float(v) > 0}
            if cleaned:
                rec_config_module.COSINE_BLEND = cleaned
                logger.info(f"cosine blend applied from artifact: {cleaned}")
        SCORER_SOURCE = (f"learned (custom: {custom})" if custom
                         else "learned (bundled weights.json)")
        return blended
    except Exception as e:
        if custom:
            logger.warning(
                f"CINE_REC_WEIGHTS load failed from {custom!r} ({e}) — "
                "falling back to heuristic weights")
        else:
            logger.warning(f"learned weights unavailable ({e}) — using heuristic")
        return None


SCORER_SOURCE = "heuristic"


HEURISTIC_WEIGHTS_LIST = list(HEURISTIC_WEIGHTS.values())[:27]

LEARNED_WEIGHTS = _load_learned_weights()
ACTIVE_WEIGHTS = LEARNED_WEIGHTS or HEURISTIC_WEIGHTS
HEURISTIC_ACTIVE_LIST = list(ACTIVE_WEIGHTS.values())


def _py_entity_profile(entity: dict, entity_data: Optional[dict],
                    genres_override: Optional[list] = None) -> dict:
    """Derive-once view of one side of a (seed, candidate) pair.

    The scoring loop evaluates candidates × seeds; everything that depends
    on ONE side only (canonical genres, keyword/cast/company sets, style,
    modality, overview bigrams, scalars) is computed once here instead of
    once per pair. Pure caching — semantics identical to inline calls.
    """
    data = entity_data if entity_data is not None else {}
    # Field ownership mirrors feature_vector exactly: the ENRICHED side
    # (keywords/cast/companies/networks/style/crew ids) lives on cand_data;
    # recall rows (entity) only carry genres/overview/scalars. Reading these
    # from the entity left every enriched feature silently zero.
    source = data if entity_data is not None else entity
    genres = entity.get("genres") or data.get("genres") or []
    param_genres = genres_override if genres_override is not None else genres
    overview = entity.get("overview_en", "") or entity.get("overview", "") or ""
    return {
        # sets
        "dir_ids": set(source.get("director_ids") or []),
        "writer_ids": set(source.get("writer_ids") or []),
        "composer_ids": set(source.get("composer_ids") or []),
        "dp_ids": set(source.get("dp_ids") or []),
        "cast5": set(
            (source.get("cast_ids") or source.get("cast_list") or [])[:5]
        ),
        "keywords": keyword_set(source.get("keywords", []) or []),
        "companies": set(source.get("companies", []) or []),
        "networks": set(source.get("networks", []) or []),
        "can_genres": canonical_genres(param_genres),
        "can_genres_raw": set(param_genres),
        "param_genres": param_genres,
        "style": set(detect_style(source.get("keywords", []) or [])),
        # scalars / strings
        "genres": genres,
        "modality": _modality(
            param_genres,
            source.get("original_language") or entity.get("original_language"),
        ),
        "director": source.get("director"),
        "coll": source.get("collection_id"),
        "overview_bigrams": bigram_set(overview) if overview else None,
        "overview": overview,
        "year": entity.get("release_year"),
        "votes": entity.get("vote_count", 0) or 0,
        "rating": float(entity.get("vote_average") or 0.0),
        "narrative": data.get("narrative") if data else entity.get("narrative"),
        "pacing": data.get("pacing") if data else entity.get("pacing"),
        "emotional_arc": data.get("emotional_arc") if data else entity.get("emotional_arc"),
        "arc_set": {
            t.lower().strip()
            for t in ((data.get("emotional_arc") if data else entity.get("emotional_arc")) or [])
        },
        "adult": bool(entity.get("adult")),
        "via": entity.get("via"),
        # cand-side precomputed pair inputs
        "cand_coll": source.get("collection_id"),
        "tmdb_rec_decay": (
            max(0.25, 1.0 - (int(entity.get("tmdb_rank") or 1) - 1) * 0.15)
            if entity.get("via") == "tmdb" else 0.0
        ),
    }


def _py_vec_from_profiles(sp: dict, cp: dict, cand: dict, cand_data: dict,
                       overview_sim: float) -> list:
    """feature_vector over precomputed profiles — same 28-entry output."""
    # director_match (name fallback mirrors feature_vector)
    director_match = float(
        bool(
            (sp["dir_ids"] and cp["dir_ids"] and sp["dir_ids"] & cp["dir_ids"])
            or (sp["director"] and sp["director"] == cand_data.get("director"))
        )
    )

    cand_coll = cand_data.get("collection_id") or cand.get("collection_id")
    shared_collection = float(bool(sp["coll"] and cand_coll and sp["coll"] == cand_coll))

    director_channel = 0.0
    if cand.get("via") == "director":
        if cp["rating"] >= 7.0:
            director_channel = 1.0
        else:
            director_channel = max(0.3, (cp["rating"] - 5.5) / 1.5)
        if shared_collection:
            director_channel *= 0.5

    writer_match = float(
        bool(
            (sp["writer_ids"] and cp["writer_ids"] and sp["writer_ids"] & cp["writer_ids"])
            or cand.get("via") == "writer"
        )
    )

    medium_mismatch = float(sp["modality"] != cp["modality"])

    composer_match = float(bool(sp["composer_ids"] & cp["composer_ids"]))
    dp_match = float(bool(sp["dp_ids"] & cp["dp_ids"]))

    tmdb_rec_decay = 0.0
    if cand.get("via") == "tmdb":
        tmdb_rank = int(cand.get("tmdb_rank") or 1)
        tmdb_rec_decay = max(0.25, 1.0 - (tmdb_rank - 1) * 0.15)

    if overview_sim > 0.0:
        cosine_sim = overview_sim
    elif sp["overview_bigrams"] and cp["overview_bigrams"]:
        inter = len(sp["overview_bigrams"] & cp["overview_bigrams"])
        union = len(sp["overview_bigrams"] | cp["overview_bigrams"])
        cosine_sim = inter / union if union else 0.0
    else:
        cosine_sim = 0.0

    skw, ckw = sp["keywords"], cp["keywords"]
    if skw and ckw:
        union = len(skw | ckw)
        keyword_sim = len(skw & ckw) / union if union else 0.0
    else:
        keyword_sim = 0.0

    sc, cc = sp["cast5"], cp["cast5"]
    total_cast = len(sc) + len(cc)
    cast_sim = (2.0 * len(sc & cc)) / total_cast if total_cast else 0.0

    genre_priority_sum = sum(
        (GENRE_PRIORITY.get(g, 1.0) for g in sorted(sp["can_genres"] & cp["can_genres"]))
    )

    style_match = float(bool(sp["style"] & cp["style"]))

    spc, cpc = sp["companies"], cp["companies"]
    if spc and cpc:
        union = len(spc | cpc)
        company_sim = len(spc & cpc) / union if union else 0.0
    else:
        company_sim = 0.0

    shared_network = float(bool(sp["networks"] and cp["networks"]
                               and sp["networks"] & cp["networks"]))

    year_le5 = year_le10 = year_gt25 = 0.0
    if sp["year"] and cp["year"]:
        year_diff = abs(int(sp["year"]) - int(cp["year"]))
        if year_diff <= 5:
            year_le5 = 1.0
        elif year_diff <= 10:
            year_le10 = 1.0
        elif year_diff > 25:
            year_gt25 = 1.0

    votes_gt15k = float(cp["votes"] > 15000)
    low_votes_high_rating = float(cp["votes"] < 2000 and cp["rating"] > 7.0)
    rating_bonus = min(max(cp["rating"] - 6.0, 0.0), 2.0) * 0.5
    pop_action_leak = float(cp["votes"] > 10000 and cosine_sim < POP_ACTION_SIM_FLOOR)

    narrative_match = 0.0
    seed_narr, cand_narr = sp["narrative"], cp["narrative"]
    if seed_narr and cand_narr:
        if seed_narr == cand_narr:
            narrative_match = 1.0
        elif {seed_narr, cand_narr} == {"linear", "anthology"}:
            narrative_match = -1.0

    tone_compat = tone_score(
        sp["genres"], cp["genres"],
        sp["emotional_arc"] or None, cp["emotional_arc"] or None,
        seed_canonical=sp["can_genres"], cand_canonical=cp["can_genres"],
    )
    sa, ca = sp["arc_set"], cp["arc_set"]
    arc_match = len(sa & ca) / len(sa | ca) if (sa and ca) else 0.0
    pacing_match = 1.0 if (sp["pacing"] and sp["pacing"] == cp["pacing"]) else 0.0

    audience_compat = audience_score(
        sp["genres"], cp["genres"],
        seed_adult=sp["adult"], cand_adult=cp["adult"],
        seed_canonical=sp["can_genres"], cand_canonical=cp["can_genres"],
    )

    vec = [
        cosine_sim,
        keyword_sim,
        cast_sim,
        genre_priority_sum,
        director_match,
        director_channel,
        writer_match,
        tmdb_rec_decay,
        shared_collection,
        shared_network,
        style_match,
        company_sim,
        year_le5,
        year_le10,
        year_gt25,
        votes_gt15k,
        low_votes_high_rating,
        rating_bonus,
        pop_action_leak,
        composer_match,
        dp_match,
        medium_mismatch,
        narrative_match,
        tone_compat,
        arc_match,
        pacing_match,
        audience_compat,
    ]
    heuristic_score = sum(w * v for w, v in zip(HEURISTIC_WEIGHTS_LIST, vec))
    vec.append(heuristic_score)
    return vec


def _py_feature_vector(
    cand: dict,
    cand_data: dict,
    cand_genres: list,
    seed: dict,
    overview_sim: float = 0.0,
) -> list:
    """Raw (unweighted) feature vector of (seed, candidate).

    The SAME function feeds the heuristic engine, the LTR trainer, and
    the learned scorer — the single source of truth for the feature
    space. Delegates to the profile-cached fast path (identical output;
    profiles are built per call here, once per entity in hot loops).
    """
    sp = _py_entity_profile(seed, None)
    cp = _py_entity_profile(cand, cand_data, genres_override=cand_genres)
    return _py_vec_from_profiles(sp, cp, cand, cand_data, overview_sim)



def _py_score_pair_fast(sp: dict, cp: dict, cand: dict, cand_data: dict,
                       cand_genres_set: Optional[set], overview_sim: float,
                       w_list: list) -> float:
    """_score_candidate_vs_seed over precomputed profiles + weight list.

    Identical semantics (genre gate, via exemptions, dot product) with a
    single accumulation pass — no intermediate feature list. Locked to
    _score_candidate_vs_seed by tests/test_golden_parity.py.
    """
    if (
        sp["genres"]
        and cand_genres_set
        and not cp["via"]
        and not (sp["can_genres_raw"] & cand_genres_set)
    ):
        return 0.0

    w = w_list  # feature order == FEATURE_NAMES (locked by tests)

    director_match = (
        (sp["dir_ids"] and cp["dir_ids"] and sp["dir_ids"] & cp["dir_ids"])
        or (sp["director"] and sp["director"] == cand_data.get("director"))
    )
    shared_collection = bool(sp["coll"] and cp["cand_coll"] and sp["coll"] == cp["cand_coll"])

    director_channel = 0.0
    if cp["via"] == "director":
        if cp["rating"] >= 7.0:
            director_channel = 1.0
        else:
            director_channel = max(0.3, (cp["rating"] - 5.5) / 1.5)
        if shared_collection:
            director_channel *= 0.5

    writer_match = float(bool(
        (sp["writer_ids"] and cp["writer_ids"] and sp["writer_ids"] & cp["writer_ids"])
        or cp["via"] == "writer"
    ))

    medium_mismatch = float(sp["modality"] != cp["modality"])
    composer_match = float(bool(sp["composer_ids"] & cp["composer_ids"]))
    dp_match = float(bool(sp["dp_ids"] & cp["dp_ids"]))

    if overview_sim > 0.0:
        cosine_sim = overview_sim
    elif sp["overview_bigrams"] and cp["overview_bigrams"]:
        inter = len(sp["overview_bigrams"] & cp["overview_bigrams"])
        union = len(sp["overview_bigrams"] | cp["overview_bigrams"])
        cosine_sim = inter / union if union else 0.0
    else:
        cosine_sim = 0.0

    skw, ckw = sp["keywords"], cp["keywords"]
    if skw and ckw:
        union = len(skw | ckw)
        keyword_sim = len(skw & ckw) / union if union else 0.0
    else:
        keyword_sim = 0.0

    sc, cc = sp["cast5"], cp["cast5"]
    total_cast = len(sc) + len(cc)
    cast_sim = (2.0 * len(sc & cc)) / total_cast if total_cast else 0.0

    genre_priority_sum = sum(
        (GENRE_PRIORITY.get(g, 1.0) for g in sorted(sp["can_genres"] & cp["can_genres"]))
    )
    style_match = float(bool(sp["style"] & cp["style"]))

    spc, cpc = sp["companies"], cp["companies"]
    if spc and cpc:
        union = len(spc | cpc)
        company_sim = len(spc & cpc) / union if union else 0.0
    else:
        company_sim = 0.0

    shared_network = float(bool(sp["networks"] and cp["networks"]
                               and sp["networks"] & cp["networks"]))

    year_le5 = year_le10 = year_gt25 = 0.0
    if sp["year"] and cp["year"]:
        year_diff = abs(int(sp["year"]) - int(cp["year"]))
        if year_diff <= 5:
            year_le5 = 1.0
        elif year_diff <= 10:
            year_le10 = 1.0
        elif year_diff > 25:
            year_gt25 = 1.0

    votes_gt15k = float(cp["votes"] > 15000)
    low_votes_high_rating = float(cp["votes"] < 2000 and cp["rating"] > 7.0)
    rating_bonus = min(max(cp["rating"] - 6.0, 0.0), 2.0) * 0.5
    pop_action_leak = float(cp["votes"] > 10000 and cosine_sim < POP_ACTION_SIM_FLOOR)

    narrative_match = 0.0
    seed_narr, cand_narr = sp["narrative"], cp["narrative"]
    if seed_narr and cand_narr:
        if seed_narr == cand_narr:
            narrative_match = 1.0
        elif {seed_narr, cand_narr} == {"linear", "anthology"}:
            narrative_match = -1.0

    tone_compat = tone_score(
        sp["genres"], cp["genres"],
        sp["emotional_arc"] or None, cp["emotional_arc"] or None,
        seed_canonical=sp["can_genres"], cand_canonical=cp["can_genres"],
    )
    sa, ca = sp["arc_set"], cp["arc_set"]
    arc_match = len(sa & ca) / len(sa | ca) if (sa and ca) else 0.0
    pacing_match = 1.0 if (sp["pacing"] and sp["pacing"] == cp["pacing"]) else 0.0

    audience_compat = audience_score(
        sp["genres"], cp["genres"],
        seed_adult=sp["adult"], cand_adult=cp["adult"],
        seed_canonical=sp["can_genres"], cand_canonical=cp["can_genres"],
    )

    score = (
        w[0] * cosine_sim
        + w[1] * keyword_sim
        + w[2] * cast_sim
        + w[3] * genre_priority_sum
        + w[4] * float(bool(director_match))
        + w[5] * director_channel
        + w[6] * writer_match
        + w[7] * cp["tmdb_rec_decay"]
        + w[8] * float(shared_collection)
        + w[9] * shared_network
        + w[10] * style_match
        + w[11] * company_sim
        + w[12] * year_le5
        + w[13] * year_le10
        + w[14] * year_gt25
        + w[15] * votes_gt15k
        + w[16] * low_votes_high_rating
        + w[17] * rating_bonus
        + w[18] * pop_action_leak
        + w[19] * composer_match
        + w[20] * dp_match
        + w[21] * medium_mismatch
        + w[22] * narrative_match
        + w[23] * tone_compat
        + w[24] * arc_match
        + w[25] * pacing_match
        + w[26] * audience_compat
    )
    if w[27]:
        # residual heuristic score (LTR input; weight 0.0 in the baseline)
        score += w[27] * (
            HEURISTIC_WEIGHTS_LIST[0] * cosine_sim
            + HEURISTIC_WEIGHTS_LIST[1] * keyword_sim
            + HEURISTIC_WEIGHTS_LIST[2] * cast_sim
            + HEURISTIC_WEIGHTS_LIST[3] * genre_priority_sum
            + HEURISTIC_WEIGHTS_LIST[4] * float(bool(director_match))
            + HEURISTIC_WEIGHTS_LIST[5] * director_channel
            + HEURISTIC_WEIGHTS_LIST[6] * writer_match
            + HEURISTIC_WEIGHTS_LIST[7] * cp["tmdb_rec_decay"]
            + HEURISTIC_WEIGHTS_LIST[8] * float(shared_collection)
            + HEURISTIC_WEIGHTS_LIST[9] * shared_network
            + HEURISTIC_WEIGHTS_LIST[10] * style_match
            + HEURISTIC_WEIGHTS_LIST[11] * company_sim
            + HEURISTIC_WEIGHTS_LIST[12] * year_le5
            + HEURISTIC_WEIGHTS_LIST[13] * year_le10
            + HEURISTIC_WEIGHTS_LIST[14] * year_gt25
            + HEURISTIC_WEIGHTS_LIST[15] * votes_gt15k
            + HEURISTIC_WEIGHTS_LIST[16] * low_votes_high_rating
            + HEURISTIC_WEIGHTS_LIST[17] * rating_bonus
            + HEURISTIC_WEIGHTS_LIST[18] * pop_action_leak
            + HEURISTIC_WEIGHTS_LIST[19] * composer_match
            + HEURISTIC_WEIGHTS_LIST[20] * dp_match
            + HEURISTIC_WEIGHTS_LIST[21] * medium_mismatch
            + HEURISTIC_WEIGHTS_LIST[22] * narrative_match
            + HEURISTIC_WEIGHTS_LIST[23] * tone_compat
            + HEURISTIC_WEIGHTS_LIST[24] * arc_match
            + HEURISTIC_WEIGHTS_LIST[25] * pacing_match
            + HEURISTIC_WEIGHTS_LIST[26] * audience_compat
        )
    return score


def _score_candidate_vs_seed(
    cand: dict,
    cand_data: dict,
    cand_genres: list,
    seed: dict,
    overview_sim: float = 0.0,
    weights: Optional[dict] = None,
) -> float:
    """Score a candidate against a single seed: w · φ(seed, candidate).

    Returns 0.0 if the candidate fails the genre gate (classic-recall
    candidates must share a genre; KNN/TMDB/auteur recalls are exempt —
    their proximity signal is the qualification). Weights come from
    HEURISTIC_WEIGHTS (default) or learned LTR coefficients
    (REC_SCORER=learned + learned_weights.json). `weights` is a
    per-request override (a solo /set_model model re-prices cosine_sim);
    the residual heuristic_score inside the feature vector always uses
    the global coefficients — it is an LTR input, not the score.
    """
    base_genres = seed.get("genres", [])
    if (
        base_genres
        and cand_genres
        and not cand.get("via")
        and not (set(base_genres) & set(cand_genres))
    ):
        return 0.0

    vec = feature_vector(cand, cand_data, cand_genres, seed, overview_sim)
    w_map = weights if weights is not None else ACTIVE_WEIGHTS
    return sum(w * v for w, v in zip(w_map.values(), vec))


# ── Optional Rust acceleration (rust-port phase 2) ─────────────────
# Public dispatchers: identical to the pure implementations until the
# rebind below. Everything hot calls THESE names.
_entity_profile = _py_entity_profile
_score_pair_fast = _py_score_pair_fast
feature_vector = _py_feature_vector

# The _py_* implementations above are the SOURCE OF TRUTH; when the
# optional compiled extension (`cine_rec_engine._scoring_rs`) is
# importable the three dispatcher names rebind to it — output-identical
# by the golden parity contract (rust/scoring harness, both golden
# sets). CINE_REC_SCORING_RS=0 forces the pure-Python path. A missing
# or broken extension must never break the engine: degrade, never fail.


def _install_rust_scoring() -> str:
    flag = os.getenv("CINE_REC_SCORING_RS", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return "disabled (CINE_REC_SCORING_RS=0)"
    try:
        from . import _scoring_accel
        return _scoring_accel.install(globals())
    except Exception as e:  # noqa: BLE001 — fail-safe by contract
        logger.info(f"rust scoring not active ({type(e).__name__}: {e})")
        return "python-fallback"


SCORING_RS = _install_rust_scoring()


