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
import random
from typing import Dict, List, Optional, Tuple, Union

import asyncpg
from loguru import logger

from . import config as rec_config_module
from .config import (
    AUTEUR_DECAY_FACTORS,
    COLLECTION_MATCH_BONUS,
    COMPANY_SIMILARITY_WEIGHT,
    DIRECTOR_CHANNEL_BONUS,
    DIRECTOR_MATCH_BONUS,
    EMBEDDING_COLUMN,
    GENRE_MISMATCH_AUTEUR_FACTOR,
    GENRE_PRIORITY,
    KNN_CANDIDATES_PER_SEED,
    MMR_ENABLED,
    MMR_LAMBDA,
    NETWORK_MATCH_BONUS,
    POP_ACTION_PENALTY,
    POP_ACTION_SIM_FLOOR,
    TMDB_REC_BONUS,
    WRITER_BONUS,
)
from .queries import (
    enrich_candidates_batch,
    generate_candidates,
    generate_director_candidates,
    generate_knn_candidates,
    generate_knn_finetuned_candidates,
    generate_tmdb_rec_candidates,
    generate_user_vector_candidates,
    get_movie_info_batch,
)
from .scoring import (
    audience_score,
    canonical_genres,
    detect_style,
    bigram_set,
    keyword_set,
    optimize_candidate_selection,
    semantic_overview_similarity,
    tone_score,
)
from .tmdb_recs import ensure_seeds_synced
from .model_spaces import REC_MODELS, normalize_model



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
    contrib = sorted(
        ((weights.get(name, 0.0) * val, name)
         for name, val in zip(FEATURE_NAMES, vec)),
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
            logger.info("RecommendationService initialized with shared pool")

    async def _ensure_initialized(self) -> None:
        """Ensure the connection pool is set (thread-safe)."""
        if self.pool is None:
            raise RuntimeError(
                "RecommendationService not initialized. "
                "Call await recommendation_service.initialize(pool) first."
            )

    async def _apply_user_filters(
        self,
        results: List[dict],
        exclude: Optional[set],
        seed_ids: Optional[set] = None,
    ) -> List[dict]:
        """Per-user post-processing: watched filtering + saga advancement
        + final seed guard.

        One merged pass (the order matters — a naive filter-then-collapse
        would DELETE a saga slot whose cached member the user watched,
        instead of advancing it):

        - non-saga item the user watched → dropped, slot refilled later
          by the media balancing.
        - saga slot (movie with collection_id) → replaced by the
          collection's FIRST UNWATCHED member: the series start for a
          fresh viewer, the next installment for someone who saw the
          start (saw #1 → #2, saw #1+#2 → #3...). Whole saga watched →
          the slot drops. The slot keeps its score.
        - the SEED titles (what the user asked recommendations FOR) are
          persona non grata TOGETHER WITH THEIR WHOLE SAGA: asking for
          Harry Potter 1 never yields HP2/HP3 — recommending the seed's
          own sequel is trivial, not discovery. Other sagas keep their
          advancement. Bare seed ids are dropped too (the recall excludes
          them, but the saga replacement re-introduces them).
        - swap collisions between slots are deduped.

        Failures (members fetch) degrade to plain watched-filtering.
        """
        watched = exclude or set()
        seeds = seed_ids or set()

        # The seeds' own sagas are off-limits entirely: every member of
        # any collection the seeds belong to is blocked (movies only).
        blocked: set = set(seeds)
        if seeds:
            try:
                rows = await self.pool.fetch(
                    """
                    SELECT DISTINCT m2.id
                    FROM tmdb_media m2
                    JOIN tmdb_media s ON s.collection_id = m2.collection_id
                    WHERE s.id = ANY($1::bigint[])
                      AND s.collection_id > 0
                      AND m2.media_type = 'movie'
                    """,
                    sorted(seeds),
                )
                blocked |= {row["id"] for row in rows}
            except Exception as e:
                logger.debug(f"seed-saga block fetch failed: {e}")

        # Collect saga ids first; plain filter when there are none.
        coll_ids = sorted(
            {
                r["collection_id"]
                for r in results
                if r.get("media_type") == "movie" and r.get("collection_id")
            }
        )
        members: dict[int, list[dict]] = {}
        if coll_ids:
            try:
                rows = await self.pool.fetch(
                    """
                    SELECT m.collection_id, m.id, m.title, m.title_en,
                           m.vote_average::float AS vote_average, m.poster_path,
                           EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int
                               AS release_year
                    FROM tmdb_media m
                    WHERE m.collection_id = ANY($1::int[]) AND m.media_type = 'movie'
                    ORDER BY m.collection_id,
                             COALESCE(m.release_date, m.first_air_date) ASC NULLS LAST,
                             m.popularity DESC
                    """,
                    coll_ids,
                )
                for row in rows:
                    members.setdefault(row["collection_id"], []).append(dict(row))
            except Exception as e:
                logger.debug(f"saga members fetch failed: {e}")

        out: List[dict] = []
        seen_ids: set = set()
        for r in results:
            # Final guard — the seeds and their whole sagas never return
            # (bare id, matching the recall-level seed exclusion).
            if r["tmdb_id"] in blocked:
                continue
            saga = members.get(r.get("collection_id") or 0) if members else None
            if saga:
                target = next(
                    (
                        m
                        for m in saga
                        if m["id"] not in blocked and (m["id"], "movie") not in watched
                    ),
                    None,
                )
                if target is None:
                    continue  # the user watched (or seeded) the entire saga
                if target["id"] != r["tmdb_id"]:
                    r = {
                        **r,
                        "tmdb_id": target["id"],
                        "title": target["title"],
                        "title_en": target["title_en"],
                        "rating": target["vote_average"] or 0,
                        "poster_path": target["poster_path"],
                        "release_year": target["release_year"],
                    }
            elif (r["tmdb_id"], r.get("media_type")) in watched:
                continue
            key = (r["tmdb_id"], r.get("media_type"))
            if key not in seen_ids:
                seen_ids.add(key)
                out.append(r)
        return out

    async def _merged_exclusions(
        self, exclude: Optional[set], user_id: Optional[int]
    ) -> Optional[set]:
        """Explicit exclude ∪ the user's watched/rated pairs (when both given).

        A watched-history lookup failure must not wipe an explicit exclude
        set — degrade to what the caller passed."""
        if user_id is None or self.pool is None:
            return exclude
        merged = set(exclude or set())
        try:
            from .watched import (
                get_user_dislikes,
                get_user_recommendation_exclusions,
            )

            merged |= await get_user_recommendation_exclusions(self.pool, user_id)
            merged |= await get_user_dislikes(self.pool, user_id)
        except Exception as e:
            logger.debug(f"user exclusions unavailable for {user_id}: {e}")
        return merged


    async def recommend_for_user(
        self,
        user_id: int,
        limit: int = 30,
        randomness: float = 0.0,
        vector_space: Optional[str] = None,
        include_why: bool = False,
    ) -> dict:
        """§F end-to-end: user_id in, personalized list out.

        Returns a dict — the caller never assembles the pipeline:
          results:      ranked titles (score, media_type, ...)
          reason:       "personalized" | "watchlist" | "cold_start"
                        (cold_start ⇒ results is EMPTY by design —
                        never a silent blockbuster list)
          seeds:        the top-10 (tmdb_id, media_type, w_i) used
          vector_used:  whether the user-vector ANN channel ran
          why:          per-result top contributing features
                        (include_why=True; costs one extra pass)

        Saga rule (documented): a seed never returns its own saga —
        watched Rocky I blocks I AND recommends II via advancement.
        """
        from .queries import enrich_candidates_batch
        from .user_vector import load_user_vector, top_weighted_seeds

        weighted = await top_weighted_seeds(self.pool, user_id, limit=20)
        watchlist_rows = await self.pool.fetch(
            """SELECT f.tmdb_id, f.media_type FROM user_feedback f
               WHERE f.user_id = $1 AND f.kind = 'watchlist'
               ORDER BY f.created_at DESC LIMIT 10""",
            user_id,
        )
        watchlist = [(r["tmdb_id"], r["media_type"], 1.0) for r in watchlist_rows]
        mode, seeds = recommendation_plan(weighted, watchlist)

        if mode == "cold_start":
            logger.debug(f"recommend user={user_id}: cold_start (no signal)")
            return {
                "results": [], "reason": "cold_start",
                "seeds": [], "vector_used": False, "why": {},
            }

        extra: Optional[List[dict]] = None
        vector_used = False
        try:
            uvec = await load_user_vector(self.pool, user_id, vector_space)
            if uvec:
                extra = await generate_user_vector_candidates(
                    self.pool, uvec, limit=60
                )
                vector_used = bool(extra)
        except Exception as e:
            logger.debug(f"user-vector recall skipped for {user_id}: {e}")

        results = await self.find_similar(
            [(tmdb_id, mt) for tmdb_id, mt, _w in seeds],  # composite key!
            # (bare ids would mix movie 155 with tv 155 — the engine's
            # own invariant; see AGENTS.md)
            limit=limit,
            randomness=randomness,
            user_id=user_id,
            extra_candidates=extra,
        )

        why: dict = {}
        if include_why and results:
            enriched = await enrich_candidates_batch(
                self.pool, [r["tmdb_id"] for r in results]
            )
            top_seed = await self._seed_info_for(seeds[0])
            if top_seed:
                for r in results:
                    cand = dict(enriched.get(r["tmdb_id"], {}))
                    cand.setdefault("id", r["tmdb_id"])
                    for k, v in r.items():
                        cand.setdefault(k, v)
                    vec = feature_vector(
                        cand, cand, cand.get("genres", []), top_seed)
                    why[str(r["tmdb_id"])] = explain_features(vec, ACTIVE_WEIGHTS)

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
        }

    async def _seed_info_for(self, seed: tuple) -> Optional[dict]:
        """Full info dict for one (id, media_type) seed (why rendering)."""
        try:
            info = await get_movie_info_batch(self.pool, [seed[0]])
            if info:
                row = dict(info[seed[0]])
                row.setdefault("media_type", seed[1])
                return row
        except Exception as e:
            logger.debug(f"seed info fetch failed for {seed}: {e}")
        return None

    async def find_similar(
        self,
        tmdb_id: Union[int, List[int], List[Tuple[int, str]]],
        limit: Optional[int] = None,
        randomness: float = 0.0,
        allow_cross_media: bool = False,
        seed_weights: Optional[Dict[int, float]] = None,
        media_type: Optional[str] = None,
        rec_model: Optional[str] = None,
        exclude: Optional[set] = None,
        seed_vectors: Optional[Dict[int, Dict[str, list]]] = None,
        seed_info_override: Optional[dict] = None,
        user_id: Optional[int] = None,
        extra_candidates: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Find similar movies/series, ranked by score.

        Supports single seed (tmdb_id: int) or multi-seed (tmdb_id: List[int]).
        Multi-seed scores each candidate against ALL seeds and averages the scores.
        Results are balanced by media_type proportion of the seeds.

        Results are cached in Redis (TTL: 1h).
        Subsequent calls for the same tmdb_id(s) return instantly.

        Args:
            tmdb_id: TMDB ID (int), list of TMDB IDs, or list of (tmdb_id, media_type)
                tuples. If int, treated as [int]. Up to 20 seeds (MAX_SEEDS).
            limit: Max results to return (None = no limit, return all).
            randomness: Noise factor for controlled shuffling (0.0 = deterministic).
                0.0 = exact score order (best match first)
                0.1-0.3 = slight shuffle, close scores may swap
                0.5+ = significant randomness, still biased toward top scores
            allow_cross_media: If True, candidates from both movies and series
                are included regardless of the seed's media_type. Default False
                preserves the same-type-only behavior.
            seed_weights: tmdb_id -> INFLUENCE (higher = MORE influence;
                internally inverted to 1/w). Passing personalization w_i
                values directly is correct; passing inverse weights is the
                classic mistake.
                Controls the relative influence of each seed on scoring via
                Final_Score(c) = Σ(Score(c, sᵢ) · 1/wᵢ) / Σ(1/wᵢ).
                Weight 1 = full influence (default), weight 2 = half influence,
                weight N = 1/N influence. Only relative ratios matter
                (e.g., 1:2 is the same as 50:100). Works with any positive value.
            rec_model: Per-user model key from user_models.REC_MODELS
                (/set_model). None or 'ensemble' keeps the process-global
                config; a solo key swaps the KNN retrieval column, collapses
                the cosine blend to that column, and applies its tuned
                cosine feature weight.
            exclude: Optional set of (tmdb_id, media_type) pairs the user
                already watched — filtered from the results AFTER the cache
                (the cached payload stays user-agnostic) and BEFORE the
                media-type balancing, in both the cache-hit and fresh paths.
            extra_candidates: Pre-recalled rows (e.g. user-vector ANN)
                merged into the candidate pool; carry `via` to bypass the
                genre gate, `knn_similarity` to feed the cosine feature.
            user_id: Convenience — when given (and a pool is attached),
                loads that user's watched ∪ rated pairs from
                user_watches/title_ratings (engine/watched.py) and merges
                them into `exclude`. Explicit `exclude` entries are kept.

        Returns:
            List of dicts sorted by score (descending), each containing:
            - tmdb_id: int
            - score: float
            - title: str
            - media_type: str
            - rating: float
            - genres: list[str]
            - poster_path: str | None
        """
        await self._ensure_initialized()

        # Per-user model override (/set_model): resolve the spec once and
        # thread it through retrieval + scoring. Unknown keys degrade to
        # the process default — same as no override at all.
        model_key = normalize_model(rec_model)
        model_spec = REC_MODELS.get(model_key) if model_key else None
        emb_column = model_spec["column"] if model_spec else None
        sem_cols = [emb_column] if emb_column else None
        sem_wts = [1.0] if emb_column else None
        score_weights = None
        if model_spec and model_spec["cosine_weight"] is not None:
            score_weights = dict(ACTIVE_WEIGHTS)
            score_weights["cosine_sim"] = float(model_spec["cosine_weight"])
            logger.debug(
                f"Recommendation: per-user model={model_key} column={emb_column} "
                f"cosine_w={model_spec['cosine_weight']}"
            )

        # Normalize input and extract per-seed media_types
        seed_media_types: dict[int, str] = {}
        if isinstance(tmdb_id, int):
            tmdb_ids = [tmdb_id]
        elif tmdb_id and isinstance(tmdb_id[0], tuple):
            # List[Tuple[int, str]] — per-seed media_type
            tmdb_ids = []
            for sid, smt in tmdb_id:
                tmdb_ids.append(sid)
                seed_media_types[sid] = smt
        else:
            tmdb_ids = list(tmdb_id)  # type: ignore[arg-type]

        # Cap seed count to avoid excessive candidate generation
        MAX_SEEDS = 20
        if len(tmdb_ids) > MAX_SEEDS:
            logger.debug(f"Recommendation: capping {len(tmdb_ids)} seeds to {MAX_SEEDS}")
            tmdb_ids = tmdb_ids[:MAX_SEEDS]
            if seed_media_types:
                seed_media_types = {
                    sid: mt for sid, mt in seed_media_types.items() if sid in set(tmdb_ids)
                }

        # --- Redis cache check ---
        # Build media_type fingerprint for cache key
        if seed_media_types:
            mt_fingerprint = ",".join(
                f"{sid}={seed_media_types[sid]}" for sid in sorted(seed_media_types)
            )
        elif media_type:
            mt_fingerprint = media_type
        else:
            mt_fingerprint = "any"

        # Every parameter that changes the result MUST be in the key —
        # limit/randomness/seed_weights/embedding column/extra candidates
        # were missing, so a hit could serve stale/wrong lists.
        cache_key = (
            f"{self.CACHE_PREFIX}{','.join(str(i) for i in sorted(map(str, tmdb_ids)))}"
            f":cm={int(allow_cross_media)}:mt={mt_fingerprint}"
            f":m={model_key or 'def'}"
            f":li={limit if limit is not None else -1}"
            f":rnd={round(randomness, 4)}"
            f":sw={sorted((k, round(v, 4)) for k, v in (seed_weights or {}).items())}"
            f":emb={EMBEDDING_COLUMN}"
            f":xc={len(extra_candidates) if extra_candidates else 0}"
            f":v=2"  # cache format version — bump on any result-shape change
        )
        cache = await _get_cache()
        if cache:
            try:
                cached = await cache.get(cache_key)
                if cached:
                    logger.debug(f"Recommendation cache HIT: {len(tmdb_ids)} seeds")
                    # Cached format: {"movie_ratio": float, "results": list,
                    #                 "seed_ids": list, "raw_per_seed": dict}
                    if isinstance(cached, dict):
                        movie_ratio = cached["movie_ratio"]
                        result = cached["results"]

                        # Re-weight if seed_weights provided and raw data available
                        if (
                            seed_weights is not None
                            and "raw_per_seed" in cached
                            and "seed_ids" in cached
                        ):
                            result = _reweight_from_raw(
                                cached["seed_ids"],
                                cached["raw_per_seed"],
                                cached["results"],
                                seed_weights,
                            )
                    else:
                        # Legacy cache format (list only) — backward compat
                        movie_count = sum(1 for r in cached if r["media_type"] == "movie")
                        movie_ratio = movie_count / len(cached) if cached else 1.0
                        result = cached
                    # Per-user post-processing on the cached copy only —
                    # the stored payload stays user-agnostic. Seeds count
                    # as taken (see _apply_user_filters). `is not None`:
                    # an EMPTY watch set is still a user context — a
                    # truthiness check here would skip the seed guard
                    # entirely for users with no history.
                    if exclude is not None:
                        result = await self._apply_user_filters(
                            result,
                            await self._merged_exclusions(exclude, user_id),
                            set(tmdb_ids),
                        )
                    # Balance FIRST, then noise on the final list (see step 9)
                    if limit is not None:
                        result = _balance_by_media_type(result, movie_ratio, limit)
                    if randomness > 0:
                        result = _apply_noise(result, randomness)
                        for r in result:
                            r["score"] = round(r["score"], 3)
                    return result
            except Exception as e:
                logger.debug(f"Recommendation cache error: {e}")

        # --- Compute recommendations ---

        # 1. Get all seed infos — batch query (1 round-trip instead of N×7).
        # seed_info_override: a VIRTUAL seed (already-shaped info dict for the
        # first tmdb_id) supplied by callers whose seed isn't in tmdb_media —
        # the engine-not-found fallback builds it from the cloud tables and
        # never writes tmdb_media at all.
        if seed_info_override and tmdb_ids:
            seed_infos = [dict(seed_info_override, id=tmdb_ids[0])]
        elif seed_media_types:
            # Group IDs by media_type so each group is a single query
            groups: dict[str, list[int]] = {}
            for sid in tmdb_ids:
                mt = seed_media_types.get(sid, media_type or "")
                groups.setdefault(mt, []).append(sid)
            batch_tasks = [
                get_movie_info_batch(self.pool, ids, media_type=mt if mt else None)
                for mt, ids in groups.items()
            ]
            batch_results = await asyncio.gather(*batch_tasks)
        else:
            batch_results = [await get_movie_info_batch(self.pool, tmdb_ids, media_type=media_type)]

        seed_map: dict[int, dict] = {}
        for br in batch_results:
            seed_map.update(br)
        seed_infos = [seed_map[sid] for sid in tmdb_ids if sid in seed_map]
        if not seed_infos:
            logger.warning(f"Recommendation: none of {tmdb_ids} found")
            return []

        # 1b. Seed cinematic structure tags (narrative/pacing/emotional_arc)
        # — powers narrative_match + the mood features. Missing → None.
        try:
            async with self.pool.acquire() as conn:
                narr_rows = await conn.fetch(
                    """
                    SELECT media_id, narrative_complexity, pacing, emotional_arc
                    FROM tmdb_cinematic
                    WHERE media_id = ANY($1::bigint[])
                    """,
                    [s["id"] for s in seed_infos],
                )
            narr_map = {r["media_id"]: r["narrative_complexity"] for r in narr_rows}
            pacing_m = {r["media_id"]: r["pacing"] for r in narr_rows}
            arc_m = {r["media_id"]: list(r["emotional_arc"] or []) for r in narr_rows}
            for s in seed_infos:
                s["narrative"] = narr_map.get(s["id"])
                s["pacing"] = pacing_m.get(s["id"])
                s["emotional_arc"] = arc_m.get(s["id"], [])
        except Exception as e:
            logger.debug(f"seed cinematic fetch skipped: {e}")

        # Calculate movie_ratio from seeds
        movie_count = sum(1 for s in seed_infos if s["media_type"] == "movie")
        movie_ratio = movie_count / len(seed_infos)

        # 1.5 On-demand TMDB sync: seeds without locally cached behavioral
        #     recommendations get them fetched now (bounded, rate-limited).
        #     After the first query of a seed this is a no-op DB check.
        try:
            await ensure_seeds_synced(self.pool, [(s["id"], s["media_type"]) for s in seed_infos])
        except Exception as e:
            logger.debug(f"Recommendation: on-demand TMDB sync failed: {e}")

        # 2. Generate candidates per seed in parallel, then union.
        #    Four recall paths per seed: classic genre+popularity, pgvector
        #    KNN over overview embeddings, locally synced TMDB behavioral
        #    recommendations ("people also liked"), and the seed director's
        #    other works (auteur DNA). Non-classic failures degrade silently.
        channel_coros: list[tuple[str, object]] = []
        for s in seed_infos:
            channel_coros.append(
                (
                    "genre",
                    generate_candidates(
                        self.pool,
                        s["id"],
                        s["genres"],
                        s["media_type"],
                        limit=60,
                        allow_cross_media=allow_cross_media,
                    ),
                )
            )
            channel_coros.append(
                (
                    "KNN",
                    generate_knn_candidates(
                        self.pool,
                        s["id"],
                        s["media_type"],
                        limit=KNN_CANDIDATES_PER_SEED,
                        allow_cross_media=allow_cross_media,
                        emb_column=emb_column,
                    ),
                )
            )
            channel_coros.append(
                ("TMDB-recs", generate_tmdb_rec_candidates(self.pool, s["id"], s["media_type"]))
            )
            channel_coros.append(
                ("director", generate_director_candidates(self.pool, s["id"], s["media_type"]))
            )
            # Fifth channel: fine-tuned KNN — direct retrieval from the
            # contrastively trained space. Gated by ENABLE_KNN_FINETUNED
            # (default true) for instant revert.
            if os.getenv("ENABLE_KNN_FINETUNED", "true").lower() != "false":
                channel_coros.append(
                    (
                        "knn_ft",
                        generate_knn_finetuned_candidates(
                            self.pool,
                            s["id"],
                            s["media_type"],
                            limit=KNN_CANDIDATES_PER_SEED,
                            emb_column=emb_column,
                        ),
                    )
                )
        all_results = await asyncio.gather(
            *(coro for _, coro in channel_coros), return_exceptions=True
        )

        seed_id_set = set(tmdb_ids)
        seen_ids: set[int] = set()
        candidates = []
        # Seed-indexed semantic similarities harvested from the KNN results —
        # the ranking query already computed the cosine, no need to redo it.
        knn_sim_by_seed: dict[int, dict[int, float]] = {}
        # Each seed owns a contiguous slice of channel_coros (4 channels)
        # Channel count includes the optional fine-tuned KNN channel when enabled
        n_channels = len(channel_coros) // max(len(seed_infos), 1)
        for i, res in enumerate(all_results):
            seed = seed_infos[i // n_channels]
            channel = channel_coros[i][0]
            if isinstance(res, BaseException):
                logger.debug(
                    f"Recommendation: {channel} recall failed for seed {seed['id']}: {res}"
                )
                continue
            if channel == "KNN" and res:
                sims = {c["id"]: c["knn_similarity"] for c in res if c.get("knn_similarity")}
                if sims:
                    knn_sim_by_seed[seed["id"]] = sims
            for c in res:
                if c["id"] not in seen_ids and c["id"] not in seed_id_set:
                    seen_ids.add(c["id"])
                    candidates.append(c)

        if not candidates:
            logger.debug(f"Recommendation: no candidates for {tmdb_ids}")
            return []

        # 3. Merge all seed genres for pre-filtering
        all_genres: list[str] = []
        for s in seed_infos:
            all_genres.extend(s["genres"])

        # 4. Pre-filter candidates
        candidates = optimize_candidate_selection(candidates, all_genres, target_count=150)

        # 5. Batch-enrich candidates with director, cast, keywords
        candidate_ids = [c["id"] for c in candidates]
        enriched = await enrich_candidates_batch(
            self.pool, candidate_ids, media_types=[c["media_type"] for c in candidates]
        )

        # 5b. Precompute semantic overview similarity for every seed × candidate
        # pair via pgvector. `_score_candidate_vs_seed` is synchronous, so we
        # fetch all similarities up-front in one query per seed and pass them
        # in. Similarities already harvested from the KNN ranking results are
        # kept (they are the same cosine, computed for free). Missing
        # embeddings silently degrade to legacy bigram similarity.
        seed_sim_by_seed: dict[int, dict[int, float]] = knn_sim_by_seed
        if candidate_ids:
            sim_tasks = [
                semantic_overview_similarity(
                    self.pool,
                    s["id"],
                    candidate_ids,
                    columns=sem_cols,
                    weights=sem_wts,
                    seed_vectors=(seed_vectors.get(s["id"]) if seed_vectors else None),
                )
                for s in seed_infos
            ]
            sim_results = await asyncio.gather(*sim_tasks, return_exceptions=True)
            for seed, res in zip(seed_infos, sim_results):
                if isinstance(res, dict) and res:
                    existing = seed_sim_by_seed.setdefault(seed["id"], {})
                    for cid, sim in res.items():
                        existing.setdefault(cid, sim)

        # 5b. Caller-injected candidates (e.g. user-vector ANN, §F.1):
        # merged through the same dedup + seed guard as recall channels.
        if extra_candidates:
            for c in extra_candidates:
                if c["id"] not in seen_ids and c["id"] not in seed_id_set:
                    seen_ids.add(c["id"])
                    candidates.append(c)

        # 6. Score each candidate vs ALL seeds, weighted average
        weights = _build_seed_weights(seed_infos, seed_weights)
        total_weight = sum(weights)

        raw_per_seed: dict[int, dict[int, float]] = {}
        results = []
        # Per-entity profiles built ONCE (not once per pair) + the request's
        # weights as a list — the pair loop below is pure set ops + math.
        seed_profiles = [_entity_profile(seed, None) for seed in seed_infos]
        w_list = list(score_weights.values()) if score_weights else HEURISTIC_ACTIVE_LIST
        for cand in candidates:
            cid = cand["id"]
            cand_data = enriched.get(cid, {})
            cand_genres = cand.get("genres", [])
            cp = _entity_profile(cand, cand_data, genres_override=cand_genres)
            cand_genres_set = set(cand_genres) if cand_genres else None

            seed_score_map: dict[int, float] = {}
            weighted_scores: list[float] = []
            for (seed, w), sp in zip(zip(seed_infos, weights), seed_profiles):
                overview_sim = seed_sim_by_seed.get(seed["id"], {}).get(cid, 0.0)
                s = _score_pair_fast(
                    sp, cp, cand, cand_data, cand_genres_set,
                    overview_sim, w_list,
                )
                if s > 0:
                    seed_score_map[seed["id"]] = s
                    weighted_scores.append(s * w)

            if not weighted_scores:
                continue

            # Display guard: a candidate with no renderable title produces an
            # empty Telegram button ("[ 🎬 ]" with no label — live bug on
            # Arrival's page 1). The title backfill covers known rows; this
            # skips anything the sync writes title-less in the future.
            if not ((cand.get("title_en") or "").strip() or (cand.get("title") or "").strip()):
                continue

            raw_per_seed[cid] = seed_score_map
            avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0

            results.append(
                {
                    "tmdb_id": cid,
                    "score": avg_score,
                    "title": cand.get("title", ""),
                    "title_en": cand.get("title_en"),
                    "media_type": cand.get("media_type", "movie"),
                    "rating": cand.get("vote_average") or 0,
                    "genres": cand_genres,
                    "poster_path": cand.get("poster_path"),
                    # Story identity for same-collection/sequel dedup and
                    # the collapse-to-series-start step (the candidate SQL
                    # carries both; they were missing here until now, which
                    # silently disabled the collection branch of
                    # _story_key — sequels of one saga sat in separate slots).
                    "collection_id": cand.get("collection_id"),
                    "release_year": cand.get("release_year"),
                    # For intra-list diversity metrics (evaluation-side);
                    # person_ids of this title's Director/Creator rows.
                    "director_ids": enriched.get(cid, {}).get("director_ids", []),
                }
            )

        # 7. Sort by score descending (deterministic base order), then keep
        # one representative per story (remake/re-series of the same title
        # or same collection) so duplicates stop wasting list slots — done
        # BEFORE caching so cached lists are deduped too.
        results.sort(key=lambda x: x["score"], reverse=True)
        before_dedup = len(results)
        results = _dedup_same_story(results)
        if len(results) != before_dedup:
            logger.debug(f"same-story dedup: {before_dedup} -> {len(results)}")

        # 7b. Auteur diversification — geometric decay per over-represented
        # director. The seed director's FIRST title keeps its full score,
        # the second decays, the third and beyond decay harder. A
        # Villeneuve seed still surfaces Dune and Enemy, but Prisoners and
        # Sicario can no longer squat in every slot while cerebral matches
        # starve — a recall channel bug must surface loudly.
        seed_director_ids: set[int] = set()
        for s in seed_infos:
            seed_director_ids.update(s.get("director_ids") or [])
        if seed_director_ids and len(AUTEUR_DECAY_FACTORS) > 1:
            per_director_seen: dict[int, int] = {}
            for r in results:
                cand_dir_ids = enriched.get(r["tmdb_id"], {}).get("director_ids") or []
                shared = seed_director_ids & set(cand_dir_ids)
                if not shared:
                    continue
                occurrence = max(per_director_seen.get(p, 0) + 1 for p in shared)
                for p in shared:
                    per_director_seen[p] = per_director_seen.get(p, 0) + 1
                idx = min(occurrence, len(AUTEUR_DECAY_FACTORS)) - 1
                r["score"] *= AUTEUR_DECAY_FACTORS[idx]
            # Re-sort: the decay moved scores, the list order must follow —
            # otherwise the limit slice keeps the pre-decay ranking.
            results.sort(key=lambda x: x["score"], reverse=True)

        # 7c. Genre-conditional auteur decay: a SECOND-plus title from the
        # seed director whose genres do NOT overlap the seed's is exactly
        # the Sicario-for-Arrival failure — auteur brand with zero thematic
        # bridge. Same-director titles that DO share the seed's genre world
        # (Dune for Arrival) keep their full score.
        if seed_director_ids:
            seed_genre_set = set()
            for s in seed_infos:
                seed_genre_set.update(canonical_genres(s.get("genres", [])))
            for r in results:
                cand_dir_ids = enriched.get(r["tmdb_id"], {}).get("director_ids") or []
                shared = seed_director_ids & set(cand_dir_ids)
                if not shared:
                    continue
                cand_genre_set = canonical_genres(r.get("genres") or [])
                if not (seed_genre_set & cand_genre_set):
                    r["score"] *= GENRE_MISMATCH_AUTEUR_FACTOR
            results.sort(key=lambda x: x["score"], reverse=True)

        # 7d. MMR selection — THE diversifier. Greedy Maximal Marginal
        # Relevance over the fully re-ranked list, immediately before the
        # final limit: each pick maximizes λ·relevance_norm −
        # (1−λ)·max_sim(c, already_selected). Relevance is min-max
        # normalized per list (both terms in [0,1], λ = true 70/30).
        # Similarity graded: collection 1.0 > director 0.85 > genre
        # Jaccard capped 0.5. λ=1.0 control skips the loop entirely.
        if MMR_ENABLED and limit and MMR_LAMBDA < 1.0 and len(results) > (limit or 0):
            results = _mmr_order(results, MMR_LAMBDA, limit)

        # 8. Round scores (noise moves AFTER media balancing — applying it
        # before let noise-shuffled movies win slots that the tv-balance then
        # dropped, leaving random survivors like Family Guy; 2026-08-23)
        for r in results:
            r["score"] = round(r["score"], 3)

        logger.info(
            f"Recommendation: {len(tmdb_ids)} seeds -> {len(results)} results "
            f"(movie_ratio={movie_ratio:.0%}, limit={limit}, "
            f"randomness={randomness})"
        )

        # --- Store in Redis cache ---
        if cache:
            try:
                await cache.set(
                    cache_key,
                    {
                        "movie_ratio": movie_ratio,
                        "results": results,
                        "seed_ids": [s["id"] for s in seed_infos],
                        "raw_per_seed": raw_per_seed,
                    },
                    ttl=self.CACHE_TTL,
                )
                logger.debug(f"Recommendation cache STORE: {tmdb_ids}")
            except Exception as e:
                logger.debug(f"Recommendation cache store error: {e}")

        # 9. Apply limit with media type balancing, THEN variety noise on
        # the final list (top-3 pinned) — never before the media cut.
        # The per-user pass (watched filter + saga advancement) runs BEFORE
        # the balancing so a freed slot is refilled by the next-best
        # candidate, and after the cache store above so the shared cache
        # stays user-agnostic. Skipped without a user context
        # (benchmark/eval calls) to keep comparisons stable.
        if exclude is not None:  # empty set = real user with no history yet
            results = await self._apply_user_filters(
                results,
                await self._merged_exclusions(exclude, user_id),
                set(tmdb_ids),
            )
        if limit is not None:
            results = _balance_by_media_type(results, movie_ratio, limit)
        if randomness > 0:
            results = _apply_noise(results, randomness)
            for r in results:
                r["score"] = round(r["score"], 3)

        return results


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


def _modality(genres, original_language) -> str:
    """'anime' | 'animation' | 'live' — the medium, not the genre.

    Animation genre + Japanese origin is anime (Attack on Titan); Animation
    without ja is western animation (Over the Garden Wall); everything
    else is live action. The distinction that matters: dark war
    anime and British crime dramas shared Drama/Action tags and ratings.
    """

    if "Animation" in canonical_genres(genres or []):
        return "anime" if (original_language or "").lower().startswith("ja") else "animation"
    return "live"


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
    # Cinematic-DNA features start at 0.0 — pure learned-scorer inputs until
    # prices them (Johann Johannsson scored BOTH Arrival and Sicario; the
    # heuristic has no principled prior for them yet).
    # 0.0 — heuristic pricing lost
    # (19.9/19.2 vs 20.5). The LTR oracle priced them 3.86/2.52 as
    # directions, but partial catalog coverage makes them noisy at
    # heuristic strength. LTR-only inputs until coverage completes.
    "composer_match": 0.0,
    "dp_match": 0.0,
    # The medium signal must be LEARNED, not imposed. Bonus form
    # (+1.5) measured 22.4/22.7 and penalty form (-2.0) 22.6/22.6 — both
    # under the 22.9/23.7 baseline, because the evaluation data deliberately
    # encodes cross-medium taste (The Boys -> INVINCIBLE) that a global
    # rule contradicts. The feature stays for the LTR (with N4 modality
    # hard negatives in the miner) to learn WHEN medium matters.
    # → Re-enabled 2026-09: user feedback showed anime→live-action dark

    # The evaluation set's cross-medium edge case is rarer than the damage.
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
def _story_key(item: dict) -> str:
    coll = item.get("collection_id")
    if coll:
        return f"coll:{coll}"
    t = (item.get("title_en") or item.get("title") or "").strip().lower()
    return "t:" + "".join(ch for ch in t if ch.isalnum())


def _dedup_same_story(items: List[dict]) -> List[dict]:
    """One slot per story.

    Sequel sagas (same collection_id) keep their EARLIEST member — the
    series start — rather than the top-scored sequel, so a list that
    surfaced "John Wick 3" collapses toward the first John Wick.
    Same-title remakes keep the top-scored version (no meaningful
    "first" there). Falls back to top-scored when years are unknown.
    """
    best: dict[str, dict] = {}
    for it in sorted(items, key=lambda x: x["score"], reverse=True):
        key = _story_key(it)
        cur = best.get(key)
        if cur is None:
            best[key] = it
            continue
        if key.startswith("coll:"):
            ya, yb = it.get("release_year"), cur.get("release_year")
            if ya and yb and ya < yb:
                best[key] = it
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def _load_ensemble_config() -> None:
    """Apply the Optuna-tuned ensemble profile when REC_ENSEMBLE_CONFIG points
    at a saved config (per-model weights + heuristic knobs).

    Pattern: production-decision config file produced by
    scripts/eval/tune_ensemble_optuna.py; sets the multi-column cosine blend
    (config.COSINE_BLEND) and the tuned heuristic weights. Any failure leaves
    the defaults untouched — a corrupt artifact must never break the engine.
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

    Partial application: the artifact may cover a subset of FEATURE_NAMES
    (the published weights.json predates the mood features). Learned
    values override the heuristic baseline feature-by-feature; features
    the artifact lacks keep their heuristic coefficients. An artifact
    covering nothing (or a corrupt file) returns None — the engine then
    stays fully on the heuristic weights (fail-safe: a bad artifact must
    never break recommendations)."""
    if os.getenv("CINE_REC_SCORER", os.getenv("REC_SCORER", "heuristic")) != "learned":
        return None
    try:
        import json
        from pathlib import Path

        artifact = Path(__file__).parent / "weights.json"
        data = json.loads(artifact.read_text())
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
        return blended
    except Exception as e:
        logger.warning(f"learned weights unavailable ({e}) — using heuristic")
        return None


HEURISTIC_WEIGHTS_LIST = list(HEURISTIC_WEIGHTS.values())[:27]

LEARNED_WEIGHTS = _load_learned_weights()
ACTIVE_WEIGHTS = LEARNED_WEIGHTS or HEURISTIC_WEIGHTS
HEURISTIC_ACTIVE_LIST = list(ACTIVE_WEIGHTS.values())


def _entity_profile(entity: dict, entity_data: Optional[dict],
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


def _vec_from_profiles(sp: dict, cp: dict, cand: dict, cand_data: dict,
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


def feature_vector(
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
    sp = _entity_profile(seed, None)
    cp = _entity_profile(cand, cand_data, genres_override=cand_genres)
    return _vec_from_profiles(sp, cp, cand, cand_data, overview_sim)



def _pair_similarity(a: dict, b: dict) -> float:
    """Graded diversity similarity between two result rows (in-memory only).

    1.0 same collection (sequel pile-up is the strongest redundancy),
    0.85 shared director person_id (a genuine second masterpiece can
    still surface when its relevance is exceptional), genre Jaccard
    capped at 0.5 (genre overlap must not erase good picks).
    """

    ca, cb = a.get("collection_id"), b.get("collection_id")
    if ca and cb and ca == cb:
        return 1.0
    da = set(a.get("director_ids") or [])
    if da & set(b.get("director_ids") or []):
        return 0.85
    ga, gb = canonical_genres(a.get("genres") or []), canonical_genres(b.get("genres") or [])
    if not ga or not gb:
        return 0.0
    return min(len(ga & gb) / len(ga | gb), 1.0) * 0.5


def _mmr_pair_key(item: dict) -> tuple:
    """Per-item precomputed inputs of _pair_similarity (identical logic)."""
    return (
        item.get("collection_id"),
        frozenset(item.get("director_ids") or []),
        frozenset(canonical_genres(item.get("genres") or [])),
    )


def _pair_similarity_fast(ka: tuple, kb: tuple) -> float:
    """_pair_similarity over precomputed keys — same values."""
    ca, da, ga = ka
    cb, db, gb = kb
    if ca and cb and ca == cb:
        return 1.0
    if da & db:
        return 0.85
    if not ga or not gb:
        return 0.0
    return min(len(ga & gb) / len(ga | gb), 1.0) * 0.5


def _mmr_order(results: list, lam: float, k: int) -> list:
    """Greedy MMR: build the final order by repeatedly picking
    argmax λ·rel_norm − (1−λ)·max_sim(c, selected); the remainder keeps
    relevance order after the selected prefix.

    Incremental: each pool item keeps its running MAX similarity against
    the selected set — adding a selection updates only the new pairs.
    n·k similarity computations instead of k²·n/2, same pick order
    (max is associative; the argmax comparison is untouched).
    """
    lo = min(r["score"] for r in results)
    hi = max(r["score"] for r in results)
    spread = (hi - lo) or 1.0
    keys = {id(r): _mmr_pair_key(r) for r in results}
    pool = list(results)
    pen = {id(r): 0.0 for r in results}  # running max sim vs selected
    selected: list = []
    while pool and len(selected) < k:
        best, best_val = None, -1e9
        for c in pool:
            val = lam * ((c["score"] - lo) / spread) - (1 - lam) * pen[id(c)]
            if val > best_val:
                best, best_val = c, val
        selected.append(best)
        pool.remove(best)
        bk = keys[id(best)]
        for c in pool:
            sim = _pair_similarity_fast(bk, keys[id(c)])
            if sim > pen[id(c)]:
                pen[id(c)] = sim
    return selected + pool


def _score_pair_fast(sp: dict, cp: dict, cand: dict, cand_data: dict,
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


def _balance_by_media_type(
    results: List[dict],
    movie_ratio: float,
    limit: int,
) -> List[dict]:
    """Split results into movies/series by ratio, then interleave by score.

    Args:
        results: Already sorted by score (descending).
        movie_ratio: 0.0 (all series) to 1.0 (all movies).
        limit: Max total results to return.

    Returns:
        Balanced list, sorted by score, with exactly `limit` items.
    """
    if not results:
        return []

    # Pure single type — no balancing needed, just filter to that type
    if movie_ratio in (0.0, 1.0):
        target_type = "movie" if movie_ratio == 1.0 else "tv"
        return [r for r in results if r["media_type"] == target_type][:limit]

    MIN_FRACTION = 0.1  # At least 10% for the minority type

    movie_slots = round(limit * movie_ratio)
    tv_slots = limit - movie_slots

    # Enforce minimum fraction for the minority type
    if movie_slots > 0 and tv_slots < limit * MIN_FRACTION:
        tv_slots = max(1, round(limit * MIN_FRACTION))
        movie_slots = limit - tv_slots
    elif tv_slots > 0 and movie_slots < limit * MIN_FRACTION:
        movie_slots = max(1, round(limit * MIN_FRACTION))
        tv_slots = limit - movie_slots

    movies = [r for r in results if r["media_type"] == "movie"]
    series = [r for r in results if r["media_type"] == "tv"]

    # Take top X from each type (already sorted by score)
    selected = movies[:movie_slots] + series[:tv_slots]

    # Re-sort by score so movies and series are interleaved
    selected.sort(key=lambda x: x["score"], reverse=True)

    return selected


def _build_seed_weights(
    seed_infos: list[dict],
    seed_weights: Optional[Dict[int, float]],
) -> list[float]:
    """Build influence weights (1/w) aligned with seed_infos.

    Uses the formula: Final_Score(c) = Σ(Score(c, sᵢ) · 1/wᵢ) / Σ(1/wᵢ)
    Weight 1 = full influence, weight 2 = half influence, weight N = 1/N influence.
    """
    weights = []
    for seed in seed_infos:
        w = 1.0
        if seed_weights is not None and seed["id"] in seed_weights:
            w = float(seed_weights[seed["id"]])
        weights.append(1.0 / w)  # Invert: higher weight = lower influence
    return weights


def _reweight_from_raw(
    seed_ids: list[int],
    raw_per_seed: dict[int, dict[int, float]],
    cached_results: list[dict],
    seed_weights: Dict[int, float],
) -> list[dict]:
    """Re-compute weighted average from cached raw per-seed scores.

    Used when a cached result is hit but the caller provides different seed_weights.
    """
    weights = [1.0 / float(seed_weights.get(sid, 1.0)) for sid in seed_ids]
    total_weight = sum(weights)
    if total_weight <= 0:
        return cached_results

    result_map = {r["tmdb_id"]: r for r in cached_results}
    result = []
    for cid, seed_map in raw_per_seed.items():
        if not seed_map:
            continue
        weighted = sum(
            seed_map.get(sid, 0) * w for sid, w in zip(seed_ids, weights) if sid in seed_map
        )
        if weighted <= 0:
            continue
        avg = weighted / total_weight
        orig = result_map.get(cid)
        if orig:
            result.append({**orig, "score": avg})
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


async def _get_cache():
    """Get the global AsyncCacheClient (lazy init, same as PostgresTMDBService pattern).

    Returns None if Redis is unavailable.
    """
    try:
        from .cache import get_cache

        return await get_cache()
    except Exception:
        return None


def _apply_noise(items: List[dict], randomness: float) -> List[dict]:
    """Apply controlled noise to sorted results (like LLM temperature).

    Uses the score as confidence: high-score items resist shuffling,
    low-score items are more likely to swap positions. The top-3 are
    pinned — variety must never bury the engine's best answers (users
    judge a list by its head; 2026-08-23 feedback).

    Args:
        items: Already sorted by score (descending)
        randomness: 0.0 (no noise) to 1.0+ (heavy shuffle)

    Returns:
        Re-sorted list with noise applied.
    """
    if not items or randomness <= 0:
        return items

    head, tail = items[:3], items[3:]
    if not tail:
        return head

    scores = [item["score"] for item in tail]
    max_score = max(scores) if scores else 1.0
    score_range = max_score - min(scores) if len(scores) > 1 else 1.0

    for item in tail:
        noise = random.uniform(-1, 1) * randomness * score_range
        item["score"] += noise

    tail.sort(key=lambda x: x["score"], reverse=True)
    return head + tail
