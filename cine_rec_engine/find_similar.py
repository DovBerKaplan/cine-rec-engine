"""find_similar — the seed-based recommendation flow.

Split from service.py (phase 5a, final cut): channel fan-out, candidate
pool assembly, the scoring loop, MMR, media balancing, caching and the
per-user post-passes. `find_similar` takes the service instance (`svc`);
rebindable scoring names resolve through the service module at call
time (see the seam note at the top of the function).
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger

from . import db
from . import metrics as _metrics
from .config import (
    AUTEUR_DECAY_FACTORS,
    EMBEDDING_COLUMN,
    EXPLORE_SHARE,
    GENRE_MISMATCH_AUTEUR_FACTOR,
    KNN_CANDIDATES_PER_SEED,
    MMR_ENABLED,
    MMR_LAMBDA,
)
from .model_spaces import REC_MODELS, normalize_model
from .queries import (
    apply_recall_filters,
    enrich_candidates_batch,
    filters_fingerprint,
    generate_candidates,
    generate_director_candidates,
    generate_director_candidates_batch,
    generate_knn_candidates,
    generate_tmdb_rec_candidates,
    generate_tmdb_rec_candidates_batch,
    get_movie_info_batch,
    normalize_filters,
)
from .scoring import (
    canonical_genres,
    optimize_candidate_selection,
    semantic_overview_similarity_batch,
)
from .tmdb_recs import ensure_seeds_synced

async def find_similar(
    svc,
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
    extra_factory: Optional[Callable[[], Awaitable[List[dict]]]] = None,
    extra_limit: int = 60,
    filters: Optional[dict] = None,
    explore: Optional[float] = None,
    user_context: Optional[dict] = None,
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
        extra_factory: Lazy form of extra_candidates — awaited only on
            the compute path (a cache hit never calls it, so the ANN
            query behind it is skipped when the answer is already
            cached). The cache key records extra_limit instead of the
            fetched row count, so the key stays stable across requests.
        extra_limit: The limit extra_factory recalls with — part of
            the cache key when extra_factory is given.
        filters: Session/context filters (RFC §3): year_min, year_max,
            genre_ids, exclude_genre_ids, max_runtime (minutes,
            movies only — tv has no runtime). Validated by
            queries.normalize_filters (ValueError on bad input),
            hashed into the cache key, and applied at the recall
            boundary BEFORE scoring and the limit cut.
        explore: Exploration budget share override (RFC §2), 0.0–0.5.
            None = config EXPLORE_SHARE (default 0.12); 0 disables.
            Post-cache and per-user, deterministic: eligible picks
            are genre-disjoint from the user's top clusters AND
            affinity ≥ the main list's median.
        user_context: Precomputed per-request context {"exclusions":
            set, "uvec": list} — a page computes it once and threads
            it through every row instead of re-querying per row.
            Pure optimization; results are identical without it.
        user_id: Per-user context — merges the user's watched ∪ rated
            ∪ disliked pairs into `exclude`, advances sagas, then
            applies a GENTLE taste tilt: near-tied candidates break
            toward the user's profile (boost-only, capped by
            USER_TILT_ALPHA, applied after the hard filters, per
            request — the shared cache stays user-agnostic). The
            seed's relevance stays primary; this reorders near-ties,
            never the candidate pool. Explicit `exclude` kept.

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
    # rebindable service attributes (the Rust scoring ext swaps
    # them post-install) — resolve through the module at CALL time
    from . import service as _svc

    await svc._ensure_initialized()

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
        score_weights = dict(_svc.ACTIVE_WEIGHTS)
        score_weights["cosine_sim"] = float(model_spec["cosine_weight"])
        logger.debug(
            f"Recommendation: per-user model={model_key} column={emb_column} "
            f"cosine_w={model_spec['cosine_weight']}"
        )

    # Session filters (RFC §3): validated once, hashed into the cache
    # key, applied at the recall boundary. ValueError on bad input —
    # a filter that silently did nothing is worse than a loud error.
    session_filters = normalize_filters(filters)
    # Exploration budget (RFC §2) — post-cache and per-user, so it is
    # deliberately NOT part of the cache key.
    explore_share = EXPLORE_SHARE if explore is None else float(explore)
    if not 0.0 <= explore_share <= 0.5:
        raise ValueError(f"explore must be in [0.0, 0.5], got {explore}")

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
        f"{svc.CACHE_PREFIX}{','.join(str(i) for i in sorted(map(str, tmdb_ids)))}"
        f":cm={int(allow_cross_media)}:mt={mt_fingerprint}"
        f":m={model_key or 'def'}"
        f":li={limit if limit is not None else -1}"
        f":rnd={round(randomness, 4)}"
        f":sw={sorted((k, round(v, 4)) for k, v in (seed_weights or {}).items())}"
        f":emb={EMBEDDING_COLUMN}"
        f":xc={len(extra_candidates) if extra_candidates else (f'f{extra_limit}' if extra_factory else 0)}"
        f":fl={filters_fingerprint(session_filters)}"
        f":v=3"  # cache format version — bump on any result-shape change
    )
    # Per-user prefetch: the exclusion set and the seed-saga block
    # set depend only on the request, never on the result list —
    # start them alongside the cache read (and, on a miss, the
    # recall fan-out) so the post-processing pass awaits background
    # work instead of serializing two query chains after it. Both
    # coroutines degrade internally (they never raise), so a task
    # abandoned by an error elsewhere in the hit branch cannot leak
    # an unretrieved exception.
    prefetch: dict = {}
    if exclude is not None:
        if (user_context or {}).get("exclusions") is None:
            prefetch["merged"] = asyncio.ensure_future(
                svc._merged_exclusions(exclude, user_id))
        prefetch["blocked"] = asyncio.ensure_future(
            svc._seed_saga_blocked(set(tmdb_ids)))

    cache = await _get_cache()
    if cache:
        try:
            cached = await cache.get(cache_key)
            _metrics.observe_cache("hit" if cached else "miss")
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
                tilt_ctx = user_context if isinstance(user_context, dict) else {}
                if exclude is not None:
                    merged = (user_context or {}).get("exclusions")
                    if merged is None and "merged" in prefetch:
                        merged = await prefetch["merged"]
                    elif exclude:
                        merged = set(merged) | set(exclude)
                    blocked = (
                        await prefetch["blocked"]
                        if "blocked" in prefetch else None)
                    result = await svc._apply_user_filters(
                        result, merged, set(tmdb_ids), blocked=blocked,
                    )
                    if user_id is not None:
                        result = await svc._apply_user_tilt(
                            result, user_id, model_key, context=tilt_ctx)
                # Balance FIRST, then noise on the final list (see step 9)
                if limit is not None:
                    if user_id is not None and explore_share > 0 \
                            and tilt_ctx.get("affinities"):
                        result = await svc._apply_exploration(
                            result, user_id, movie_ratio, limit,
                            explore_share, tilt_ctx)
                    else:
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
    else:
        if seed_media_types:
            # Group IDs by media_type so each group is a single query
            groups: dict[str, list[int]] = {}
            for sid in tmdb_ids:
                mt = seed_media_types.get(sid, media_type or "")
                groups.setdefault(mt, []).append(sid)
            batch_tasks = [
                get_movie_info_batch(svc.pool, ids, media_type=mt if mt else None)
                for mt, ids in groups.items()
            ]
            batch_results = await asyncio.gather(*batch_tasks)
        else:
            batch_results = [
                await get_movie_info_batch(svc.pool, tmdb_ids, media_type=media_type)]

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
        pool = svc.pool
        assert pool is not None  # initialize() ran — caller contract
        async with pool.acquire() as conn:
            narr_rows = await db.fetch(conn,
                """
                SELECT media_id, narrative_complexity, pacing, emotional_arc
                FROM {t_tmdb_cinematic}
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
        await ensure_seeds_synced(svc.pool, [(s["id"], s["media_type"]) for s in seed_infos])
    except Exception as e:
        logger.debug(f"Recommendation: on-demand TMDB sync failed: {e}")

    # 2. Generate candidates per seed in parallel, then union.
    #    Four recall paths per seed: classic genre+popularity, pgvector
    #    KNN over overview embeddings, locally synced TMDB behavioral
    #    recommendations ("people also liked"), and the seed director's
    #    other works (auteur DNA). Non-classic failures degrade silently.
    #    Channels with a batched form run ONE query for all seeds; the
    #    merge below walks a seed-major x channel-minor matrix in the
    #    fixed channel order, so dedup winners, via tags, and the KNN
    #    similarity harvest stay identical to the per-seed fan-out no
    #    matter which channels batch.
    n_seeds = len(seed_infos)
    # Both KNN channels share the same embedding column in every
    # configuration, so when the fine-tuned channel is enabled ONE
    # query per seed (at its looser vote floor) covers both result
    # sets — the strict-floor list is a near-subset of the loose one.
    # Gated by ENABLE_KNN_FINETUNED (default true) for instant revert
    # to the two-channel form.
    knn_ft_enabled = os.getenv("ENABLE_KNN_FINETUNED", "true").lower() != "false"

    async def _tmdb_recs_channel() -> List[List[dict]]:
        """One query for every seed's TMDB recs, or the per-seed
        channel when the batch fails (each seed then degrades
        alone, exactly like the pre-batch fan-out)."""
        try:
            return await generate_tmdb_rec_candidates_batch(
                svc.pool,
                [(s["id"], s["media_type"]) for s in seed_infos],
            )
        except Exception as e:
            logger.debug(
                f"Recommendation: TMDB-recs batch failed, retrying per seed: {e}")
            return await asyncio.gather(
                *[generate_tmdb_rec_candidates(svc.pool, s["id"], s["media_type"])
                  for s in seed_infos],
                return_exceptions=True)

    async def _director_channel() -> List[List[dict]]:
        """Same contract for the auteur channel: one query for all
        seeds, per-seed fallback when the batch fails."""
        try:
            return await generate_director_candidates_batch(
                svc.pool,
                [(s["id"], s["media_type"]) for s in seed_infos],
            )
        except Exception as e:
            logger.debug(
                f"Recommendation: director batch failed, retrying per seed: {e}")
            return await asyncio.gather(
                *[generate_director_candidates(svc.pool, s["id"], s["media_type"])
                  for s in seed_infos],
                return_exceptions=True)

    genre_task = asyncio.ensure_future(asyncio.gather(
        *[generate_candidates(
            svc.pool, s["id"], s["genres"], s["media_type"],
            limit=60, allow_cross_media=allow_cross_media,
        ) for s in seed_infos],
        return_exceptions=True))
    knn_task = asyncio.ensure_future(asyncio.gather(
        *[generate_knn_candidates(
            svc.pool, s["id"], s["media_type"],
            limit=KNN_CANDIDATES_PER_SEED,
            allow_cross_media=allow_cross_media,
            emb_column=emb_column,
            vote_floor=100 if knn_ft_enabled else None,
        ) for s in seed_infos],
        return_exceptions=True))
    tmdb_task = asyncio.ensure_future(_tmdb_recs_channel())
    director_task = asyncio.ensure_future(_director_channel())
    (
        genre_results,
        knn_results,
        tmdb_batch,
        director_batch,
    ) = await asyncio.gather(
        genre_task, knn_task, tmdb_task, director_task,
        return_exceptions=True)

    def _per_seed(res, n: int) -> list:
        """A channel's per-seed result list; a stray Exception for
        the whole channel (cancellation) degrades every seed at
        once — the inner gathers otherwise never raise."""
        if isinstance(res, list):
            return res
        return [res] * n

    # channel_rows[channel_index][seed_index] — batched channels
    # arrive as ONE per-seed-indexed list (or an Exception expanded
    # so every seed degrades individually)
    channel_rows: list = [
        _per_seed(genre_results, n_seeds),
        _per_seed(knn_results, n_seeds),
        _per_seed(tmdb_batch, n_seeds),
        _per_seed(director_batch, n_seeds),
    ]
    channel_names = ("genre", "KNN", "TMDB-recs", "director")

    seed_id_set = set(tmdb_ids)
    seen_ids: set[int] = set()
    candidates = []
    # Seed-indexed semantic similarities harvested from the KNN results —
    # the ranking query already computed the cosine, no need to redo it.
    knn_sim_by_seed: dict[int, dict[int, float]] = {}
    for si, seed in enumerate(seed_infos):
        for channel, rows_per_seed in zip(channel_names, channel_rows):
            res = rows_per_seed[si]
            if isinstance(res, BaseException):
                logger.debug(
                    f"Recommendation: {channel} recall failed for seed {seed['id']}: {res}"
                )
                _metrics.observe_channel(channel, 0)
                continue
            _metrics.observe_channel(channel, len(res))
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

    # 2b. Session filters at the recall boundary — before scoring and
    # the limit cut, so limit semantics stay honest (RFC §3).
    if session_filters:
        candidates = await apply_recall_filters(
            svc.pool, candidates, session_filters)
        if not candidates:
            logger.debug(
                f"Recommendation: filters {filters_fingerprint(session_filters)}"
                f" removed every candidate for {tmdb_ids}")
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
        svc.pool, candidate_ids, media_types=[c["media_type"] for c in candidates]
    )

    # 5b. Precompute semantic overview similarity for every seed × candidate
    # pair via pgvector — ONE batched cross-join query per blend column
    # instead of one query per seed. `_score_candidate_vs_seed` is
    # synchronous, so we fetch all similarities up-front and pass them
    # in. Similarities already harvested from the KNN ranking results
    # are kept (they are the same cosine, computed for free). Missing
    # embeddings silently degrade to legacy bigram similarity.
    seed_sim_by_seed: dict[int, dict[int, float]] = knn_sim_by_seed
    if candidate_ids:
        try:
            batched = await semantic_overview_similarity_batch(
                svc.pool,
                [s["id"] for s in seed_infos],
                candidate_ids,
                columns=sem_cols,
                weights=sem_wts,
                seed_vectors=seed_vectors,
            )
        except Exception as e:
            logger.debug(f"batched semantic similarity failed: {e}")
            batched = {}
        for sid, cmap in batched.items():
            existing = seed_sim_by_seed.setdefault(sid, {})
            for cid, sim in cmap.items():
                existing.setdefault(cid, sim)

    # 5b. Caller-injected candidates (e.g. user-vector ANN, §F.1):
    # merged through the same dedup + seed guard as recall channels.
    # The factory form resolves HERE — a cache hit returns before this
    # point, so its ANN query never runs when the list is already
    # cached. A factory failure degrades to no extras, like the eager
    # path in recommend_for_user always has.
    if extra_factory and not extra_candidates:
        try:
            extra_candidates = await extra_factory()
        except Exception as e:
            logger.debug(f"extra candidates factory failed: {e}")
            extra_candidates = None
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
    seed_profiles = [_svc._entity_profile(seed, None) for seed in seed_infos]
    w_list = list(score_weights.values()) if score_weights else _svc.HEURISTIC_ACTIVE_LIST
    for cand in candidates:
        cid = cand["id"]
        cand_data = enriched.get(cid, {})
        cand_genres = cand.get("genres", [])
        cp = _svc._entity_profile(cand, cand_data, genres_override=cand_genres)
        cand_genres_set = set(cand_genres) if cand_genres else None

        seed_score_map: dict[int, float] = {}
        weighted_scores: list[float] = []
        for (seed, w), sp in zip(zip(seed_infos, weights), seed_profiles):
            overview_sim = seed_sim_by_seed.get(seed["id"], {}).get(cid, 0.0)
            # `score`, not `s`: `s` is a seed dict in the channel loop
            # above — one name, one type keeps checkers honest.
            score = _svc._score_pair_fast(
                sp, cp, cand, cand_data, cand_genres_set,
                overview_sim, w_list,
            )
            if score > 0:
                seed_score_map[seed["id"]] = score
                weighted_scores.append(score * w)

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
                ttl=svc.CACHE_TTL,
            )
            _metrics.observe_cache("store")
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
        merged = (user_context or {}).get("exclusions")
        if merged is None and "merged" in prefetch:
            merged = await prefetch["merged"]
        elif exclude:
            merged = set(merged) | set(exclude)
        blocked = (
            await prefetch["blocked"]
            if "blocked" in prefetch else None)
        results = await svc._apply_user_filters(
            results, merged, set(tmdb_ids), blocked=blocked,
        )
        tilt_ctx = user_context if isinstance(user_context, dict) else {}
        if user_id is not None:
            results = await svc._apply_user_tilt(
                results, user_id, model_key, context=tilt_ctx)
    else:
        tilt_ctx = {}
    if limit is not None:
        if user_id is not None and explore_share > 0 \
                and tilt_ctx.get("affinities"):
            results = await svc._apply_exploration(
                results, user_id, movie_ratio, limit,
                explore_share, tilt_ctx)
        else:
            results = _balance_by_media_type(results, movie_ratio, limit)
    if randomness > 0:
        results = _apply_noise(results, randomness)
        for r in results:
            r["score"] = round(r["score"], 3)

    return results


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


# (FEATURE_NAMES / HEURISTIC_WEIGHTS definitions live in service.py —
# the shared feature space, rebind-sensitive; restored there in 5a)


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
