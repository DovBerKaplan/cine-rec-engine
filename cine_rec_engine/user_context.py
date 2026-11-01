"""Per-user context passes — filtering, tilt, exploration, exclusions.

Split from service.py (phase 5a, call-flow split): everything that
post-processes a result list FOR ONE USER after the shared,
user-agnostic compute. Order is a contract: filters → tilt → balance
→ noise (balance/noise live with the find-similar flow).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from . import db
from .config import USER_TILT_ALPHA
from .model_spaces import column_for
from .queries import genre_names_for_ids, top_user_genre_ids


async def seed_saga_blocked(pool, seed_ids: set) -> Optional[set]:
    """The seeds plus every member of their collections — a seed's
    whole saga is persona non grata (see apply_user_filters).

    Standalone (vs. inline in the filter pass) so it can be prefetched
    concurrently with the cache read / recall: it depends only on the
    seed ids, never on the result list. Returns None on fetch failure
    so the caller falls back to computing it inside the pass, which
    degrades exactly the same way.
    """
    if not seed_ids:
        return set()
    try:
        rows = await db.fetch(pool,
            """
                SELECT DISTINCT m2.id
                FROM {t_tmdb_movies} m2
                WHERE m2.collection_id > 0
                  AND m2.collection_id IN (
                    SELECT s.collection_id FROM {t_tmdb_movies} s
                    WHERE s.id = ANY($1::bigint[])
                      AND s.collection_id > 0
                )
                """,
            sorted(seed_ids),
        )
        return set(seed_ids) | {row["id"] for row in rows}
    except Exception as e:
        logger.debug(f"seed-saga block fetch failed: {e}")
        return None


async def apply_user_filters(
    pool,
    results: List[dict],
    exclude: Optional[set],
    seed_ids: Optional[set] = None,
    blocked: Optional[set] = None,
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
    # any collection the seeds belong to is blocked (movies only —
    # tv rows carry a NULL collection_id, so only movie seeds can
    # contribute). Driven seed-first so it runs on the pkey + the
    # collection index instead of a catalog-wide join over the view.
    # `blocked` arrives prefetched from the request path when the
    # caller had the seed ids before the results existed.
    if blocked is None:
        blocked = set(seeds)
        if seeds:
            blocked = await seed_saga_blocked(pool, seeds) or set(seeds)

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
        # Straight at tmdb_movies (sagas are movie-only; the view's
        # COALESCE/title_en are no-ops on its movie branch) so the
        # collection index serves it — the same query through the
        # compatibility view planned a catalog-wide union+sort.
        try:
            rows = await db.fetch(pool,
                """
                    SELECT m.collection_id, m.id, m.title, m.title AS title_en,
                           m.vote_average::float AS vote_average, m.poster_path,
                           EXTRACT(YEAR FROM m.release_date)::int
                               AS release_year
                    FROM {t_tmdb_movies} m
                    WHERE m.collection_id > 0
                      AND m.collection_id = ANY($1::int[])
                    ORDER BY m.collection_id,
                             m.release_date ASC NULLS LAST,
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


async def apply_user_tilt(
    pool,
    results: List[dict],
    user_id: int,
    model_key: Optional[str],
    context: Optional[dict] = None,
) -> List[dict]:
    """Controlled personalization: a gentle re-rank toward the user's
    taste, AFTER the hard filters and BEFORE balancing/noise. Boost-
    only and capped (USER_TILT_ALPHA) — the seed stays primary; this
    breaks near-ties, it never reshapes the list. The user vector is
    ensured on demand in the SAME model space as the request's
    embedding column (or taken from `context["uvec"]` — a page
    computes it once and threads it through every row). Every
    failure degrades to the untitled list. When `context` is a dict,
    it receives {"uvec", "affinities"} for the exploration pass.
    """
    if not results:
        return results
    try:
        uvec = (context or {}).get("uvec")
        if uvec is None:
            from .user_vector import ensure_user_vector

            uvec = await ensure_user_vector(pool, user_id, model_key)
        if not uvec:
            return results
        if context is not None:
            context["uvec"] = uvec
        # function-level import, exactly like the pre-split code: tests
        # patch cine_rec_engine.queries.fetch_embeddings_batch and the
        # call must see the CURRENT module attribute, not a frozen one
        from .queries import fetch_embeddings_batch

        embs = await fetch_embeddings_batch(
            pool,
            [(r["tmdb_id"], r.get("media_type")) for r in results],
            column_for(model_key),
        )
        if not embs:
            return results
        affinities = {key: _cosine(uvec, vec) for key, vec in embs.items()}
        if context is not None:
            context["affinities"] = affinities
        return tilt_scores(results, affinities, USER_TILT_ALPHA)
    except Exception as e:
        logger.debug(f"user tilt skipped for {user_id}: {e}")
        return results


async def user_top_genres(pool, user_id: int) -> set:
    """The user's top genre cluster NAMES (ids → names across both
    per-medium lists). Degrades to an empty set — exploration skips."""
    try:
        ids = await top_user_genre_ids(pool, user_id, 3)
        if not ids:
            return set()
        return await genre_names_for_ids(pool, ids)
    except Exception:
        return set()


async def apply_exploration(
    pool,
    results: List[dict],
    user_id: int,
    movie_ratio: float,
    limit: int,
    share: float,
    tilt_ctx: dict,
) -> List[dict]:
    """Deterministic discovery slice (RFC §2): reserve share·limit
    downstream slots for adjacent-cluster candidates (novel AND
    relevant — see exploration_slice), then rebalance the main
    portion to the freed size and re-insert the picks at fixed
    slots. Degrades to plain balancing whenever the relevance signal
    is missing — exploration without relevance is noise."""
    user_genres = await user_top_genres(pool, user_id)
    main, picks = exploration_slice(
        results, limit, user_genres,
        tilt_ctx.get("affinities") or {}, share)
    # _balance_by_media_type stays with the find-similar flow in
    # service.py until its own 5a cut — read at CALL time (service
    # imports this module first; a module-level import would cycle)
    from .service import _balance_by_media_type

    if not picks:
        return _balance_by_media_type(results, movie_ratio, limit)
    main = _balance_by_media_type(main, movie_ratio, limit - len(picks))
    return insert_exploration(main, picks, limit)


async def merged_exclusions(
    pool, exclude: Optional[set], user_id: Optional[int]
) -> Optional[set]:
    """Explicit exclude ∪ the user's watched/rated pairs (when both given).

    A watched-history lookup failure must not wipe an explicit exclude
    set — degrade to what the caller passed."""
    if user_id is None or pool is None:
        return exclude
    merged = set(exclude or set())
    try:
        from .watched import (
            get_user_dislikes,
            get_user_recommendation_exclusions,
        )

        merged |= await get_user_recommendation_exclusions(pool, user_id)
        merged |= await get_user_dislikes(pool, user_id)
    except Exception as e:
        logger.debug(f"user exclusions unavailable for {user_id}: {e}")
    return merged


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine with explicit norms — catalog embeddings are not guaranteed
    unit-length (pgvector's <=> normalizes internally; a raw dot doesn't)."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def tilt_scores(
    results: List[dict],
    affinities: dict,
    alpha: float,
) -> List[dict]:
    """User-taste tilt: score' = score · (1 + α · max(cos, 0)).

    Multiplicative and boost-only, so the seed's relevance stays primary
    (a 2× better candidate can never be overtaken; negative affinity
    never buries). Re-packages dicts — the shared cached payload is
    never mutated. Candidates without an affinity pass through
    untouched, in place.
    """
    out = []
    for r in results:
        cos = affinities.get((r["tmdb_id"], r.get("media_type")))
        if cos is None:
            out.append(r)
            continue
        boost = max(cos, 0.0)
        out.append({
            **r,
            "score": r["score"] * (1.0 + alpha * boost),
            "user_affinity": round(boost, 3),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def exploration_slice(
    results: List[dict],
    limit: int,
    user_genres: set,
    affinities: dict,
    share: float,
) -> Tuple[List[dict], List[dict]]:
    """Split results into (main, discovery_picks) — RFC §2.

    An eligible pick is NOVEL (genre set disjoint from the user's top
    clusters) AND RELEVANT (affinity ≥ the main list's median) —
    serendipity, never random noise. Highest-scoring eligible first,
    capped at ceil(share·limit) and never intruding on the top-3.
    Pure and deterministic: same inputs, same slice."""
    if share <= 0 or not results or limit is None or not user_genres \
            or not affinities:
        return results, []
    n = min(max(1, math.ceil(share * limit)), max(0, limit - 3))
    if n <= 0:
        return results, []

    def _aff(r):
        return affinities.get((r["tmdb_id"], r.get("media_type")))

    main_affs = sorted(
        a for a in (_aff(r) for r in results[:limit]) if a is not None)
    if not main_affs:
        return results, []
    median = main_affs[len(main_affs) // 2]

    picks: List[dict] = []
    picked: set = set()
    for r in results:  # score order
        if len(picks) >= n:
            break
        key = (r["tmdb_id"], r.get("media_type"))
        if key in picked:
            continue
        rg = r.get("genres") or []
        if not rg or (user_genres & set(rg)):
            continue  # not novel (or unclassifiable)
        aff = _aff(r)
        if aff is None or aff < median:
            continue  # not relevant
        picks.append(r)
        picked.add(key)
    if not picks:
        return results, []
    main = [r for r in results
            if (r["tmdb_id"], r.get("media_type")) not in picked]
    return main, picks


def insert_exploration(
    main: List[dict], picks: List[dict], limit: int,
) -> List[dict]:
    """Re-insert discovery picks at fixed downstream slots (top-3
    pinned): slot_i = 3 + (i+1)·(limit−3)//(len(picks)+1)."""
    out = list(main)
    for i, p in enumerate(picks):
        pos = min(3 + ((i + 1) * (limit - 3)) // (len(picks) + 1), len(out))
        out.insert(pos, p)
    return out
