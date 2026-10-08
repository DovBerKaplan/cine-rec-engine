"""Pure scoring functions for the recommendation engine.

Pure similarity functions —
  - tmdb.py: sentence_similarity (line 1275), detect_style (line 1293)
  - performance_optimizer.py: fast_keyword_similarity (line 115),
    fast_cast_similarity (line 126), optimize_candidate_selection (line 164)

All functions are synchronous with no I/O dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .config import KEYWORDS_STYLE, MIN_VOTE_COUNT, VOTE_FLOOR_TV


def bigram_set(text: str) -> set:
    """Character bigrams of the lowercased string (cached by callers)."""
    low = text.lower()
    return {low[i : i + 2] for i in range(len(low) - 1)}


def sentence_similarity_legacy(a: str, b: str) -> float:
    """Character-bigram Jaccard similarity.

    Much faster than SequenceMatcher while still capturing substring
    overlap. Falls back to 0.0 for empty inputs.

    Retained as a fallback for when pgvector embeddings are unavailable.
    """
    if not a or not b:
        return 0.0
    bigrams_a = bigram_set(a)
    bigrams_b = bigram_set(b)
    inter = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    return inter / union if union else 0.0


# Backwards-compat alias — older code/import sites still work.
sentence_similarity = sentence_similarity_legacy


def detect_style(keywords: List[str]) -> List[str]:
    """Map keyword lists to style labels.

    Returns style labels whose keyword sets overlap with the input.
    """
    if not keywords:
        return []
    return [style for style, keys in KEYWORDS_STYLE.items() if set(keys) & set(keywords)]


def canonical_genres(genres: List[str]) -> set:
    """Translate local Hebrew genre labels to canonical English names.

    The local catalog stores Hebrew labels ('מותחן') while GENRE_PRIORITY
    and style matching use English — without this, genre weighting is dead
    code on real data.
    """
    from .config import GENRE_LABELS_EN

    return {GENRE_LABELS_EN.get(str(g).strip(), str(g).strip()) for g in genres or []}


# ── Mood/tone scoring (2026-09 user feedback) ──────────────────────

LIGHT_GENRES = {"Comedy", "Family", "Music", "Romance", "Soap", "Kids", "Animation"}
DARK_GENRES = {"Horror", "Thriller", "Mystery", "Crime"}
LIGHT_ARC_TAGS = {"feel-good", "heartwarming", "uplifting", "hopeful", "cozy", "feel-good"}
DARK_ARC_TAGS = {"dark", "gritty", "bleak", "disturbing", "tense", "suspenseful"}

# Youth/audience scoring (teen telenovela
# was recommended alongside "Marvelous Mrs. Maisel" TV-MA adult drama)
YOUTH_GENRES = {"Kids", "Family", "Animation", "Soap"}
ADULT_LEANING_GENRES = {"Crime", "Horror", "Thriller"}


def audience_score(
    seed_genres: List[str],
    cand_genres: List[str],
    seed_adult: bool = False,
    cand_adult: bool = False,
    seed_canonical: Optional[set] = None,
    cand_canonical: Optional[set] = None,
) -> float:
    """Are these two titles for the same target audience?

    1.0 = same audience; 0.0 = youth seed vs adult candidate; 0.5 = neutral.
    ``*_canonical`` lets hot loops pass precomputed canonical genre sets.
    """
    seed_g = seed_canonical if seed_canonical is not None else canonical_genres(seed_genres or [])
    cand_g = cand_canonical if cand_canonical is not None else canonical_genres(cand_genres or [])
    seed_y = bool(seed_g & YOUTH_GENRES)
    cand_y = bool(cand_g & YOUTH_GENRES)
    cand_adult_signal = cand_adult or (
        not cand_y and bool(cand_g & {"Crime", "Horror", "Thriller"})
    )

    if seed_y and cand_adult_signal:
        return 0.0
    if seed_y == cand_y:
        return 1.0
    return 0.5


def tone_score(
    seed_genres: List[str],
    cand_genres: List[str],
    seed_arc: Optional[List[str]] = None,
    cand_arc: Optional[List[str]] = None,
    seed_canonical: Optional[set] = None,
    cand_canonical: Optional[set] = None,
) -> float:
    """Are these two titles in the same emotional register?

    1.0 = both light or both dark; 0.0 = opposite; 0.2 = strongly one-sided
    (seed clearly light + candidate has ZERO light signal); 0.5 = unknown.
    Derived from canonical genres + emotional_arc tags — plot keywords
    can overlap while the vibe is inverted.
    Genres in NEITHER the light nor dark set are treated as unknown,
    not neutral agreement.
    """
    seed_g = seed_canonical if seed_canonical is not None else canonical_genres(seed_genres or [])
    cand_g = cand_canonical if cand_canonical is not None else canonical_genres(cand_genres or [])

    seed_light_count = len(seed_g & LIGHT_GENRES)
    cand_light_count = len(cand_g & LIGHT_GENRES)
    seed_dark_count = len(seed_g & DARK_GENRES)
    cand_dark_count = len(cand_g & DARK_GENRES)

    seed_light = seed_light_count > 0
    seed_dark = seed_dark_count > 0
    cand_light = cand_light_count > 0
    cand_dark = cand_dark_count > 0

    # Emotional arc tags refine when genres alone are ambiguous
    if seed_arc:
        seed_light = seed_light or bool(set(t.lower() for t in seed_arc) & LIGHT_ARC_TAGS)
        seed_dark = seed_dark or bool(set(t.lower() for t in seed_arc) & DARK_ARC_TAGS)
    if cand_arc:
        cand_light = cand_light or bool(set(t.lower() for t in cand_arc) & LIGHT_ARC_TAGS)
        cand_dark = cand_dark or bool(set(t.lower() for t in cand_arc) & DARK_ARC_TAGS)

    # Both in the same register
    if (seed_light and cand_light) or (seed_dark and cand_dark):
        return 1.0
    # Opposite registers (explicit light vs dark)
    if (seed_light and cand_dark) or (seed_dark and cand_light):
        return 0.0
    # Strongly one-sided: seed is CLEARLY light (2+ genres) but candidate
    # has ZERO light signal → penalize. A Family+Comedy+Music title
    # vs "His Dark Materials" (Fantasy+Drama) = 0.2, not 0.5.
    if seed_light_count >= 2 and cand_light_count == 0:
        return 0.2
    if seed_dark_count >= 2 and cand_dark_count == 0:
        return 0.2
    # Unknown or neutral
    return 0.5


def emotional_arc_jaccard(seed_arc: Optional[List[str]], cand_arc: Optional[List[str]]) -> float:
    """Jaccard overlap of emotional_arc tag lists (feel-good, tense...)."""
    if not seed_arc or not cand_arc:
        return 0.0
    s, c = set(t.lower().strip() for t in seed_arc), set(t.lower().strip() for t in cand_arc)
    if not s or not c:
        return 0.0
    return len(s & c) / len(s | c)


# Meta-tag stoplist: TMDB keywords that describe PRODUCTION facts, not
# themes. "based on novel or book" (1,958 uses), "sequel", "woman
# director", "duringcreditsstinger"... carry zero taste signal but massive
# cross-genre pull — every sequel "resembling" every other sequel is noise
# that inflates keyword similarity between unrelated titles (quantified
# 2026-08-20; the same noise is the prime suspect in the enrichment
# production regression).
META_KEYWORD_STOPWORDS = frozenset(
    {
        "based on novel or book",
        "based on book",
        "based on novel",
        "based on play",
        "based on true story",
        "based on comic",
        "based on manga",
        "based on video game",
        "based on tv series",
        "woman director",
        "sequel",
        "prequel",
        "remake",
        "spin off",
        "duringcreditsstinger",
        "aftercreditsstinger",
        "short film",
        "biography",
        "silent film",
        "3d",
        "imax",
        "anime",  # medium, not theme — genre already carries it
        "adaptation",
        "reboot",
    }
)


def keyword_set(keywords: List[str]) -> set:
    """Filtered keyword set (meta-tags removed) — cached by callers."""
    return {k for k in keywords if k not in META_KEYWORD_STOPWORDS}


def fast_keyword_similarity(keywords1: List[str], keywords2: List[str]) -> float:
    """Jaccard similarity on keyword sets, meta-tags filtered out.

    intersection / union — a single shared generic tag no longer scores as a
    perfect match (the old overlap-coefficient divided by min(|A|,|B|), so a
    one-keyword candidate matching any seed keyword got 1.0).
    """
    if not keywords1 or not keywords2:
        return 0.0

    set1 = keyword_set(keywords1)
    set2 = keyword_set(keywords2)
    if not set1 or not set2:
        return 0.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


def fast_cast_similarity(cast1: List[str], cast2: List[str]) -> float:
    """Dice coefficient over the top 5 cast members.

    2*intersection / (|A|+|B|) — one shared actor out of a full top-5 pair
    now scores 0.2 instead of the old 1.0 (min-denominator inflation).
    """
    if not cast1 or not cast2:
        return 0.0

    set1, set2 = set(cast1[:5]), set(cast2[:5])  # Top 5 cast members
    total = len(set1) + len(set2)
    return (2.0 * len(set1 & set2)) / total if total else 0.0


def company_similarity(companies1: List[str], companies2: List[str]) -> float:
    """Jaccard similarity for production companies.

    A shared studio (Warner Bros appears in half the catalog) must not read
    as a strong signal — union-based similarity keeps it proportional.
    """
    if not companies1 or not companies2:
        return 0.0

    set1, set2 = set(companies1), set(companies2)
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


def optimize_candidate_selection(
    candidates: List[dict],
    base_genres: List[str],
    target_count: int = 15,
) -> List[dict]:
    """Pre-filter candidates before expensive scoring.

    Filters:
    - Must share at least one genre with the seed — checked against the FULL
      genre lists on both sides. The old [:3] slices dropped candidates that
      matched the seed's 4th+ genre (e.g. Mystery thrillers under an
      Action/Sci-Fi-first seed like Inception).
    - Rating >= 5.5

    Candidates carrying a `via` tag — `"knn"` (pgvector semantic neighbors,
    kept in semantic-distance order) or `"tmdb"` (behavioral recommendations
    synced from TMDB, kept in TMDB rank order) — are exempt from the genre
    gate AND the vote floor: their proximity signal (TMDB's own behavioral
    engine ranked them; the ANN index returned them) is their qualification,
    and the graph head often carries mid-vote titles the local floor would
    silently drop. They still pass the rating floor.
    Returns up to target_count * 2 candidates (~70% curated intake).
    """
    if not candidates:
        return []

    curated_pool: List[dict] = []
    genre_pool: List[dict] = []
    base_genre_set = set(base_genres)
    for cand in candidates:
        # Rating floor (applies to every recall path).
        if (cand.get("vote_average") or 0) < 5.5:
            continue

        if cand.get("via"):
            curated_pool.append(cand)  # keep the channel's own ordering
            continue

        # Vote-count floor for the genre channel. TV vote counts run an
        # order of magnitude below movies on TMDB, so series candidates
        # get their own floor — a flat 500 starves tv seeds.
        vote_floor = VOTE_FLOOR_TV if cand.get("media_type") == "tv" else MIN_VOTE_COUNT
        if (cand.get("vote_count") or 0) < vote_floor:
            continue

        cand_genres = cand.get("genres", [])

        # Genre overlap against the FULL seed genre list
        if base_genre_set and cand_genres and not (base_genre_set & set(cand_genres)):
            continue

        genre_pool.append(cand)

    def sort_score(movie: dict) -> float:
        # Blended, normalized ranking: rating leads, popularity mildly
        # matters, and raw vote counts enter on a log scale so thousands of
        # votes no longer dominate a 1-10 rating scale. (The old tuple key
        # sorted lexicographically — effectively by rating alone.)
        vote_avg = (movie.get("vote_average") or 0.0) / 10.0
        pop = min((movie.get("popularity") or 0.0) / 100.0, 1.0)
        vote_cnt = min(math.log10((movie.get("vote_count") or 1) + 1) / 5.0, 1.0)
        return (vote_avg * 0.5) + (pop * 0.2) + (vote_cnt * 0.3)

    genre_pool.sort(key=sort_score, reverse=True)

    # Curated (semantic/behavioral) candidates get the lion's share of the
    # intake; the genre channel backfills the rest.
    curated_limit = int(target_count * 0.7)
    combined = curated_pool[:curated_limit] + genre_pool
    return combined[: target_count * 2]  # Double target for safety


def _fact_emb_union(cols_sql: str) -> str:
    """id + embedding columns from the fact tables, both media.

    The output ALWAYS starts with `id` — pass only the embedding columns;
    prefixing 'id, ' yourself duplicates the output column and makes
    every outer reference to it ambiguous (AmbiguousColumnError).

    The embedding columns live on tmdb_movies/tmdb_tv — the tmdb_media
    compatibility view does not expose columns added after its CREATE,
    so reading them through the view silently fails on any schema
    upgraded in place.
    """
    return (
        f"(SELECT id, {cols_sql} FROM tmdb_movies "
        f"UNION ALL SELECT id, {cols_sql} FROM tmdb_tv)"
    )


async def semantic_overview_similarity(
    pool,
    seed_id: int,
    candidate_ids: List[int],
    columns: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    seed_vectors: Optional[Dict[str, list]] = None,
) -> Dict[int, float]:
    """Cosine similarity between the seed's overview embedding and each
    candidate's embedding, via pgvector.

    Uses `<=>` (cosine distance) on the precomputed embedding columns of
    the fact tables. Returns a dict mapping `candidate_id -> similarity`
    in the range [0.0, 1.0]. Entries for candidates without an embedding
    are omitted.

    Args:
        pool: asyncpg connection pool (the same one shared with
            PostgresTMDBService). The pgvector Python package must be
            registered on the connection's codec via
            `pgvector.asyncpg.register_vector` before this query runs;
            we register lazily here so callers don't need to.
        seed_id: TMDB id whose embedding acts as the reference vector.
        candidate_ids: TMDB ids to score against the seed.
        columns: per-request column override (/set_model). When given
            together with `weights`, they replace the global COSINE_BLEND
            entirely — a solo model passes [(column, 1.0)].
        weights: weights matching `columns`, normalized internally.
        seed_vectors: optional per-column override for the SEED's vectors
            (column -> plain float list). Fills columns the DB row lacks —
            e.g. a freshly enriched seed whose embeddings were computed
            on demand and cached in Redis. Columns present in the DB row
            keep their stored values; overrides win where given.

    Returns:
        Dict[int, float] — possibly empty if the seed or all candidates
        lack embeddings, or if the `embedding` column is missing.
    """
    if not candidate_ids:
        return {}

    import asyncpg

    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return {}  # optional dependency — cosine channel off without it

    async with pool.acquire() as conn:
        # Register the vector codec on this connection so asyncpg can
        # deserialize `vector` columns. Idempotent and cheap.
        try:
            await register_vector(conn)
        except Exception:
            # Extension/column may not exist yet — graceful degradation.
            return {}

        from .config import COSINE_BLEND, EMBEDDING_COLUMN

        # Column selection order: explicit per-request override (a solo
        # /set_model choice) → the global REC_COSINE_BLEND (the tuned
        # ensemble rescore) → the single global embedding column. Weights
        # are normalized to 1 over the active (non-NULL seed) columns.
        if columns and weights:
            active_spec = list(zip(columns, weights))
        elif COSINE_BLEND:
            active_spec = list(COSINE_BLEND.items())
        else:
            active_spec = [(EMBEDDING_COLUMN, 1.0)]
        cols = [c for c, _ in active_spec]
        col_weights = [w for _, w in active_spec]

        seed_cols = ", ".join(cols)
        # Pull the seed embedding(s). NULL or missing -> no semantic signal
        # for that column — unless an on-demand override fills it below.
        try:
            seed_row = await conn.fetchrow(
                f"SELECT {seed_cols} FROM {_fact_emb_union(seed_cols)} m"
                f" WHERE id = $1 LIMIT 1",
                seed_id,
            )
        except asyncpg.UndefinedColumnError:
            if not seed_vectors:
                return {}
            seed_row = None
        except asyncpg.UndefinedObjectError:
            # `vector` type not available — extension not installed.
            return {}

        seed_vecs = {c: (seed_row[c] if seed_row else None) for c in cols}
        if seed_vectors:
            import numpy as _np

            for c in cols:
                override = seed_vectors.get(c)
                if override:
                    seed_vecs[c] = _np.asarray(override, dtype=_np.float32)
        if all(v is None for v in seed_vecs.values()):
            return {}

        select_sims = ", ".join(
            f"GREATEST(0.0, 1 - ({c} <=> ${i + 1}::vector)) AS sim_{i}"
            for i, c in enumerate(cols)
            if seed_vecs[c] is not None
        )
        active_cols = [c for c in cols if seed_vecs[c] is not None]
        active_w = [w for c, w in zip(cols, col_weights) if seed_vecs[c] is not None]
        params = [seed_vecs[c] for c in active_cols]
        params.append(list(candidate_ids))

        try:
            rows = await conn.fetch(
                f"""
                SELECT m.id, {select_sims}
                FROM {_fact_emb_union(", ".join(active_cols))} m
                WHERE m.id = ANY(${len(active_cols) + 1}::bigint[])
                  AND {" OR ".join(f"m.{c} IS NOT NULL" for c in active_cols)}
                """,
                *params,
            )
        except (asyncpg.UndefinedColumnError, asyncpg.UndefinedObjectError):
            return {}

    out = {}
    for row in rows:
        sims = [float(row[f"sim_{i}"]) for i in range(len(active_cols))]
        if any(s is None for s in sims):
            continue
        w_sum = sum(active_w)
        out[row["id"]] = sum(w * s for w, s in zip(active_w, sims)) / max(w_sum, 1e-9)
    return out


async def semantic_overview_similarity_batch(
    pool,
    seed_ids: List[int],
    candidate_ids: List[int],
    columns: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    seed_vectors: Optional[Dict[int, Dict[str, list]]] = None,
) -> Dict[int, Dict[int, float]]:
    """semantic_overview_similarity for ALL seeds in one round trip per
    blend column instead of one per seed — multi-seed requests (the
    personalized row) fire 15–20 of those otherwise.

    Same blend math and the same skip rule as the single-seed version: a
    candidate needs every column that is active for the seed (a NULL
    similarity drops the pair). Returns {seed_id: {candidate_id: sim}};
    seeds with no usable vector are omitted.
    """
    if not candidate_ids or not seed_ids:
        return {}

    import asyncpg

    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return {}  # optional dependency — cosine channel off without it

    async with pool.acquire() as conn:
        try:
            await register_vector(conn)
        except Exception:
            return {}

        from .config import COSINE_BLEND, EMBEDDING_COLUMN

        if columns and weights:
            active_spec = list(zip(columns, weights))
        elif COSINE_BLEND:
            active_spec = list(COSINE_BLEND.items())
        else:
            active_spec = [(EMBEDDING_COLUMN, 1.0)]
        cols = [c for c, _ in active_spec]
        col_weights = dict(active_spec)

        # All seed vectors in one query (last row wins on a colliding
        # movie/tv id — same tie behavior as the single-seed path).
        try:
            seed_rows = await conn.fetch(
                f"SELECT id, {', '.join(cols)} FROM {_fact_emb_union(', '.join(cols))} m"
                f" WHERE id = ANY($1::bigint[])",
                list(seed_ids),
            )
        except (asyncpg.UndefinedColumnError, asyncpg.UndefinedObjectError):
            return {}

        seed_vecs: Dict[int, Dict[str, object]] = {}
        for row in seed_rows:
            seed_vecs[row["id"]] = {c: row[c] for c in cols}
        if seed_vectors:
            for sid, overrides in seed_vectors.items():
                if overrides:
                    merged = dict(seed_vecs.get(sid, {c: None for c in cols}))
                    merged.update({c: v for c, v in overrides.items() if v})
                    seed_vecs[sid] = merged

        usable = {
            sid: vecs for sid, vecs in seed_vecs.items()
            if any(vecs.get(c) is not None for c in cols)
        }
        if not usable:
            return {}

        # One cross-join query per column with at least one usable seed:
        # seeds × candidates cosine pairs computed server-side.
        sims_by_col: Dict[str, Dict[Tuple[int, int], float]] = {}
        for col in cols:
            if not any(vecs.get(col) is not None for vecs in usable.values()):
                continue
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT s.id AS seed_id, c.id AS cand_id,
                           GREATEST(0.0, 1 - (c.{col} <=> s.{col})) AS sim
                    FROM {_fact_emb_union(col)} s
                    CROSS JOIN {_fact_emb_union(col)} c
                    WHERE s.id = ANY($1::bigint[]) AND c.id = ANY($2::bigint[])
                      AND s.{col} IS NOT NULL AND c.{col} IS NOT NULL
                    """,
                    list(usable.keys()), list(candidate_ids),
                )
            except (asyncpg.UndefinedColumnError, asyncpg.UndefinedObjectError):
                return {}
            sims_by_col[col] = {
                (r["seed_id"], r["cand_id"]): float(r["sim"]) for r in rows
            }

        out: Dict[int, Dict[int, float]] = {}
        for sid, vecs in usable.items():
            active_cols = [c for c in cols if vecs.get(c) is not None]
            w_sum = sum(col_weights[c] for c in active_cols)
            col_maps = [sims_by_col.get(c, {}) for c in active_cols]
            per_cand: Dict[int, float] = {}
            cand_ids_seen = {
                cid for m in col_maps for (_s, cid) in m if _s == sid
            }
            for cid in cand_ids_seen:
                pair_sims = [m.get((sid, cid)) for m in col_maps]
                if any(s is None for s in pair_sims):
                    continue  # candidate lacks one of the seed's columns
                per_cand[cid] = sum(
                    col_weights[c] * s
                    for c, s in zip(active_cols, pair_sims)
                ) / max(w_sum, 1e-9)
            if per_cand:
                out[sid] = per_cand
        return out
