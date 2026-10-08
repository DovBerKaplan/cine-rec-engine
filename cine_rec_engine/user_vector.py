"""User vector construction (spec v0.2 §E).

    u_raw = Σ w_i · v_i          (only items with w_i > 0 AND an embedding)
    u     = u_raw / ‖u_raw‖₂

Items without an embedding are skipped for the vector (they still feed the
SQL recall). No weighted items → NO vector at all — never a fake zero
vector for search. Pure math (aggregate) is split from the DB plumbing so
it is testable offline. Multi-persona (§E.2) is deliberately not built
yet — the spec defers K-Means until real smearing shows up in data; the
storage (user_vectors.persona_id) is ready for it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import asyncpg
from loguru import logger

from .config import USER_VECTOR_MAX_AGE_HOURS
from .model_spaces import column_for


def _as_list(v) -> list:
    """Normalize a pgvector cell to a plain list.

    With the asyncpg codec registered the value arrives as pgvector.Vector
    (no __iter__ — `list(v)` raises TypeError); other callers (offline
    tests) may hand us a list/tuple already.
    """
    if isinstance(v, (list, tuple)):
        return list(v)
    return v.to_list()


def aggregate(
    vectors: Sequence[Sequence[float]], weights: Sequence[float]
) -> Optional[List[float]]:
    """Weighted sum + L2 normalize. None when nothing weighs in.

    Raises ValueError on dimension mismatch — a mixed-dimension sum is a
    configuration error (two encoders in one space), not silence.
    """
    if not vectors or not weights:
        return None
    dim = len(vectors[0])
    acc = [0.0] * dim
    total_w = 0.0
    for vec, w in zip(vectors, weights):
        if w <= 0:
            continue
        if len(vec) != dim:
            raise ValueError(
                f"embedding dimension mismatch: {len(vec)} vs {dim}"
            )
        for i, x in enumerate(vec):
            acc[i] += w * x
        total_w += w
    if total_w <= 0:
        return None
    norm = math.sqrt(sum(x * x for x in acc))
    if norm == 0.0:
        return None
    return [x / norm for x in acc]


async def build_user_vector(
    pool: asyncpg.Pool,
    user_id: int,
    space: Optional[str] = None,
    column: Optional[str] = None,
) -> Optional[List[float]]:
    """Rebuild and store the user's vector (persona 0). None if not buildable.

    Reads w_i > 0 items from user_title_stats joined to the embedding
    column on the media view; dislikes and dropped items weigh 0 and never
    reach the sum. Updates user_stats.vector_updated_at + persona_count.
    """
    # call-time config read (not a module-level from-import): tests and
    # the smoke scripts switch the embedding space at runtime, and a
    # stale copy silently builds from a column that may not exist
    from .config import EMBEDDING_COLUMN

    emb_col = column or EMBEDDING_COLUMN
    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        logger.debug("user vector: pgvector missing — none built")
        return None

    try:
        async with pool.acquire() as conn:
            try:
                await register_vector(conn)
            except Exception:
                return None  # no vector infrastructure — no user vector
            rows = await conn.fetch(
                f"""
                SELECT uts.w_item, m.emb
                FROM user_title_stats uts
                JOIN (
                    SELECT id, 'movie'::text AS media_type, {emb_col} AS emb
                    FROM tmdb_movies WHERE {emb_col} IS NOT NULL
                    UNION ALL
                    SELECT id, 'tv'::text, {emb_col}
                    FROM tmdb_tv WHERE {emb_col} IS NOT NULL
                ) m ON m.id = uts.tmdb_id AND m.media_type = uts.media_type
                WHERE uts.user_id = $1 AND uts.w_item > 0
                ORDER BY uts.w_item DESC
                """,
                user_id,
            )
    except asyncpg.UndefinedColumnError:
        logger.debug(f"user vector: embedding column {emb_col} missing — none built")
        return None

    vectors = [_as_list(r["emb"]) for r in rows]
    weights = [float(r["w_item"]) for r in rows]
    vec = aggregate(vectors, weights)

    async with pool.acquire() as conn:
        if vec is None:
            await conn.execute(
                "DELETE FROM user_vectors WHERE user_id = $1 AND persona_id = 0",
                user_id,
            )
            await conn.execute(
                """UPDATE user_stats
                   SET vector_updated_at = now(), persona_count = 0
                   WHERE user_id = $1""",
                user_id,
            )
            return None

        await register_vector(conn)
        await conn.execute(
            """
            INSERT INTO user_vectors (user_id, persona_id, space, embedding,
                                      weight, updated_at)
            VALUES ($1, 0, $2, $3, 1.0, now())
            ON CONFLICT (user_id, persona_id, space) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                weight = 1.0,
                updated_at = now()
            """,
            user_id, space or "default", vec,
        )
        await conn.execute(
            """UPDATE user_stats
               SET vector_updated_at = now(), persona_count = 1
               WHERE user_id = $1""",
            user_id,
        )
    return vec


async def ensure_user_vector(
    pool: asyncpg.Pool,
    user_id: int,
    space: Optional[str] = None,
    column: Optional[str] = None,
    max_age_hours: float = USER_VECTOR_MAX_AGE_HOURS,
) -> Optional[List[float]]:
    """The stored vector if fresh; rebuild it first when absent or stale.

    The plug-and-play path: recommend_for_user calls this so the ANN
    channel works without the nightly cron. Costs one probe query per
    request and at most one rebuild per max_age window per user
    (concurrent probes may race — both rebuild, the upsert is idempotent).

    A registered-model space (model_spaces.REC_MODELS) rebuilds from its
    OWN column — without the resolution, a solo space missing its vector
    would silently rebuild from the default column under the wrong label.

    Degrades to None on any error (missing table, no pgvector, no
    embedding column) — never raises on the request path.
    """
    col = column or column_for(space)
    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        logger.debug("user vector: pgvector missing — none ensured")
        return None
    try:
        async with pool.acquire() as conn:
            await register_vector(conn)
            row = await conn.fetchrow(
                """SELECT embedding FROM user_vectors
                   WHERE user_id = $1 AND persona_id = 0
                     AND space = COALESCE($2, space)
                     AND updated_at > now() - ($3 || ' hours')::interval""",
                user_id, space, str(max_age_hours),
            )
        if row is not None and row["embedding"] is not None:
            return _as_list(row["embedding"])
    except Exception as e:
        # A failed probe means the vector layer is unusable (missing
        # table, missing extension) — a rebuild would fail the same way.
        logger.debug(f"ensure_user_vector probe failed for {user_id}: {e}")
        return None
    try:
        return await build_user_vector(pool, user_id, space, col)
    except Exception as e:
        logger.debug(f"user vector rebuild failed for {user_id}: {e}")
        return None


async def load_user_vector(
    pool: asyncpg.Pool, user_id: int, space: Optional[str] = None
) -> Optional[List[float]]:
    """Fetch the stored vector for retrieval (None when absent)."""
    from pgvector.asyncpg import register_vector

    async with pool.acquire() as conn:
        await register_vector(conn)
        row = await conn.fetchrow(
            """SELECT embedding FROM user_vectors
               WHERE user_id = $1 AND persona_id = 0
                 AND space = COALESCE($2, space)""",
            user_id, space,
        )
        if row is None or row["embedding"] is None:
            return None
        return _as_list(row["embedding"])


async def top_weighted_seeds(
    pool: asyncpg.Pool, user_id: int, limit: int = 20
) -> List[Tuple[int, str, float]]:
    """Highest-w_i (id, media_type, w) tuples — the LTR seed set (§F.1)."""
    rows = await pool.fetch(
        """SELECT tmdb_id, media_type, w_item FROM user_title_stats
           WHERE user_id = $1 AND w_item > 0
           ORDER BY w_item DESC, last_watched_at DESC LIMIT $2""",
        user_id, limit,
    )
    return [(r["tmdb_id"], r["media_type"], float(r["w_item"])) for r in rows]
