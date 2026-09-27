"""PostgreSQL queries for the recommendation engine.

All candidate recall is SQL against your local mirror of TMDB data.
All functions are async and use asyncpg.
"""

from __future__ import annotations

from typing import List, Optional

import asyncpg
from loguru import logger

# ---------------------------------------------------------------------------
# Single combined query (replaces 7 round-trips with 1)
# ---------------------------------------------------------------------------

# SQL with keywords subquery
_QUERY_WITH_KEYWORDS = """
    SELECT
        m.id,
        m.title,
        m.title_en,
        m.media_type,
        m.overview,
        m.overview_en,
        m.original_language,
        m.vote_average::float as rating,
        m.poster_path,
        m.collection_id,
        m.adult,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
        ARRAY(
            SELECT g.name
            FROM tmdb_media_genres mg
            JOIN tmdb_genres g ON g.id = mg.genre_id
            WHERE mg.media_id = m.id
        ) as genres,
        (
            SELECT p.name
            FROM tmdb_crew c
            JOIN tmdb_people p ON c.person_id = p.id
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job IN ('Director', 'Creator')
            LIMIT 1
        ) as director,
        ARRAY(
            SELECT c.person_id
            FROM tmdb_crew c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job IN ('Director', 'Creator')
        ) as director_ids,
        ARRAY(
            SELECT c.person_id
            FROM tmdb_crew c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Writer'
        ) as writer_ids,
        ARRAY(
            SELECT c.person_id
            FROM tmdb_crew c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Original Music Composer'
        ) as composer_ids,
        ARRAY(
            SELECT c.person_id
            FROM tmdb_crew c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Director of Photography'
        ) as dp_ids,
        ARRAY(
            SELECT p.name
            FROM tmdb_cast c
            JOIN tmdb_people p ON c.person_id = p.id
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
            ORDER BY c.cast_order ASC NULLS LAST
            LIMIT 5
        ) as cast_list,
        ARRAY(
            SELECT c.person_id
            FROM tmdb_cast c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
            ORDER BY c.cast_order ASC NULLS LAST
            LIMIT 5
        ) as cast_ids,
        ARRAY(
            SELECT k.name
            FROM tmdb_media_keywords mk
            JOIN tmdb_keywords k ON k.id = mk.keyword_id
            WHERE mk.media_id = m.id
        ) as keywords,
        ARRAY(
            SELECT pc.name
            FROM tmdb_media_companies mc
            JOIN tmdb_production_companies pc ON mc.company_id = pc.id
            WHERE mc.media_id = m.id
        ) as companies,
        ARRAY(
            SELECT n.name
            FROM tmdb_media_networks mn
            JOIN tmdb_networks n ON n.id = mn.network_id
            WHERE mn.media_id = m.id
        ) as networks
    FROM tmdb_media m
    WHERE m.id = ANY($1::bigint[])
"""

# Same query without keywords (fallback when tmdb_keywords table missing)
_QUERY_NO_KEYWORDS = _QUERY_WITH_KEYWORDS.replace(
    """        ARRAY(
            SELECT k.name
            FROM tmdb_media_keywords mk
            JOIN tmdb_keywords k ON k.id = mk.keyword_id
            WHERE mk.media_id = m.id
        ) as keywords,""",
    "        ARRAY[]::text[] as keywords,",
)


def _row_to_movie_info(row: asyncpg.Record) -> dict:
    """Convert a combined-query row to the movie info dict."""
    return {
        "id": row["id"],
        "title": row["title"],
        "title_en": row.get("title_en"),
        "media_type": row["media_type"],
        "rating": float(row["rating"] or 0),
        "poster_path": row["poster_path"],
        "overview": row["overview"] or "",
        "original_language": row.get("original_language"),
        "overview_en": row.get("overview_en"),
        "genres": list(row["genres"] or []),
        "director": row["director"],
        "director_ids": list(row.get("director_ids") or []),
        "writer_ids": list(row.get("writer_ids") or []),
        "composer_ids": list(row.get("composer_ids") or []),
        "dp_ids": list(row.get("dp_ids") or []),
        "keywords": list(row["keywords"] or []),
        "cast_list": list(row["cast_list"] or []),
        "cast_ids": list(row.get("cast_ids") or []),
        "collection_id": row.get("collection_id"),
        "release_year": row.get("release_year"),
        "companies": list(row["companies"] or []),
        "networks": list(row["networks"] or []),
    }


async def get_movie_info_batch(
    pool: asyncpg.Pool,
    movie_ids: List[int],
    media_type: Optional[str] = None,
) -> dict[int, dict]:
    """Get full movie info for multiple IDs in a single SQL query.

    Replaces N × 7 individual queries with 1 combined query.
    Returns: dict mapping movie_id -> info dict (same shape as get_movie_info).
    """
    if not movie_ids:
        return {}

    async with pool.acquire() as conn:
        try:
            if media_type:
                rows = await conn.fetch(
                    _QUERY_WITH_KEYWORDS + " AND m.media_type = $2",
                    movie_ids,
                    media_type,
                )
            else:
                rows = await conn.fetch(_QUERY_WITH_KEYWORDS, movie_ids)
        except asyncpg.UndefinedTableError:
            # tmdb_keywords / tmdb_media_keywords tables don't exist — retry without
            logger.debug("tmdb_keywords table not found — retrying without keywords")
            if media_type:
                rows = await conn.fetch(
                    _QUERY_NO_KEYWORDS + " AND m.media_type = $2",
                    movie_ids,
                    media_type,
                )
            else:
                rows = await conn.fetch(_QUERY_NO_KEYWORDS, movie_ids)

    return {row["id"]: _row_to_movie_info(row) for row in rows}


async def get_movie_info(
    pool: asyncpg.Pool,
    movie_id: int,
    media_type: Optional[str] = None,
) -> Optional[dict]:
    """Get full movie info from PostgreSQL (single combined query).

    Replaces _get_movie_info_async from the original engine.
    Returns dict with: id, title, media_type, rating, poster_path,
                       overview, genres, director, keywords, cast_list,
                       companies, networks.
    """
    result = await get_movie_info_batch(pool, [movie_id], media_type)
    return result.get(movie_id)


async def _get_keywords(conn, media_id: int) -> list[str]:
    """Get keywords for a media item.

    Returns empty list if tmdb_keywords / tmdb_media_keywords tables
    don't exist in PostgreSQL (they may only exist in the MySQL DB).
    """
    try:
        rows = await conn.fetch(
            """
            SELECT k.name
            FROM tmdb_keywords k
            JOIN tmdb_media_keywords mk ON k.id = mk.keyword_id
            WHERE mk.media_id = $1
            """,
            media_id,
        )
        return [r["name"] for r in rows]
    except asyncpg.UndefinedTableError:
        logger.debug("tmdb_keywords table not found — keyword scoring disabled")
        return []


from .config import VOTE_FLOOR_MOVIE, VOTE_FLOOR_TV

QUALITY_SLICE = 40  # rating-ordered complement to the popularity slice


def _vote_floor(media_type: Optional[str]) -> int:
    return VOTE_FLOOR_TV if media_type == "tv" else VOTE_FLOOR_MOVIE


async def generate_candidates(
    pool: asyncpg.Pool,
    base_movie_id: int,
    base_genres: List[str],
    media_type: str,
    limit: int = 60,
    allow_cross_media: bool = False,
) -> List[dict]:
    """Generate candidate movies from local PostgreSQL.

    Replaces the TMDB /recommendations API call.
    Finds movies with same media_type and at least one overlapping genre,
    ordered by popularity then vote_average.

    Args:
        allow_cross_media: If True, candidates from both movies and series
            are included regardless of the seed's media_type.
    """
    if not base_genres:
        return []

    async with pool.acquire() as conn:
        if allow_cross_media:
            rows = await conn.fetch(
                """
                SELECT
                    m.id,
                    m.title,
                    m.title_en,
                    m.media_type,
                    m.overview,
                    m.overview_en,
                    m.vote_average::float,
                    m.vote_count::int,
                    m.popularity::float,
                    m.poster_path,
                    m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
                    ARRAY(
                        SELECT g.name
                        FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id
                    ) as genres
                FROM tmdb_media m
                WHERE m.id != $1
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE mg.media_id = m.id
                        AND g.name = ANY($2::text[])
                  )
                ORDER BY m.popularity DESC, m.vote_average DESC
                LIMIT $3
                """,
                base_movie_id,
                base_genres,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    m.id,
                    m.title,
                    m.title_en,
                    m.media_type,
                    m.overview,
                    m.overview_en,
                    m.vote_average::float,
                    m.vote_count::int,
                    m.popularity::float,
                    m.poster_path,
                    m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
                    ARRAY(
                        SELECT g.name
                        FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id
                    ) as genres
                FROM tmdb_media m
                WHERE m.media_type = $1
                  AND m.id != $2
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE mg.media_id = m.id
                        AND g.name = ANY($3::text[])
                  )
                ORDER BY m.popularity DESC, m.vote_average DESC
                LIMIT $4
                """,
                media_type,
                base_movie_id,
                base_genres,
                limit,
            )

        candidates = [dict(row) for row in rows]

        # Quality slice: rating-ordered among well-voted titles sharing a
        # genre — mid-tail gems (an 8.5-rated anime with modest popularity)
        # never make the popularity slice but belong in the shortlist.
        if allow_cross_media:
            q_rows = await conn.fetch(
                """
                SELECT m.id, m.title, m.title_en, m.media_type, m.overview,
                       m.overview_en, m.vote_average::float, m.vote_count::int,
                       m.popularity::float, m.poster_path, m.collection_id,
                       EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int
                           AS release_year,
                       ARRAY(
                           SELECT g.name FROM tmdb_genres g
                           JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                           WHERE mg.media_id = m.id
                       ) as genres
                FROM tmdb_media m
                WHERE m.id != $1 AND m.vote_average >= 7.0 AND m.vote_count >= $4
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE mg.media_id = m.id AND g.name = ANY($2::text[])
                  )
                ORDER BY m.vote_average DESC, m.vote_count DESC
                LIMIT $3
                """,
                base_movie_id,
                base_genres,
                QUALITY_SLICE,
                VOTE_FLOOR_MOVIE,  # cross-media slice spans both types
            )
        else:
            q_rows = await conn.fetch(
                """
                SELECT m.id, m.title, m.title_en, m.media_type, m.overview,
                       m.overview_en, m.vote_average::float, m.vote_count::int,
                       m.popularity::float, m.poster_path, m.collection_id,
                       EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int
                           AS release_year,
                       ARRAY(
                           SELECT g.name FROM tmdb_genres g
                           JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                           WHERE mg.media_id = m.id AND mg.media_type = m.media_type
                       ) as genres
                FROM tmdb_media m
                WHERE m.media_type = $1 AND m.id != $2
                  AND m.vote_average >= 7.0 AND m.vote_count >= $4
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE mg.media_id = m.id AND mg.media_type = m.media_type
                        AND g.name = ANY($3::text[])
                  )
                ORDER BY m.vote_average DESC, m.vote_count DESC
                LIMIT $5
                """,
                media_type,
                base_movie_id,
                base_genres,
                _vote_floor(media_type),
                QUALITY_SLICE,
            )

        seen = {c["id"] for c in candidates}
        for r in q_rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                candidates.append(dict(r))
        return candidates


async def enrich_candidates_batch(
    pool: asyncpg.Pool,
    candidate_ids: List[int],
    media_types: Optional[List[str]] = None,
) -> dict[int, dict]:
    """Fetch director, cast, keywords for multiple candidates at once.

    Replaces N individual _get_movie_info_async calls with 3 batch queries.
    Returns: dict mapping candidate_id -> {director, director_ids, cast_list,
    keywords, companies, networks, collection_id}.

    Args:
        media_types: Per-candidate media_type (aligned with candidate_ids).
            When provided, director matching is (id, media_type)-scoped so a
            movie and a tv title sharing a numeric id can't bleed crew data
            into each other. Falls back to id-only matching when omitted.
    """
    if not candidate_ids:
        return {}

    result: dict[int, dict] = {}
    async with pool.acquire() as conn:
        # Batch: directors — names for display/string-fallback plus the
        # person_id array the scoring layer matches on.
        if media_types is not None:
            director_rows = await conn.fetch(
                """
                SELECT c.media_id,
                       MIN(p.name) as director,
                       ARRAY_AGG(c.person_id) as director_ids
                FROM tmdb_crew c
                JOIN tmdb_people p ON c.person_id = p.id
                WHERE (c.media_id, c.media_type) IN (
                          SELECT * FROM unnest($1::bigint[], $2::text[])
                      )
                  AND c.job IN ('Director', 'Creator')
                GROUP BY c.media_id
                """,
                candidate_ids,
                media_types,
            )
        else:
            director_rows = await conn.fetch(
                """
                SELECT c.media_id,
                       MIN(p.name) as director,
                       ARRAY_AGG(c.person_id) as director_ids
                FROM tmdb_crew c
                JOIN tmdb_people p ON c.person_id = p.id
                WHERE c.media_id = ANY($1::bigint[])
                  AND c.job IN ('Director', 'Creator')
                GROUP BY c.media_id
                """,
                candidate_ids,
            )
        directors = {r["media_id"]: r["director"] for r in director_rows}
        director_ids_map = {r["media_id"]: list(r.get("director_ids") or []) for r in director_rows}

        # Batch: writers (person_ids for the writer-DNA match)
        if media_types is not None:
            writer_rows = await conn.fetch(
                """
                SELECT c.media_id, ARRAY_AGG(DISTINCT c.person_id) as writer_ids
                FROM tmdb_crew c
                WHERE (c.media_id, c.media_type) IN (
                          SELECT * FROM unnest($1::bigint[], $2::text[])
                      )
                  AND c.job = 'Writer'
                GROUP BY c.media_id
                """,
                candidate_ids,
                media_types,
            )
        else:
            writer_rows = await conn.fetch(
                """
                SELECT c.media_id, ARRAY_AGG(DISTINCT c.person_id) as writer_ids
                FROM tmdb_crew c
                WHERE c.media_id = ANY($1::bigint[])
                  AND c.job = 'Writer'
                GROUP BY c.media_id
                """,
                candidate_ids,
            )
        writer_ids_map = {r["media_id"]: list(r.get("writer_ids") or []) for r in writer_rows}

        # Batch: composer + cinematographer ids (cinematic DNA beyond the
        # director — the visual/audio language of a body of work)
        dna_rows = await conn.fetch(
            """
            SELECT c.media_id,
                   ARRAY_AGG(c.person_id) FILTER (WHERE c.job = 'Original Music Composer')
                       AS composer_ids,
                   ARRAY_AGG(c.person_id) FILTER (WHERE c.job = 'Director of Photography')
                       AS dp_ids
            FROM tmdb_crew c
            WHERE c.media_id = ANY($1::bigint[])
              AND c.job IN ('Original Music Composer', 'Director of Photography')
            GROUP BY c.media_id
            """,
            candidate_ids,
        )
        composer_map = {r["media_id"]: list(r.get("composer_ids") or []) for r in dna_rows}
        dp_map = {r["media_id"]: list(r.get("dp_ids") or []) for r in dna_rows}

        # Batch: cast (top 5 per movie, ordered) — names for display,
        # person_ids for transliteration-proof similarity.
        cast_rows = await conn.fetch(
            """
            SELECT c.media_id, p.name, c.person_id
            FROM tmdb_cast c
            JOIN tmdb_people p ON c.person_id = p.id
            WHERE c.media_id = ANY($1::bigint[])
            ORDER BY c.cast_order ASC NULLS LAST
            """,
            candidate_ids,
        )
        cast_map: dict[int, list[str]] = {}
        cast_ids_map: dict[int, list[int]] = {}
        for r in cast_rows:
            mid = r["media_id"]
            if len(cast_map.get(mid, [])) < 5:
                cast_map.setdefault(mid, []).append(r["name"])
                cast_ids_map.setdefault(mid, []).append(r["person_id"])

        # Batch: keywords (graceful degradation if tables missing)
        keywords_map: dict[int, list[str]] = {}
        try:
            kw_rows = await conn.fetch(
                """
                SELECT mk.media_id, k.name
                FROM tmdb_media_keywords mk
                JOIN tmdb_keywords k ON mk.keyword_id = k.id
                WHERE mk.media_id = ANY($1::bigint[])
                """,
                candidate_ids,
            )
            for r in kw_rows:
                keywords_map.setdefault(r["media_id"], [])
                keywords_map[r["media_id"]].append(r["name"])
        except asyncpg.UndefinedTableError:
            logger.debug("tmdb_keywords table not found — keyword scoring disabled")

        # Batch: production companies
        company_map: dict[int, list[str]] = {}
        comp_rows = await conn.fetch(
            """
            SELECT mc.media_id, pc.name
            FROM tmdb_media_companies mc
            JOIN tmdb_production_companies pc ON mc.company_id = pc.id
            WHERE mc.media_id = ANY($1::bigint[])
            """,
            candidate_ids,
        )
        for r in comp_rows:
            company_map.setdefault(r["media_id"], [])
            company_map[r["media_id"]].append(r["name"])

        # Batch: networks (TV)
        network_map: dict[int, list[str]] = {}
        net_rows = await conn.fetch(
            """
            SELECT mn.media_id, n.name
            FROM tmdb_media_networks mn
            JOIN tmdb_networks n ON mn.network_id = n.id
            WHERE mn.media_id = ANY($1::bigint[])
            """,
            candidate_ids,
        )
        for r in net_rows:
            network_map.setdefault(r["media_id"], [])
            network_map[r["media_id"]].append(r["name"])

        # Batch: collection_ids + original_language (modality feature) from
        # tmdb_media
        coll_rows = await conn.fetch(
            """
            SELECT id, collection_id, original_language
            FROM tmdb_media
            WHERE id = ANY($1::bigint[])
            """,
            candidate_ids,
        )
        collection_ids = {r["id"]: r["collection_id"] for r in coll_rows}
        languages = {r["id"]: r.get("original_language") for r in coll_rows}

        # Batch: cinematic structure tags (narrative/pacing/emotional_arc)
        # for the narrative-match + mood features. Graceful when enrichment
        # is partial.
        narrative_map: dict[int, str] = {}
        pacing_map: dict[int, str] = {}
        arc_map: dict[int, list] = {}
        try:
            cin_rows = await conn.fetch(
                """
                SELECT media_id, narrative_complexity, pacing, emotional_arc
                FROM tmdb_cinematic
                WHERE media_id = ANY($1::bigint[])
                """,
                candidate_ids,
            )
            for r in cin_rows:
                narrative_map[r["media_id"]] = r["narrative_complexity"]
                pacing_map[r["media_id"]] = r["pacing"]
                arc_map[r["media_id"]] = list(r["emotional_arc"] or [])
        except asyncpg.UndefinedTableError:
            logger.debug("tmdb_cinematic table missing — narrative feature off")

    for cid in candidate_ids:
        result[cid] = {
            "director": directors.get(cid),
            "director_ids": director_ids_map.get(cid, []),
            "writer_ids": writer_ids_map.get(cid, []),
            "composer_ids": composer_map.get(cid, []),
            "dp_ids": dp_map.get(cid, []),
            "cast_list": cast_map.get(cid, []),
            "cast_ids": cast_ids_map.get(cid, []),
            "keywords": keywords_map.get(cid, []),
            "companies": company_map.get(cid, []),
            "networks": network_map.get(cid, []),
            "collection_id": collection_ids.get(cid),
            "original_language": languages.get(cid),
            "narrative": narrative_map.get(cid),
            "pacing": pacing_map.get(cid),
            "emotional_arc": arc_map.get(cid, []),
        }

    return result


# ---------------------------------------------------------------------------
# Semantic (pgvector KNN) candidate recall
# ---------------------------------------------------------------------------


# Columns/infrastructure known-missing (first failure latches until restart).
# An absent embedding column must not re-run the failing SQL per seed —
# the failed attempt costs a round trip each time.
_KNN_DISABLED: set = set()


def _knn_available(emb_column: Optional[str]) -> bool:
    return (emb_column or "default") not in _KNN_DISABLED


async def generate_knn_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 40,
    allow_cross_media: bool = False,
    emb_column: Optional[str] = None,
) -> List[dict]:
    """Fetch candidates whose overview embedding is nearest to the seed's.

    Uses the pgvector cosine-distance operator (`<=>`) on the precomputed
    `embedding` column — the ivfflat index from scripts/setup_pgvector.sql
    serves this query. Unlike `generate_candidates`, recall is NOT gated on
    genre overlap or popularity ordering: titles that are close in spirit
    but live in other genres surface here.

    Rows carry `via="knn"` and their `knn_similarity` (0..1) so downstream
    stages can exempt them from the genre gate and reuse the similarity.

    Args:
        emb_column: per-request embedding column override (/set_model).
            None = the process-global EMBEDDING_COLUMN.

    Returns:
        Up to `limit` candidates (same row shape as `generate_candidates`
        plus `via` and `knn_similarity`), or an empty list when the seed or
        the vector infrastructure is unavailable (graceful degradation).
    """
    import asyncpg as _asyncpg

    if not _knn_available(emb_column):
        return []

    try:
        try:
            from pgvector.asyncpg import register_vector
        except ImportError:
            return []  # optional dependency — KNN channel off without it
        async with pool.acquire() as conn:
            try:
                await register_vector(conn)
            except Exception:
                return []

            # Subquery for the seed vector — passing it as an asyncpg
            # parameter produced wrong cosine distances (codec mismatch).
            from .config import EMBEDDING_COLUMN

            _emb = emb_column or EMBEDDING_COLUMN
            await conn.execute("SET hnsw.ef_search = 400")
            rows = await conn.fetch(
                """
                WITH seed AS (
                    SELECT {emb} FROM tmdb_media
                    WHERE (id, media_type) = ($1, $2)
                )
                SELECT
                    m.id, m.title, m.title_en, m.media_type,
                    m.overview, m.overview_en,
                    m.vote_average::float, m.vote_count::int,
                    m.popularity::float, m.poster_path, m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int
                        AS release_year,
                    ARRAY(
                        SELECT g.name FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id
                    ) as genres,
                    GREATEST(0.0, 1 - (m.{emb} <=> (SELECT {emb} FROM seed)))
                        AS knn_similarity
                FROM tmdb_media m
                WHERE m.{emb} IS NOT NULL
                  AND (m.id, m.media_type) != ($1, $2)
                  AND m.vote_average >= 5.5
                  AND m.vote_count >= $5
                  AND ($3::text IS NULL OR m.media_type = $3::text)
                ORDER BY m.{emb} <=> (SELECT {emb} FROM seed)
                LIMIT $4
                """.replace("{emb}", _emb),
                seed_id,
                seed_media_type,
                None if allow_cross_media else seed_media_type,
                limit,
                _vote_floor(seed_media_type if not allow_cross_media else None),
            )
    except (
        _asyncpg.UndefinedColumnError,
        _asyncpg.UndefinedObjectError,
        _asyncpg.UndefinedTableError,
    ):
        _KNN_DISABLED.add(emb_column or "default")
        return []

    candidates = []
    for r in rows:
        c = dict(r)
        c["via"] = "knn"
        c["knn_similarity"] = float(c.get("knn_similarity") or 0.0)
        candidates.append(c)
    return candidates


# ---------------------------------------------------------------------------
# User-vector ANN recall — candidates nearest to u (spec §F.1 ch.2)
# ---------------------------------------------------------------------------

# Embedding columns live on the split fact tables (schema.sql §2), not on
# the compatibility view — so this reads both sides with UNION ALL.
_USER_VECTOR_RECALL = """
    SELECT * FROM (
        SELECT id, title, title AS title_en, 'movie'::text AS media_type,
               overview, overview AS overview_en,
               vote_average::float, vote_count::int,
               popularity::float, poster_path, collection_id,
               EXTRACT(YEAR FROM release_date)::int AS release_year,
               1 - ({emb} <=> $1) AS knn_similarity
        FROM tmdb_movies WHERE {emb} IS NOT NULL
        UNION ALL
        SELECT id, name, name AS title_en, 'tv'::text,
               overview, overview,
               vote_average::float, vote_count::int,
               popularity::float, poster_path, NULL::bigint,
               EXTRACT(YEAR FROM first_air_date)::int,
               1 - ({emb} <=> $1)
        FROM tmdb_tv WHERE {emb} IS NOT NULL
    ) m
    WHERE m.vote_average >= 5.5
    ORDER BY m.knn_similarity DESC
    LIMIT $2
"""


async def generate_user_vector_candidates(
    pool,
    user_vector: list,
    limit: int = 60,
    emb_column: Optional[str] = None,
) -> List[dict]:
    """Titles closest to the user's vector (cosine on unit vectors).

    Rows carry via="uvec" + knn_similarity like the KNN channel, so the
    genre gate exempts them and the cosine feature reuses the distance.
    Degrades to [] when the column/extension is missing.
    """
    import asyncpg as _asyncpg

    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return []
    if not _knn_available(emb_column):
        return []
    from .config import EMBEDDING_COLUMN

    col = emb_column or EMBEDDING_COLUMN
    try:
        async with pool.acquire() as conn:
            try:
                await register_vector(conn)
            except Exception:
                return []
            rows = await conn.fetch(
                _USER_VECTOR_RECALL.format(emb=col), user_vector, limit
            )
    except (_asyncpg.UndefinedColumnError, _asyncpg.UndefinedObjectError,
            _asyncpg.UndefinedTableError):
        _KNN_DISABLED.add(col)
        return []
    out = []
    for r in rows:
        c = dict(r)
        c["via"] = "uvec"
        c["knn_similarity"] = float(c.get("knn_similarity") or 0.0)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Fine-tuned KNN recall — direct retrieval from the contrastive space
# ---------------------------------------------------------------------------


async def generate_knn_finetuned_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 60,
    emb_column: Optional[str] = None,
) -> List[dict]:
    """Semantic neighbors from the contrastively fine-tuned embedding space.

    Identical SQL to generate_knn_candidates — the difference is the
    vectors themselves (from the MNRL fine-tuned model after re-embed).
    This channel surfaces thematic neighbors the vanilla MiniLM could
    never see (Annihilation for Arrival, Vinland Saga for AoT).

    Args:
        emb_column: per-request embedding column override (/set_model).
            None = the process-global EMBEDDING_COLUMN.
    """
    if not _knn_available(emb_column):
        return []

    try:
        async with pool.acquire() as conn:
            from .config import EMBEDDING_COLUMN

            _emb2 = emb_column or EMBEDDING_COLUMN
            await conn.execute("SET hnsw.ef_search = 400")
            rows = await conn.fetch(
                """
                WITH seed AS (
                    SELECT {emb} FROM tmdb_media
                    WHERE (id, media_type) = ($1, $2)
                )
                SELECT
                    m.id, m.title, m.title_en, m.media_type,
                    m.overview, m.overview_en,
                    m.vote_average::float, m.vote_count::int, m.popularity::float,
                    m.poster_path, m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int
                        AS release_year,
                    ARRAY(
                        SELECT g.name FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id AND mg.media_type = m.media_type
                    ) as genres,
                    GREATEST(0.0, 1 - (m.{emb} <=> (SELECT {emb} FROM seed)))
                        AS knn_similarity
                FROM tmdb_media m
                WHERE m.{emb} IS NOT NULL
                  AND (m.id, m.media_type) != ($1, $2)
                  AND m.media_type = $2
                  AND m.vote_average >= 5.5
                  AND m.vote_count >= 100
                ORDER BY m.{emb} <=> (SELECT {emb} FROM seed)
                LIMIT $3
                """.replace("{emb}", _emb2),
                seed_id,
                seed_media_type,
                limit,
            )
    except (
        asyncpg.UndefinedColumnError,
        asyncpg.UndefinedObjectError,
        asyncpg.UndefinedTableError,
    ):
        _KNN_DISABLED.add(emb_column or "default")
        return []

    candidates = []
    for r in rows:
        c = dict(r)
        c["via"] = "knn_finetuned"
        c["knn_similarity"] = float(c.get("knn_similarity") or 0.0)
        candidates.append(c)
    return candidates


# ---------------------------------------------------------------------------
# TMDB behavioral recommendations (locally synced)
# ---------------------------------------------------------------------------


async def generate_tmdb_rec_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 20,
) -> List[dict]:
    """Candidates from the locally synced TMDB recommendations table.

    These are the seed's "people also liked" titles from TMDB's behavioral
    engine (synced by app/services/recommendation/tmdb_recs.py), restricted
    to titles that exist in the local tmdb_media catalog, in TMDB's own
    rank order.

    Rows carry `via="tmdb"` so downstream stages exempt them from the genre
    gate — behavioral proximity is their qualification.

    Returns:
        Up to `limit` candidates (same row shape as `generate_candidates`
        plus `via`), or an empty list when the table is missing or the
        seed was never synced (graceful degradation).
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    m.id,
                    m.title,
                    m.title_en,
                    m.media_type,
                    m.overview,
                    m.overview_en,
                    m.vote_average::float,
                    m.vote_count::int,
                    m.popularity::float,
                    m.poster_path,
                    m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
                    r.rank AS tmdb_rank,
                    ARRAY(
                        SELECT g.name
                        FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id
                    ) as genres
                FROM tmdb_recommendations r
                JOIN tmdb_media m
                  ON (m.id, m.media_type) = (r.rec_media_id, r.rec_media_type)
                WHERE (r.media_id, r.media_type) = ($1, $2)
                ORDER BY r.rank
                LIMIT $3
                """,
                seed_id,
                seed_media_type,
                limit,
            )
    except asyncpg.UndefinedTableError:
        return []

    candidates = []
    for r in rows:
        c = dict(r)
        c["via"] = "tmdb"
        candidates.append(c)
    return candidates


# ---------------------------------------------------------------------------
# Director recall — "more from the same filmmaker"
# ---------------------------------------------------------------------------


async def generate_director_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 15,
) -> List[dict]:
    """Other works by the seed's director(s), joined to the local catalog.

    Auteur DNA is the strongest "feel-similar" signal the overview vectors
    cannot see: an Inception seed should surface Memento, The Prestige and
    Interstellar — same filmmaker, same mind-bending sensibility — even
    though their overviews are textually distant.

    Rows carry `via="director"` (exempt from the genre gate). Returns []
    when tmdb_crew is empty or the seed has no recorded director.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH seed_auteurs AS (
                    SELECT person_id
                    FROM tmdb_crew
                    WHERE (media_id, media_type) = ($1, $2)
                      AND job IN ('Director', 'Creator', 'Writer')
                )
                SELECT m.id, m.title, m.title_en, m.media_type,
                    m.overview, m.overview_en,
                    m.vote_average::float, m.vote_count::int, m.popularity::float,
                    m.poster_path, m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
                    BOOL_OR(c.job IN ('Director', 'Creator')) AS via_director,
                    ARRAY(
                        SELECT g.name FROM tmdb_genres g
                        JOIN tmdb_media_genres mg ON g.id = mg.genre_id
                        WHERE mg.media_id = m.id AND mg.media_type = m.media_type
                    ) as genres
                FROM tmdb_crew c
                JOIN tmdb_media m
                  ON (m.id, m.media_type) = (c.media_id, c.media_type)
                WHERE c.person_id IN (SELECT person_id FROM seed_auteurs)
                  AND c.job IN ('Director', 'Creator', 'Writer')
                  AND (m.id, m.media_type) != ($1, $2)
                  AND m.vote_average >= 5.5
                  AND m.vote_count >= $3
                GROUP BY m.id, m.title, m.title_en, m.media_type,
                    m.overview, m.overview_en, m.vote_average, m.vote_count,
                    m.popularity, m.poster_path, m.collection_id,
                    EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))
                ORDER BY m.id, m.popularity DESC
                """,
                seed_id,
                seed_media_type,
                _vote_floor(seed_media_type),
            )
    except asyncpg.UndefinedTableError:
        return []

    # Rank by aggregate popularity per title, capped. Writer-only links get
    # via="writer" so scoring can weight them below true director DNA
    # (a shared screenwriter is a taste signal, not an auteur stamp).
    by_id: dict = {}
    for r in rows:
        c = dict(r)
        c["via"] = "director" if c.pop("via_director") else "writer"
        prev = by_id.get(c["id"])
        if prev is None or c["popularity"] > prev["popularity"]:
            by_id[c["id"]] = c
    ranked = sorted(by_id.values(), key=lambda x: x["popularity"], reverse=True)
    return ranked[:limit]
