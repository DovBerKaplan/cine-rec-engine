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
            WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
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
            WHERE (mk.media_id, mk.media_type) = (m.id, m.media_type)
        ) as keywords,
        ARRAY(
            SELECT pc.name
            FROM tmdb_media_companies mc
            JOIN tmdb_production_companies pc ON mc.company_id = pc.id
            WHERE (mc.media_id, mc.media_type) = (m.id, m.media_type)
        ) as companies,
        ARRAY(
            SELECT n.name
            FROM tmdb_media_networks mn
            JOIN tmdb_networks n ON n.id = mn.network_id
            WHERE (mn.media_id, mn.media_type) = (m.id, m.media_type)
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
            WHERE (mk.media_id, mk.media_type) = (m.id, m.media_type)
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
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM tmdb_media m
                WHERE m.id != $1
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                        AND g.name = ANY($2::text[])
                  )
                ORDER BY
                    -- genre-overlap count first (the channel's whole point),
                    -- then a quality blend — raw popularity alone buried
                    -- classics under fresh high-pop titles (#1)
                    (
                        SELECT count(*) FROM tmdb_media_genres mg
                        JOIN tmdb_genres g ON mg.genre_id = g.id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                          AND g.name = ANY($2::text[])
                    ) DESC,
                    (m.vote_count * m.vote_average) DESC,
                    m.popularity DESC
                LIMIT $3 * 2  -- per-medium budget: the dedup below keeps
                              -- `limit` per media_type, so one hot medium
                              -- can no longer starve the other (#1)
                """,
                base_movie_id,
                base_genres,
                limit,
            )
            # per-medium budget: rank within each media_type, keep `limit`
            # per side (issue #1 — a UNION ordered globally let one medium
            # crowd out the other's classics)
            rows = sorted(
                rows,
                key=lambda r: (
                    -sum(1 for g in (r["genres"] or []) if g in set(base_genres)),
                    -(r["vote_count"] or 0) * (r["vote_average"] or 0),
                    -(r["popularity"] or 0),
                ),
            )
            kept: dict = {"movie": 0, "tv": 0}
            budgeted = []
            for r in rows:
                mt = r["media_type"]
                if kept.get(mt, 0) >= limit:
                    continue
                kept[mt] = kept.get(mt, 0) + 1
                budgeted.append(r)
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
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM tmdb_media m
                WHERE m.media_type = $1
                  AND m.id != $2
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
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
                           WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                       ) as genres
                FROM tmdb_media m
                WHERE m.id != $1 AND m.vote_average >= 7.0 AND m.vote_count >= $4
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND g.name = ANY($2::text[])
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
                           WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND mg.media_type = m.media_type
                       ) as genres
                FROM tmdb_media m
                WHERE m.media_type = $1 AND m.id != $2
                  AND m.vote_average >= 7.0 AND m.vote_count >= $4
                  AND EXISTS (
                      SELECT 1 FROM tmdb_media_genres mg
                      JOIN tmdb_genres g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND mg.media_type = m.media_type
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


# Per-medium facts for direct KNN recall. The embedding columns live on
# the split fact tables (schema.sql §2) — the tmdb_media compatibility
# view does NOT expose them (a view's column list is fixed at CREATE
# time, so later ALTER TABLE ... ADD COLUMN never shows up there) — so
# the KNN channels query the tables themselves, like the user-vector
# recall below.
_KNN_MEDIUM = {
    "movie": dict(
        table="tmdb_movies", title="title", date="release_date",
        coll="collection_id", gmap="tmdb_movie_genres_map", gid="movie_id",
    ),
    "tv": dict(
        table="tmdb_tv", title="name", date="first_air_date",
        coll="NULL::bigint", gmap="tmdb_tv_genres_map", gid="tv_id",
    ),
}


def _knn_arm_sql(mt: str, emb: str, seed_mt: str, vote_floor: int) -> str:
    """One medium's arm of a KNN recall: top-`limit` rows by cosine.

    The arm keeps its own ORDER BY + LIMIT (and the seed vector comes
    from a same-statement subquery, not a bind parameter — passing it as
    a parameter produced wrong distances) so the planner can serve it
    from the medium's ANN index instead of sorting the whole table. The
    id exclusion only applies in the seed's OWN medium: movie 155 stays a
    candidate for tv-seed 155 (different title — ids collide across
    media by design).
    """
    s = _KNN_MEDIUM[mt]
    seed = _KNN_MEDIUM[seed_mt]
    excl = "\n          AND c.id != $1" if mt == seed_mt else ""
    return f"""
        WITH seed AS (SELECT {emb} AS v FROM {seed['table']} WHERE id = $1)
        SELECT c.id, c.{s['title']} AS title, c.{s['title']} AS title_en,
               '{mt}'::text AS media_type,
               c.overview, c.overview AS overview_en,
               c.vote_average::float, c.vote_count::int,
               c.popularity::float, c.poster_path, {s['coll']} AS collection_id,
               EXTRACT(YEAR FROM c.{s['date']})::int AS release_year,
               ARRAY(
                   SELECT g.name FROM tmdb_genres g
                   JOIN {s['gmap']} gm ON gm.genre_id = g.id
                   WHERE gm.{s['gid']} = c.id
               ) AS genres,
               GREATEST(0.0, 1 - (c.{emb} <=> (SELECT v FROM seed)))
                   AS knn_similarity
        FROM {s['table']} c
        WHERE c.{emb} IS NOT NULL{excl}
          AND c.vote_average >= 5.5
          AND c.vote_count >= {int(vote_floor)}
        ORDER BY c.{emb} <=> (SELECT v FROM seed)
        LIMIT $2
    """


async def _fetch_knn_rows(pool, seed_id, seed_media_type, limit, emb,
                          media_types, vote_floor):
    """Run one KNN recall (single arm, or two arms merged in SQL).

    Returns None when the vector codec can't attach (no pgvector
    infrastructure) — the caller degrades without latching; asyncpg
    schema errors propagate so the caller can latch the channel off.
    """
    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return None  # optional dependency — vector channels off without it
    arms = [
        _knn_arm_sql(mt, emb, seed_media_type, vote_floor)
        for mt in ("movie", "tv") if mt in media_types
    ]
    if not arms:
        return []
    sql = (
        arms[0] if len(arms) == 1
        else f"({arms[0]})\nUNION ALL\n({arms[1]})\nORDER BY knn_similarity DESC\nLIMIT $2"
    )
    async with pool.acquire() as conn:
        try:
            await register_vector(conn)
        except Exception:
            return None
        # ef_search must be raised per session for pgvector's hnsw
        # indexes to return deep-enough candidate lists.
        await conn.execute("SET hnsw.ef_search = 400")
        return await conn.fetch(sql, seed_id, limit)


async def generate_knn_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 40,
    allow_cross_media: bool = False,
    emb_column: Optional[str] = None,
    vote_floor: Optional[int] = None,
) -> List[dict]:
    """Fetch candidates whose overview embedding is nearest to the seed's.

    Uses the pgvector cosine-distance operator (`<=>`) on the precomputed
    embedding column of the fact tables (an hnsw/ivfflat index on that
    column serves the query). Unlike `generate_candidates`, recall is NOT
    gated on genre overlap or popularity ordering: titles that are close
    in spirit but live in other genres surface here.

    Rows carry `via="knn"` and their `knn_similarity` (0..1) so downstream
    stages can exempt them from the genre gate and reuse the similarity.

    Args:
        emb_column: per-request embedding column override (/set_model).
            None = the process-global EMBEDDING_COLUMN.
        vote_floor: override the per-medium vote floor. find_similar
            passes the fine-tuned channel's floor here so ONE query
            serves both KNN channels (they share the same space and
            only differ in floor; the looser floor's top-k covers both
            old result sets).

    Returns:
        Up to `limit` candidates (same row shape as `generate_candidates`
        plus `via` and `knn_similarity`), or an empty list when the seed or
        the vector infrastructure is unavailable (graceful degradation).
    """
    import asyncpg as _asyncpg

    if not _knn_available(emb_column):
        return []

    from .config import EMBEDDING_COLUMN

    _emb = emb_column or EMBEDDING_COLUMN
    media = {"movie", "tv"} if allow_cross_media else {seed_media_type}
    try:
        rows = await _fetch_knn_rows(
            pool, seed_id, seed_media_type, limit, _emb, media,
            vote_floor if vote_floor is not None
            else _vote_floor(seed_media_type if not allow_cross_media else None),
        )
    except (
        _asyncpg.UndefinedColumnError,
        _asyncpg.UndefinedObjectError,
        _asyncpg.UndefinedTableError,
    ):
        _KNN_DISABLED.add(emb_column or "default")
        return []
    if not rows:
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
# the compatibility view — so this reads both sides with UNION ALL. Each
# arm keeps its own ORDER BY + LIMIT so the medium's ANN index serves the
# arm; the outer merge only sorts the (at most 2×limit) survivors. A
# single ORDER BY over the whole UNION would force a full sort of both
# tables on every request.
_USER_VECTOR_RECALL = """
    SELECT * FROM (
        (SELECT id, title, title AS title_en, 'movie'::text AS media_type,
                overview, overview AS overview_en,
                vote_average::float, vote_count::int,
                popularity::float, poster_path, collection_id,
                EXTRACT(YEAR FROM release_date)::int AS release_year,
                1 - ({emb} <=> $1) AS knn_similarity
         FROM tmdb_movies
         WHERE {emb} IS NOT NULL AND vote_average >= 5.5
         ORDER BY {emb} <=> $1
         LIMIT $2)
        UNION ALL
        (SELECT id, name, name AS title_en, 'tv'::text,
                overview, overview,
                vote_average::float, vote_count::int,
                popularity::float, poster_path, NULL::bigint,
                EXTRACT(YEAR FROM first_air_date)::int,
                1 - ({emb} <=> $1)
         FROM tmdb_tv
         WHERE {emb} IS NOT NULL AND vote_average >= 5.5
         ORDER BY {emb} <=> $1
         LIMIT $2)
    ) m
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
        # Latch the key the guard checks (emb_column or "default"), not
        # the resolved column name — a latched key that nothing reads
        # would retry the failing query on every request forever.
        _KNN_DISABLED.add(emb_column or "default")
        return []
    out = []
    for r in rows:
        c = dict(r)
        c["via"] = "uvec"
        c["knn_similarity"] = float(c.get("knn_similarity") or 0.0)
        out.append(c)
    return out


def _vec_as_list(v) -> Optional[list]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    return v.to_list()  # pgvector Vector


async def fetch_embeddings_batch(
    pool,
    pairs: list,
    emb_column: Optional[str] = None,
) -> dict:
    """(id, media_type) -> embedding vector, for the user-tilt pass.

    Two small queries against the split fact tables (the column lives
    there, not on the view). Degrades to {} on any missing piece —
    no embeddings, no tilt, never an error.
    """
    if not pairs:
        return {}
    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return {}
    from .config import EMBEDDING_COLUMN

    col = emb_column or EMBEDDING_COLUMN
    movie_ids = sorted({i for i, mt in pairs if mt != "tv"})
    tv_ids = sorted({i for i, mt in pairs if mt == "tv"})
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            try:
                await register_vector(conn)
            except Exception:
                return {}
            if movie_ids:
                rows = await conn.fetch(
                    f"SELECT id, {col} AS emb FROM tmdb_movies "
                    f"WHERE id = ANY($1::bigint[]) AND {col} IS NOT NULL",
                    movie_ids,
                )
                for r in rows:
                    emb = _vec_as_list(r["emb"])
                    if emb is not None:
                        out[(r["id"], "movie")] = emb
            if tv_ids:
                rows = await conn.fetch(
                    f"SELECT id, {col} AS emb FROM tmdb_tv "
                    f"WHERE id = ANY($1::bigint[]) AND {col} IS NOT NULL",
                    tv_ids,
                )
                for r in rows:
                    emb = _vec_as_list(r["emb"])
                    if emb is not None:
                        out[(r["id"], "tv")] = emb
    except Exception:
        return {}
    return out


# ---------------------------------------------------------------------------
# Session filters (RFC §3, M1) — recall-boundary predicates
# ---------------------------------------------------------------------------

_FILTER_KEYS = ("year_min", "year_max", "genre_ids", "exclude_genre_ids",
                "max_runtime")


def normalize_filters(raw: Optional[dict]) -> Optional[dict]:
    """Validate + normalize a session-filter dict; None when empty.

    Raises ValueError on anything a caller typo'd — a filter that
    silently did nothing is worse than a loud 400.
    """
    if not raw:
        return None
    unknown = set(raw) - set(_FILTER_KEYS)
    if unknown:
        raise ValueError(f"unknown filter keys: {sorted(unknown)}")

    out: dict = {}

    def _year(key):
        v = raw.get(key)
        if v is None:
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer year")
        if not 1900 <= v <= 2100:
            raise ValueError(f"{key} out of range (1900-2100): {v}")
        return v

    out["year_min"] = _year("year_min")
    out["year_max"] = _year("year_max")
    if out["year_min"] is not None and out["year_max"] is not None \
            and out["year_min"] > out["year_max"]:
        raise ValueError("year_min must be <= year_max")

    def _ids(key):
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, str):
            v = [p.strip() for p in v.split(",") if p.strip()]
        try:
            ids = sorted({int(x) for x in v})
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a list of integer genre ids")
        if any(i <= 0 for i in ids):
            raise ValueError(f"{key} must contain positive genre ids")
        return ids or None

    out["genre_ids"] = _ids("genre_ids")
    out["exclude_genre_ids"] = _ids("exclude_genre_ids")
    if out["genre_ids"] and out["exclude_genre_ids"] \
            and set(out["genre_ids"]) & set(out["exclude_genre_ids"]):
        raise ValueError("a genre cannot be both pinned and excluded")

    rt = raw.get("max_runtime")
    if rt is not None:
        try:
            rt = int(rt)
        except (TypeError, ValueError):
            raise ValueError("max_runtime must be integer minutes")
        if not 1 <= rt <= 1440:
            raise ValueError("max_runtime out of range (1-1440 minutes)")
    out["max_runtime"] = rt

    if not any(v is not None for v in out.values()):
        return None
    return out


def filters_fingerprint(filters: Optional[dict]) -> str:
    """Stable cache-key component: same filters ⇒ same string, any
    parameter of the dict order. 'none' when filters are absent."""
    if not filters:
        return "none"
    parts = []
    if filters.get("year_min") is not None:
        parts.append(f"ym={filters['year_min']}")
    if filters.get("year_max") is not None:
        parts.append(f"yx={filters['year_max']}")
    if filters.get("genre_ids"):
        parts.append("g=" + ",".join(str(i) for i in sorted(filters["genre_ids"])))
    if filters.get("exclude_genre_ids"):
        parts.append("xg=" + ",".join(
            str(i) for i in sorted(filters["exclude_genre_ids"])))
    if filters.get("max_runtime") is not None:
        parts.append(f"rt={filters['max_runtime']}")
    return "|".join(parts) or "none"


async def apply_recall_filters(pool, candidates: list, filters: dict) -> list:
    """Drop candidates failing the session filters — one eligibility
    query per medium, BEFORE scoring and the limit cut so limit
    semantics stay honest. Runtime applies to movies only (the tv
    schema has no runtime). Degrades to the unfiltered list when the
    filter tables are missing — a filter never 500s a query path.
    """
    if not filters or not candidates:
        return candidates
    movie_ids = sorted({c["id"] for c in candidates if c.get("media_type") != "tv"})
    tv_ids = sorted({c["id"] for c in candidates if c.get("media_type") == "tv"})
    eligible: set = set()

    def _year_clause(col, params):
        clauses, p = [], list(params)
        if filters.get("year_min") is not None:
            p.append(filters["year_min"])
            clauses.append(f"EXTRACT(YEAR FROM {col}) >= ${len(p)}")
        if filters.get("year_max") is not None:
            p.append(filters["year_max"])
            clauses.append(f"EXTRACT(YEAR FROM {col}) <= ${len(p)}")
        return clauses, p

    try:
        async with pool.acquire() as conn:
            if movie_ids:
                clauses, p = _year_clause("m.release_date", [movie_ids])
                if filters.get("max_runtime") is not None:
                    p.append(filters["max_runtime"])
                    clauses.append(f"m.runtime <= ${len(p)}")
                if filters.get("genre_ids"):
                    p.append(filters["genre_ids"])
                    clauses.append(
                        f"EXISTS (SELECT 1 FROM tmdb_movie_genres_map g "
                        f"WHERE g.movie_id = m.id AND g.genre_id = ANY(${len(p)}))")
                if filters.get("exclude_genre_ids"):
                    p.append(filters["exclude_genre_ids"])
                    clauses.append(
                        f"NOT EXISTS (SELECT 1 FROM tmdb_movie_genres_map g "
                        f"WHERE g.movie_id = m.id AND g.genre_id = ANY(${len(p)}))")
                where = " AND ".join(["m.id = ANY($1)"] + clauses)
                rows = await conn.fetch(
                    f"SELECT m.id FROM tmdb_movies m WHERE {where}", *p)
                eligible |= {(r["id"], "movie") for r in rows}
            if tv_ids:
                clauses, p = _year_clause("t.first_air_date", [tv_ids])
                if filters.get("genre_ids"):
                    p.append(filters["genre_ids"])
                    clauses.append(
                        f"EXISTS (SELECT 1 FROM tmdb_tv_genres_map g "
                        f"WHERE g.tv_id = t.id AND g.genre_id = ANY(${len(p)}))")
                if filters.get("exclude_genre_ids"):
                    p.append(filters["exclude_genre_ids"])
                    clauses.append(
                        f"NOT EXISTS (SELECT 1 FROM tmdb_tv_genres_map g "
                        f"WHERE g.tv_id = t.id AND g.genre_id = ANY(${len(p)}))")
                where = " AND ".join(["t.id = ANY($1)"] + clauses)
                rows = await conn.fetch(
                    f"SELECT t.id FROM tmdb_tv t WHERE {where}", *p)
                eligible |= {(r["id"], "tv") for r in rows}
    except Exception:
        return candidates  # filter tables missing — degrade open
    return [c for c in candidates
            if (c["id"], c.get("media_type")) in eligible]


# ---------------------------------------------------------------------------
# Page rows (RFC §1, M1) — direct rows outside the LTR pipeline
# ---------------------------------------------------------------------------

HIDDEN_GEMS_RATING_MIN = 7.0   # quality floor — a gem must actually be good
HIDDEN_GEMS_VOTES_MAX = 2000   # above this it's a hit, not hidden
HIDDEN_GEMS_VOTES_MIN = 30     # below this is 12-votes noise


async def top_user_genre_ids(pool, user_id: int, limit: int = 3) -> list:
    """The user's top genre clusters by weighted watch share. Degrades
    to [] — the caller falls back to global rows."""
    try:
        rows = await pool.fetch(
            """SELECT genre_id, SUM(weighted_sum) AS w
               FROM user_genre_stats WHERE user_id = $1
               GROUP BY genre_id ORDER BY w DESC NULLS LAST, genre_id LIMIT $2""",
            user_id, limit,
        )
        return [r["genre_id"] for r in rows]
    except Exception:
        return []


async def genre_names_for_ids(pool, genre_ids: list) -> set:
    """Genre NAMES for ids across both per-medium genre lists (the
    engine's genre vocabulary is names; the stats layer stores ids).
    Degrades to an empty set — exploration then skips, never misfires."""
    if not genre_ids:
        return set()
    try:
        rows = await pool.fetch(
            """SELECT name FROM tmdb_movie_genres WHERE id = ANY($1::int[])
               UNION
               SELECT name FROM tmdb_tv_genres WHERE id = ANY($1::int[])""",
            sorted(set(genre_ids)),
        )
        return {r["name"] for r in rows}
    except Exception:
        return set()


async def generate_popular_candidates(
    pool, limit: int = 120, filters: Optional[dict] = None,
) -> List[dict]:
    """Popularity slice above the vote floor — the /by-text recall
    fallback when no encoder is configured (RFC §3 Phase B). Rows carry
    via="text" so the genre gate exempts them. Deterministic; degrades
    to []."""
    rows = await _gem_or_trending_rows(
        pool, limit, None, filters, mode="popular")
    for r in rows:
        r["id"] = r["tmdb_id"]  # internal recall-pool shape (find_similar)
        r["via"] = "text"
    return rows


_DISCOVERY_RECALL = """
    SELECT * FROM (
        SELECT m.id, m.title, m.title AS title_en, 'movie'::text AS media_type,
               m.overview, m.overview AS overview_en,
               m.vote_average::float, m.vote_count::int,
               m.popularity::float, m.poster_path, m.collection_id,
               EXTRACT(YEAR FROM m.release_date)::int AS release_year,
               1 - (m.{emb} <=> $1) AS knn_similarity
        FROM tmdb_movies m
        WHERE m.{emb} IS NOT NULL AND m.vote_average >= 5.5
          AND NOT EXISTS (SELECT 1 FROM tmdb_movie_genres_map g
                          WHERE g.movie_id = m.id AND g.genre_id = ANY($2))
        UNION ALL
        SELECT t.id, t.name, t.name AS title_en, 'tv'::text,
               t.overview, t.overview,
               t.vote_average::float, t.vote_count::int,
               t.popularity::float, t.poster_path, NULL::bigint,
               EXTRACT(YEAR FROM t.first_air_date)::int,
               1 - (t.{emb} <=> $1)
        FROM tmdb_tv t
        WHERE t.{emb} IS NOT NULL AND t.vote_average >= 5.5
          AND NOT EXISTS (SELECT 1 FROM tmdb_tv_genres_map g
                          WHERE g.tv_id = t.id AND g.genre_id = ANY($2))
    ) d
    ORDER BY d.knn_similarity DESC, d.id ASC
    LIMIT $3
"""


async def generate_discovery_candidates(
    pool, user_vector: list, exclude_genre_ids: list,
    limit: int = 60, emb_column: Optional[str] = None,
) -> List[dict]:
    """The dedicated discovery row (RFC §2): titles NEAREST the user
    vector among those OUTSIDE her top genre clusters — novel by the
    SQL exclusion, relevant by ANN distance. Deterministic; degrades
    to [] (missing column/extension/tables)."""
    import asyncpg as _asyncpg

    if not _knn_available(emb_column):
        return []
    from .config import EMBEDDING_COLUMN

    col = emb_column or EMBEDDING_COLUMN
    try:
        from pgvector.asyncpg import register_vector
    except ImportError:
        return []
    try:
        async with pool.acquire() as conn:
            try:
                await register_vector(conn)
            except Exception:
                return []
            rows = await conn.fetch(
                _DISCOVERY_RECALL.format(emb=col),
                user_vector, sorted(set(exclude_genre_ids)) or [0], limit,
            )
    except (_asyncpg.UndefinedColumnError, _asyncpg.UndefinedObjectError,
            _asyncpg.UndefinedTableError, Exception):
        return []
    out = []
    for r in rows:
        c = dict(r)
        c["via"] = "uvec"
        c["knn_similarity"] = float(c.get("knn_similarity") or 0.0)
        out.append(c)
    return out


async def generate_hidden_gems(
    pool, limit: int = 12, genre_ids: Optional[list] = None,
    filters: Optional[dict] = None,
) -> List[dict]:
    """High-rated, under-seen titles (optionally inside given genres).

    Ranked rating DESC, votes ASC — the better AND lesser-known a title
    is, the higher. Deterministic. Degrades to [] when tables are
    missing. Rows are API-shaped (tmdb_id, rating, ...), not scored.
    """
    return await _gem_or_trending_rows(
        pool, limit, genre_ids, filters, mode="gems")


async def generate_trending_genre(
    pool, genre_id: int, limit: int = 12,
    filters: Optional[dict] = None,
) -> List[dict]:
    """Most-popular titles in one genre, above the per-medium vote
    floor. Deterministic (popularity DESC, id ASC). Degrades to []."""
    return await _gem_or_trending_rows(
        pool, limit, [genre_id], filters, mode="trending")


async def _gem_or_trending_rows(pool, limit, genre_ids, filters, mode) -> List[dict]:
    per_medium = max(limit, 1)
    out: List[dict] = []

    # movies title column is `title`, tv's is `name` (the split schema's
    # one naming asymmetry — _USER_VECTOR_RECALL does the same aliasing)
    async def _fetch(conn, table, title_col, date_col, media_type, gmap, gcol):
        where, p = ["vote_count >= $1"], [_vote_floor(media_type)]
        if mode == "gems":
            p.extend([HIDDEN_GEMS_RATING_MIN,
                      HIDDEN_GEMS_VOTES_MIN, HIDDEN_GEMS_VOTES_MAX])
            where += [f"vote_average >= ${len(p) - 2}",
                      f"vote_count BETWEEN ${len(p) - 1} AND ${len(p)}"]
        if genre_ids:
            p.append(list(genre_ids))
            where.append(
                f"EXISTS (SELECT 1 FROM {gmap} g WHERE g.{gcol} = m.id "
                f"AND g.genre_id = ANY(${len(p)}))")
        f_ = filters or {}
        if f_.get("year_min") is not None:
            p.append(f_["year_min"])
            where.append(f"EXTRACT(YEAR FROM {date_col}) >= ${len(p)}")
        if f_.get("year_max") is not None:
            p.append(f_["year_max"])
            where.append(f"EXTRACT(YEAR FROM {date_col}) <= ${len(p)}")
        order = ("vote_average DESC, vote_count ASC, id ASC" if mode == "gems"
                 else "popularity DESC, id ASC")
        return [dict(r) for r in await conn.fetch(
            f"SELECT m.id AS tmdb_id, m.{title_col} AS title, "
            f"m.{title_col} AS title_en, "
            f"'{media_type}'::text AS media_type, m.vote_average::float AS rating, "
            f"m.vote_count::int AS vote_count, m.popularity::float AS popularity, "
            f"m.poster_path, EXTRACT(YEAR FROM {date_col})::int AS release_year "
            f"FROM {table} m WHERE {' AND '.join(where)} "
            f"ORDER BY {order} LIMIT ${len(p) + 1}",
            *p, per_medium,
        )]

    movies: List[dict] = []
    tvs: List[dict] = []
    try:
        async with pool.acquire() as conn:
            try:
                movies = await _fetch(
                    conn, "tmdb_movies", "title", "release_date", "movie",
                    "tmdb_movie_genres_map", "movie_id")
            except Exception as e:
                logger.debug(f"page row ({mode}) movie side degraded: {e}")
            try:
                tvs = await _fetch(
                    conn, "tmdb_tv", "name", "first_air_date", "tv",
                    "tmdb_tv_genres_map", "tv_id")
            except Exception as e:
                logger.debug(f"page row ({mode}) tv side degraded: {e}")
    except Exception as e:
        logger.debug(f"page row ({mode}) degraded: {e}")
        return []

    # interleave by rank (best movie, best tv, 2nd movie, …) so neither
    # medium monopolizes the row
    while len(out) < limit and (movies or tvs):
        for bucket in (movies, tvs):
            if bucket and len(out) < limit:
                out.append(bucket.pop(0))
    return out
# ---------------------------------------------------------------------------


async def generate_knn_finetuned_candidates(
    pool: asyncpg.Pool,
    seed_id: int,
    seed_media_type: str,
    limit: int = 60,
    emb_column: Optional[str] = None,
) -> List[dict]:
    """Semantic neighbors from the contrastively fine-tuned embedding space.

    Identical recall to generate_knn_candidates (same-medium only, higher
    vote floor) — the difference is the vectors themselves (from the
    MNRL fine-tuned model after re-embed). This channel surfaces thematic
    neighbors the vanilla MiniLM could never see (Annihilation for
    Arrival, Vinland Saga for AoT).

    Args:
        emb_column: per-request embedding column override (/set_model).
            None = the process-global EMBEDDING_COLUMN.
    """
    if not _knn_available(emb_column):
        return []

    from .config import EMBEDDING_COLUMN

    _emb = emb_column or EMBEDDING_COLUMN
    try:
        rows = await _fetch_knn_rows(
            pool, seed_id, seed_media_type, limit, _emb,
            {seed_media_type}, vote_floor=100,
        )
    except (
        asyncpg.UndefinedColumnError,
        asyncpg.UndefinedObjectError,
        asyncpg.UndefinedTableError,
    ):
        _KNN_DISABLED.add(emb_column or "default")
        return []
    if not rows:
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
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
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
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND mg.media_type = m.media_type
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
