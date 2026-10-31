//! Recall channels, part 2 — the ANN/vector channels, the behavioral
//! (TMDB-recs) and auteur (director) channels, and the page rows
//! (gems/trending/popular). Faithful ports of the remaining
//! `queries.py` functions; part 1 (`recall.rs`) holds the genre,
//! enrichment and seed-KNN channels.

use std::collections::HashMap;

use crate::executor::{row, Bind, Db, DbError};
use crate::recall::CandidateRow;

// ── user-vector ANN recall (spec §F.1) ─────────────────────────────

/// `_USER_VECTOR_RECALL` — candidates nearest to the USER's vector.
/// Each arm keeps its own ORDER BY + LIMIT so the medium's ANN index
/// serves the arm; the outer merge only sorts the survivors. The
/// `{emb}` placeholder is substituted by the CALLER (after the {t_*}
/// resolver — same two-stage rule as Python).
const USER_VECTOR_RECALL: &str = r#"
    SELECT * FROM (
        (SELECT id, title, title AS title_en, 'movie'::text AS media_type,
                overview, overview AS overview_en,
                vote_average::float, vote_count::int,
                popularity::float, poster_path, collection_id,
                EXTRACT(YEAR FROM release_date)::int AS release_year,
                1 - ({emb} <=> $1) AS knn_similarity
         FROM {t_tmdb_movies}
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
         FROM {t_tmdb_tv}
         WHERE {emb} IS NOT NULL AND vote_average >= 5.5
         ORDER BY {emb} <=> $1
         LIMIT $2)
    ) m
    ORDER BY m.knn_similarity DESC
    LIMIT $2
"#;

/// `generate_user_vector_candidates` — titles closest to the user's
/// vector (cosine on unit vectors). Rows carry `via="uvec"` so the
/// genre gate exempts them. Degrades to `[]` when the column /
/// extension / tables are missing.
pub async fn generate_user_vector_candidates(
    db: &Db,
    user_vector: &[f32],
    limit: i64,
    emb_column: &str,
) -> Result<Vec<CandidateRow>, DbError> {
    // two-stage substitution: {t_*} through the resolver, then {emb}
    let sql = db.sql(USER_VECTOR_RECALL)?.replace("{emb}", emb_column);
    let rows = match db
        .fetch(
            &sql,
            &[
                ("vec", Bind::Vector(user_vector.to_vec())),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("uvec") {
            DbError::ChannelOff { .. } => return Ok(Vec::new()),
            real => return Err(real),
        },
    };
    Ok(rows
        .iter()
        .map(|r| {
            let mut c = CandidateRow::from_recall_row(r);
            c.via = Some("uvec".to_string());
            c.knn_similarity = Some(c.knn_similarity.unwrap_or(0.0));
            c
        })
        .collect())
}

/// `_DISCOVERY_RECALL` — titles NEAREST the user vector among those
/// OUTSIDE her top genre clusters: novel by the SQL exclusion,
/// relevant by ANN distance.
const DISCOVERY_RECALL: &str = r#"
    SELECT * FROM (
        SELECT m.id, m.title, m.title AS title_en, 'movie'::text AS media_type,
               m.overview, m.overview AS overview_en,
               m.vote_average::float, m.vote_count::int,
               m.popularity::float, m.poster_path, m.collection_id,
               EXTRACT(YEAR FROM m.release_date)::int AS release_year,
               1 - (m.{emb} <=> $1) AS knn_similarity
        FROM {t_tmdb_movies} m
        WHERE m.{emb} IS NOT NULL AND m.vote_average >= 5.5
          AND NOT EXISTS (SELECT 1 FROM {t_tmdb_movie_genres_map} g
                          WHERE g.movie_id = m.id AND g.genre_id = ANY($2))
        UNION ALL
        SELECT t.id, t.name, t.name AS title_en, 'tv'::text,
               t.overview, t.overview,
               t.vote_average::float, t.vote_count::int,
               t.popularity::float, t.poster_path, NULL::bigint,
               EXTRACT(YEAR FROM t.first_air_date)::int,
               1 - (t.{emb} <=> $1)
        FROM {t_tmdb_tv} t
        WHERE t.{emb} IS NOT NULL AND t.vote_average >= 5.5
          AND NOT EXISTS (SELECT 1 FROM {t_tmdb_tv_genres_map} g
                          WHERE g.tv_id = t.id AND g.genre_id = ANY($2))
    ) d
    ORDER BY d.knn_similarity DESC, d.id ASC
    LIMIT $3
"#;

/// `generate_discovery_candidates` — same shape as the user-vector
/// channel with the genre-cluster exclusion. Degrades to `[]`.
pub async fn generate_discovery_candidates(
    db: &Db,
    user_vector: &[f32],
    exclude_genre_ids: &[i64],
    limit: i64,
    emb_column: &str,
) -> Result<Vec<CandidateRow>, DbError> {
    // Python: sorted(set(...)) or [0] — the sentinel 0 keeps ANY($2)
    // well-typed when the exclusion list is empty
    let mut genres: Vec<i64> = exclude_genre_ids.to_vec();
    genres.sort_unstable();
    genres.dedup();
    if genres.is_empty() {
        genres.push(0);
    }
    let sql = db.sql(DISCOVERY_RECALL)?.replace("{emb}", emb_column);
    let rows = match db
        .fetch(
            &sql,
            &[
                ("vec", Bind::Vector(user_vector.to_vec())),
                ("genres", Bind::I64Array(genres)),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("uvec") {
            DbError::ChannelOff { .. } => return Ok(Vec::new()),
            real => return Err(real),
        },
    };
    Ok(rows
        .iter()
        .map(|r| {
            let mut c = CandidateRow::from_recall_row(r);
            c.via = Some("uvec".to_string());
            c.knn_similarity = Some(c.knn_similarity.unwrap_or(0.0));
            c
        })
        .collect())
}

/// `fetch_embeddings_batch` — `(id, media_type) -> embedding`, for the
/// user-tilt pass. Degrades to an EMPTY map on any missing piece: no
/// embeddings, no tilt, never an error.
pub async fn fetch_embeddings_batch(
    db: &Db,
    pairs: &[(i64, String)],
    emb_column: &str,
) -> Result<HashMap<(i64, String), Vec<f32>>, DbError> {
    if pairs.is_empty() {
        return Ok(HashMap::new());
    }
    let mut movie_ids: Vec<i64> = pairs
        .iter()
        .filter(|(_, mt)| mt != "tv")
        .map(|(id, _)| *id)
        .collect();
    movie_ids.sort_unstable();
    movie_ids.dedup();
    let mut tv_ids: Vec<i64> = pairs
        .iter()
        .filter(|(_, mt)| mt == "tv")
        .map(|(id, _)| *id)
        .collect();
    tv_ids.sort_unstable();
    tv_ids.dedup();

    let mut out = HashMap::new();
    for (ids, table, mt) in [
        (&movie_ids, "tmdb_movies", "movie"),
        (&tv_ids, "tmdb_tv", "tv"),
    ] {
        if ids.is_empty() {
            continue;
        }
        let sql = format!(
            "SELECT id, {col} AS emb FROM {table} \
             WHERE id = ANY($1::bigint[]) AND {col} IS NOT NULL",
            col = emb_column,
            table = db.map().name(table)?,
        );
        match db
            .fetch(&sql, &[("ids", Bind::I64Array(ids.clone()))])
            .await
        {
            Ok(rows) => {
                for r in rows {
                    if let Some(emb) = row::opt_vector(&r, "emb") {
                        out.insert(
                            (row::opt_i64(&r, "id").unwrap_or_default(), mt.to_string()),
                            emb,
                        );
                    }
                }
            }
            Err(e) => match e.into_degraded("embeddings") {
                // no embeddings, no tilt — never an error
                DbError::ChannelOff { .. } => return Ok(HashMap::new()),
                real => return Err(real),
            },
        }
    }
    Ok(out)
}

// ── TMDB behavioral recommendations (synced graph) ─────────────────

const TMDB_REC_SELECT: &str = r#"
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
                        FROM {t_tmdb_genres} g
                        JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM {t_tmdb_recommendations} r
                JOIN {t_tmdb_media} m
                  ON (m.id, m.media_type) = (r.rec_media_id, r.rec_media_type)
                WHERE (r.media_id, r.media_type) = ($1, $2)
                ORDER BY r.rank
                LIMIT $3
                "#;

/// `generate_tmdb_rec_candidates` — the seed's "people also liked"
/// titles from the locally synced TMDB graph, in TMDB's rank order.
/// Rows carry `via="tmdb"` (genre-gate exempt). Degrades to `[]` when
/// the table is missing.
pub async fn generate_tmdb_rec_candidates(
    db: &Db,
    seed_id: i64,
    seed_media_type: &str,
    limit: i64,
) -> Result<Vec<CandidateRow>, DbError> {
    let rows = match db
        .fetch(
            TMDB_REC_SELECT,
            &[
                ("id", Bind::I64(seed_id)),
                ("mt", Bind::Text(seed_media_type.to_string())),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("tmdb") {
            DbError::ChannelOff { .. } => return Ok(Vec::new()),
            real => return Err(real),
        },
    };
    Ok(rows
        .iter()
        .map(|r| {
            let mut c = CandidateRow::from_recall_row(r);
            c.via = Some("tmdb".to_string());
            c
        })
        .collect())
}

/// `generate_tmdb_rec_candidates_batch` — per-seed TMDB-recs in ONE
/// query. The per-seed top-k is preserved exactly (row_number() over
/// the seed partition). Returns one list per input seed, in input
/// order; the missing-table degrade returns empty lists.
pub async fn generate_tmdb_rec_candidates_batch(
    db: &Db,
    seeds: &[(i64, String)],
    limit: i64,
) -> Result<Vec<Vec<CandidateRow>>, DbError> {
    let mut out: Vec<Vec<CandidateRow>> = vec![Vec::new(); seeds.len()];
    if seeds.is_empty() {
        return Ok(out);
    }
    const SQL: &str = r#"
                WITH seeds AS (
                    SELECT * FROM unnest($1::bigint[], $2::text[])
                        AS s(media_id, media_type)
                ),
                ranked AS (
                    SELECT r.media_id, r.media_type,
                           r.rec_media_id, r.rec_media_type, r.rank,
                           row_number() OVER (
                               PARTITION BY r.media_id, r.media_type
                               ORDER BY r.rank
                           ) AS rn
                    FROM {t_tmdb_recommendations} r
                    JOIN seeds s
                      ON (r.media_id, r.media_type) = (s.media_id, s.media_type)
                )
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
                    rk.rank AS tmdb_rank,
                    rk.media_id AS _seed_id,
                    rk.media_type AS _seed_media_type,
                    ARRAY(
                        SELECT g.name
                        FROM {t_tmdb_genres} g
                        JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM ranked rk
                JOIN {t_tmdb_media} m
                  ON (m.id, m.media_type) = (rk.rec_media_id, rk.rec_media_type)
                WHERE rk.rn <= $3
                ORDER BY rk.media_id, rk.media_type, rk.rank
                "#;
    let ids: Vec<i64> = seeds.iter().map(|(id, _)| *id).collect();
    let mts: Vec<String> = seeds.iter().map(|(_, mt)| mt.clone()).collect();
    let rows = match db
        .fetch(
            SQL,
            &[
                ("ids", Bind::I64Array(ids)),
                ("mts", Bind::TextArray(mts)),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("tmdb") {
            DbError::ChannelOff { .. } => return Ok(out),
            real => return Err(real),
        },
    };
    let mut by_seed: HashMap<(i64, String), Vec<CandidateRow>> = HashMap::new();
    for r in rows {
        let mut c = CandidateRow::from_recall_row(&r);
        c.via = Some("tmdb".to_string());
        let key = (
            row::opt_i64(&r, "_seed_id").unwrap_or_default(),
            row::opt_text(&r, "_seed_media_type").unwrap_or_default(),
        );
        by_seed.entry(key).or_default().push(c);
    }
    for (i, (id, mt)) in seeds.iter().enumerate() {
        out[i] = by_seed.get(&(*id, mt.clone())).cloned().unwrap_or_default();
    }
    Ok(out)
}

// ── auteur (director/writer) recall ────────────────────────────────

const DIRECTOR_SQL: &str = r#"
                WITH seed_auteurs AS (
                    SELECT person_id
                    FROM {t_tmdb_crew}
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
                        SELECT g.name FROM {t_tmdb_genres} g
                        JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND mg.media_type = m.media_type
                    ) as genres
                FROM {t_tmdb_crew} c
                JOIN {t_tmdb_media} m
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
                "#;

/// Rank by aggregate popularity per title, capped. Writer-only links
/// get `via="writer"` — a shared screenwriter is a taste signal, not
/// an auteur stamp.
fn rank_auteur_works(rows: Vec<sqlx::postgres::PgRow>, limit: usize) -> Vec<CandidateRow> {
    let mut by_id: HashMap<i64, CandidateRow> = HashMap::new();
    for r in rows {
        let mut c = CandidateRow::from_recall_row(&r);
        let via_director: bool = row::opt_i64(&r, "via_director")
            .map(|v| v != 0)
            .or_else(|| row::opt_f64(&r, "via_director").map(|v| v != 0.0))
            .unwrap_or(false);
        c.via = Some(if via_director { "director" } else { "writer" }.to_string());
        let prev = by_id.get(&c.id);
        if prev.is_none() || c.popularity.unwrap_or(0.0) > prev.unwrap().popularity.unwrap_or(0.0) {
            by_id.insert(c.id, c);
        }
    }
    let mut ranked: Vec<CandidateRow> = by_id.into_values().collect();
    ranked.sort_by(|a, b| {
        b.popularity
            .unwrap_or(0.0)
            .partial_cmp(&a.popularity.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit);
    ranked
}

/// `generate_director_candidates` — other works by the seed's
/// director(s)/writer(s), joined to the local catalog. Degrades to
/// `[]` when `tmdb_crew` is missing.
pub async fn generate_director_candidates(
    db: &Db,
    seed_id: i64,
    seed_media_type: &str,
    limit: usize,
) -> Result<Vec<CandidateRow>, DbError> {
    let floor = crate::recall::vote_floor_for(Some(seed_media_type));
    let rows = match db
        .fetch(
            DIRECTOR_SQL,
            &[
                ("id", Bind::I64(seed_id)),
                ("mt", Bind::Text(seed_media_type.to_string())),
                ("floor", Bind::I64(floor)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("director") {
            DbError::ChannelOff { .. } => return Ok(Vec::new()),
            real => return Err(real),
        },
    };
    Ok(rank_auteur_works(rows, limit))
}

/// `generate_director_candidates_batch` — per-seed auteur works in ONE
/// query; each seed's auteur set, exclusion and floor stay per-seed
/// (parallel unnest arrays). Result lists are identical to N
/// individual calls.
pub async fn generate_director_candidates_batch(
    db: &Db,
    seeds: &[(i64, String)],
    limit: usize,
) -> Result<Vec<Vec<CandidateRow>>, DbError> {
    let mut out: Vec<Vec<CandidateRow>> = vec![Vec::new(); seeds.len()];
    if seeds.is_empty() {
        return Ok(out);
    }
    const SQL: &str = r#"
                WITH seeds AS (
                    SELECT * FROM unnest($1::bigint[], $2::text[], $3::int[])
                        WITH ORDINALITY AS s(media_id, media_type, vote_floor, seed_ordinal)
                ),
                seed_auteurs AS (
                    SELECT s.seed_ordinal, s.media_id, s.media_type, c.person_id
                    FROM seeds s
                    JOIN {t_tmdb_crew} c
                      ON (c.media_id, c.media_type) = (s.media_id, s.media_type)
                    WHERE c.job IN ('Director', 'Creator', 'Writer')
                ),
                works AS (
                    SELECT sa.seed_ordinal, sa.media_id AS seed_media_id,
                           sa.media_type AS seed_media_type, m.id, m.title, m.title_en,
                           m.media_type, m.overview, m.overview_en,
                           m.vote_average::float, m.vote_count::int, m.popularity::float,
                           m.poster_path, m.collection_id,
                           EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))::int AS release_year,
                           BOOL_OR(c.job IN ('Director', 'Creator')) AS via_director,
                           ARRAY(
                               SELECT g.name FROM {t_tmdb_genres} g
                               JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                               WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type) AND mg.media_type = m.media_type
                           ) as genres
                    FROM seed_auteurs sa
                    JOIN {t_tmdb_crew} c
                      ON c.person_id = sa.person_id
                     AND c.job IN ('Director', 'Creator', 'Writer')
                    JOIN {t_tmdb_media} m
                      ON (m.id, m.media_type) = (c.media_id, c.media_type)
                    WHERE NOT (m.id = sa.media_id AND m.media_type = sa.media_type)
                      AND m.vote_average >= 5.5
                      AND m.vote_count >= (
                          SELECT s2.vote_floor FROM seeds s2
                          WHERE s2.seed_ordinal = sa.seed_ordinal)
                    GROUP BY sa.seed_ordinal, sa.media_id, sa.media_type,
                             m.id, m.title, m.title_en, m.media_type, m.overview,
                             m.overview_en, m.vote_average, m.vote_count, m.popularity,
                             m.poster_path, m.collection_id,
                             EXTRACT(YEAR FROM COALESCE(m.release_date, m.first_air_date))
                    ORDER BY sa.seed_ordinal, m.id, m.popularity DESC
                )
                SELECT * FROM works
                "#;
    let ids: Vec<i64> = seeds.iter().map(|(id, _)| *id).collect();
    let mts: Vec<String> = seeds.iter().map(|(_, mt)| mt.clone()).collect();
    let floors: Vec<i64> = seeds
        .iter()
        .map(|(_, mt)| crate::recall::vote_floor_for(Some(mt.as_str())))
        .collect();
    let rows = match db
        .fetch(
            SQL,
            &[
                ("ids", Bind::I64Array(ids)),
                ("mts", Bind::TextArray(mts)),
                ("floors", Bind::I64Array(floors)),
            ],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => match e.into_degraded("director") {
            DbError::ChannelOff { .. } => return Ok(out),
            real => return Err(real),
        },
    };
    // split rows per seed by ordinal, then the same dedup/rank/cap
    let mut per_seed: HashMap<usize, Vec<sqlx::postgres::PgRow>> = HashMap::new();
    for r in rows {
        let ord = row::opt_i64(&r, "seed_ordinal").unwrap_or_default() as usize;
        per_seed.entry(ord.saturating_sub(1)).or_default().push(r);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = rank_auteur_works(per_seed.remove(&i).unwrap_or_default(), limit);
    }
    Ok(out)
}

// ── page rows: hidden gems / trending / popular ────────────────────

/// A page row (API-shaped, not scored): `tmdb_id`, `title`, rating.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PageRow {
    pub tmdb_id: i64,
    pub title: Option<String>,
    pub media_type: Option<String>,
    pub rating: Option<f64>,
    pub vote_count: Option<i64>,
    pub popularity: Option<f64>,
    pub poster_path: Option<String>,
    pub release_year: Option<i64>,
    pub via: Option<String>,
}

/// `_gem_or_trending_rows` — one medium's arm, built dynamically like
/// the Python `_fetch`. `mode` ∈ {"gems", "trending", "popular"}.
async fn gem_or_trending_arm(
    db: &Db,
    mode: &str,
    media_type: &str,
    limit: i64,
    genre_ids: Option<&[i64]>,
    year_min: Option<i64>,
    year_max: Option<i64>,
) -> Result<Vec<PageRow>, DbError> {
    let (table, title_col, date_col, gmap, gcol) = if media_type == "tv" {
        (
            "tmdb_tv",
            "name",
            "first_air_date",
            "tmdb_tv_genres_map",
            "tv_id",
        )
    } else {
        (
            "tmdb_movies",
            "title",
            "release_date",
            "tmdb_movie_genres_map",
            "movie_id",
        )
    };
    // gems thresholds mirror config.HIDDEN_GEMS_* (taste constants)
    const GEMS_RATING_MIN: f64 = 7.5;
    const GEMS_VOTES_MIN: i64 = 50;
    const GEMS_VOTES_MAX: i64 = 2000;

    let mut params: Vec<(&'static str, Bind)> = vec![(
        "floor",
        Bind::I64(crate::recall::vote_floor_for(Some(media_type))),
    )];
    let mut where_clauses: Vec<String> = vec!["vote_count >= $1".to_string()];
    if mode == "gems" {
        params.push(("rmin", Bind::F64(GEMS_RATING_MIN)));
        params.push(("vmin", Bind::I64(GEMS_VOTES_MIN)));
        params.push(("vmax", Bind::I64(GEMS_VOTES_MAX)));
        let n = params.len();
        where_clauses.push(format!("vote_average >= ${}", n - 2));
        where_clauses.push(format!("vote_count BETWEEN ${} AND ${}", n - 1, n));
    }
    if let Some(genres) = genre_ids.filter(|g| !g.is_empty()) {
        params.push(("genres", Bind::I64Array(genres.to_vec())));
        where_clauses.push(format!(
            "EXISTS (SELECT 1 FROM {} g WHERE g.{} = m.id AND g.genre_id = ANY(${}))",
            db.map().name(gmap)?,
            gcol,
            params.len()
        ));
    }
    if let Some(y) = year_min {
        params.push(("ymin", Bind::I64(y)));
        where_clauses.push(format!(
            "EXTRACT(YEAR FROM {}) >= ${}",
            date_col,
            params.len()
        ));
    }
    if let Some(y) = year_max {
        params.push(("ymax", Bind::I64(y)));
        where_clauses.push(format!(
            "EXTRACT(YEAR FROM {}) <= ${}",
            date_col,
            params.len()
        ));
    }
    let order = if mode == "gems" {
        "vote_average DESC, vote_count ASC, id ASC"
    } else {
        "popularity DESC, id ASC"
    };
    params.push(("limit", Bind::I64(limit)));

    let sql = format!(
        "SELECT m.id AS tmdb_id, m.{title_col} AS title, m.{title_col} AS title_en, \
         '{media_type}'::text AS media_type, m.vote_average::float AS rating, \
         m.vote_count::int AS vote_count, m.popularity::float AS popularity, \
         m.poster_path, EXTRACT(YEAR FROM {date_col})::int AS release_year \
         FROM {table} m WHERE {where} ORDER BY {order} LIMIT ${n}",
        title_col = title_col,
        media_type = media_type,
        date_col = date_col,
        table = db.map().name(table)?,
        where = where_clauses.join(" AND "),
        order = order,
        n = params.len(),
    );
    Ok(db
        .fetch(&sql, &params)
        .await?
        .iter()
        .map(|r| PageRow {
            tmdb_id: row::opt_i64(r, "tmdb_id").unwrap_or_default(),
            title: row::opt_text(r, "title"),
            media_type: row::opt_text(r, "media_type"),
            rating: row::opt_f64(r, "rating"),
            vote_count: row::opt_i64(r, "vote_count"),
            popularity: row::opt_f64(r, "popularity"),
            poster_path: row::opt_text(r, "poster_path"),
            release_year: row::opt_i64(r, "release_year"),
            via: None,
        })
        .collect())
}

/// `generate_hidden_gems` / `generate_trending_genre` /
/// `generate_popular_candidates` — the shared arm pair with the
/// rank-interleave (best movie, best tv, 2nd movie, …) so neither
/// medium monopolizes the row. Each side degrades independently.
pub async fn gem_or_trending_rows(
    db: &Db,
    mode: &str,
    limit: usize,
    genre_ids: Option<&[i64]>,
    year_min: Option<i64>,
    year_max: Option<i64>,
) -> Result<Vec<PageRow>, DbError> {
    let per_medium = limit.max(1) as i64;
    // each side degrades on its own (Python logs and continues)
    let movies = gem_or_trending_arm(db, mode, "movie", per_medium, genre_ids, year_min, year_max)
        .await
        .unwrap_or_default();
    let tvs = gem_or_trending_arm(db, mode, "tv", per_medium, genre_ids, year_min, year_max)
        .await
        .unwrap_or_default();

    let mut movies = movies;
    let mut tvs = tvs;
    let mut out: Vec<PageRow> = Vec::with_capacity(limit);
    while out.len() < limit && !(movies.is_empty() && tvs.is_empty()) {
        for bucket in [&mut movies, &mut tvs] {
            if !bucket.is_empty() && out.len() < limit {
                out.push(bucket.remove(0));
            }
        }
    }
    Ok(out)
}

/// `generate_popular_candidates` — popularity slice above the vote
/// floor, the /by-text recall fallback. Rows carry `via="text"`.
pub async fn generate_popular_candidates(
    db: &Db,
    limit: usize,
    year_min: Option<i64>,
    year_max: Option<i64>,
) -> Result<Vec<PageRow>, DbError> {
    let mut rows = gem_or_trending_rows(db, "popular", limit, None, year_min, year_max).await?;
    for r in &mut rows {
        r.via = Some("text".to_string());
    }
    Ok(rows)
}
