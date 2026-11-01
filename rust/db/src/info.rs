//! The phase-3 tail that phase 5 needs: seed-info batch, cinematic
//! tags, recall-boundary filters, and the batched semantic-overview
//! similarity — the four pieces `find_similar`'s compute path calls
//! that lived in Python's `queries.py`/`scoring.py` until now.
//!
//! SQL is copied verbatim from the Python sources (principle 7); the
//! one deviation is documented inline (seed vectors decode and are
//! discarded — only their NULL-ness drives the blend's skip rule).

use std::collections::{HashMap, HashSet};

use crate::executor::{row, Bind, Db, DbError};

// ---------------------------------------------------------------------------
// Seed info (queries.py `_QUERY_WITH_KEYWORDS` / `_row_to_movie_info`)
// ---------------------------------------------------------------------------

/// `_row_to_movie_info` — the seed/candidate info dict as a value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeedInfo {
    pub id: i64,
    pub title: String,
    pub title_en: Option<String>,
    pub media_type: String,
    pub rating: f64,
    pub poster_path: Option<String>,
    pub overview: String,
    pub original_language: Option<String>,
    pub overview_en: Option<String>,
    pub genres: Vec<String>,
    pub director: Option<String>,
    pub director_ids: Vec<i64>,
    pub writer_ids: Vec<i64>,
    pub composer_ids: Vec<i64>,
    pub dp_ids: Vec<i64>,
    pub keywords: Vec<String>,
    pub cast_list: Vec<String>,
    pub cast_ids: Vec<i64>,
    pub collection_id: Option<i64>,
    pub release_year: Option<i64>,
    pub companies: Vec<String>,
    pub networks: Vec<String>,
    /// cinematic tags (filled separately by [`fetch_cinematic_tags`]);
    /// Python mutates them onto the info dict — same shape, two reads
    pub narrative: Option<String>,
    pub pacing: Option<String>,
    pub emotional_arc: Vec<String>,
}

const SEED_INFO_SQL_HEAD: &str = r#"
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
            FROM {t_tmdb_media_genres} mg
            JOIN {t_tmdb_genres} g ON g.id = mg.genre_id
            WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
        ) as genres,
        (
            SELECT p.name
            FROM {t_tmdb_crew} c
            JOIN {t_tmdb_people} p ON c.person_id = p.id
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job IN ('Director', 'Creator')
            LIMIT 1
        ) as director,
        ARRAY(
            SELECT c.person_id
            FROM {t_tmdb_crew} c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job IN ('Director', 'Creator')
        ) as director_ids,
        ARRAY(
            SELECT c.person_id
            FROM {t_tmdb_crew} c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Writer'
        ) as writer_ids,
        ARRAY(
            SELECT c.person_id
            FROM {t_tmdb_crew} c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Original Music Composer'
        ) as composer_ids,
        ARRAY(
            SELECT c.person_id
            FROM {t_tmdb_crew} c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
              AND c.job = 'Director of Photography'
        ) as dp_ids,
        ARRAY(
            SELECT p.name
            FROM {t_tmdb_cast} c
            JOIN {t_tmdb_people} p ON c.person_id = p.id
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
            ORDER BY c.cast_order ASC NULLS LAST
            LIMIT 5
        ) as cast_list,
        ARRAY(
            SELECT c.person_id
            FROM {t_tmdb_cast} c
            WHERE (c.media_id, c.media_type) = (m.id, m.media_type)
            ORDER BY c.cast_order ASC NULLS LAST
            LIMIT 5
        ) as cast_ids,
        ARRAY(
            SELECT k.name
            FROM {t_tmdb_media_keywords} mk
            JOIN {t_tmdb_keywords} k ON k.id = mk.keyword_id
            WHERE (mk.media_id, mk.media_type) = (m.id, m.media_type)
        ) as keywords,
        ARRAY(
            SELECT pc.name
            FROM {t_tmdb_media_companies} mc
            JOIN {t_tmdb_production_companies} pc ON mc.company_id = pc.id
            WHERE (mc.media_id, mc.media_type) = (m.id, m.media_type)
        ) as companies,
        ARRAY(
            SELECT n.name
            FROM {t_tmdb_media_networks} mn
            JOIN {t_tmdb_networks} n ON n.id = mn.network_id
            WHERE (mn.media_id, mn.media_type) = (m.id, m.media_type)
        ) as networks
    FROM {t_tmdb_media} m
    WHERE m.id = ANY($1::bigint[])"#;

/// `get_movie_info_batch` — full seed infos in one query per
/// media-type group. Falls back to the keywords-less query when the
/// keywords tables are missing (Python catches UndefinedTableError —
/// SQLSTATE 42P01 ONLY, not the broader degrade set).
pub async fn get_movie_info_batch(
    db: &Db,
    ids: &[i64],
    media_type: Option<&str>,
) -> Result<HashMap<i64, SeedInfo>, DbError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let with_keywords_base = db.sql(SEED_INFO_SQL_HEAD)?;
    // Python: resolve(query) + " AND m.media_type = $2" — the filter is
    // appended AFTER resolution, same here
    let with_keywords = match media_type {
        Some(_) => format!("{with_keywords_base} AND m.media_type = $2"),
        None => with_keywords_base.clone(),
    };
    // Python derives the no-keywords query by string-replacing the
    // keywords ARRAY block before name resolution; this is the result
    let no_keywords = with_keywords.replace(
        "        ARRAY(\n            SELECT k.name\n            FROM {t_tmdb_media_keywords} mk\n            JOIN {t_tmdb_keywords} k ON k.id = mk.keyword_id\n            WHERE (mk.media_id, mk.media_type) = (m.id, m.media_type)\n        ) as keywords,",
        "        ARRAY[]::text[] as keywords,",
    );
    // resolve AFTER the replace, exactly like Python
    let no_keywords = db.sql(&no_keywords)?;
    let no_keywords = match media_type {
        Some(_) => format!("{no_keywords} AND m.media_type = $2"),
        None => no_keywords,
    };

    let seed_binds = |mt: Option<&str>| -> Vec<(&'static str, Bind)> {
        let mut binds: Vec<(&'static str, Bind)> =
            vec![("ids", Bind::I64Array(ids.to_vec()))];
        if let Some(mt) = mt {
            binds.push(("mt", Bind::Text(mt.to_string())));
        }
        binds
    };

    let rows = match db.fetch(&with_keywords, &seed_binds(media_type)).await {
        Ok(r) => r,
        Err(e) if matches!(&e, DbError::Sqlx(se) if crate::degrade::pg_code(se) == Some("42P01")) => {
            db.fetch(&no_keywords, &seed_binds(media_type)).await?
        }
        Err(e) => return Err(e),
    };

    let mut out = HashMap::with_capacity(rows.len());
    for r in &rows {
        let info = seed_info_from_row(r)?;
        out.insert(info.id, info);
    }
    Ok(out)
}

fn seed_info_from_row(r: &sqlx::postgres::PgRow) -> Result<SeedInfo, DbError> {
    Ok(SeedInfo {
        id: row::opt_i64(r, "id")?.unwrap_or_default(),
        title: row::opt_text(r, "title")?.unwrap_or_default(),
        title_en: row::opt_text(r, "title_en")?,
        media_type: row::opt_text(r, "media_type")?.unwrap_or_default(),
        // float(row["rating"] or 0)
        rating: row::opt_f64(r, "rating")?.unwrap_or(0.0),
        poster_path: row::opt_text(r, "poster_path")?,
        overview: row::opt_text(r, "overview")?.unwrap_or_default(),
        original_language: row::opt_text(r, "original_language")?,
        overview_en: row::opt_text(r, "overview_en")?,
        genres: row::opt_text_array(r, "genres")?.unwrap_or_default(),
        director: row::opt_text(r, "director")?,
        director_ids: row::opt_i64_array(r, "director_ids")?.unwrap_or_default(),
        writer_ids: row::opt_i64_array(r, "writer_ids")?.unwrap_or_default(),
        composer_ids: row::opt_i64_array(r, "composer_ids")?.unwrap_or_default(),
        dp_ids: row::opt_i64_array(r, "dp_ids")?.unwrap_or_default(),
        keywords: row::opt_text_array(r, "keywords")?.unwrap_or_default(),
        cast_list: row::opt_text_array(r, "cast_list")?.unwrap_or_default(),
        cast_ids: row::opt_i64_array(r, "cast_ids")?.unwrap_or_default(),
        collection_id: row::opt_i64(r, "collection_id")?,
        release_year: row::opt_i64(r, "release_year")?,
        companies: row::opt_text_array(r, "companies")?.unwrap_or_default(),
        networks: row::opt_text_array(r, "networks")?.unwrap_or_default(),
        narrative: None,
        pacing: None,
        emotional_arc: Vec::new(),
    })
}

/// Seed cinematic structure tags (step 1b) — narrative/pacing/
/// emotional_arc powers narrative_match + the mood features. Missing
/// rows leave the seed's tags unset (Python degrades the whole read).
pub async fn fetch_cinematic_tags(
    db: &Db,
    ids: &[i64],
) -> Result<HashMap<i64, (Option<String>, Option<String>, Vec<String>)>, DbError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = db.sql(
        "SELECT media_id, narrative_complexity, pacing, emotional_arc \
         FROM {t_tmdb_cinematic} WHERE media_id = ANY($1::bigint[])",
    )?;
    let rows = match db
        .fetch(&sql, &[("ids", Bind::I64Array(ids.to_vec()))])
        .await
    {
        Ok(r) => r,
        Err(_) => return Ok(HashMap::new()), // degrade — tags stay unset
    };
    let mut out = HashMap::with_capacity(rows.len());
    for r in &rows {
        let id = row::opt_i64(r, "media_id")?.unwrap_or_default();
        out.insert(
            id,
            (
                row::opt_text(r, "narrative_complexity")?,
                row::opt_text(r, "pacing")?,
                row::opt_text_array(r, "emotional_arc")?.unwrap_or_default(),
            ),
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Recall-boundary session filters (queries.py `apply_recall_filters`)
// ---------------------------------------------------------------------------

/// The RFC §3 session filter set (validated upstream by the caller —
/// `normalize_filters` is a Python-boundary concern).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecallFilters {
    pub year_min: Option<i64>,
    pub year_max: Option<i64>,
    pub genre_ids: Option<Vec<i64>>,
    pub exclude_genre_ids: Option<Vec<i64>>,
    /// minutes; movies only — tv has no runtime
    pub max_runtime: Option<i64>,
}

/// Result of the eligibility query: the eligible (id, media_type) set,
/// or `degraded` when the filter tables are missing (caller keeps the
/// unfiltered list — a filter never 500s a query path).
#[derive(Debug, Default)]
pub struct FilterOutcome {
    pub eligible: HashSet<(i64, String)>,
    pub degraded: bool,
}

pub async fn apply_recall_filters(
    db: &Db,
    candidates: &[(i64, String)],
    filters: &RecallFilters,
) -> Result<FilterOutcome, DbError> {
    if filters == &RecallFilters::default() || candidates.is_empty() {
        return Ok(FilterOutcome {
            eligible: candidates.iter().cloned().collect(),
            degraded: false,
        });
    }
    let movie_ids: Vec<i64> = {
        let mut v: Vec<i64> = candidates
            .iter()
            .filter(|(_, mt)| mt != "tv")
            .map(|(id, _)| *id)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let tv_ids: Vec<i64> = {
        let mut v: Vec<i64> = candidates
            .iter()
            .filter(|(_, mt)| mt == "tv")
            .map(|(id, _)| *id)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut eligible: HashSet<(i64, String)> = HashSet::new();

    // clause builder mirroring Python's _year_clause + per-medium
    // appends; params are positional $1.. and bound in build order
    let year_clauses = |col: &str, p: &mut Vec<Bind>| -> Vec<String> {
        let mut clauses = Vec::new();
        if let Some(y) = filters.year_min {
            p.push(Bind::I64(y));
            clauses.push(format!("EXTRACT(YEAR FROM {col}) >= ${}", p.len()));
        }
        if let Some(y) = filters.year_max {
            p.push(Bind::I64(y));
            clauses.push(format!("EXTRACT(YEAR FROM {col}) <= ${}", p.len()));
        }
        clauses
    };

    let res = async {
        if !movie_ids.is_empty() {
            let mut p = vec![Bind::I64Array(movie_ids.clone())];
            let mut clauses = year_clauses("m.release_date", &mut p);
            if let Some(rt) = filters.max_runtime {
                p.push(Bind::I64(rt));
                clauses.push(format!("m.runtime <= ${}", p.len()));
            }
            if let Some(g) = &filters.genre_ids {
                p.push(Bind::I64Array(g.clone()));
                clauses.push(format!(
                    "EXISTS (SELECT 1 FROM {} g WHERE g.movie_id = m.id AND g.genre_id = ANY(${}))",
                    db.map().name("tmdb_movie_genres_map")?,
                    p.len()
                ));
            }
            if let Some(g) = &filters.exclude_genre_ids {
                p.push(Bind::I64Array(g.clone()));
                clauses.push(format!(
                    "NOT EXISTS (SELECT 1 FROM {} g WHERE g.movie_id = m.id AND g.genre_id = ANY(${}))",
                    db.map().name("tmdb_movie_genres_map")?,
                    p.len()
                ));
            }
            let where_ = std::iter::once("m.id = ANY($1)".to_string())
                .chain(clauses)
                .collect::<Vec<_>>()
                .join(" AND ");
            let sql = format!(
                "SELECT m.id FROM {} m WHERE {where_}",
                db.map().name("tmdb_movies")?
            );
            for r in fetch_positional(db, &sql, p).await? {
                eligible.insert((
                    row::opt_i64(&r, "id")?.unwrap_or_default(),
                    "movie".to_string(),
                ));
            }
        }
        if !tv_ids.is_empty() {
            let mut p = vec![Bind::I64Array(tv_ids.clone())];
            let mut clauses = year_clauses("t.first_air_date", &mut p);
            if let Some(g) = &filters.genre_ids {
                p.push(Bind::I64Array(g.clone()));
                clauses.push(format!(
                    "EXISTS (SELECT 1 FROM {} g WHERE g.tv_id = t.id AND g.genre_id = ANY(${}))",
                    db.map().name("tmdb_tv_genres_map")?,
                    p.len()
                ));
            }
            if let Some(g) = &filters.exclude_genre_ids {
                p.push(Bind::I64Array(g.clone()));
                clauses.push(format!(
                    "NOT EXISTS (SELECT 1 FROM {} g WHERE g.tv_id = t.id AND g.genre_id = ANY(${}))",
                    db.map().name("tmdb_tv_genres_map")?,
                    p.len()
                ));
            }
            let where_ = std::iter::once("t.id = ANY($1)".to_string())
                .chain(clauses)
                .collect::<Vec<_>>()
                .join(" AND ");
            let sql = format!(
                "SELECT t.id FROM {} t WHERE {where_}",
                db.map().name("tmdb_tv")?
            );
            for r in fetch_positional(db, &sql, p).await? {
                eligible.insert((
                    row::opt_i64(&r, "id")?.unwrap_or_default(),
                    "tv".to_string(),
                ));
            }
        }
        Ok::<(), DbError>(())
    }
    .await;

    match res {
        Ok(()) => Ok(FilterOutcome { eligible, degraded: false }),
        // filter tables missing — degrade open (Python: except Exception)
        Err(_) => Ok(FilterOutcome {
            eligible: candidates.iter().cloned().collect(),
            degraded: true,
        }),
    }
}

/// named-bind labels for the dynamic filter queries (positional $N in
/// SQL order — labels are decorative for sqlx)
const BOXED_NAMES: [&str; 8] = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"];

async fn fetch_positional(
    db: &Db,
    sql: &str,
    binds: Vec<Bind>,
) -> Result<Vec<sqlx::postgres::PgRow>, DbError> {
    let named: Vec<(&'static str, Bind)> = binds
        .into_iter()
        .enumerate()
        .map(|(i, b)| (BOXED_NAMES[i], b))
        .collect();
    db.fetch(sql, &named).await
}

// ---------------------------------------------------------------------------
// Batched semantic-overview similarity (scoring.py)
// ---------------------------------------------------------------------------

fn fact_emb_union(db: &Db, cols_sql: &str) -> Result<String, DbError> {
    Ok(format!(
        "(SELECT id, {cols_sql} FROM {movies} UNION ALL SELECT id, {cols_sql} FROM {tv})",
        movies = db.map().name("tmdb_movies")?,
        tv = db.map().name("tmdb_tv")?,
    ))
}

/// `semantic_overview_similarity_batch` — all seed×candidate cosines
/// in one cross-join query per blend column. Same blend math and skip
/// rule as the single-seed version: a candidate needs EVERY column
/// active for the seed. Returns {seed_id: {candidate_id: sim}};
/// seeds with no usable vector are omitted.
///
/// `spec` overrides the deployment blend (`columns`+`weights` in
/// Python); `default_column` is EMBEDDING_COLUMN — the caller owns
/// config, this layer stays config-free.
///
/// DEVIATION (documented): Python reads the seed VECTORS and keeps
/// them in memory; downstream only tests them for None. We decode and
/// discard — the null-ness drives everything, bytes are identical.
pub async fn semantic_overview_similarity_batch(
    db: &Db,
    seed_ids: &[i64],
    candidate_ids: &[i64],
    spec: Option<&[(String, f64)]>,
    default_column: &str,
    seed_vectors: Option<&HashMap<i64, HashMap<String, Vec<f32>>>>,
) -> Result<HashMap<i64, HashMap<i64, f64>>, DbError> {
    if candidate_ids.is_empty() || seed_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let default_spec: Vec<(String, f64)> = vec![(default_column.to_string(), 1.0)];
    let active_spec: &[(String, f64)] = spec.unwrap_or(&default_spec);
    let cols: Vec<&str> = active_spec.iter().map(|(c, _)| c.as_str()).collect();
    let col_weight = |c: &str| -> f64 {
        active_spec
            .iter()
            .find(|(k, _)| k == c)
            .map(|(_, w)| *w)
            .unwrap_or(0.0)
    };

    let cols_sql = cols.join(", ");
    let union = fact_emb_union(db, &cols_sql)?;
    // All seed vectors in one query (last row wins on a colliding
    // movie/tv id — same tie behavior as the single-seed path: the
    // HashMap insert overwrites, like Python's dict assignment)
    let seed_sql = format!(
        "SELECT id, {cols_sql} FROM {union} m WHERE id = ANY($1::bigint[])"
    );
    let seed_rows = match db
        .fetch(&seed_sql, &[("ids", Bind::I64Array(seed_ids.to_vec()))])
        .await
    {
        Ok(r) => r,
        // UndefinedColumn/UndefinedObject → {} (channel off), verbatim
        Err(e) if matches!(&e, DbError::Sqlx(se) if crate::degrade::pg_code(se) == Some("42703") || crate::degrade::pg_code(se) == Some("42704")) => {
            return Ok(HashMap::new())
        }
        Err(e) => return Err(e),
    };

    // seed id -> per-column presence (decoded vectors are discarded)
    let mut seed_cols: HashMap<i64, Vec<bool>> = HashMap::new();
    for r in &seed_rows {
        let id = row::opt_i64(r, "id")?.unwrap_or_default();
        let present: Vec<bool> = cols
            .iter()
            .map(|c| row::opt_vector(r, c).ok().flatten().is_some())
            .collect();
        seed_cols.insert(id, present);
    }
    if let Some(overrides) = seed_vectors {
        for (sid, ovr) in overrides {
            if ovr.is_empty() {
                continue;
            }
            let mut merged = seed_cols
                .get(sid)
                .cloned()
                .unwrap_or_else(|| vec![false; cols.len()]);
            for (i, c) in cols.iter().enumerate() {
                if ovr.get(*c).is_some_and(|v| !v.is_empty()) {
                    merged[i] = true;
                }
            }
            seed_cols.insert(*sid, merged);
        }
    }

    let usable: HashMap<i64, &Vec<bool>> = seed_cols
        .iter()
        .filter(|(_, present)| present.iter().any(|p| *p))
        .map(|(k, v)| (*k, v))
        .collect();
    if usable.is_empty() {
        return Ok(HashMap::new());
    }

    // one cross-join per column with at least one usable seed
    let mut sims_by_col: Vec<HashMap<(i64, i64), f64>> = vec![HashMap::new(); cols.len()];
    let usable_ids: Vec<i64> = usable.keys().copied().collect();
    for (ci, col) in cols.iter().enumerate() {
        let col_active = usable.values().any(|p| p[ci]);
        if !col_active {
            continue;
        }
        let sql = format!(
            r#"
                    SELECT s.id AS seed_id, c.id AS cand_id,
                           GREATEST(0.0, 1 - (c.{col} <=> s.{col})) AS sim
                    FROM {union} s
                    CROSS JOIN {union} c
                    WHERE s.id = ANY($1::bigint[]) AND c.id = ANY($2::bigint[])
                      AND s.{col} IS NOT NULL AND c.{col} IS NOT NULL
                    "#,
        );
        let rows = match db
            .fetch(
                &sql,
                &[
                    ("seeds", Bind::I64Array(usable_ids.clone())),
                    ("cands", Bind::I64Array(candidate_ids.to_vec())),
                ],
            )
            .await
        {
            Ok(r) => r,
            Err(e) if matches!(&e, DbError::Sqlx(se) if crate::degrade::pg_code(se) == Some("42703") || crate::degrade::pg_code(se) == Some("42704")) => {
                return Ok(HashMap::new())
            }
            Err(e) => return Err(e),
        };
        for r in &rows {
            let s = row::opt_i64(r, "seed_id")?.unwrap_or_default();
            let c = row::opt_i64(r, "cand_id")?.unwrap_or_default();
            sims_by_col[ci].insert(
                (s, c),
                row::opt_f64(r, "sim")?.unwrap_or_default(),
            );
        }
    }

    // blend per seed: candidate needs every column active FOR THE SEED
    let mut out: HashMap<i64, HashMap<i64, f64>> = HashMap::new();
    for (sid, present) in &usable {
        let active_cols: Vec<usize> = present
            .iter()
            .enumerate()
            .filter(|(_, p)| **p)
            .map(|(i, _)| i)
            .collect();
        let w_sum: f64 = active_cols.iter().map(|&i| col_weight(cols[i])).sum();
        let mut per_cand: HashMap<i64, f64> = HashMap::new();
        // candidate ids seen in ANY active column for this seed
        let mut cand_seen: HashSet<i64> = HashSet::new();
        for &ci in &active_cols {
            for (s, c) in sims_by_col[ci].keys() {
                if *s == *sid {
                    cand_seen.insert(*c);
                }
            }
        }
        for cid in cand_seen {
            let mut pair_sims: Vec<f64> = Vec::with_capacity(active_cols.len());
            let mut missing = false;
            for &ci in &active_cols {
                match sims_by_col[ci].get(&(*sid, cid)) {
                    Some(v) => pair_sims.push(*v),
                    None => {
                        missing = true; // candidate lacks one of the seed's columns
                        break;
                    }
                }
            }
            if missing {
                continue;
            }
            let blended: f64 = active_cols
                .iter()
                .zip(&pair_sims)
                .map(|(&ci, s)| col_weight(cols[ci]) * s)
                .sum::<f64>()
                / f64::max(w_sum, 1e-9);
            per_cand.insert(cid, blended);
        }
        if !per_cand.is_empty() {
            out.insert(*sid, per_cand);
        }
    }
    Ok(out)
}
