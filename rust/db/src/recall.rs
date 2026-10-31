//! Recall channels — the Rust port of `queries.py` (phase 3 scope:
//! the genre channel, batch enrichment, and KNN; the remaining
//! channels land with the rest of the phase).
//!
//! Faithfulness rules:
//! * SQL text is copied VERBATIM from the Python source (including the
//!   comment banners that explain query-shape decisions — a planner
//!   hint documented only in a comment is part of the behavior).
//! * Every query goes through the `{t_*}` resolver (or builds names
//!   via `SchemaMap::name` where Python did — the KNN arms).
//! * Degrade points mirror the Python excepts exactly: keywords and
//!   cinematic tolerate a missing TABLE; KNN latches off on missing
//!   column/object/table.

use std::collections::HashMap;

use crate::executor::{row, Bind, Db, DbError};

/// A recall candidate row — the Python dict shape of
/// `generate_candidates` / KNN rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CandidateRow {
    pub id: i64,
    pub title: Option<String>,
    pub title_en: Option<String>,
    pub media_type: Option<String>,
    pub overview: Option<String>,
    pub overview_en: Option<String>,
    pub vote_average: Option<f64>,
    pub vote_count: Option<i64>,
    pub popularity: Option<f64>,
    pub poster_path: Option<String>,
    pub collection_id: Option<i64>,
    pub release_year: Option<i64>,
    pub genres: Vec<String>,
    /// recall channel tag ("knn" for the semantic channel)
    pub via: Option<String>,
    pub knn_similarity: Option<f64>,
}

impl CandidateRow {
    fn from_recall_row(r: &sqlx::postgres::PgRow) -> Self {
        Self {
            id: row::opt_i64(r, "id").unwrap_or_default(),
            title: row::opt_text(r, "title"),
            title_en: row::opt_text(r, "title_en"),
            media_type: row::opt_text(r, "media_type"),
            overview: row::opt_text(r, "overview"),
            overview_en: row::opt_text(r, "overview_en"),
            vote_average: row::opt_f64(r, "vote_average"),
            vote_count: row::opt_i64(r, "vote_count"),
            popularity: row::opt_f64(r, "popularity"),
            poster_path: row::opt_text(r, "poster_path"),
            collection_id: row::opt_i64(r, "collection_id"),
            release_year: row::opt_i64(r, "release_year"),
            genres: row::opt_text_array(r, "genres").unwrap_or_default(),
            via: None,
            knn_similarity: row::opt_f64(r, "knn_similarity"),
        }
    }
}

/// `generate_candidates` — the genre channel: same media_type and at
/// least one overlapping genre, ordered by popularity then rating.
///
/// `allow_cross_media` keeps a per-medium budget of `limit` (a global
/// order let one medium crowd out the other's classics).
pub async fn generate_candidates(
    db: &Db,
    base_movie_id: i64,
    base_genres: &[String],
    media_type: &str,
    limit: i64,
    allow_cross_media: bool,
) -> Result<Vec<CandidateRow>, DbError> {
    if base_genres.is_empty() {
        return Ok(Vec::new());
    }

    const CROSS_SQL: &str = r#"
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
                        FROM {t_tmdb_genres} g
                        JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM {t_tmdb_media} m
                WHERE m.id != $1
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM {t_tmdb_media_genres} mg
                      JOIN {t_tmdb_genres} g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                        AND g.name = ANY($2::text[])
                  )
                ORDER BY
                    -- genre-overlap count first (the channel's whole point),
                    -- then a quality blend — raw popularity alone buried
                    -- classics under fresh high-pop titles (#1)
                    (
                        SELECT count(*) FROM {t_tmdb_media_genres} mg
                        JOIN {t_tmdb_genres} g ON mg.genre_id = g.id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                          AND g.name = ANY($2::text[])
                    ) DESC,
                    (m.vote_count * m.vote_average) DESC,
                    m.popularity DESC
                LIMIT $3 * 2  -- per-medium budget: the dedup below keeps
                              -- `limit` per media_type, so one hot medium
                              -- can no longer starve the other (#1)
                "#;

    const SAME_SQL: &str = r#"
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
                        FROM {t_tmdb_genres} g
                        JOIN {t_tmdb_media_genres} mg ON g.id = mg.genre_id
                        WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                    ) as genres
                FROM {t_tmdb_media} m
                WHERE m.media_type = $1
                  AND m.id != $2
                  AND m.vote_average >= 5.5
                  AND EXISTS (
                      SELECT 1 FROM {t_tmdb_media_genres} mg
                      JOIN {t_tmdb_genres} g ON mg.genre_id = g.id
                      WHERE (mg.media_id, mg.media_type) = (m.id, m.media_type)
                        AND g.name = ANY($3::text[])
                  )
                ORDER BY m.popularity DESC, m.vote_average DESC
                LIMIT $4
                "#;

    let rows = if allow_cross_media {
        db.fetch(
            CROSS_SQL,
            &[
                ("id", Bind::I64(base_movie_id)),
                ("genres", Bind::TextArray(base_genres.to_vec())),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await?
    } else {
        db.fetch(
            SAME_SQL,
            &[
                ("media_type", Bind::Text(media_type.to_string())),
                ("id", Bind::I64(base_movie_id)),
                ("genres", Bind::TextArray(base_genres.to_vec())),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await?
    };

    let mut cands: Vec<CandidateRow> = rows.iter().map(CandidateRow::from_recall_row).collect();

    if allow_cross_media {
        // per-medium budget: rank within each media_type, keep `limit`
        // per side (issue #1 — a UNION ordered globally let one medium
        // crowd out the other's classics)
        let base_set: std::collections::HashSet<&String> = base_genres.iter().collect();
        cands.sort_by(|a, b| {
            let overlap = |c: &CandidateRow| {
                c.genres.iter().filter(|g| base_set.contains(g)).count()
            };
            let vc = |c: &CandidateRow| (c.vote_count.unwrap_or(0) as f64)
                * c.vote_average.unwrap_or(0.0);
            overlap(b)
                .cmp(&overlap(a))
                .then_with(|| vc(b).partial_cmp(&vc(a)).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| {
                    b.popularity
                        .partial_cmp(&a.popularity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let mut kept: HashMap<String, i64> = HashMap::new();
        cands.retain(|c| {
            let mt = c.media_type.clone().unwrap_or_default();
            let n = kept.entry(mt).or_insert(0);
            if *n >= limit {
                false
            } else {
                *n += 1;
                true
            }
        });
    }
    Ok(cands)
}

/// One enrichment bundle — the per-candidate dict shape of
/// `enrich_candidates_batch`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Enrichment {
    pub director: Option<String>,
    pub director_ids: Vec<i64>,
    pub writer_ids: Vec<i64>,
    pub composer_ids: Vec<i64>,
    pub dp_ids: Vec<i64>,
    pub cast_list: Vec<String>,
    pub cast_ids: Vec<i64>,
    pub keywords: Vec<String>,
    pub companies: Vec<String>,
    pub networks: Vec<String>,
    pub collection_id: Option<i64>,
    pub original_language: Option<String>,
    pub narrative: Option<String>,
    pub pacing: Option<String>,
    pub emotional_arc: Vec<String>,
}

/// `enrich_candidates_batch` — director/writer/composer/dp/cast/
/// keywords/companies/networks/collection/language/narrative for many
/// candidates in batched queries.
///
/// Degrade: keywords and cinematic tolerate a MISSING TABLE (channel
/// off in place); everything else must exist.
pub async fn enrich_candidates_batch(
    db: &Db,
    candidate_ids: &[i64],
    media_types: Option<&[String]>,
) -> Result<HashMap<i64, Enrichment>, DbError> {
    if candidate_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let mut out: HashMap<i64, Enrichment> = candidate_ids
        .iter()
        .map(|id| (*id, Enrichment::default()))
        .collect();

    // Batch: directors — names for display/string-fallback plus the
    // person_id array the scoring layer matches on.
    let director_sql = if media_types.is_some() {
        r#"
                SELECT c.media_id,
                       MIN(p.name) as director,
                       ARRAY_AGG(c.person_id) as director_ids
                FROM {t_tmdb_crew} c
                JOIN {t_tmdb_people} p ON c.person_id = p.id
                WHERE (c.media_id, c.media_type) IN (
                          SELECT * FROM unnest($1::bigint[], $2::text[])
                      )
                  AND c.job IN ('Director', 'Creator')
                GROUP BY c.media_id
                "#
    } else {
        r#"
                SELECT c.media_id,
                       MIN(p.name) as director,
                       ARRAY_AGG(c.person_id) as director_ids
                FROM {t_tmdb_crew} c
                JOIN {t_tmdb_people} p ON c.person_id = p.id
                WHERE c.media_id = ANY($1::bigint[])
                  AND c.job IN ('Director', 'Creator')
                GROUP BY c.media_id
                "#
    };
    let director_binds: Vec<(&'static str, Bind)> = match media_types {
        Some(mts) => vec![
            ("ids", Bind::I64Array(candidate_ids.to_vec())),
            ("mts", Bind::TextArray(mts.to_vec())),
        ],
        None => vec![("ids", Bind::I64Array(candidate_ids.to_vec()))],
    };
    for r in db.fetch(director_sql, &director_binds).await? {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        let e = out.entry(mid).or_default();
        e.director = row::opt_text(&r, "director");
        e.director_ids = row::opt_i64_array(&r, "director_ids").unwrap_or_default();
    }

    // Batch: writers (person_ids for the writer-DNA match)
    let writer_sql = if media_types.is_some() {
        r#"
                SELECT c.media_id, ARRAY_AGG(DISTINCT c.person_id) as writer_ids
                FROM {t_tmdb_crew} c
                WHERE (c.media_id, c.media_type) IN (
                          SELECT * FROM unnest($1::bigint[], $2::text[])
                      )
                  AND c.job = 'Writer'
                GROUP BY c.media_id
                "#
    } else {
        r#"
                SELECT c.media_id, ARRAY_AGG(DISTINCT c.person_id) as writer_ids
                FROM {t_tmdb_crew} c
                WHERE c.media_id = ANY($1::bigint[])
                  AND c.job = 'Writer'
                GROUP BY c.media_id
                "#
    };
    for r in db.fetch(writer_sql, &director_binds).await? {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        out.entry(mid)
            .or_default()
            .writer_ids = row::opt_i64_array(&r, "writer_ids").unwrap_or_default();
    }

    // Batch: composer + cinematographer ids (cinematic DNA beyond the
    // director — the visual/audio language of a body of work)
    const DNA_SQL: &str = r#"
            SELECT c.media_id,
                   ARRAY_AGG(c.person_id) FILTER (WHERE c.job = 'Original Music Composer')
                       AS composer_ids,
                   ARRAY_AGG(c.person_id) FILTER (WHERE c.job = 'Director of Photography')
                       AS dp_ids
            FROM {t_tmdb_crew} c
            WHERE c.media_id = ANY($1::bigint[])
              AND c.job IN ('Original Music Composer', 'Director of Photography')
            GROUP BY c.media_id
            "#;
    for r in db
        .fetch(
            DNA_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await?
    {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        let e = out.entry(mid).or_default();
        e.composer_ids = row::opt_i64_array(&r, "composer_ids").unwrap_or_default();
        e.dp_ids = row::opt_i64_array(&r, "dp_ids").unwrap_or_default();
    }

    // Batch: cast (top 5 per movie, ordered) — names for display,
    // person_ids for transliteration-proof similarity.
    const CAST_SQL: &str = r#"
            SELECT c.media_id, p.name, c.person_id
            FROM {t_tmdb_cast} c
            JOIN {t_tmdb_people} p ON c.person_id = p.id
            WHERE c.media_id = ANY($1::bigint[])
            ORDER BY c.cast_order ASC NULLS LAST
            "#;
    for r in db
        .fetch(
            CAST_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await?
    {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        let e = out.entry(mid).or_default();
        if e.cast_list.len() < 5 {
            if let Some(name) = row::opt_text(&r, "name") {
                e.cast_list.push(name);
            }
            if let Some(pid) = row::opt_i64(&r, "person_id") {
                e.cast_ids.push(pid);
            }
        }
    }

    // Batch: keywords (graceful degradation if tables missing)
    const KEYWORDS_SQL: &str = r#"
                SELECT mk.media_id, k.name
                FROM {t_tmdb_media_keywords} mk
                JOIN {t_tmdb_keywords} k ON mk.keyword_id = k.id
                WHERE mk.media_id = ANY($1::bigint[])
                "#;
    match db
        .fetch(
            KEYWORDS_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await
    {
        Ok(rows) => {
            for r in rows {
                let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
                if let Some(name) = row::opt_text(&r, "name") {
                    out.entry(mid).or_default().keywords.push(name);
                }
            }
        }
        Err(e) => match e.into_degraded("keywords") {
            // keywords table not found — keyword scoring disabled
            DbError::ChannelOff { .. } => {}
            real => return Err(real),
        },
    }

    // Batch: production companies
    const COMPANY_SQL: &str = r#"
            SELECT mc.media_id, pc.name
            FROM {t_tmdb_media_companies} mc
            JOIN {t_tmdb_production_companies} pc ON mc.company_id = pc.id
            WHERE mc.media_id = ANY($1::bigint[])
            "#;
    for r in db
        .fetch(
            COMPANY_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await?
    {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        if let Some(name) = row::opt_text(&r, "name") {
            out.entry(mid).or_default().companies.push(name);
        }
    }

    // Batch: networks (TV)
    const NETWORK_SQL: &str = r#"
            SELECT mn.media_id, n.name
            FROM {t_tmdb_media_networks} mn
            JOIN {t_tmdb_networks} n ON mn.network_id = n.id
            WHERE mn.media_id = ANY($1::bigint[])
            "#;
    for r in db
        .fetch(
            NETWORK_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await?
    {
        let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
        if let Some(name) = row::opt_text(&r, "name") {
            out.entry(mid).or_default().networks.push(name);
        }
    }

    // Batch: collection_ids + original_language (modality feature)
    const COLL_SQL: &str = r#"
            SELECT id, collection_id, original_language
            FROM {t_tmdb_media}
            WHERE id = ANY($1::bigint[])
            "#;
    for r in db
        .fetch(
            COLL_SQL,
            &[("ids", Bind::I64Array(candidate_ids.to_vec()))],
        )
        .await?
    {
        let id = row::opt_i64(&r, "id").unwrap_or_default();
        let e = out.entry(id).or_default();
        e.collection_id = row::opt_i64(&r, "collection_id");
        e.original_language = row::opt_text(&r, "original_language");
    }

    // Batch: cinematic structure tags (narrative/pacing/emotional_arc)
    // for the narrative-match + mood features. Graceful when enrichment
    // is partial.
    const CIN_SQL: &str = r#"
                SELECT media_id, narrative_complexity, pacing, emotional_arc
                FROM {t_tmdb_cinematic}
                WHERE media_id = ANY($1::bigint[])
                "#;
    match db
        .fetch(CIN_SQL, &[("ids", Bind::I64Array(candidate_ids.to_vec()))])
        .await
    {
        Ok(rows) => {
            for r in rows {
                let mid = row::opt_i64(&r, "media_id").unwrap_or_default();
                let e = out.entry(mid).or_default();
                e.narrative = row::opt_text(&r, "narrative_complexity");
                e.pacing = row::opt_text(&r, "pacing");
                e.emotional_arc =
                    row::opt_text_array(&r, "emotional_arc").unwrap_or_default();
            }
        }
        Err(e) => match e.into_degraded("cinematic") {
            // tmdb_cinematic table missing — narrative feature off
            DbError::ChannelOff { .. } => {}
            real => return Err(real),
        },
    }

    Ok(out)
}

/// `_vote_floor` — TV vote counts run an order of magnitude below
/// movies on TMDB; a flat floor starves tv seeds.
fn vote_floor(media_type: Option<&str>) -> i64 {
    match media_type {
        Some("tv") => 50,
        _ => 500,
    }
}

/// Per-medium KNN facts. The embedding columns live on the split fact
/// tables (schema.sql §2) — the tmdb_media compatibility view does NOT
/// expose them (a view's column list is fixed at CREATE time), so the
/// KNN channels query the tables themselves.
struct KnnMedium {
    table: &'static str,
    title: &'static str,
    date: &'static str,
    coll: &'static str,
    gmap: &'static str,
    gid: &'static str,
}

const KNN_MOVIE: KnnMedium = KnnMedium {
    table: "tmdb_movies",
    title: "title",
    date: "release_date",
    coll: "collection_id",
    gmap: "tmdb_movie_genres_map",
    gid: "movie_id",
};
const KNN_TV: KnnMedium = KnnMedium {
    table: "tmdb_tv",
    title: "name",
    date: "first_air_date",
    coll: "NULL::bigint",
    gmap: "tmdb_tv_genres_map",
    gid: "tv_id",
};

/// One medium's arm of a KNN recall: top-`limit` rows by cosine.
///
/// The arm keeps its own ORDER BY + LIMIT (and the seed vector comes
/// from a same-statement subquery, not a bind parameter — passing it
/// as a parameter produced wrong distances) so the planner can serve
/// it from the medium's ANN index instead of sorting the whole table.
/// The id exclusion only applies in the seed's OWN medium: movie 155
/// stays a candidate for tv-seed 155 (ids collide across media).
fn knn_arm_sql(
    db: &Db,
    mt: &str,
    emb: &str,
    seed_mt: &str,
    vote_floor: i64,
) -> Result<String, DbError> {
    let s = if mt == "tv" { &KNN_TV } else { &KNN_MOVIE };
    let seed = if seed_mt == "tv" { &KNN_TV } else { &KNN_MOVIE };
    let excl = if mt == seed_mt {
        "\n          AND c.id != $1"
    } else {
        ""
    };
    Ok(format!(
        r#"
        WITH seed AS (SELECT {emb} AS v FROM {seed_table} WHERE id = $1)
        SELECT c.id, c.{title} AS title, c.{title} AS title_en,
               '{mt}'::text AS media_type,
               c.overview, c.overview AS overview_en,
               c.vote_average::float, c.vote_count::int,
               c.popularity::float, c.poster_path, {coll} AS collection_id,
               EXTRACT(YEAR FROM c.{date})::int AS release_year,
               ARRAY(
                   SELECT g.name FROM {genres_table} g
                   JOIN {gmap} gm ON gm.genre_id = g.id
                   WHERE gm.{gid} = c.id
               ) AS genres,
               GREATEST(0.0, 1 - (c.{emb} <=> (SELECT v FROM seed)))
                   AS knn_similarity
        FROM {table} c
        WHERE c.{emb} IS NOT NULL{excl}
          AND c.vote_average >= 5.5
          AND c.vote_count >= {vote_floor}
        ORDER BY c.{emb} <=> (SELECT v FROM seed)
        LIMIT $2
    "#,
        emb = emb,
        seed_table = db.map().name(seed.table)?,
        title = s.title,
        mt = mt,
        coll = s.coll,
        date = s.date,
        genres_table = db.map().name("tmdb_genres")?,
        gmap = db.map().name(s.gmap)?,
        gid = s.gid,
        table = db.map().name(s.table)?,
        excl = excl,
        vote_floor = vote_floor,
    ))
}

/// `generate_knn_candidates` — candidates whose overview embedding is
/// nearest to the seed's, via pgvector `<=>`. NOT gated on genre
/// overlap: close-in-spirit titles from other genres surface here.
/// Rows carry `via="knn"` + `knn_similarity`.
///
/// Degrade: a missing embedding column / vector extension / fact table
/// returns an EMPTY list (the Python latches the channel off per
/// column until restart — the latch itself lands with the service
/// layer, which owns process-global state).
pub async fn generate_knn_candidates(
    db: &Db,
    seed_id: i64,
    seed_media_type: &str,
    limit: i64,
    allow_cross_media: bool,
    emb_column: &str,
    floor_override: Option<i64>,
) -> Result<Vec<CandidateRow>, DbError> {
    let media: Vec<&str> = if allow_cross_media {
        vec!["movie", "tv"]
    } else {
        vec![seed_media_type]
    };
    let floor = floor_override.unwrap_or_else(|| {
        vote_floor((!allow_cross_media).then_some(seed_media_type))
    });

    let arms: Vec<String> = media
        .iter()
        .map(|mt| knn_arm_sql(db, mt, emb_column, seed_media_type, floor))
        .collect::<Result<_, _>>()?;
    if arms.is_empty() {
        return Ok(Vec::new());
    }
    let sql = if arms.len() == 1 {
        arms.into_iter().next().unwrap()
    } else {
        // two arms merged in SQL; each arm's own ORDER BY + LIMIT lets
        // its ANN index serve it, the outer merge sorts the survivors
        format!(
            "({a})\nUNION ALL\n({b})\nORDER BY knn_similarity DESC\nLIMIT $2",
            a = arms[0],
            b = arms[1]
        )
    };

    // ef_search must be raised per session for pgvector's hnsw indexes
    // to return deep-enough candidate lists.
    let _ = sqlx::query(&db.sql("SET hnsw.ef_search = 400")?)
        .execute(db.pool())
        .await;

    let rows = match db
        .fetch(
            &sql, // already resolved (built via SchemaMap::name, no placeholders)
            &[
                ("seed", Bind::I64(seed_id)),
                ("limit", Bind::I64(limit)),
            ],
        )
        .await
    {
        Ok(r) => r,
        // only the four missing-dependency codes degrade the channel;
        // anything else (connection loss, syntax) is a real failure
        Err(e) => match e.into_degraded("knn") {
            DbError::ChannelOff { .. } => return Ok(Vec::new()),
            real => return Err(real),
        },
    };

    Ok(rows
        .iter()
        .map(|r| {
            let mut c = CandidateRow::from_recall_row(r);
            c.via = Some("knn".to_string());
            c.knn_similarity = Some(c.knn_similarity.unwrap_or(0.0));
            c
        })
        .collect())
}
