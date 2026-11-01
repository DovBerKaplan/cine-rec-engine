//! Per-user context passes — the Rust port of `user_context.py`
//! (phase 5c): watched filtering + saga advancement, the boost-only
//! taste tilt, deterministic exploration, and merged exclusions.
//! Order is the contract: filters → tilt → balance/exploration.

use std::collections::{HashMap, HashSet};

use cine_rec_db::executor::row;
use cine_rec_db::executor::{Bind, Db, DbError};
use cine_rec_db::info::{genre_names_for_ids, top_user_genre_ids};
use cine_rec_db::recall2::fetch_embeddings_batch;
use cine_rec_db::user_vector::ensure_user_vector;
use cine_rec_db::watched::{get_user_dislikes, get_user_recommendation_exclusions};

use crate::similar::{balance_by_media_type, ScoredItem};

/// `USER_TILT_ALPHA` (config.py) — the caller-owned constant, mirrored
/// here as the deployment default.
pub const USER_TILT_ALPHA: f64 = 0.15;

// ---------------------------------------------------------------------------
// Pure helpers (tilt + exploration)
// ---------------------------------------------------------------------------

/// `_cosine` — explicit norms; catalog embeddings are NOT guaranteed
/// unit-length (pgvector's <=> normalizes internally; a raw dot doesn't).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `tilt_scores` — score' = score · (1 + α · max(cos, 0)).
/// Multiplicative and boost-only (the seed stays primary); re-packages
/// (never mutates the shared payload); stable re-sort by score.
pub fn tilt_scores(
    results: Vec<ScoredItem>,
    affinities: &HashMap<(i64, String), f64>,
    alpha: f64,
) -> Vec<ScoredItem> {
    let mut out: Vec<ScoredItem> = results
        .into_iter()
        .map(|r| {
            let cos = affinities.get(&(r.tmdb_id, r.media_type.clone()));
            match cos {
                None => r,
                Some(c) => {
                    let boost = c.max(0.0);
                    let mut r = r;
                    r.score *= 1.0 + alpha * boost;
                    r.user_affinity = Some((boost * 1000.0).round() / 1000.0); // round(boost, 3)
                    r
                }
            }
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// `exploration_slice` — (main, discovery_picks): picks are NOVEL
/// (genre-disjoint from the user's top clusters) AND RELEVANT
/// (affinity ≥ the main list's median), capped ceil(share·limit),
/// never intruding on the top-3. Pure and deterministic.
#[allow(clippy::too_many_arguments)]
pub fn exploration_slice(
    results: &[ScoredItem],
    limit: usize,
    user_genres: &HashSet<String>,
    affinities: &HashMap<(i64, String), f64>,
    share: f64,
) -> (Vec<ScoredItem>, Vec<ScoredItem>) {
    if share <= 0.0
        || results.is_empty()
        || user_genres.is_empty()
        || affinities.is_empty()
    {
        return (results.to_vec(), Vec::new());
    }
    let n = ((share * limit as f64).ceil() as usize).max(1);
    let n = n.min(limit.saturating_sub(3));
    if n == 0 {
        return (results.to_vec(), Vec::new());
    }

    let aff = |r: &ScoredItem| affinities.get(&(r.tmdb_id, r.media_type.clone())).copied();

    let mut main_affs: Vec<f64> = results
        .iter()
        .take(limit)
        .filter_map(aff)
        .collect();
    if main_affs.is_empty() {
        return (results.to_vec(), Vec::new());
    }
    main_affs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = main_affs[main_affs.len() / 2];

    let mut picks: Vec<ScoredItem> = Vec::new();
    let mut picked: HashSet<(i64, String)> = HashSet::new();
    for r in results {
        if picks.len() >= n {
            break;
        }
        let key = (r.tmdb_id, r.media_type.clone());
        if picked.contains(&key) {
            continue;
        }
        if r.genres.is_empty() || r.genres.iter().any(|g| user_genres.contains(g)) {
            continue; // not novel (or unclassifiable)
        }
        let a = match aff(r) {
            Some(v) if v >= median => v,
            _ => continue, // not relevant
        };
        let _ = a;
        picks.push(r.clone());
        picked.insert(key);
    }
    if picks.is_empty() {
        return (results.to_vec(), Vec::new());
    }
    let main: Vec<ScoredItem> = results
        .iter()
        .filter(|r| !picked.contains(&(r.tmdb_id, r.media_type.clone())))
        .cloned()
        .collect();
    (main, picks)
}

/// `insert_exploration` — fixed downstream slots (top-3 pinned):
/// slot_i = 3 + (i+1)·(limit−3)//(len(picks)+1).
pub fn insert_exploration(
    main: Vec<ScoredItem>,
    picks: Vec<ScoredItem>,
    limit: usize,
) -> Vec<ScoredItem> {
    let mut out = main;
    let n = picks.len();
    for (i, p) in picks.into_iter().enumerate() {
        let pos = (3 + ((i + 1) * (limit.saturating_sub(3))) / (n + 1)).min(out.len());
        out.insert(pos, p);
    }
    out
}

// ---------------------------------------------------------------------------
// Async passes
// ---------------------------------------------------------------------------

/// `seed_saga_blocked` — the seeds plus every member of their
/// collections; None on fetch failure (the caller falls back to the
/// bare seed ids inside the filter pass — same degrade).
pub async fn seed_saga_blocked(
    db: &Db,
    seed_ids: &HashSet<i64>,
) -> Result<Option<HashSet<i64>>, DbError> {
    if seed_ids.is_empty() {
        return Ok(Some(HashSet::new()));
    }
    let sql = db.sql(
        "SELECT DISTINCT m2.id \
         FROM {t_tmdb_movies} m2 \
         WHERE m2.collection_id > 0 \
           AND m2.collection_id IN ( \
             SELECT s.collection_id FROM {t_tmdb_movies} s \
             WHERE s.id = ANY($1::bigint[]) AND s.collection_id > 0)",
    )?;
    let mut sorted: Vec<i64> = seed_ids.iter().copied().collect();
    sorted.sort_unstable();
    match db
        .fetch(&sql, &[("ids", Bind::I64Array(sorted))])
        .await
    {
        Ok(rows) => {
            let mut out: HashSet<i64> = seed_ids.clone();
            out.extend(
                rows.iter()
                    .map(|r| row::opt_i64(r, "id").ok().flatten().unwrap_or_default()),
            );
            Ok(Some(out))
        }
        Err(_) => Ok(None),
    }
}

/// `merged_exclusions` — explicit exclude ∪ watched/rated ∪ dislikes.
/// A lookup failure must not wipe the explicit set — degrade to what
/// the caller passed.
pub async fn merged_exclusions(
    db: &Db,
    exclude: Option<HashSet<(i64, String)>>,
    user_id: Option<i64>,
) -> Result<Option<HashSet<(i64, String)>>, DbError> {
    let Some(user_id) = user_id else {
        return Ok(exclude);
    };
    let mut merged: HashSet<(i64, String)> = exclude.unwrap_or_default();
    // both watched-set reads degrade internally to empty sets
    merged.extend(get_user_recommendation_exclusions(db, user_id).await?);
    merged.extend(get_user_dislikes(db, user_id).await?);
    Ok(Some(merged))
}

/// `user_top_genres` — top cluster NAMES (ids → names). Degrades to
/// empty — exploration then skips.
pub async fn user_top_genres(
    db: &Db,
    user_id: i64,
) -> Result<HashSet<String>, DbError> {
    let ids = top_user_genre_ids(db, user_id, 3).await?;
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    genre_names_for_ids(db, &ids).await
}

/// The tilt/exploration threading context (`user_context` dict).
#[derive(Debug, Default, Clone)]
pub struct TiltContext {
    pub uvec: Option<Vec<f32>>,
    pub affinities: HashMap<(i64, String), f64>,
}

/// `apply_user_tilt` — boost-only re-rank toward the user's taste,
/// AFTER the hard filters and BEFORE balancing. The vector comes from
/// the context when a page threaded it, else ensured on demand in the
/// request's space. Every failure degrades to the untitled list.
pub async fn apply_user_tilt(
    db: &Db,
    results: Vec<ScoredItem>,
    user_id: i64,
    emb_column: &str,
    max_age_hours: f64,
    context: Option<&mut TiltContext>,
) -> Result<Vec<ScoredItem>, DbError> {
    if results.is_empty() {
        return Ok(results);
    }
    let ctx_uvec = context.as_ref().and_then(|c| c.uvec.clone());
    let uvec = match ctx_uvec {
        Some(v) => v,
        None => {
            match ensure_user_vector(db, user_id, None, emb_column, max_age_hours).await {
                Ok(Some(v)) => v,
                Ok(None) => return Ok(results),
                Err(_) => return Ok(results), // degrade, never fail
            }
        }
    };
    if uvec.is_empty() {
        return Ok(results);
    }
    let pairs: Vec<(i64, String)> = results
        .iter()
        .map(|r| (r.tmdb_id, r.media_type.clone()))
        .collect();
    let embs = match fetch_embeddings_batch(db, &pairs, emb_column).await {
        Ok(e) => e,
        Err(_) => return Ok(results),
    };
    if embs.is_empty() {
        return Ok(results);
    }
    let affinities: HashMap<(i64, String), f64> = embs
        .iter()
        .map(|(k, v)| (k.clone(), cosine(&uvec, v)))
        .collect();
    if let Some(ctx) = context {
        ctx.uvec = Some(uvec);
        ctx.affinities = affinities.clone();
    }

    Ok(tilt_scores(results, &affinities, USER_TILT_ALPHA))
}

/// `apply_exploration` — the discovery slice, then rebalance the main
/// portion and re-insert the picks at fixed slots. Degrades to plain
/// balancing when the relevance signal is missing.
pub async fn apply_exploration(
    db: &Db,
    results: Vec<ScoredItem>,
    user_id: i64,
    movie_ratio: f64,
    limit: usize,
    share: f64,
    tilt_ctx: &TiltContext,
) -> Result<Vec<ScoredItem>, DbError> {
    let user_genres = user_top_genres(db, user_id).await?;
    let (main, picks) = exploration_slice(
        &results,
        limit,
        &user_genres,
        &tilt_ctx.affinities,
        share,
    );
    if picks.is_empty() {
        return Ok(balance_by_media_type(&results, movie_ratio, limit));
    }
    let main = balance_by_media_type(&main, movie_ratio, limit - picks.len());
    Ok(insert_exploration(main, picks, limit))
}

/// `apply_user_filters` — watched filtering + saga advancement + the
/// final seed guard, one merged pass. A saga slot advances to the
/// collection's FIRST UNWATCHED member; a whole-watched saga drops its
/// slot. Swap collisions dedup by (id, media_type).
pub async fn apply_user_filters(
    db: &Db,
    results: Vec<ScoredItem>,
    watched: &HashSet<(i64, String)>,
    seed_ids: &HashSet<i64>,
    blocked: Option<HashSet<i64>>,
) -> Result<Vec<ScoredItem>, DbError> {
    // blocked: prefetched saga block, else computed here; None on
    // fetch failure degrades to the bare seed ids
    let blocked = match blocked {
        Some(b) => b,
        None => seed_saga_blocked(db, seed_ids)
            .await?
            .unwrap_or_else(|| seed_ids.clone()),
    };

    // saga members (movie collections), release-date order
    let coll_ids: Vec<i64> = {
        let mut v: Vec<i64> = results
            .iter()
            .filter(|r| r.media_type == "movie")
            .filter_map(|r| r.collection_id)
            .filter(|c| *c != 0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut members: HashMap<i64, Vec<SagaMember>> = HashMap::new();
    if !coll_ids.is_empty() {
        let sql = db.sql(
            "SELECT m.collection_id, m.id, m.title, m.title AS title_en, \
                    m.vote_average::float AS vote_average, m.poster_path, \
                    EXTRACT(YEAR FROM m.release_date)::int AS release_year \
             FROM {t_tmdb_movies} m \
             WHERE m.collection_id > 0 AND m.collection_id = ANY($1::int[]) \
             ORDER BY m.collection_id, m.release_date ASC NULLS LAST, m.popularity DESC",
        )?;
        if let Ok(rows) = db
            .fetch(&sql, &[("colls", Bind::I64Array(coll_ids))])
            .await
        {
            for r in &rows {
                let coll = row::opt_i64(r, "collection_id")
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                members.entry(coll).or_default().push(SagaMember {
                    id: row::opt_i64(r, "id").ok().flatten().unwrap_or_default(),
                    title: row::opt_text(r, "title").ok().flatten().unwrap_or_default(),
                    title_en: row::opt_text(r, "title_en").ok().flatten(),
                    rating: row::opt_f64(r, "vote_average").ok().flatten().unwrap_or(0.0),
                    poster_path: row::opt_text(r, "poster_path").ok().flatten(),
                    release_year: row::opt_i64(r, "release_year").ok().flatten(),
                });
            }
        }
    }

    let mut out: Vec<ScoredItem> = Vec::new();
    let mut seen_keys: HashSet<(i64, String)> = HashSet::new();
    for mut r in results {
        if blocked.contains(&r.tmdb_id) {
            continue;
        }
        let saga = members.get(&r.collection_id.unwrap_or(0));
        if let Some(saga) = saga {
            let target = saga.iter().find(|m| {
                !blocked.contains(&m.id) && !watched.contains(&(m.id, "movie".to_string()))
            });
            let Some(target) = target else {
                continue; // the user watched (or seeded) the entire saga
            };
            if target.id != r.tmdb_id {
                r.tmdb_id = target.id;
                r.title = target.title.clone();
                r.title_en = target.title_en.clone();
                r.rating = target.rating;
                r.poster_path = target.poster_path.clone();
                r.release_year = target.release_year;
            }
        } else if watched.contains(&(r.tmdb_id, r.media_type.clone())) {
            continue;
        }
        let key = (r.tmdb_id, r.media_type.clone());
        if seen_keys.insert(key) {
            out.push(r);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct SagaMember {
    id: i64,
    title: String,
    title_en: Option<String>,
    rating: f64,
    poster_path: Option<String>,
    release_year: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(id: i64, score: f64, genres: &[&str]) -> ScoredItem {
        ScoredItem {
            tmdb_id: id,
            media_type: "movie".into(),
            score,
            genres: genres.iter().map(|g| g.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn tilt_matches_python() {
        // verified against user_context.tilt_scores on these rows:
        // 1.0·(1+0.15·0.5)=1.075, affinity rounded 3; untouched pass
        // through in place; negative affinity never buries (boost 0)
        let rows = vec![it(1, 1.0, &[]), it(2, 0.9, &[])];
        let mut aff = HashMap::new();
        aff.insert((1i64, "movie".to_string()), 0.5f64);
        let out = tilt_scores(rows, &aff, 0.15);
        assert!((out[0].score - 1.075).abs() < 1e-12);
        assert_eq!(out[0].user_affinity, Some(0.5));
        assert_eq!(out[1].user_affinity, None);

        let mut neg = HashMap::new();
        neg.insert((1i64, "movie".to_string()), -0.9f64);
        let out = tilt_scores(vec![it(1, 1.0, &[])], &neg, 0.15);
        assert!((out[0].score - 1.0).abs() < 1e-12);
        assert_eq!(out[0].user_affinity, Some(0.0));
    }

    #[test]
    fn exploration_slice_matches_python() {
        // verified rows: 10 novel+relevant is picked; 11 shares a user
        // genre (not novel); 12 novel but below median affinity
        let rows = vec![
            it(10, 5.0, &["Sci-Fi"]),
            it(11, 4.5, &["Action"]),
            it(12, 4.0, &["Horror"]),
        ];
        let ug: HashSet<String> = ["Action".to_string()].into_iter().collect();
        let mut aff = HashMap::new();
        aff.insert((10i64, "movie".to_string()), 0.9);
        aff.insert((11i64, "movie".to_string()), 0.9);
        aff.insert((12i64, "movie".to_string()), 0.1);
        let (main, picks) = exploration_slice(&rows, 10, &ug, &aff, 0.2);
        assert_eq!(picks.iter().map(|p| p.tmdb_id).collect::<Vec<_>>(), vec![10]);
        assert_eq!(main.iter().map(|m| m.tmdb_id).collect::<Vec<_>>(), vec![11, 12]);
        // fixed downstream slot: 3+1·(10-3)//2 = 6 → clamped to end
        let merged = insert_exploration(main.clone(), picks, 10);
        assert_eq!(
            merged.iter().map(|r| r.tmdb_id).collect::<Vec<_>>(),
            vec![11, 12, 10]
        );
    }

    #[test]
    fn slice_skips_without_signals() {
        let rows = vec![it(1, 1.0, &["Sci-Fi"])];
        let ug: HashSet<String> = HashSet::new(); // no user genres
        let aff = HashMap::new();
        let (main, picks) = exploration_slice(&rows, 10, &ug, &aff, 0.2);
        assert!(picks.is_empty());
        assert_eq!(main.len(), 1);
        // share<=0 → identity too
        let ug2: HashSet<String> = ["X".to_string()].into_iter().collect();
        let (_, picks2) = exploration_slice(&rows, 10, &ug2, &aff, 0.0);
        assert!(picks2.is_empty());
    }
}
