//! recommend_for_user — the §F pipeline: plan → gather →
//! the compute path with per-user passes → exploration/balance. The
//! Rust port of `recommend.py` + the user-pass tail of find_similar.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use cine_rec_db::executor::row;
use cine_rec_db::executor::{Bind, Db, DbError};
use cine_rec_db::user_vector::{ensure_user_vector, top_weighted_seeds};

use crate::similar::{
    balance_by_media_type, compute_find_similar, ComputeSpec, ScoredItemSerde, WeightLists,
};
use crate::user_context::{
    apply_exploration, apply_user_filters, apply_user_tilt, merged_exclusions,
    seed_saga_blocked, TiltContext,
};

/// `COLD_START_MIN_TITLES` / `USER_SEEDS` / `EXPLORE_SHARE` /
/// `USER_VECTOR_MAX_AGE_HOURS` (config.py) — deployment defaults the
/// caller may override in the spec.
pub const COLD_START_MIN_TITLES: usize = 3;
pub const USER_SEEDS: i64 = 20;
pub const EXPLORE_SHARE: f64 = 0.12;
pub const USER_VECTOR_MAX_AGE_HOURS: f64 = 24.0;

/// `recommendation_plan` — pure: how to recommend for a user.
/// Returns (mode, seeds-as-(id, media_type, w)).
pub fn recommendation_plan(
    weighted_seeds: &[(i64, String, f64)],
    watchlist_seeds: &[(i64, String, f64)],
    min_titles: usize,
) -> (&'static str, Vec<(i64, String, f64)>) {
    if weighted_seeds.len() >= min_titles {
        return ("personalized", weighted_seeds.to_vec());
    }
    if !weighted_seeds.is_empty() {
        // some history but not enough: blend, watchlist dedup vs seeds
        let seed_ids: HashSet<i64> = weighted_seeds.iter().map(|s| s.0).collect();
        let mut blended = weighted_seeds.to_vec();
        blended.extend(
            watchlist_seeds
                .iter()
                .filter(|w| !seed_ids.contains(&w.0))
                .cloned(),
        );
        return ("personalized", blended);
    }
    if !watchlist_seeds.is_empty() {
        return ("watchlist", watchlist_seeds.to_vec());
    }
    ("cold_start", Vec::new())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecommendSpec {
    pub user_id: i64,
    pub limit: Option<usize>,
    /// None → EXPLORE_SHARE; 0 disables; validated [0.0, 0.5]
    pub explore: Option<f64>,
    pub emb_column: String,
    pub knn_ft_enabled: bool,
    pub weights: WeightLists,
    #[serde(default)]
    pub decay: crate::similar::DecayConfig,
    /// the uvec for the ANN extras: taken from the STORED vector (the
    /// parity fixture freezes it); None → ensure on demand
    pub uvec: Option<Vec<f32>>,
    /// frozen inputs for parity replay: when present, these override
    /// the live DB reads (the gate works on any DB that ran the
    /// generator, not just the one that created the fixture)
    #[serde(default)]
    pub frozen_exclusions: Option<Vec<(i64, String)>>,
    #[serde(default)]
    pub frozen_user_genres: Option<Vec<String>>,
    /// frozen seeds: (id, media_type, w) — when present, skip the
    /// live top_weighted_seeds read (the gate works across time;
    /// w_item decays with recency)
    #[serde(default)]
    pub frozen_seeds: Option<Vec<(i64, String, f64)>>,
}

#[derive(Debug, Default, Serialize)]
pub struct RecommendResult {
    pub results: Vec<ScoredItemSerde>,
    pub reason: String,
    /// the top seeds used, (id, media_type, w rounded 3)
    pub seeds: Vec<(i64, String, f64)>,
    pub vector_used: bool,
}

pub async fn recommend_for_user(
    db: &Db,
    spec: &RecommendSpec,
) -> Result<RecommendResult, DbError> {
    // 1. the three request inputs (Python gathers concurrently — the
    //    wall time shape differs, the result does not; seeds/vector
    //    are independent reads)
    let weighted = match &spec.frozen_seeds {
        Some(frozen) => frozen.clone(),
        None => top_weighted_seeds(db, spec.user_id, USER_SEEDS).await?,
    };
    let watchlist = watchlist_seeds(db, spec.user_id).await?;
    let uvec = match &spec.uvec {
        Some(v) => Some(v.clone()),
        None => {
            ensure_user_vector(
                db,
                spec.user_id,
                None,
                &spec.emb_column,
                USER_VECTOR_MAX_AGE_HOURS,
            )
            .await
            .unwrap_or(None)
        }
    };

    // 2. plan
    let (mode, seeds) =
        recommendation_plan(&weighted, &watchlist, COLD_START_MIN_TITLES);
    if mode == "cold_start" {
        return Ok(RecommendResult {
            results: Vec::new(),
            reason: "cold_start".into(),
            seeds: Vec::new(),
            vector_used: false,
        });
    }

    // 3. the compute path (seeds as composite keys — movie 155 ≠ tv 155)
    let limit = spec.limit;
    let compute_spec = ComputeSpec {
        seeds: seeds.iter().map(|(id, mt, _)| (*id, Some(mt.clone()))).collect(),
        media_type: None,
        limit,
        allow_cross_media: false,
        // Python's recommend does NOT thread w_i into find_similar —
        // seeds score at equal influence; w_i shapes only the seed
        // SELECTION and the returned seeds list
        seed_weights: None,
        emb_column: spec.emb_column.clone(),
        knn_ft_enabled: spec.knn_ft_enabled,
        filters: Default::default(),
        weights: spec.weights.clone(),
        decay: spec.decay.clone(),
        uvec_extra: uvec.as_ref().map(|v| (v.clone(), 60)),
        balance: false, // the user passes + exploration run BEFORE balance
    };
    let computed = compute_find_similar(db, &compute_spec).await?;

    // 4. per-user passes in contract order: filters → tilt →
    //    exploration/balance (the compute already balanced on rounded
    //    scores; the user pass RE-balances after filtering — exactly
    //    Python's double-balance shape)
    let seed_id_set: HashSet<i64> = seeds.iter().map(|s| s.0).collect();
    let merged = match &spec.frozen_exclusions {
        Some(frozen) => Some(frozen.iter().cloned().collect()),
        None => merged_exclusions(db, Some(HashSet::new()), Some(spec.user_id)).await?,
    };
    let blocked = seed_saga_blocked(db, &seed_id_set)
        .await?
        .unwrap_or_else(|| seed_id_set.clone());

    let mut tilt_ctx = TiltContext {
        uvec: uvec.clone(),
        affinities: HashMap::new(),
    };
    let items: Vec<crate::similar::ScoredItem> = computed
        .results
        .iter()
        .map(|r| crate::similar::ScoredItem {
            tmdb_id: r.tmdb_id,
            media_type: r.media_type.clone(),
            score: r.score,
            title: r.title.clone(),
            title_en: r.title_en.clone(),
            rating: r.rating,
            poster_path: r.poster_path.clone(),
            collection_id: r.collection_id,
            release_year: r.release_year,
            director_ids: r.director_ids.clone(),
            genres: r.genres.clone(),
            user_affinity: None,
            raw_score: 0.0,
        })
        .collect();

    let filtered =
        apply_user_filters(db, items, &merged.unwrap_or_default(), &seed_id_set, Some(blocked))
            .await?;
    let tilted = apply_user_tilt(
        db,
        filtered,
        spec.user_id,
        &spec.emb_column,
        USER_VECTOR_MAX_AGE_HOURS,
        Some(&mut tilt_ctx),
    )
    .await?;

    let explore_share = spec.explore.unwrap_or(EXPLORE_SHARE);
    // Python validates [0.0, 0.5] and raises; the bridge must too
    if !(0.0..=0.5).contains(&explore_share) {
        return Err(cine_rec_db::DbError::Invalid(format!(
            "explore must be in [0.0, 0.5], got {explore_share}"
        )));
    }
    let final_items = if let Some(limit) = limit {
        if !tilt_ctx.affinities.is_empty() && explore_share > 0.0 {
            // frozen genres override the live read (parity replay)
            let genres = match &spec.frozen_user_genres {
                Some(g) => g.iter().cloned().collect(),
                None => HashSet::new(), // unused — see apply_exploration_frozen
            };
            if spec.frozen_user_genres.is_some() {
                apply_exploration_frozen(
                    tilted,
                    &genres,
                    computed.movie_ratio,
                    limit,
                    explore_share,
                    &tilt_ctx,
                )
            } else {
                apply_exploration(
                    db,
                    tilted,
                    spec.user_id,
                    computed.movie_ratio,
                    limit,
                    explore_share,
                    &tilt_ctx,
                )
                .await?
            }
        } else {
            balance_by_media_type(&tilted, computed.movie_ratio, limit)
        }
    } else {
        tilted
    };

    Ok(RecommendResult {
        results: final_items
            .into_iter()
            .map(|r| ScoredItemSerde {
                tmdb_id: r.tmdb_id,
                score: r.score,
                title: r.title,
                title_en: r.title_en,
                media_type: r.media_type,
                rating: r.rating,
                genres: r.genres,
                poster_path: r.poster_path,
                collection_id: r.collection_id,
                release_year: r.release_year,
                director_ids: r.director_ids,
                user_affinity: r.user_affinity,
                raw_score: r.raw_score,
            })
            .collect(),
        reason: mode.to_string(),
        seeds: seeds
            .iter()
            .take(10)
            .map(|(id, mt, w)| (*id, mt.clone(), crate::similar::py_round3(*w)))
            .collect(),
        vector_used: uvec.is_some(),
    })
}

/// the watchlist query (recommend.py): latest 10 watchlist entries
async fn watchlist_seeds(
    db: &Db,
    user_id: i64,
) -> Result<Vec<(i64, String, f64)>, DbError> {
    let sql = db.sql(
        "SELECT f.tmdb_id, f.media_type FROM {t_user_feedback} f \
         WHERE f.user_id = $1 AND f.kind = 'watchlist' \
         ORDER BY f.created_at DESC LIMIT 10",
    )?;
    let rows = match db.fetch(&sql, &[("user", Bind::I64(user_id))]).await {
        Ok(r) => r,
        Err(_) => return Ok(Vec::new()),
    };
    Ok(rows
        .iter()
        .map(|r| {
            (
                row::opt_i64(r, "tmdb_id").ok().flatten().unwrap_or_default(),
                row::opt_text(r, "media_type")
                    .ok()
                    .flatten()
                    .unwrap_or_default(),
                1.0,
            )
        })
        .collect())
}


/// The frozen-input exploration (no DB reads — the fixture supplies
/// the user genres). Same math as apply_exploration.
pub fn apply_exploration_frozen(
    results: Vec<crate::similar::ScoredItem>,
    user_genres: &HashSet<String>,
    movie_ratio: f64,
    limit: usize,
    share: f64,
    tilt_ctx: &TiltContext,
) -> Vec<crate::similar::ScoredItem> {
    let (main, picks) = crate::user_context::exploration_slice(
        &results, limit, user_genres, &tilt_ctx.affinities, share);
    if picks.is_empty() {
        return crate::similar::balance_by_media_type(&results, movie_ratio, limit);
    }
    let main = crate::similar::balance_by_media_type(&main, movie_ratio, limit - picks.len());
    crate::user_context::insert_exploration(main, picks, limit)
}
