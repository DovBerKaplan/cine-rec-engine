//! find_similar's compute path (phase 5b): seed normalization, the
//! recall-channel fan-out over `cine-rec-db`, candidate-pool assembly,
//! the scoring loop over `cine-rec-scoring`, MMR and media balancing.
//!
//! Parity target: Python's `find_similar.py` compute path (cache
//! bypassed) on frozen seeds. Every helper here is a line-faithful
//! port; the QUIRKS that matter are called out inline (banker's
//! rounding, `(hi-lo) or 1.0`, first-wins argmax ties).

use std::collections::HashSet;

use cine_rec_scoring::real::canonical_genres;

/// One scored result row in the orchestration pipeline — the fields
/// the pure helpers consume (the dict Python threads through the loop).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScoredItem {
    pub tmdb_id: i64,
    pub media_type: String,
    pub score: f64,
    pub collection_id: Option<i64>,
    pub director_ids: Vec<i64>,
    pub genres: Vec<String>,
}

/// Python's `round()` is BANKER'S rounding (ties to even); Rust's
/// `f64::round` ties away from zero. Every slot count below must match
/// Python exactly, so half-cases go to the even neighbor.
fn py_round(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor as i64
    } else if diff > 0.5 {
        floor as i64 + 1
    } else {
        // exactly .5 → to EVEN
        if (floor as i64) % 2 == 0 {
            floor as i64
        } else {
            floor as i64 + 1
        }
    }
}

/// `_mmr_pair_key` — per-item precomputed similarity inputs.
#[derive(Debug, Clone, Default)]
struct MmrKey {
    collection_id: Option<i64>,
    directors: HashSet<i64>,
    genres: HashSet<String>,
}

fn mmr_pair_key(item: &ScoredItem) -> MmrKey {
    MmrKey {
        collection_id: item.collection_id,
        directors: item.director_ids.iter().copied().collect(),
        genres: canonical_genres(&item.genres),
    }
}

/// `_pair_similarity_fast` over precomputed keys — same values as the
/// dict version: 1.0 same collection, 0.85 shared director, genre
/// Jaccard capped at 0.5, 0.0 when either genre set is empty.
fn pair_similarity_fast(ka: &MmrKey, kb: &MmrKey) -> f64 {
    if let (Some(ca), Some(cb)) = (ka.collection_id, kb.collection_id) {
        if ca == cb && ca != 0 {
            // Python truthiness: `if ca and cb and ca == cb` — id 0 is
            // falsy and never matches
            return 1.0;
        }
    }
    if !ka.directors.is_disjoint(&kb.directors) {
        return 0.85;
    }
    if ka.genres.is_empty() || kb.genres.is_empty() {
        return 0.0;
    }
    let inter = ka.genres.intersection(&kb.genres).count();
    let union = ka.genres.union(&kb.genres).count();
    (inter as f64 / union as f64).min(1.0) * 0.5
}

/// `_mmr_order` — greedy MMR with the running-max-similarity
/// optimization. Pick order is IDENTICAL to Python's: strict `>`
/// comparison means the FIRST pool item wins ties, and the pool keeps
/// its original order as items are taken (lowest index first).
pub fn mmr_order(results: &[ScoredItem], lam: f64, k: usize) -> Vec<ScoredItem> {
    if results.is_empty() {
        return Vec::new();
    }
    let lo = results.iter().map(|r| r.score).fold(f64::INFINITY, f64::min);
    let hi = results
        .iter()
        .map(|r| r.score)
        .fold(f64::NEG_INFINITY, f64::max);
    // Python: `spread = (hi - lo) or 1.0` — zero (or NaN) becomes 1.0
    let spread = if hi - lo != 0.0 { hi - lo } else { 1.0 };

    let keys: Vec<MmrKey> = results.iter().map(mmr_pair_key).collect();
    let mut taken = vec![false; results.len()];
    let mut pen: Vec<f64> = vec![0.0; results.len()];
    let mut selected: Vec<usize> = Vec::new();

    while selected.len() < k && selected.len() < results.len() {
        let mut best: Option<usize> = None;
        let mut best_val = -1e9_f64;
        for i in 0..results.len() {
            if taken[i] {
                continue;
            }
            let val = lam * ((results[i].score - lo) / spread) - (1.0 - lam) * pen[i];
            if val > best_val {
                best = Some(i);
                best_val = val;
            }
        }
        let best = best.expect("non-empty pool always yields a pick");
        taken[best] = true;
        selected.push(best);
        for i in 0..results.len() {
            if taken[i] {
                continue;
            }
            let sim = pair_similarity_fast(&keys[best], &keys[i]);
            if sim > pen[i] {
                pen[i] = sim;
            }
        }
    }
    // remainder keeps relevance (original) order after the selected
    let mut out: Vec<ScoredItem> = selected.iter().map(|&i| results[i].clone()).collect();
    for (i, item) in results.iter().enumerate() {
        if !taken[i] {
            out.push(item.clone());
        }
    }
    out
}

/// `_balance_by_media_type` — split by ratio, interleave by score,
/// exactly `limit` items. The 10% minority floor and BANKER'S rounding
/// are contract (Python's round is ties-to-even — see `py_round`).
pub fn balance_by_media_type(
    results: &[ScoredItem],
    movie_ratio: f64,
    limit: usize,
) -> Vec<ScoredItem> {
    if results.is_empty() {
        return Vec::new();
    }
    // pure single type — filter only
    if movie_ratio == 0.0 || movie_ratio == 1.0 {
        let target = if movie_ratio == 1.0 { "movie" } else { "tv" };
        return results
            .iter()
            .filter(|r| r.media_type == target)
            .take(limit)
            .cloned()
            .collect();
    }
    const MIN_FRACTION: f64 = 0.1;

    let mut movie_slots = py_round(limit as f64 * movie_ratio);
    let mut tv_slots = limit as i64 - movie_slots;

    if movie_slots > 0 && (tv_slots as f64) < limit as f64 * MIN_FRACTION {
        tv_slots = 1.max(py_round(limit as f64 * MIN_FRACTION));
        movie_slots = limit as i64 - tv_slots;
    } else if tv_slots > 0 && (movie_slots as f64) < limit as f64 * MIN_FRACTION {
        movie_slots = 1.max(py_round(limit as f64 * MIN_FRACTION));
        tv_slots = limit as i64 - movie_slots;
    }

    let mut selected: Vec<ScoredItem> = Vec::new();
    for r in results {
        if r.media_type == "movie" && movie_slots > 0 {
            selected.push(r.clone());
            movie_slots -= 1;
        }
    }
    for r in results {
        if r.media_type == "tv" && tv_slots > 0 {
            selected.push(r.clone());
            tv_slots -= 1;
        }
    }
    // re-sort by score descending (Python list.sort is stable; items
    // arrive in score order so ties keep the movies-first insertion)
    selected.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    selected
}

/// `_build_seed_weights` — INVERT to 1/w: a bigger w_i means LESS
/// influence. Missing ids default to full influence (1.0).
pub fn build_seed_weights(
    seed_ids: &[i64],
    seed_weights: Option<&std::collections::HashMap<i64, f64>>,
) -> Vec<f64> {
    seed_ids
        .iter()
        .map(|sid| match seed_weights {
            Some(map) if map.contains_key(sid) => 1.0 / map[sid],
            _ => 1.0,
        })
        .collect()
}

/// `_apply_noise` — for randomness <= 0 the Python function returns
/// the list UNTOUCHED; that is the parity surface (the harness always
/// runs deterministic). Nonzero noise consumes Python's `random`
/// module stream (Mersenne Twister) — reproducing it bit-for-bit is a
/// separate decision; until then the bridge routes nonzero randomness
/// to the Python path.
pub fn apply_noise_identity(items: Vec<ScoredItem>, randomness: f64) -> Vec<ScoredItem> {
    if items.is_empty() || randomness <= 0.0 {
        return items;
    }
    // unreachable in the parity path; the bridge guarantees r <= 0
    unreachable!("nonzero randomness is served by the Python path")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: i64, mt: &str, score: f64) -> ScoredItem {
        ScoredItem {
            tmdb_id: id,
            media_type: mt.to_string(),
            score,
            ..Default::default()
        }
    }

    #[test]
    fn py_round_is_bankers() {
        // the exact class where Rust's round() would diverge
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
        assert_eq!(py_round(-2.5), -2);
        assert_eq!(py_round(2.4), 2);
        assert_eq!(py_round(2.6), 3);
        // limit*ratio shapes: 30*0.75=22.5 → 22 (Python), not 23
        assert_eq!(py_round(30.0 * 0.75), 22);
    }

    #[test]
    fn balance_pure_type_filters() {
        let rows = vec![item(1, "movie", 9.0), item(2, "tv", 8.0), item(3, "movie", 7.0)];
        let out = balance_by_media_type(&rows, 1.0, 10);
        assert_eq!(
            out.iter().map(|r| r.tmdb_id).collect::<Vec<_>>(),
            vec![1, 3]
        );
        let out = balance_by_media_type(&rows, 0.0, 10);
        assert_eq!(out.iter().map(|r| r.tmdb_id).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn balance_ratio_and_minority_floor() {
        // 10 slots at 0.75 → bankers round(7.5) = 8 movies / 2 tv
        let mut rows = Vec::new();
        for i in 0..12 {
            rows.push(item(i, "movie", 10.0 - i as f64));
        }
        for i in 100..112 {
            rows.push(item(i, "tv", 9.5 - (i - 100) as f64));
        }
        let out = balance_by_media_type(&rows, 0.75, 10);
        let movies = out.iter().filter(|r| r.media_type == "movie").count();
        let tvs = out.iter().filter(|r| r.media_type == "tv").count();
        assert_eq!((movies, tvs), (8, 2), "banker's 7.5→(8 movies, 2 tv)");
        assert_eq!(out.len(), 10);
        // interleaved by score: first two are the two top scores
        assert!(out[0].score >= out[1].score);
    }

    #[test]
    fn mmr_penalizes_same_collection() {
        let mk = |id: i64, score: f64, coll: Option<i64>| ScoredItem {
            tmdb_id: id,
            score,
            collection_id: coll,
            media_type: "movie".into(),
            ..Default::default()
        };
        // verified against Python's _mmr_order on these exact rows:
        // pick1=1 (top); item2's same-saga penalty (1−λ)·1.0=0.3 beats
        // its small relevance edge (0.7·⅓≈0.23), so 3 goes second
        let rows = vec![
            mk(1, 9.0, Some(10)),
            mk(2, 8.8, Some(10)), // same saga — pushed behind 3
            mk(3, 8.7, None),
        ];
        let out = mmr_order(&rows, 0.7, 3);
        let ids: Vec<i64> = out.iter().map(|r| r.tmdb_id).collect();
        assert_eq!(ids, vec![1, 3, 2], "same-saga runner-up penalized");
        // and when the gap is wide the relevance DOES win (also Python)
        let rows = vec![mk(1, 9.0, Some(10)), mk(2, 8.9, Some(10)), mk(3, 8.5, None)];
        let out = mmr_order(&rows, 0.7, 3);
        assert_eq!(
            out.iter().map(|r| r.tmdb_id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn mmr_spread_zero_does_not_divide_by_zero() {
        let rows = vec![item(1, "movie", 5.0), item(2, "movie", 5.0)];
        let out = mmr_order(&rows, 0.7, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tmdb_id, 1, "ties keep original order");
    }

    #[test]
    fn seed_weights_invert() {
        let mut map = std::collections::HashMap::new();
        map.insert(7i64, 2.0f64);
        let w = build_seed_weights(&[7, 8], Some(&map));
        assert_eq!(w, vec![0.5, 1.0]);
        assert_eq!(build_seed_weights(&[7, 8], None), vec![1.0, 1.0]);
    }


    // ---- recall stage: optimize + pool walk (Python-verified) ----

    fn crow(id: i64, rating: f64, votes: i64, pop: f64) -> CandidateRow {
        CandidateRow {
            id,
            vote_average: Some(rating),
            vote_count: Some(votes),
            popularity: Some(pop),
            media_type: Some("movie".into()),
            genres: vec!["Action".into()],
            ..Default::default()
        }
    }

    #[test]
    fn optimize_matches_python_reference() {
        // exact rows verified against cine_rec_engine.scoring on the
        // Python side (case A/B/C) — see the phase-5 journal
        let mk = |id: i64, rating: f64, votes: i64, pop: f64, via: Option<&str>,
                  mt: &str, genres: &[&str]| CandidateRow {
            id,
            vote_average: Some(rating),
            vote_count: Some(votes),
            popularity: Some(pop),
            via: via.map(String::from),
            media_type: Some(mt.into()),
            genres: genres.iter().map(|g| g.to_string()).collect(),
            ..Default::default()
        };
        let cands = vec![
            mk(1, 8.0, 2000, 90.0, None, "movie", &["Action"]),
            mk(2, 8.0, 2000, 90.0, Some("knn"), "movie", &["Action"]),
            mk(3, 8.0, 2000, 90.0, Some(""), "movie", &["Action"]),
            mk(4, 5.0, 2000, 90.0, Some("tmdb"), "movie", &["Action"]),
            mk(5, 8.0, 100, 90.0, None, "movie", &["Action"]),
            mk(6, 8.0, 100, 90.0, Some("knn"), "movie", &["Action"]),
            mk(7, 8.0, 2000, 90.0, None, "movie", &["Romance"]),
            mk(8, 8.0, 2000, 90.0, None, "movie", &[]),
        ];
        let base = vec!["Action".to_string()];
        let out = optimize_candidate_selection(cands, &base, 10);
        let ids: Vec<i64> = out.iter().map(|c| c.id).collect();
        // Python case A: curated(2,6) + rating-floor(4 out) + vote-floor
        // (5 out) + genre-disjoint (7 out) + empty-genres-passes (8)
        assert_eq!(ids, vec![2, 6, 1, 3, 8]);

        // case B: equal sort scores keep RECALL order (stable sort)
        let eq = vec![crow(21, 7.0, 1000, 50.0), crow(22, 7.0, 1000, 50.0), crow(23, 7.0, 1000, 50.0)];
        let out = optimize_candidate_selection(eq, &base, 10);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), vec![21, 22, 23]);

        // case C: target=1 → curated_limit=0 → curated dropped entirely
        let mix = vec![
            mk(31, 8.0, 2000, 90.0, Some("knn"), "movie", &["Action"]),
            crow(32, 8.0, 2000, 90.0),
        ];
        let out = optimize_candidate_selection(mix, &base, 1);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), vec![32]);
    }

    #[test]
    fn pool_walk_slot_contract() {
        use super::ChannelSlot;
        let seed = |id: i64, mt: &str| SeedInfo {
            id,
            media_type: mt.to_string(),
            ..Default::default()
        };
        let krow = |id: i64, sim: Option<f64>| CandidateRow {
            id,
            knn_similarity: sim,
            ..Default::default()
        };
        let seeds = vec![seed(1, "movie"), seed(2, "tv")];
        let channels = [
            // genre: seed1 has one row (dup with seed2's), seed2 failed
            vec![
                ChannelSlot::Rows(vec![crow(10, 8.0, 2000, 90.0)]),
                ChannelSlot::Failed("channel down".into()),
            ],
            // KNN: seed1 rows (0.0 sim EXCLUDED from harvest, dup id 10
            // already seen), seed2 row with a live sim
            vec![
                ChannelSlot::Rows(vec![krow(10, Some(0.0)), krow(11, Some(0.7))]),
                ChannelSlot::Rows(vec![krow(12, Some(0.5))]),
            ],
            vec![
                ChannelSlot::Rows(vec![crow(1, 8.0, 2000, 90.0)]), // a SEED id — excluded
                ChannelSlot::Rows(vec![]),
            ],
            vec![
                ChannelSlot::Rows(vec![crow(13, 8.0, 2000, 90.0)]),
                ChannelSlot::Failed("no director".into()),
            ],
        ];
        let (pool, knn) = assemble_pool(&seeds, channels);
        let ids: Vec<i64> = pool.iter().map(|c| c.id).collect();
        // walk order: seed1 genre(10) → seed1 knn(11) → seed1 tmdb(seed
        // excluded) → seed1 director(13) → seed2 knn(12)
        assert_eq!(ids, vec![10, 11, 13, 12]);
        // KNN harvest: seed1 has only 11 (0.0 is falsy in Python);
        // seed2 has 12
        assert_eq!(knn.get(&1).map(|m| m.get(&11)), Some(Some(&0.7)));
        assert!(!knn[&1].contains_key(&10));
        assert_eq!(knn.get(&2).map(|m| m.get(&12)), Some(Some(&0.5)));
    }

    #[test]
    fn pair_similarity_truthiness() {
        // collection id 0 is FALSY in Python — never a saga match
        let a = MmrKey {
            collection_id: Some(0),
            ..Default::default()
        };
        let b = MmrKey {
            collection_id: Some(0),
            ..Default::default()
        };
        assert_eq!(pair_similarity_fast(&a, &b), 0.0);
    }
}

// ---------------------------------------------------------------------------
// Recall fan-out + pool assembly (steps 2–4 of the compute path)
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use cine_rec_db::info::SeedInfo;
use cine_rec_db::recall::{
    enrich_candidates_batch, generate_candidates, generate_knn_candidates, CandidateRow,
    Enrichment,
};
use cine_rec_db::recall2::{
    generate_director_candidates, generate_director_candidates_batch,
    generate_tmdb_rec_candidates, generate_tmdb_rec_candidates_batch,
};
use cine_rec_db::Db;

/// `KNN_CANDIDATES_PER_SEED` (config.py) — the service layer owns the
/// config constants it ports, documented per const.
pub const KNN_CANDIDATES_PER_SEED: i64 = 60;
/// Python defaults for the behavioral/auteur batch channels.
pub const TMDB_REC_BATCH_LIMIT: i64 = 20;
pub const DIRECTOR_BATCH_LIMIT: usize = 15;

/// One channel's outcome for ONE seed. Python threads Exceptions
/// through the slots (`gather(..., return_exceptions=True)`) and the
/// merge loop isinstance-handles them IN PLACE — never filtered, the
/// slot indexes stay aligned to seed positions.
#[derive(Debug)]
pub enum ChannelSlot {
    Rows(Vec<CandidateRow>),
    /// the seed's channel failed — its slot stays, contributing nothing
    Failed(String),
}

/// The fixed channel order of the seed-major × channel-minor walk —
/// dedup winners and via tags depend on it exactly as in Python.
pub const CHANNEL_NAMES: [&str; 4] = ["genre", "KNN", "TMDB-recs", "director"];

/// The pure pool walk (step 2's merge). Returns the deduped candidate
/// pool in walk order plus the KNN similarity harvest (the ranking
/// query already computed the cosine — `knn_similarity` truthiness:
/// None and 0.0 are both falsy and excluded).
pub fn assemble_pool(
    seed_infos: &[SeedInfo],
    channels: [Vec<ChannelSlot>; 4],
) -> (Vec<CandidateRow>, HashMap<i64, HashMap<i64, f64>>) {
    let seed_id_set: HashSet<i64> = seed_infos.iter().map(|s| s.id).collect();
    let mut seen_ids: HashSet<i64> = HashSet::new();
    let mut candidates: Vec<CandidateRow> = Vec::new();
    let mut knn_sim_by_seed: HashMap<i64, HashMap<i64, f64>> = HashMap::new();

    for (si, seed) in seed_infos.iter().enumerate() {
        for (ci, _name) in CHANNEL_NAMES.iter().enumerate() {
            let slot = match &channels[ci].get(si) {
                Some(ChannelSlot::Rows(rows)) => rows,
                Some(ChannelSlot::Failed(_why)) => {
                    // Python logs + observes channel(0) and continues
                    continue;
                }
                None => continue, // shorter slot list — nothing to add
            };
            if ci == 1 && !slot.is_empty() {
                let sims: HashMap<i64, f64> = slot
                    .iter()
                    .filter_map(|c| match c.knn_similarity {
                        // Python truthiness on the float: 0.0 is falsy
                        Some(v) if v != 0.0 => Some((c.id, v)),
                        _ => None,
                    })
                    .collect();
                if !sims.is_empty() {
                    knn_sim_by_seed.insert(seed.id, sims);
                }
            }
            for c in slot {
                if !seen_ids.contains(&c.id) && !seed_id_set.contains(&c.id) {
                    seen_ids.insert(c.id);
                    candidates.push(c.clone());
                }
            }
        }
    }
    (candidates, knn_sim_by_seed)
}

/// Fan-out knobs resolved by the caller (config stays out of this fn).
#[derive(Debug, Clone)]
pub struct RecallArgs {
    pub allow_cross_media: bool,
    /// resolved embedding column (default space or the per-user model's)
    pub emb_column: String,
    /// ENABLE_KNN_FINETUNED (read at call time like Python's os.getenv)
    pub knn_ft_enabled: bool,
}

/// Steps 2–2b: all four recall channels concurrently, then the pool
/// walk, then session filters. Channel failures land in their seed's
/// slot (`Failed`) — a failed channel never fails the request.
pub async fn recall_pool(
    db: &Db,
    seed_infos: &[SeedInfo],
    args: &RecallArgs,
    filters: &cine_rec_db::info::RecallFilters,
) -> Result<(Vec<CandidateRow>, HashMap<i64, HashMap<i64, f64>>), cine_rec_db::DbError> {
    let n = seed_infos.len();

    // per-seed channels — each seed degrades alone
    let mut genre_futs = Vec::with_capacity(n);
    let mut knn_futs = Vec::with_capacity(n);
    for s in seed_infos {
        genre_futs.push(generate_candidates(
            db,
            s.id,
            &s.genres,
            &s.media_type,
            60,
            args.allow_cross_media,
        ));
        knn_futs.push(generate_knn_candidates(
            db,
            s.id,
            &s.media_type,
            KNN_CANDIDATES_PER_SEED,
            args.allow_cross_media,
            &args.emb_column,
            if args.knn_ft_enabled { Some(100) } else { None },
        ));
    }
    let seeds_pairs: Vec<(i64, String)> = seed_infos
        .iter()
        .map(|s| (s.id, s.media_type.clone()))
        .collect();

    // batched channels with per-seed fallback (the batch's error
    // retries seed-by-seed; each seed then degrades alone, exactly
    // like the pre-batch fan-out)
    let tmdb_channel = async {
        match generate_tmdb_rec_candidates_batch(db, &seeds_pairs, TMDB_REC_BATCH_LIMIT).await {
            Ok(per_seed) => per_seed
                .into_iter()
                .map(ChannelSlot::Rows)
                .collect::<Vec<_>>(),
            Err(_) => {
                let mut futs = Vec::with_capacity(n);
                for (sid, mt) in &seeds_pairs {
                    futs.push(generate_tmdb_rec_candidates(db, *sid, mt, TMDB_REC_BATCH_LIMIT));
                }
                let results = futures_join_all_tmdb(futs).await;
                results
                    .into_iter()
                    .map(|r| match r {
                        Ok(rows) => ChannelSlot::Rows(rows),
                        Err(e) => ChannelSlot::Failed(e.to_string()),
                    })
                    .collect::<Vec<_>>()
            }
        }
    };
    let director_channel = async {
        match generate_director_candidates_batch(db, &seeds_pairs, DIRECTOR_BATCH_LIMIT).await {
            Ok(per_seed) => per_seed
                .into_iter()
                .map(ChannelSlot::Rows)
                .collect::<Vec<_>>(),
            Err(_) => {
                let mut futs = Vec::with_capacity(n);
                for (sid, mt) in &seeds_pairs {
                    futs.push(generate_director_candidates(db, *sid, mt, DIRECTOR_BATCH_LIMIT));
                }
                let results = futures_join_all_director(futs).await;
                results
                    .into_iter()
                    .map(|r| match r {
                        Ok(rows) => ChannelSlot::Rows(rows),
                        Err(e) => ChannelSlot::Failed(e.to_string()),
                    })
                    .collect::<Vec<_>>()
            }
        }
    };

    // all four channels concurrently — a slow channel is waited on,
    // not serialised (asyncio.gather parity)
    let (genre_results, knn_results, tmdb_slots, director_slots) =
        tokio::join!(
            futures_join_all_genre(genre_futs),
            futures_join_all_knn(knn_futs),
            tmdb_channel,
            director_channel,
        );

    let to_slots = |results: Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>>| -> Vec<ChannelSlot> {
        results
            .into_iter()
            .map(|r| match r {
                Ok(rows) => ChannelSlot::Rows(rows),
                Err(e) => ChannelSlot::Failed(e.to_string()),
            })
            .collect()
    };
    let channels = [
        to_slots(genre_results),
        to_slots(knn_results),
        tmdb_slots,
        director_slots,
    ];

    let (mut candidates, knn_sims) = assemble_pool(seed_infos, channels);

    // 2b. session filters at the recall boundary
    if filters != &cine_rec_db::info::RecallFilters::default() {
        let cands_meta: Vec<(i64, String)> = candidates
            .iter()
            .map(|c| (c.id, c.media_type.clone().unwrap_or_default()))
            .collect();
        let outcome = cine_rec_db::info::apply_recall_filters(db, &cands_meta, filters).await?;
        if !outcome.degraded {
            candidates.retain(|c| {
                outcome
                    .eligible
                    .contains(&(c.id, c.media_type.clone().unwrap_or_default()))
            });
        }
    }
    Ok((candidates, knn_sims))
}

// join_all without pulling the futures crate: wrap the three per-seed
// future lists (distinct fn item types) behind a tiny helper each
async fn futures_join_all_genre(
    futs: Vec<
        impl std::future::Future<Output = Result<Vec<CandidateRow>, cine_rec_db::DbError>>,
    >,
) -> Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>> {
    let mut out = Vec::with_capacity(futs.len());
    for f in futs {
        out.push(f.await);
    }
    out
}
async fn futures_join_all_knn(
    futs: Vec<
        impl std::future::Future<Output = Result<Vec<CandidateRow>, cine_rec_db::DbError>>,
    >,
) -> Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>> {
    futures_join_all_generic(futs).await
}
async fn futures_join_all_tmdb(
    futs: Vec<
        impl std::future::Future<Output = Result<Vec<CandidateRow>, cine_rec_db::DbError>>,
    >,
) -> Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>> {
    futures_join_all_generic(futs).await
}
async fn futures_join_all_director(
    futs: Vec<
        impl std::future::Future<Output = Result<Vec<CandidateRow>, cine_rec_db::DbError>>,
    >,
) -> Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>> {
    futures_join_all_generic(futs).await
}
async fn futures_join_all_generic(
    futs: Vec<
        impl std::future::Future<Output = Result<Vec<CandidateRow>, cine_rec_db::DbError>>,
    >,
) -> Vec<Result<Vec<CandidateRow>, cine_rec_db::DbError>> {
    let mut out = Vec::with_capacity(futs.len());
    for f in futs {
        out.push(f.await);
    }
    out
}

// ---------------------------------------------------------------------------
// optimize_candidate_selection (scoring.py) — pre-score intake filter
// ---------------------------------------------------------------------------

/// config.py floors, identical for both media in this deployment.
pub const MIN_VOTE_COUNT: i64 = 500;
pub const VOTE_FLOOR_TV: i64 = 500;
/// the rating floor applies to EVERY recall path (via included)
const OPTIMIZE_RATING_FLOOR: f64 = 5.5;

/// `optimize_candidate_selection` over recall rows. Rating floor for
/// all; via-tagged rows (knn/tmdb) skip the genre gate AND the vote
/// floor — their proximity signal is the qualification. Stable
/// score-order on the genre pool (Python's sorted is stable).
pub fn optimize_candidate_selection(
    candidates: Vec<CandidateRow>,
    base_genres: &[String],
    target_count: usize,
) -> Vec<CandidateRow> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let base_genre_set: HashSet<&str> =
        base_genres.iter().map(|g| g.as_str()).collect();
    let mut curated_pool: Vec<CandidateRow> = Vec::new();
    let mut genre_pool: Vec<(f64, CandidateRow)> = Vec::new();

    for cand in candidates {
        // rating floor (applies to every recall path)
        if cand.vote_average.unwrap_or(0.0) < OPTIMIZE_RATING_FLOOR {
            continue;
        }
        // Python truthiness on via: an EMPTY tag is falsy
        let via = cand.via.as_deref().is_some_and(|v| !v.is_empty());
        if via {
            curated_pool.push(cand); // keep the channel's own ordering
            continue;
        }
        let vote_floor = if cand.media_type.as_deref() == Some("tv") {
            VOTE_FLOOR_TV
        } else {
            MIN_VOTE_COUNT
        };
        if cand.vote_count.unwrap_or(0) < vote_floor {
            continue;
        }
        let cand_genres: HashSet<&str> =
            cand.genres.iter().map(|g| g.as_str()).collect();
        // genre overlap against the FULL seed genre list
        if !base_genre_set.is_empty()
            && !cand_genres.is_empty()
            && base_genre_set.is_disjoint(&cand_genres)
        {
            continue;
        }
        // blended intake score — rating leads, popularity mildly
        // matters, votes on a log scale (see scoring.py)
        let vote_avg = cand.vote_average.unwrap_or(0.0) / 10.0;
        let pop = (cand.popularity.unwrap_or(0.0) / 100.0).min(1.0);
        let vote_cnt =
            (((cand.vote_count.unwrap_or(1).max(1)) as f64 + 1.0).log10() / 5.0).min(1.0);
        genre_pool.push((
            (vote_avg * 0.5) + (pop * 0.2) + (vote_cnt * 0.3),
            cand,
        ));
    }
    // stable descending: equal scores keep recall order
    genre_pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let curated_limit = (target_count as f64 * 0.7) as usize; // int() truncates
    let mut combined: Vec<CandidateRow> = curated_pool.into_iter().take(curated_limit).collect();
    combined.extend(genre_pool.into_iter().map(|(_, c)| c));
    combined.truncate(target_count * 2);
    combined
}

/// Step 5: batch-enrich the surviving candidates.
pub async fn enrich_pool(
    db: &Db,
    candidates: &[CandidateRow],
) -> Result<HashMap<i64, Enrichment>, cine_rec_db::DbError> {
    let ids: Vec<i64> = candidates.iter().map(|c| c.id).collect();
    let media: Vec<String> = candidates
        .iter()
        .map(|c| c.media_type.clone().unwrap_or_default())
        .collect();
    enrich_candidates_batch(db, &ids, Some(&media)).await
}
