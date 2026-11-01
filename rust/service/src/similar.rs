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
