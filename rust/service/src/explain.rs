//! WHY rendering — the Rust port of `explain.py`:
//! explain_features / explain_contributions / explain_results.
//! Pure published-weights arithmetic only.

use std::collections::HashMap;

use cine_rec_db::executor::{Db, DbError};
use cine_rec_db::info::SeedInfo;
use cine_rec_db::recall::enrich_candidates_batch;
use cine_rec_scoring::real::{entity_profile, vec_from_profiles};

use crate::similar::{cand_entity, enrichment_entity, seed_entity, WeightLists};

/// FEATURE_NAMES order — the contract for vector ↔ name mapping (the
/// scoring crate's vec_from_profiles emits in this order; locked by
/// the golden files).
pub const FEATURE_NAMES: [&str; 28] = [
    "cosine_sim",
    "keyword_sim",
    "cast_sim",
    "genre_priority_sum",
    "director_match",
    "director_channel",
    "writer_match",
    "tmdb_rec_decay",
    "shared_collection",
    "shared_network",
    "style_match",
    "company_sim",
    "year_le5",
    "year_le10",
    "year_gt25",
    "votes_gt15k",
    "low_votes_high_rating",
    "rating_bonus",
    "pop_action_leak",
    "composer_match",
    "dp_match",
    "medium_mismatch",
    "narrative_match",
    "tone_compatibility",
    "emotional_arc_match",
    "pacing_match",
    "audience_compatibility",
    "heuristic_score",
];

/// FEATURE_LABELS (explain.py) — human-readable WHY.
pub const FEATURE_LABELS: &[(&str, &str)] = &[
    ("cosine_sim", "plot similarity"),
    ("keyword_sim", "shared keywords"),
    ("cast_sim", "shared cast"),
    ("director_match", "same director"),
    ("director_channel", "director recall"),
    ("writer_match", "same writer"),
    ("composer_match", "same composer"),
    ("dp_match", "same cinematographer"),
    ("tmdb_rec_decay", "TMDB behavior graph"),
    ("shared_collection", "same saga"),
    ("shared_network", "same network"),
    ("style_match", "same style tags"),
    ("company_sim", "same studio"),
    ("genre_priority_sum", "genre overlap"),
    ("tone_compatibility", "same tone"),
    ("narrative_match", "same narrative structure"),
    ("emotional_arc_match", "same emotional arc"),
    ("pacing_match", "same pacing"),
    ("audience_compatibility", "same audience"),
];

/// `_BORING_FEATURES` — near-constant features hidden from the why by
/// default: they explain a score but not a CHOICE.
const BORING_FEATURES: [&str; 8] = [
    "audience_compatibility",
    "tone_compatibility",
    "rating_bonus",
    "votes_gt15k",
    "low_votes_high_rating",
    "year_le5",
    "year_le10",
    "heuristic_score",
];

fn label_of(name: &str) -> String {
    FEATURE_LABELS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.to_string())
        .unwrap_or_else(|| name.to_string())
}

fn is_boring(name: &str) -> bool {
    BORING_FEATURES.contains(&name)
}

/// `explain_features` — top contributing (label, contribution) pairs.
/// An empty answer is honest — no fallback to always-firing features.
pub fn explain_features(
    vec: &[f64],
    weights_by_name: &HashMap<String, f64>,
    top: usize,
) -> Vec<(String, f64)> {
    let mut contrib: Vec<(f64, usize)> = FEATURE_NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < vec.len())
        .map(|(i, name)| {
            let w = weights_by_name.get(*name).copied().unwrap_or(0.0);
            (w * vec[i], i)
        })
        .collect();
    // Python sorts (contribution, name) tuples descending — ties break
    // by name REVERSED (tuple compare) — then filters score>0 and
    // non-boring, takes top
    contrib.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| FEATURE_NAMES[b.1].cmp(FEATURE_NAMES[a.1]))
    });
    contrib
        .into_iter()
        .filter(|(score, i)| *score > 0.0 && !is_boring(FEATURE_NAMES[*i]))
        .take(top)
        .map(|(score, i)| (label_of(FEATURE_NAMES[i]), score))
        .collect()
}

/// `explain_contributions` — {feature name: weight×value} for the same
/// top non-boring picks (the numbers the label path drops), rounded 4.
pub fn explain_contributions(
    vec: &[f64],
    weights_by_name: &HashMap<String, f64>,
    top: usize,
) -> Vec<(String, f64)> {
    let mut contrib: Vec<(f64, usize)> = FEATURE_NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < vec.len())
        .map(|(i, name)| {
            let w = weights_by_name.get(*name).copied().unwrap_or(0.0);
            (w * vec[i], i)
        })
        .collect();
    contrib.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| FEATURE_NAMES[b.1].cmp(FEATURE_NAMES[a.1]))
    });
    contrib
        .into_iter()
        .filter(|(score, i)| *score > 0.0 && !is_boring(FEATURE_NAMES[*i]))
        .take(top)
        .map(|(score, i)| {
            (
                FEATURE_NAMES[i].to_string(),
                format!("{:.4}", score).parse().unwrap_or(score),
            )
        })
        .collect()
}

/// per-result why: (labels, contributions)
pub type WhyEntry = (Vec<String>, Vec<(String, f64)>);

/// `explain_results` — per-result why against the FIRST seed. Re-derives
/// the feature vector post-hoc (an approximation by design). Degrades
/// to empty on any read failure.
pub async fn explain_results(
    db: &Db,
    result_ids: &[(i64, String)],
    genres_by_id: &HashMap<i64, Vec<String>>,
    seed: &SeedInfo,
    weights: &WeightLists,
    weights_by_name: &HashMap<String, f64>,
) -> Result<HashMap<i64, WhyEntry>, DbError> {
    if result_ids.is_empty() {
        return Ok(HashMap::new());
    }
    // enrich candidates (degrade to empty per candidate)
    let ids: Vec<i64> = result_ids.iter().map(|(id, _)| *id).collect();
    let media: Vec<String> = result_ids.iter().map(|(_, mt)| mt.clone()).collect();
    let enriched = enrich_candidates_batch(db, &ids, Some(&media)).await?;

    let seed_ent = seed_entity(seed);
    let sp = entity_profile(&seed_ent, None, None);
    let empty_enrichment = Default::default();
    let mut out: HashMap<i64, WhyEntry> = HashMap::new();

    for (cid, mt) in result_ids {
        let e = enriched.get(cid).unwrap_or(&empty_enrichment);
        let data_ent = enrichment_entity(e);
        // candidate Entity mirrors what Python's enriched dict carries;
        // the enriched row REPLACES the recall row wholesale there
        let mut cand_ent = cand_entity(&cand_row_of(*cid, mt, genres_by_id));
        cand_ent.director = e.director.clone();
        let genres = genres_by_id.get(cid).cloned().unwrap_or_default();
        let cp = entity_profile(&cand_ent, Some(&data_ent), Some(&genres));
        let vec = vec_from_profiles(
            &sp,
            &cp,
            &cand_ent,
            &data_ent,
            0.0, // explain path never had overview sims (Python parity)
            &weights.heuristic27,
        );
        out.insert(
            *cid,
            (
                explain_features(&vec, weights_by_name, 3)
                    .into_iter()
                    .map(|(l, _)| l)
                    .collect(),
                explain_contributions(&vec, weights_by_name, 3),
            ),
        );
    }
    Ok(out)
}

/// minimal cand row carrier for the explain path (the results carry
/// id/media_type/genres only; everything else the enrichment supplies)
fn cand_row_of(cid: i64, mt: &str, genres_by_id: &HashMap<i64, Vec<String>>) -> cine_rec_db::recall::CandidateRow {
    cine_rec_db::recall::CandidateRow {
        id: cid,
        media_type: Some(mt.to_string()),
        genres: genres_by_id.get(&cid).cloned().unwrap_or_default(),
        ..Default::default()
    }
}
