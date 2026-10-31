//! Entity row and derived profile — the data shapes of the scoring
//! interface, mirroring the Python dicts field-for-field.
//!
//! Field ownership follows the Python code exactly: the ENRICHED side
//! (keywords / cast / companies / networks / crew ids) is read from
//! `cand_data`, while recall rows carry genres / overview / scalars.
//! The Rust port must preserve that split or every enriched feature
//! silently reads zero.

use std::collections::HashSet;

/// One enriched catalog row (as produced by recall / `bench_scoring`).
///
/// `#[serde(default)]` mirrors Python `dict.get` semantics: a missing
/// key deserializes to the field default instead of failing.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct Entity {
    pub id: i64,
    pub media_type: String,
    pub title: Option<String>,
    pub title_en: Option<String>,
    pub genres: Vec<String>,
    pub keywords: Vec<String>,
    pub cast_ids: Vec<i64>,
    pub director_ids: Vec<i64>,
    pub writer_ids: Vec<i64>,
    pub composer_ids: Vec<i64>,
    pub dp_ids: Vec<i64>,
    pub companies: Vec<String>,
    pub networks: Vec<String>,
    pub original_language: Option<String>,
    pub overview: Option<String>,
    pub overview_en: Option<String>,
    pub release_year: Option<i64>,
    pub vote_average: Option<f64>,
    pub vote_count: Option<i64>,
    pub collection_id: Option<i64>,
    pub adult: bool,
    pub via: Option<String>,
    pub tmdb_rank: Option<i64>,
    // Personalization-era narrative fields (absent in the bench rows;
    // present on real enriched rows).
    pub narrative: Option<String>,
    pub pacing: Option<String>,
    pub emotional_arc: Option<Vec<String>>,
    pub director: Option<String>,
}

/// Derive-once view of one side of a (seed, candidate) pair — the Rust
/// shape of the Python `_entity_profile` dict. Everything that depends
/// on ONE side only is computed once here instead of once per pair.
///
/// Phase 1 carries the struct as interface only; the phase-2 backend
/// fills it with the exact Python derivation.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    // id sets
    pub dir_ids: HashSet<i64>,
    pub writer_ids: HashSet<i64>,
    pub composer_ids: HashSet<i64>,
    pub dp_ids: HashSet<i64>,
    /// Top-5 cast ids.
    pub cast5: HashSet<i64>,
    /// Meta-stopword-filtered keyword set.
    pub keywords: HashSet<String>,
    pub companies: HashSet<String>,
    pub networks: HashSet<String>,
    /// Canonical (Hebrew→English mapped) genre set.
    pub can_genres: HashSet<String>,
    /// Raw genre set before canonical mapping.
    pub can_genres_raw: HashSet<String>,
    pub param_genres: Vec<String>,
    /// Style labels derived from keywords.
    pub style: HashSet<String>,
    // scalars / strings
    pub genres: Vec<String>,
    /// Animated/live-action/… modality label.
    pub modality: String,
    pub director: Option<String>,
    pub coll: Option<i64>,
    /// Character-bigram set of the overview; `None` when there is no
    /// overview text (the Python side stores `None`, not an empty set).
    pub overview_bigrams: Option<HashSet<String>>,
    pub overview: String,
    pub year: Option<i64>,
    pub votes: i64,
    pub rating: f64,
    pub narrative: Option<String>,
    pub pacing: Option<String>,
    pub emotional_arc: Option<Vec<String>>,
    /// Lowercased/stripped emotional-arc tag set.
    pub arc_set: HashSet<String>,
    pub adult: bool,
    pub via: Option<String>,
    // candidate-side precomputed pair inputs
    pub cand_coll: Option<i64>,
    /// Rank-decay factor for `via == "tmdb"` candidates; 0.0 otherwise.
    pub tmdb_rec_decay: f64,
}
