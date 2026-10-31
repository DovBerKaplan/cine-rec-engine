//! Golden parity harness for the cine-rec-engine scoring core.
//!
//! The Python package is the source of truth. This crate exists to hold
//! the parity CONTRACT in executable Rust form before any scoring code
//! is ported:
//!
//! * `feature_vector` — all 28 entries bit-exact against
//!   `tests/golden_feature_vectors.json`.
//! * `_score_pair_fast` — at most 1 ULP from the golden score (the two
//!   paths accumulate in different orders; the golden file was captured
//!   from the reference path).
//!
//! Inputs are pinned, not regenerated: `tests/fixtures/entities_120_4.json`
//! freezes the exact entity rows and weight list produced by
//! `benchmarks/bench_scoring.py::build(120, 4)` — the same generator the
//! Python golden test uses. No RNG semantics ever enter the parity
//! question.
//!
//! Phase status (rust-port plan):
//! * phase 1 (this crate) — spec harness + red-green stubs proving the
//!   harness fails when it should and passes when it should.
//! * phase 2 — a real [`ScoringBackend`] translating `scoring.py` /
//!   the service hot loop with the exact operation ORDER of the Python
//!   code (ULP distances depend on it); the same harness then proves
//!   parity. The Python package keeps a transparent fallback to the
//!   pure-Python path — degrade, never fail.
//!
//! No fast-math style reordering anywhere: Rust does not reorder float
//! operations by default, and this crate must keep it that way.

pub mod backend;
pub mod goldens;
pub mod parity;
pub mod types;

pub use backend::{OracleBackend, ScoringBackend, ZeroBackend};
pub use goldens::{Fixture, GoldenCase, GoldenEntry, Harness, Probe};
pub use parity::{ulp_distance, ParityReport};
pub use types::{Entity, Profile};

/// Number of feature-vector entries the contract pins (27 features +
/// the trailing heuristic residual input).
pub const VEC_LEN: usize = 28;

/// Maximum tolerated ULP distance for the fast-path score.
pub const MAX_SCORE_ULP: u64 = 1;

/// Drive a backend over every golden case exactly the way the Python
/// golden test (`tests/test_fast_path_parity.py`) drives the Python
/// functions: per pair, build the candidate and seed profiles, call
/// `_score_pair_fast` with the frozen weight list, and call
/// `feature_vector` — then compare against the golden entry.
pub fn run_parity<B: ScoringBackend>(backend: &B, harness: &Harness) -> ParityReport {
    use std::collections::HashSet;

    let mut report = ParityReport::new(harness.cases.len());
    for case in &harness.cases {
        let cand = &case.cand;
        let seed = &case.seed;

        // Mirrors: cp = _entity_profile(cand, cand, genres_override=cand["genres"])
        let cp = backend.entity_profile(cand, Some(cand), Some(&cand.genres));
        // Mirrors: sp = _entity_profile(seed, None)
        let sp = backend.entity_profile(seed, None, None);
        // Mirrors: set(cand["genres"]) or None
        let cand_genres_set: Option<HashSet<String>> = if cand.genres.is_empty() {
            None
        } else {
            Some(cand.genres.iter().cloned().collect())
        };
        let score = backend.score_pair_fast(
            &sp,
            &cp,
            cand,
            cand,
            cand_genres_set.as_ref(),
            0.0,
            &harness.w_list,
        );
        // Mirrors: feature_vector(cand, cand, cand["genres"], seed)  (overview_sim=0.0)
        let vec = backend.feature_vector(cand, cand, &cand.genres, seed, 0.0);

        report.record_pair(case.index, &vec, &case.vec, score, case.score);
    }
    report
}
