//! The scoring interface the phase-2 port must satisfy, plus the
//! red-green stubs that prove the harness itself works.
//!
//! The trait mirrors the Python entry points one-for-one so the real
//! translation is a drop-in replacement; only the backend `impl`
//! changes between phase 1 (stubs) and phase 2 (the port).

use std::cell::Cell;
use std::collections::HashSet;

use crate::goldens::Harness;
use crate::types::{Entity, Profile};

/// Scoring interface, mirroring `cine_rec_engine.service`:
///
/// * [`ScoringBackend::entity_profile`] — `_entity_profile(entity,
///   entity_data, genres_override)`. `entity_data is None` on the seed
///   side; the enriched side reads keywords/cast/companies/crew from
///   `entity_data` when given.
/// * [`ScoringBackend::feature_vector`] — `feature_vector(cand,
///   cand_data, cand_genres, seed, overview_sim)`: the raw 28-entry
///   feature vector, contract: bit-exact.
/// * [`ScoringBackend::score_pair_fast`] — `_score_pair_fast(sp, cp,
///   cand, cand_data, cand_genres_set, overview_sim, w_list)`: the hot
///   loop score, contract: ≤ [`crate::MAX_SCORE_ULP`] from the golden.
pub trait ScoringBackend {
    fn entity_profile(
        &self,
        entity: &Entity,
        entity_data: Option<&Entity>,
        genres_override: Option<&[String]>,
    ) -> Profile;

    fn feature_vector(
        &self,
        cand: &Entity,
        cand_data: &Entity,
        cand_genres: &[String],
        seed: &Entity,
        overview_sim: f64,
    ) -> Vec<f64>;

    #[allow(clippy::too_many_arguments)]
    fn score_pair_fast(
        &self,
        sp: &Profile,
        cp: &Profile,
        cand: &Entity,
        cand_data: &Entity,
        cand_genres_set: Option<&HashSet<String>>,
        overview_sim: f64,
        w_list: &[f64],
    ) -> f64;
}

/// A deliberately-wrong backend: zeros everywhere. Exists so tests can
/// prove the harness DETECTS wrongness (and counts it exactly).
pub struct ZeroBackend;

impl ScoringBackend for ZeroBackend {
    fn entity_profile(
        &self,
        _entity: &Entity,
        _entity_data: Option<&Entity>,
        _genres_override: Option<&[String]>,
    ) -> Profile {
        Profile::default()
    }

    fn feature_vector(
        &self,
        _cand: &Entity,
        _cand_data: &Entity,
        _cand_genres: &[String],
        _seed: &Entity,
        _overview_sim: f64,
    ) -> Vec<f64> {
        vec![0.0; crate::VEC_LEN]
    }

    fn score_pair_fast(
        &self,
        _sp: &Profile,
        _cp: &Profile,
        _cand: &Entity,
        _cand_data: &Entity,
        _cand_genres_set: Option<&HashSet<String>>,
        _overview_sim: f64,
        _w_list: &[f64],
    ) -> f64 {
        0.0
    }
}

/// The green-side test double: serves the golden answers, pair after
/// pair, in the harness's canonical case order — an oracle that can
/// only pass if the driver feeds pairs in exactly the Python golden
/// test's order.
///
/// One cursor plus a per-pair phase (`score_pair_fast` first, then
/// `feature_vector` — the driver's fixed order): a double call, a
/// skipped call, or swapped order panics with a loud message instead
/// of passing quietly. Entity ids are validated against the fixture on
/// every served pair, so a wrong pairing CONVENTION cannot agree with
/// itself.
pub struct OracleBackend<'h> {
    harness: &'h Harness,
    pair_cursor: Cell<usize>,
    /// false → next expected call is score_pair_fast for the current
    /// case; true → feature_vector for the current case.
    fv_phase: Cell<bool>,
}

impl<'h> OracleBackend<'h> {
    pub fn new(harness: &'h Harness) -> Self {
        Self {
            harness,
            pair_cursor: Cell::new(0),
            fv_phase: Cell::new(false),
        }
    }

    fn begin(&self, want_fv: bool) -> usize {
        // fv_phase == want_fv means the expected next call is the one
        // being made (false → score_pair_fast, true → feature_vector)
        assert_eq!(
            self.fv_phase.get(),
            want_fv,
            "oracle driven out of phase at case {} (double/skipped/swapped call)",
            self.pair_cursor.get()
        );
        self.pair_cursor.get()
    }

    fn case(&self, cursor: usize) -> &crate::goldens::GoldenCase {
        self.harness
            .cases
            .get(cursor)
            .expect("oracle driven past the last golden case")
    }
}

impl ScoringBackend for OracleBackend<'_> {
    fn entity_profile(
        &self,
        _entity: &Entity,
        _entity_data: Option<&Entity>,
        _genres_override: Option<&[String]>,
    ) -> Profile {
        Profile::default() // the oracle serves final answers only
    }

    fn feature_vector(
        &self,
        cand: &Entity,
        _cand_data: &Entity,
        _cand_genres: &[String],
        seed: &Entity,
        _overview_sim: f64,
    ) -> Vec<f64> {
        let i = self.begin(true);
        let case = self.case(i);
        assert_eq!(
            case.cand.id, cand.id,
            "feature_vector called out of golden order (case {i})"
        );
        assert_eq!(case.seed.id, seed.id, "seed out of order (case {i})");
        self.pair_cursor.set(i + 1);
        self.fv_phase.set(false);
        case.vec.clone()
    }

    fn score_pair_fast(
        &self,
        _sp: &Profile,
        _cp: &Profile,
        cand: &Entity,
        _cand_data: &Entity,
        _cand_genres_set: Option<&HashSet<String>>,
        _overview_sim: f64,
        _w_list: &[f64],
    ) -> f64 {
        let i = self.begin(false);
        let case = self.case(i);
        assert_eq!(
            case.cand.id, cand.id,
            "score_pair_fast called out of golden order (case {i})"
        );
        self.fv_phase.set(true);
        case.score
    }
}
