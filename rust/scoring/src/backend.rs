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
/// test's order AND with exactly the Python golden test's arguments.
///
/// Beyond the cursor + phase discipline (a double call, a skipped
/// call, or swapped order panics loudly), the oracle validates every
/// driver argument it can see: entity ids per side (seed profiles get
/// `entity_data = None`, candidate profiles get their own data back),
/// `cand_data.id == cand.id`, `cand_genres == cand.genres`,
/// `cand_genres_set` matching (or `None` for empty genres),
/// `overview_sim == 0.0`, and `w_list` equal to the harness's frozen
/// weight list. A driver bug (wrong weights, swapped sides, dropped
/// arguments) therefore fails HERE in phase 1 instead of surfacing in
/// phase 2 disguised as a port bug.
///
/// What the oracle-green run proves is then honest about its scope:
/// driver wiring + JSON round-trip + comparator acceptance — the
/// pairing convention is anchored separately by the fixture probes,
/// and parity itself only by the phase-2 real backend.
pub struct OracleBackend<'h> {
    harness: &'h Harness,
    cand_ids: HashSet<i64>,
    seed_ids: HashSet<i64>,
    pair_cursor: Cell<usize>,
    /// false → next expected call is score_pair_fast for the current
    /// case; true → feature_vector for the current case.
    fv_phase: Cell<bool>,
}

impl<'h> OracleBackend<'h> {
    pub fn new(harness: &'h Harness) -> Self {
        Self {
            cand_ids: harness.cases.iter().map(|c| c.cand.id).collect(),
            seed_ids: harness.cases.iter().map(|c| c.seed.id).collect(),
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
        entity: &Entity,
        entity_data: Option<&Entity>,
        genres_override: Option<&[String]>,
    ) -> Profile {
        // Driver contract: seed side gets entity_data = None; candidate
        // side gets the SAME entity back as its enriched data, and any
        // genres_override equals the entity's own genres.
        match entity_data {
            None => assert!(
                self.seed_ids.contains(&entity.id),
                "entity_profile(seed, None) called with non-seed id {}",
                entity.id
            ),
            Some(data) => {
                assert_eq!(
                    data.id, entity.id,
                    "entity_profile candidate data id {} != entity id {}",
                    data.id, entity.id
                );
                assert!(
                    self.cand_ids.contains(&entity.id),
                    "entity_profile(cand, Some) called with non-candidate id {}",
                    entity.id
                );
            }
        }
        if let Some(g) = genres_override {
            assert_eq!(
                g,
                entity.genres.as_slice(),
                "genres_override does not mirror the entity genres (id {})",
                entity.id
            );
        }
        Profile::default() // the oracle serves final answers only
    }

    fn feature_vector(
        &self,
        cand: &Entity,
        cand_data: &Entity,
        cand_genres: &[String],
        seed: &Entity,
        overview_sim: f64,
    ) -> Vec<f64> {
        let i = self.begin(true);
        let case = self.case(i);
        assert_eq!(
            case.cand.id, cand.id,
            "feature_vector called out of golden order (case {i})"
        );
        assert_eq!(case.seed.id, seed.id, "seed out of order (case {i})");
        assert_eq!(
            cand_data.id, cand.id,
            "feature_vector cand_data is not the candidate (case {i})"
        );
        assert_eq!(
            cand_genres,
            cand.genres.as_slice(),
            "feature_vector cand_genres do not mirror the candidate (case {i})"
        );
        assert_eq!(
            overview_sim, 0.0,
            "feature_vector overview_sim must be 0.0 on the golden path (case {i})"
        );
        self.pair_cursor.set(i + 1);
        self.fv_phase.set(false);
        case.vec.clone()
    }

    fn score_pair_fast(
        &self,
        _sp: &Profile,
        _cp: &Profile,
        cand: &Entity,
        cand_data: &Entity,
        cand_genres_set: Option<&HashSet<String>>,
        overview_sim: f64,
        w_list: &[f64],
    ) -> f64 {
        let i = self.begin(false);
        let case = self.case(i);
        assert_eq!(
            case.cand.id, cand.id,
            "score_pair_fast called out of golden order (case {i})"
        );
        assert_eq!(
            cand_data.id, cand.id,
            "score_pair_fast cand_data is not the candidate (case {i})"
        );
        let expected_set: HashSet<String> = cand.genres.iter().cloned().collect();
        match cand_genres_set {
            Some(set) if !set.is_empty() => assert_eq!(
                set, &expected_set,
                "cand_genres_set does not mirror the candidate genres (case {i})"
            ),
            _ => assert!(
                cand.genres.is_empty() && cand_genres_set.is_none(),
                "empty genres must surface as None, non-empty as Some (case {i})"
            ),
        }
        assert_eq!(
            overview_sim, 0.0,
            "score_pair_fast overview_sim must be 0.0 on the golden path (case {i})"
        );
        assert_eq!(
            w_list,
            self.harness.w_list.as_slice(),
            "score_pair_fast got a w_list that is not the harness weight list (case {i})"
        );
        self.fv_phase.set(true);
        case.score
    }
}
