//! Phase-1 gate: the harness itself — red-green proof.
//!
//! GREEN side: an oracle backend serving the golden answers drives the
//! full harness path (fixture → pairing → backend → comparators) to
//! zero mismatches. If the pairing convention, the JSON float
//! round-trip, or the comparators were broken, this test goes red.
//!
//! RED side: wrong backends and perturbed values must be CAUGHT, and
//! caught exactly — the expected mismatch counts are derived from the
//! goldens themselves, so the harness can neither miss nor over-report.
//!
//! `cargo test -- --ignored demo_red_end_to_end` intentionally runs
//! the GREEN assertions against a zero backend to demonstrate the full
//! failure output (it is expected to FAIL — that is the demonstration).

use std::collections::HashSet;
use std::path::Path;

use cine_rec_scoring::backend::ScoringBackend;
use cine_rec_scoring::{run_parity, Harness, OracleBackend, ParityReport, ZeroBackend, VEC_LEN};

fn load_harness() -> Harness {
    // crate lives at <repo>/rust/scoring — the repo root (holding
    // tests/golden_feature_vectors.json) is two levels up.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root two levels above the crate manifest")
        .to_path_buf();
    let fixture = manifest
        .join("tests")
        .join("fixtures")
        .join("entities_120_4.json");
    Harness::load(&repo_root, &fixture).unwrap_or_else(|e| panic!("harness must load: {e}"))
}

// ── Shape contract ─────────────────────────────────────────────────

#[test]
fn harness_shape_is_the_contract() {
    let h = load_harness();
    assert!(!h.cases.is_empty(), "empty harness must never look green");
    assert_eq!(h.n_cands, 120);
    assert_eq!(h.n_seeds, 4);
    assert_eq!(h.cases.len(), 480);
    assert_eq!(h.w_list.len(), VEC_LEN);
    assert_eq!(h.feature_names.len(), VEC_LEN);
    for c in &h.cases {
        assert_eq!(c.vec.len(), VEC_LEN, "case {}", c.index);
    }
}

// ── Pairing convention, anchored to Python ─────────────────────────

#[test]
fn pairing_convention_matches_python_probes() {
    let h = load_harness();
    assert!(!h.probes.is_empty(), "fixture must carry probes");
    for p in &h.probes {
        let case = h
            .case_for(p.cand_id, p.seed_id)
            .unwrap_or_else(|| panic!("probe pair ({}, {}) not found", p.cand_id, p.seed_id));
        assert_eq!(
            case.index, p.golden_index,
            "pairing convention disagrees with Python for ({}, {})",
            p.cand_id, p.seed_id
        );
        assert_eq!(
            case.score.to_bits(),
            p.score.to_bits(),
            "probe score for ({}, {}) did not round-trip",
            p.cand_id,
            p.seed_id
        );
        for (slot_str, value) in &p.vec_slots {
            let slot: usize = slot_str.parse().expect("probe slot key is numeric");
            assert_eq!(
                case.vec[slot].to_bits(),
                value.to_bits(),
                "probe vec[{slot}] for ({}, {}) did not round-trip",
                p.cand_id,
                p.seed_id
            );
        }
    }
}

// ── GREEN: oracle drives the full path to zero mismatches ──────────

#[test]
fn oracle_backend_is_green_end_to_end() {
    let h = load_harness();
    let oracle = OracleBackend::new(&h);
    let report = run_parity(&oracle, &h);
    assert!(report.green(), "oracle must be green:\n{report}");
    assert_eq!(report.n_pairs, 480);
}

// ── RED: wrongness is caught, and counted exactly ──────────────────

#[test]
fn zero_backend_is_caught_exactly() {
    let h = load_harness();
    let report = run_parity(&ZeroBackend, &h);
    assert!(!report.green(), "zero backend must be red:\n{report}");

    // Expected counts derived from the goldens themselves: a zero vec
    // mismatches every case whose golden vec has any nonzero slot; a
    // zero score mismatches every case whose golden score is nonzero
    // (genre-gated pairs legitimately score 0.0).
    let exp_vec = h
        .cases
        .iter()
        .filter(|c| c.vec.iter().any(|v| v.to_bits() != 0))
        .count();
    let exp_score = h.cases.iter().filter(|c| c.score != 0.0).count();
    assert_eq!(report.n_vec_mismatch, exp_vec);
    assert_eq!(report.n_score_mismatch, exp_score);
    assert!(report.first_vec_mismatch.is_some());
    assert!(report.first_score_mismatch.is_some());
}

/// Serves golden[i + 1] for pair i — classic index drift. The oracle's
/// own id-validation refuses this; a naive backend serving drifted
/// values must at least be flagged by the comparators.
struct ShiftedOracle<'h> {
    harness: &'h Harness,
    cursor: std::cell::Cell<usize>,
}

impl ScoringBackend for ShiftedOracle<'_> {
    fn entity_profile(
        &self,
        _entity: &cine_rec_scoring::Entity,
        _entity_data: Option<&cine_rec_scoring::Entity>,
        _genres_override: Option<&[String]>,
    ) -> cine_rec_scoring::Profile {
        cine_rec_scoring::Profile::default()
    }
    fn feature_vector(
        &self,
        _cand: &cine_rec_scoring::Entity,
        _cand_data: &cine_rec_scoring::Entity,
        _cand_genres: &[String],
        _seed: &cine_rec_scoring::Entity,
        _overview_sim: f64,
    ) -> Vec<f64> {
        let i = (self.cursor.get() + 1) % self.harness.cases.len();
        self.harness.cases[i].vec.clone()
    }
    fn score_pair_fast(
        &self,
        _sp: &cine_rec_scoring::Profile,
        _cp: &cine_rec_scoring::Profile,
        _cand: &cine_rec_scoring::Entity,
        _cand_data: &cine_rec_scoring::Entity,
        _cand_genres_set: Option<&HashSet<String>>,
        _overview_sim: f64,
        _w_list: &[f64],
    ) -> f64 {
        // advance once per pair, on the second call
        let i = (self.cursor.replace(self.cursor.get() + 1) + 1) % self.harness.cases.len();
        self.harness.cases[i].score
    }
}

#[test]
fn off_by_one_pairing_is_caught() {
    let h = load_harness();
    let shifted = ShiftedOracle {
        harness: &h,
        cursor: std::cell::Cell::new(0),
    };
    let report = run_parity(&shifted, &h);
    assert!(
        !report.green(),
        "index drift must be flagged by the comparators:\n{report}"
    );
    assert!(report.n_vec_mismatch > 0);
}

#[test]
fn oracle_refuses_misaligned_entities() {
    // The oracle validates the (cand, seed) ids against its cursor —
    // feeding it the candidate of a DIFFERENT case must panic loudly
    // rather than silently serve the wrong golden row. (Cases 0–3
    // share cand 0 — cand-major pairing — so the wrong cand must come
    // from another cand group: case 4 carries cand 1.)
    let h = load_harness();
    let oracle = OracleBackend::new(&h);
    let wrong_cand = h.cases[4].cand.clone();
    assert_ne!(wrong_cand.id, h.cases[0].cand.id);
    let sp = oracle.entity_profile(&h.cases[0].seed, None, None);
    let cp = oracle.entity_profile(&wrong_cand, Some(&wrong_cand), Some(&wrong_cand.genres));
    let set: HashSet<String> = wrong_cand.genres.iter().cloned().collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.score_pair_fast(
            &sp,
            &cp,
            &wrong_cand,
            &wrong_cand,
            Some(&set),
            0.0,
            &h.w_list,
        );
    }));
    assert!(
        result.is_err(),
        "oracle must refuse entities that disagree with its case cursor"
    );
}

// ── Comparator boundaries (via the public report API) ──────────────

#[test]
fn comparator_bites_on_single_bit_flip_in_vec() {
    let h = load_harness();
    // Flip the LAST bit of one slot of the oracle's answer for case 0 —
    // the contract is bit equality, so even this must be caught.
    let case = &h.cases[0];
    let mut perturbed = case.vec.clone();
    perturbed[0] = f64::from_bits(perturbed[0].to_bits() ^ 1);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &perturbed, &case.vec, case.score, case.score);
    assert_eq!(report.n_vec_mismatch, 1, "1-bit vec drift must be caught");
    assert!(!report.green());
}

#[test]
fn comparator_tolerates_exactly_one_ulp_in_score() {
    let h = load_harness();
    let case = &h.cases[0];
    assert_eq!(case.score, 0.0); // case 0 is genre-gated; a clean base
    let bumped = f64::from_bits(case.score.to_bits() + 1); // exactly 1 ULP up
    assert_eq!(cine_rec_scoring::ulp_distance(bumped, case.score), 1);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, bumped, case.score);
    assert!(report.green(), "1 ULP is inside the contract:\n{report}");
}

#[test]
fn comparator_rejects_two_ulp_in_score() {
    let h = load_harness();
    let case = &h.cases[0];
    let bumped = f64::from_bits(case.score.to_bits() + 2); // 2 ULP up
    assert_eq!(cine_rec_scoring::ulp_distance(bumped, case.score), 2);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, bumped, case.score);
    assert_eq!(report.n_score_mismatch, 1, "2 ULP is outside the contract");
}

#[test]
fn nan_output_is_always_a_mismatch() {
    let h = load_harness();
    let case = &h.cases[0];
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, f64::NAN, case.score);
    assert_eq!(report.n_score_mismatch, 1, "NaN score must be caught");
    // A NaN where the golden has a finite 0.0 must be caught by the
    // bit comparison (NaN bits never equal finite bits).
    let nan_vec: Vec<f64> = case
        .vec
        .iter()
        .map(|v| if *v == 0.0 { f64::NAN } else { *v })
        .collect();
    let mut report2 = ParityReport::new(1);
    report2.record_pair(0, &nan_vec, &case.vec, case.score, case.score);
    assert_eq!(report2.n_vec_mismatch, 1, "NaN vec entry must be caught");
}

// ── RED demo (expected to FAIL when run — that is the point) ───────

#[test]
#[ignore = "demonstration only: `cargo test -- --ignored demo_red_end_to_end` shows the harness going red against a broken backend"]
fn demo_red_end_to_end() {
    let h = load_harness();
    let report = run_parity(&ZeroBackend, &h);
    assert!(
        report.green(),
        "this line fails on purpose — harness red demo:\n{report}"
    );
}

// ── Interface sanity: the trait surface exists for phase 2 ─────────

#[test]
fn backend_trait_surface_compiles_for_real_impl() {
    // A minimal real-looking backend compiles against the trait with
    // Python-mirroring signatures (phase 2 slots in here); and any
    // placeholder impl is caught by parity, never mistaken for green.
    struct Realish;
    impl ScoringBackend for Realish {
        fn entity_profile(
            &self,
            entity: &cine_rec_scoring::Entity,
            _entity_data: Option<&cine_rec_scoring::Entity>,
            genres_override: Option<&[String]>,
        ) -> cine_rec_scoring::Profile {
            cine_rec_scoring::Profile {
                genres: genres_override
                    .map(|g| g.to_vec())
                    .unwrap_or_else(|| entity.genres.clone()),
                ..Default::default()
            }
        }
        fn feature_vector(
            &self,
            _cand: &cine_rec_scoring::Entity,
            _cand_data: &cine_rec_scoring::Entity,
            _cand_genres: &[String],
            _seed: &cine_rec_scoring::Entity,
            _overview_sim: f64,
        ) -> Vec<f64> {
            vec![0.0; VEC_LEN]
        }
        fn score_pair_fast(
            &self,
            _sp: &cine_rec_scoring::Profile,
            _cp: &cine_rec_scoring::Profile,
            _cand: &cine_rec_scoring::Entity,
            _cand_data: &cine_rec_scoring::Entity,
            _cand_genres_set: Option<&HashSet<String>>,
            _overview_sim: f64,
            _w_list: &[f64],
        ) -> f64 {
            0.0
        }
    }
    let h = load_harness();
    let report = run_parity(&Realish, &h);
    assert!(!report.green(), "a placeholder impl must not pass parity");
}
