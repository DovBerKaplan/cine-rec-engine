//! Phase-1 gate: the harness itself — red-green proof.
//!
//! Two golden sets are driven (BASE 480 pairs from the repo golden
//! file; ENRICHED 64 pairs covering the via channels, narrative/
//! pacing/emotional-arc, the director-name fallback, style keywords,
//! and the via gate-exemption — features that are constant-zero on
//! the base set).
//!
//! GREEN side: an oracle backend serving the golden answers drives
//! the full harness path (fixture → pairing → backend → comparators)
//! to zero mismatches on BOTH sets — while validating every driver
//! argument (weights, sides, genres, overview_sim), so a driver bug
//! fails here, not in phase 2 disguised as a port bug.
//!
//! RED side: wrong backends and perturbed values must be CAUGHT, and
//! caught exactly — expected mismatch counts are derived from the
//! goldens themselves, so the harness can neither miss nor over-report.
//! Vector LENGTH is part of the contract: empty / truncated / padded
//! vectors are red.
//!
//! `cargo test -- --ignored demo_red_end_to_end` intentionally runs
//! the GREEN assertions against a zero backend to demonstrate the full
//! failure output (it is expected to FAIL — that is the demonstration).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cine_rec_scoring::backend::ScoringBackend;
use cine_rec_scoring::{run_parity, Harness, OracleBackend, ParityReport, ZeroBackend, VEC_LEN};

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn repo_root() -> PathBuf {
    // crate lives at <repo>/rust/scoring — the repo root is two up
    manifest_dir()
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root two levels above the crate manifest")
        .to_path_buf()
}

fn load_base() -> Harness {
    let golden = repo_root()
        .join("tests")
        .join("golden_feature_vectors.json");
    let fixture = manifest_dir()
        .join("tests")
        .join("fixtures")
        .join("entities_120_4.json");
    Harness::load(&golden, &fixture).unwrap_or_else(|e| panic!("base harness: {e}"))
}

fn load_enriched() -> Harness {
    let dir = manifest_dir().join("tests").join("fixtures");
    Harness::load(
        &dir.join("goldens_enriched.json"),
        &dir.join("entities_enriched.json"),
    )
    .unwrap_or_else(|e| panic!("enriched harness: {e}"))
}

fn both() -> Vec<(&'static str, Harness)> {
    vec![("base", load_base()), ("enriched", load_enriched())]
}

// ── Shape contract ─────────────────────────────────────────────────

#[test]
fn harness_shape_is_the_contract() {
    let h = load_base();
    assert!(!h.cases.is_empty(), "empty harness must never look green");
    assert_eq!((h.n_cands, h.n_seeds, h.cases.len()), (120, 4, 480));
    assert_eq!(h.w_list.len(), VEC_LEN);
    assert_eq!(h.feature_names.len(), VEC_LEN);
    for c in &h.cases {
        assert_eq!(c.vec.len(), VEC_LEN, "case {}", c.index);
    }

    let e = load_enriched();
    assert_eq!((e.n_cands, e.n_seeds, e.cases.len()), (16, 4, 64));
    assert_eq!(e.w_list, h.w_list, "both sets share one weight list");
}

#[test]
fn enriched_set_actually_exercises_the_base_dead_features() {
    // On the base goldens these six features are constant 0.0 (the
    // bench rows lack via/tmdb_rank/narrative/pacing/emotional_arc/
    // style keywords). The enriched set must wake them — otherwise it
    // adds nothing and the phase-2 gate keeps its holes.
    let base = load_base();
    let enriched = load_enriched();
    for slot in [5usize, 7, 10, 22, 24, 25] {
        assert!(
            base.cases.iter().all(|c| c.vec[slot] == 0.0),
            "base slot {slot} was expected constant-zero (fixture drifted)"
        );
        assert!(
            enriched.cases.iter().any(|c| c.vec[slot] != 0.0),
            "enriched slot {slot} never fires — coverage hole"
        );
    }
}

#[test]
fn enriched_gate_exemption_is_visible_in_goldens() {
    // A genre-clashing candidate WITH a (non-empty) via tag scores
    // against every seed (exemption); the same clash WITHOUT via — or
    // with an EMPTY via string, which is falsy in Python — is gated
    // to 0.0 (pinned by the k=15 clash candidate).
    let h = load_enriched();
    let war = |c: &cine_rec_scoring::GoldenCase| c.cand.genres.as_slice() == ["War"];
    let via_truthy = |c: &cine_rec_scoring::GoldenCase| matches!(c.cand.via.as_deref(), Some(v) if !v.is_empty());
    let exempt: Vec<_> = h.cases.iter().filter(|c| war(c) && via_truthy(c)).collect();
    let gated: Vec<_> = h
        .cases
        .iter()
        .filter(|c| war(c) && !via_truthy(c))
        .collect();
    assert!(!exempt.is_empty(), "no via-tagged clash candidates found");
    assert!(!gated.is_empty(), "no plain clash candidates found");
    assert!(
        exempt.iter().all(|c| c.score != 0.0),
        "exemption not scored"
    );
    assert!(gated.iter().all(|c| c.score == 0.0), "gate not enforced");
}

// ── Pairing convention, anchored to Python ─────────────────────────

/// The full probe contract over a harness: index agreement, bit-exact
/// computed vec slots, computed score within contract tolerance, and
/// discriminating (non-zero, distinct) probes. Shared by the positive
/// test and the golden-mutation negative test.
fn check_probes(h: &Harness, label: &str) -> Result<(), String> {
    if h.probes.is_empty() {
        return Err(format!("{label}: fixture must carry probes"));
    }
    for p in &h.probes {
        let case = h.case_for(p.cand_id, p.seed_id).ok_or_else(|| {
            format!(
                "{label}: probe pair ({}, {}) not found",
                p.cand_id, p.seed_id
            )
        })?;
        if case.index != p.golden_index {
            return Err(format!(
                "{label}: pairing convention disagrees with the Python-searched index for ({}, {}): {} != {}",
                p.cand_id, p.seed_id, case.index, p.golden_index
            ));
        }
        for (slot_str, value) in &p.vec_slots {
            let slot: usize = slot_str.parse().expect("probe slot key is numeric");
            if case.vec[slot].to_bits() != value.to_bits() {
                return Err(format!(
                    "{label}: probe vec[{slot}] (computed by Python) did not land bit-exact at case {}",
                    case.index
                ));
            }
        }
        // the probe score is the COMPUTED fast-path value: within the
        // contract tolerance of the golden it anchored
        let ulp = cine_rec_scoring::ulp_distance(case.score, p.score);
        if ulp > cine_rec_scoring::MAX_SCORE_ULP {
            return Err(format!(
                "{label}: probe score for ({}, {}) is {ulp} ULP from the golden",
                p.cand_id, p.seed_id
            ));
        }
    }
    // probes must discriminate: non-zero scores, distinct cases
    let mut scores: Vec<_> = h.probes.iter().map(|p| p.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    scores.dedup();
    if scores.len() != h.probes.len() {
        return Err(format!("{label}: probes are not distinct"));
    }
    if !h.probes.iter().all(|p| p.score != 0.0) {
        return Err(format!("{label}: probes must carry non-zero scores"));
    }
    Ok(())
}

#[test]
fn pairing_convention_matches_python_probes() {
    for (label, h) in both() {
        check_probes(&h, label).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn probes_catch_a_golden_file_that_breaks_the_convention() {
    // Negative proof of the anchoring: if the golden file did NOT
    // carry each pair's outputs at the convention's index (e.g. it was
    // captured under a transposed order, or two rows were swapped),
    // the searched-index probes must disagree — verified by running
    // the FULL probe check against the mutated harness, not just by
    // eyeballing one slot.
    let h = load_base();
    let anchored: Vec<usize> = h.probes.iter().map(|p| p.golden_index).collect();
    assert!(
        anchored.contains(&1),
        "test assumes a probe anchors index 1"
    );

    let golden_path = repo_root()
        .join("tests")
        .join("golden_feature_vectors.json");
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("read goldens"))
            .expect("parse goldens");
    let entries = doc.as_array_mut().expect("goldens are a list");
    // 120 = seed-major position of pair (cand 0, seed 1) — a piece of
    // the transposition a wrong capture order would have produced
    entries.swap(1, 120);

    // PID-suffixed temp name: two concurrent cargo test runs on one
    // machine must not clobber each other's mutated goldens
    let tmp = std::env::temp_dir().join(format!(
        "cine_rec_goldens_swapped_{}.json",
        std::process::id()
    ));
    std::fs::write(&tmp, serde_json::to_string(&doc).expect("serialize")).expect("write temp");
    let fixture = manifest_dir()
        .join("tests")
        .join("fixtures")
        .join("entities_120_4.json");
    let mutated = Harness::load(&tmp, &fixture).expect("mutated harness still loads (shape ok)");
    let _ = std::fs::remove_file(&tmp);

    assert!(
        check_probes(&mutated, "mutated").is_err(),
        "swapped goldens passed the probe anchors — anchoring is broken"
    );
}

// ── GREEN: oracle drives the full path to zero mismatches ──────────

#[test]
fn oracle_backend_is_green_end_to_end() {
    for (label, h) in both() {
        let oracle = OracleBackend::new(&h);
        let report = run_parity(&oracle, &h);
        assert!(report.green(), "{label}: oracle must be green:\n{report}");
        assert_eq!(report.n_pairs, h.cases.len());
    }
}

// ── RED: wrongness is caught, and counted exactly ──────────────────

#[test]
fn zero_backend_is_caught_exactly() {
    for (label, h) in both() {
        let report = run_parity(&ZeroBackend, &h);
        assert!(
            !report.green(),
            "{label}: zero backend must be red:\n{report}"
        );

        // Expected counts derived from the goldens themselves: a zero
        // 28-entry vec mismatches exactly the cases whose golden vec
        // has a nonzero slot; a zero score mismatches every case whose
        // golden score is nonzero (gated pairs legitimately score 0.0).
        let exp_vec = h
            .cases
            .iter()
            .filter(|c| c.vec.iter().any(|v| v.to_bits() != 0))
            .count();
        let exp_score = h.cases.iter().filter(|c| c.score != 0.0).count();
        assert_eq!(report.n_vec_mismatch, exp_vec, "{label}");
        assert_eq!(report.n_score_mismatch, exp_score, "{label}");
        assert!(report.first_vec_mismatch.is_some());
        assert!(report.first_score_mismatch.is_some() || exp_score == 0);
    }
}

/// Serves golden[i + 1] for pair i — classic index drift. The oracle's
/// own id-validation refuses this; a naive backend serving drifted
/// values must still be flagged by the comparators on both channels.
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
        // score_pair_fast (the driver's FIRST call) has already bumped
        // the cursor for this pair, so [cursor + 1] is two cases ahead
        // of the pair being served.
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
        // the driver's first call of each pair: serve the NEXT case's
        // score and advance
        let i = (self.cursor.replace(self.cursor.get() + 1) + 1) % self.harness.cases.len();
        self.harness.cases[i].score
    }
}

#[test]
fn off_by_one_pairing_is_caught() {
    let h = load_base();
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
    assert!(
        report.n_score_mismatch > 0,
        "score drift must be flagged too"
    );
}

#[test]
fn oracle_refuses_misaligned_entities() {
    // The oracle validates the (cand, seed) ids against its cursor —
    // feeding it the candidate of a DIFFERENT case must panic loudly
    // rather than silently serve the wrong golden row. (Cases 0–3
    // share cand 0 — cand-major pairing — so the wrong cand must come
    // from another cand group: case 4 carries cand 1.)
    let h = load_base();
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

#[test]
fn oracle_refuses_wrong_driver_arguments() {
    // The oracle doubles as a driver-contract checker: foreign weights,
    // a non-zero overview_sim, or a seed profile built with entity_data
    // must panic, not pass quietly (these would otherwise surface in
    // phase 2 disguised as port bugs).
    let h = load_base();
    let case = &h.cases[0];
    let set: HashSet<String> = case.cand.genres.iter().cloned().collect();

    // wrong w_list
    let oracle = OracleBackend::new(&h);
    let sp = oracle.entity_profile(&case.seed, None, None);
    let cp = oracle.entity_profile(&case.cand, Some(&case.cand), Some(&case.cand.genres));
    let bad_w: Vec<f64> = vec![0.0; VEC_LEN];
    let r1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.score_pair_fast(&sp, &cp, &case.cand, &case.cand, Some(&set), 0.0, &bad_w);
    }));
    assert!(r1.is_err(), "oracle must refuse a foreign w_list");

    // non-zero overview_sim
    let oracle = OracleBackend::new(&h);
    let sp = oracle.entity_profile(&case.seed, None, None);
    let cp = oracle.entity_profile(&case.cand, Some(&case.cand), Some(&case.cand.genres));
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ =
            oracle.score_pair_fast(&sp, &cp, &case.cand, &case.cand, Some(&set), 9.9, &h.w_list);
    }));
    assert!(r2.is_err(), "oracle must refuse overview_sim != 0.0");

    // seed profile built with entity_data (side swap)
    let oracle = OracleBackend::new(&h);
    let r3 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.entity_profile(&case.seed, Some(&case.seed), None);
    }));
    assert!(
        r3.is_err(),
        "oracle must refuse entity_data on the seed side"
    );
}

#[test]
fn oracle_refuses_swapped_or_untagged_profiles() {
    // The oracle tags returned profiles with entity id + side, and
    // score_pair_fast checks the tags against the CURRENT case — so
    // swapped sides, a stale/foreign profile, or fully skipped
    // entity_profile calls (untagged default) all panic.
    let h = load_base();
    let case = &h.cases[0];
    let set: HashSet<String> = case.cand.genres.iter().cloned().collect();

    // correct profiles, passed in the WRONG order (sp <-> cp)
    let oracle = OracleBackend::new(&h);
    let sp = oracle.entity_profile(&case.seed, None, None);
    let cp = oracle.entity_profile(&case.cand, Some(&case.cand), Some(&case.cand.genres));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ =
            oracle.score_pair_fast(&cp, &sp, &case.cand, &case.cand, Some(&set), 0.0, &h.w_list);
    }));
    assert!(r.is_err(), "oracle must refuse swapped sp/cp");

    // entity_profile skipped entirely: untagged default profiles
    let oracle = OracleBackend::new(&h);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.score_pair_fast(
            &cine_rec_scoring::Profile::default(),
            &cine_rec_scoring::Profile::default(),
            &case.cand,
            &case.cand,
            Some(&set),
            0.0,
            &h.w_list,
        );
    }));
    assert!(r.is_err(), "oracle must refuse untagged (skipped) profiles");

    // stale profiles from a DIFFERENT case's entities
    let oracle = OracleBackend::new(&h);
    let other_seed = &h.cases[5].seed;
    let wrong_sp = oracle.entity_profile(other_seed, None, None);
    let cp = oracle.entity_profile(&case.cand, Some(&case.cand), Some(&case.cand.genres));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.score_pair_fast(
            &wrong_sp,
            &cp,
            &case.cand,
            &case.cand,
            Some(&set),
            0.0,
            &h.w_list,
        );
    }));
    assert!(r.is_err(), "oracle must refuse a foreign seed profile");

    // candidate profile without genres_override
    let oracle = OracleBackend::new(&h);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = oracle.entity_profile(&case.cand, Some(&case.cand), None);
    }));
    assert!(
        r.is_err(),
        "oracle must require genres_override on the candidate side"
    );
}

// ── Comparator boundaries (via the public report API) ──────────────

#[test]
fn comparator_bites_on_single_bit_flip_in_vec() {
    let h = load_base();
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
fn comparator_rejects_wrong_vector_lengths() {
    // Length is part of the contract: empty, truncated (missing the
    // heuristic_score slot), and padded vectors are ALL red — a zip
    // would have silently accepted the first two.
    let h = load_base();
    let case = &h.cases[0];

    let empty: Vec<f64> = vec![];
    let mut r = ParityReport::new(1);
    r.record_pair(0, &empty, &case.vec, case.score, case.score);
    assert_eq!(r.n_vec_mismatch, 1, "empty vector must be red");
    assert!(r.first_vec_mismatch.is_some());

    let truncated = case.vec[..VEC_LEN - 1].to_vec();
    let mut r = ParityReport::new(1);
    r.record_pair(0, &truncated, &case.vec, case.score, case.score);
    assert_eq!(r.n_vec_mismatch, 1, "27-entry vector must be red");

    let mut padded = case.vec.clone();
    padded.push(f64::NAN); // 29 entries — the extra slot must bite
    let mut r = ParityReport::new(1);
    r.record_pair(0, &padded, &case.vec, case.score, case.score);
    assert_eq!(r.n_vec_mismatch, 1, "29-entry vector must be red");

    // and the same through the full harness path: a backend returning
    // empty vectors with CORRECT scores must not look green anywhere
    struct EmptyVecBackend;
    impl ScoringBackend for EmptyVecBackend {
        fn entity_profile(
            &self,
            _e: &cine_rec_scoring::Entity,
            _d: Option<&cine_rec_scoring::Entity>,
            _g: Option<&[String]>,
        ) -> cine_rec_scoring::Profile {
            cine_rec_scoring::Profile::default()
        }
        fn feature_vector(
            &self,
            _c: &cine_rec_scoring::Entity,
            _cd: &cine_rec_scoring::Entity,
            _cg: &[String],
            _s: &cine_rec_scoring::Entity,
            _o: f64,
        ) -> Vec<f64> {
            vec![] // length violation
        }
        fn score_pair_fast(
            &self,
            _sp: &cine_rec_scoring::Profile,
            _cp: &cine_rec_scoring::Profile,
            _c: &cine_rec_scoring::Entity,
            _cd: &cine_rec_scoring::Entity,
            _g: Option<&HashSet<String>>,
            _o: f64,
            _w: &[f64],
        ) -> f64 {
            0.0
        }
    }
    let report = run_parity(&EmptyVecBackend, &h);
    assert_eq!(report.n_vec_mismatch, h.cases.len());
    assert!(
        !report.green(),
        "empty-vector backend must be red:\n{report}"
    );
}

#[test]
fn comparator_tolerates_exactly_one_ulp_in_score_around_nonzero() {
    let h = load_base();
    // the first non-zero golden score — the typical regime (case 1 on
    // the base set)
    let case = h
        .cases
        .iter()
        .find(|c| c.score != 0.0)
        .expect("a non-zero golden score to test around");
    assert_eq!(case.index, 1);

    for bump in [
        f64::from_bits(case.score.to_bits() + 1), // exactly 1 ULP up
        f64::from_bits(case.score.to_bits() - 1), // exactly 1 ULP down
    ] {
        assert_eq!(cine_rec_scoring::ulp_distance(bump, case.score), 1);
        let mut report = ParityReport::new(1);
        report.record_pair(case.index, &case.vec, &case.vec, bump, case.score);
        assert!(
            report.green(),
            "1 ULP (up or down) around a non-zero golden is inside the contract:\n{report}"
        );
    }
}

#[test]
fn comparator_rejects_two_ulp_in_score() {
    let h = load_base();
    let case = &h.cases[0];
    let bumped = f64::from_bits(case.score.to_bits() + 2); // 2 ULP up
    assert_eq!(cine_rec_scoring::ulp_distance(bumped, case.score), 2);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, bumped, case.score);
    assert_eq!(report.n_score_mismatch, 1, "2 ULP is outside the contract");
}

#[test]
fn comparator_zero_ulp_boundaries_are_the_documented_decision() {
    // Documented in parity.rs: ±0.0 compare equal for scores; the
    // tolerance is asymmetric ACROSS zero (stricter on the negative
    // side) — +smallest_subnormal is 1 ULP from +0.0 (passes),
    // -smallest is 2 (fails). Vec comparison stays bit-exact, so
    // -0.0 vs 0.0 inside a vector IS a mismatch.
    let h = load_base();
    let case = &h.cases[0];
    assert_eq!(case.score, 0.0); // gated pair — the zero baseline

    let plus = f64::from_bits(1); // smallest positive subnormal
    assert_eq!(cine_rec_scoring::ulp_distance(plus, 0.0), 1);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, plus, case.score);
    assert!(report.green(), "+1 ULP from zero passes:\n{report}");

    let minus = -plus;
    assert_eq!(cine_rec_scoring::ulp_distance(minus, 0.0), 2);
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, minus, case.score);
    assert_eq!(
        report.n_score_mismatch, 1,
        "-1 ULP below zero counts as 2 — documented asymmetry"
    );

    // -0.0 score is equal (0 ULP); -0.0 vec entry is a mismatch
    let mut report = ParityReport::new(1);
    report.record_pair(0, &case.vec, &case.vec, -0.0, case.score);
    assert!(report.green(), "-0.0 score equals 0.0");
    let mut neg_zero_vec = case.vec.clone();
    neg_zero_vec[0] = -0.0;
    let mut report = ParityReport::new(1);
    report.record_pair(0, &neg_zero_vec, &case.vec, case.score, case.score);
    assert_eq!(report.n_vec_mismatch, 1, "-0.0 vec entry is a bit mismatch");
}

#[test]
fn nan_output_is_always_a_mismatch() {
    let h = load_base();
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
    let h = load_base();
    let report = run_parity(&ZeroBackend, &h);
    assert!(
        report.green(),
        "this line fails on purpose — harness red demo:\n{report}"
    );
}
