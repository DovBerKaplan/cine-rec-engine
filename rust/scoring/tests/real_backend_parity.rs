//! Phase-2 gate: the REAL port against the phase-1 harness.
//!
//! [`RealBackend`] is the translated scoring core; the same frozen
//! goldens that pinned the contract now prove the port:
//! * `feature_vector` — bit-exact, all 28 entries, both sets
//!   (BASE 480 pairs + ENRICHED 64 pairs).
//! * `_score_pair_fast` — ≤ 1 ULP everywhere.
//!
//! The backend's residual weights are `w_list[..27]` — exactly the
//! Python relationship `HEURISTIC_WEIGHTS_LIST = list(
//! HEURISTIC_WEIGHTS.values())[:27]` for the frozen harness weights.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cine_rec_scoring::{run_parity, Harness, RealBackend, ScoringBackend};

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn load_base() -> Harness {
    let root = manifest_dir()
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .to_path_buf();
    Harness::load(
        &root.join("tests").join("golden_feature_vectors.json"),
        &manifest_dir()
            .join("tests")
            .join("fixtures")
            .join("entities_120_4.json"),
    )
    .expect("base harness")
}

fn load_enriched() -> Harness {
    let dir = manifest_dir().join("tests").join("fixtures");
    Harness::load(
        &dir.join("goldens_enriched.json"),
        &dir.join("entities_enriched.json"),
    )
    .expect("enriched harness")
}

#[test]
fn real_backend_is_bit_exact_on_base_and_enriched() {
    for (label, h) in [("base", load_base()), ("enriched", load_enriched())] {
        // HEURISTIC_WEIGHTS_LIST == w_list[:27] on the frozen harness
        let backend = RealBackend::new(h.w_list[..27].to_vec());
        let report = run_parity(&backend, &h);
        assert!(
            report.green(),
            "{label}: REAL port must pass parity:\n{report}"
        );
        assert_eq!(report.n_pairs, h.cases.len());
    }
}

#[test]
fn real_backend_scores_are_exactly_reproducible() {
    // Determinism smoke: the same backend instance scored twice gives
    // bit-identical results (no hidden state, no ordering luck).
    let h = load_enriched();
    let backend = RealBackend::new(h.w_list[..27].to_vec());
    let r1 = run_parity(&backend, &h);
    let r2 = run_parity(&backend, &h);
    assert_eq!(
        (r1.n_vec_mismatch, r1.n_score_mismatch),
        (r2.n_vec_mismatch, r2.n_score_mismatch)
    );
    assert!(r1.green());
}

#[test]
fn real_backend_gate_matches_golden_gating() {
    // The genre gate + via exemption through the REAL port must match
    // what the enriched goldens captured: clash WITH via scored,
    // clash WITHOUT via gated to 0.0.
    let h = load_enriched();
    let backend = RealBackend::new(h.w_list[..27].to_vec());
    let war = |c: &cine_rec_scoring::GoldenCase| c.cand.genres.as_slice() == ["War"];
    for case in h.cases.iter().filter(|c| war(c)) {
        let sp = backend.entity_profile(&case.seed, None, None);
        let cp = backend.entity_profile(&case.cand, Some(&case.cand), Some(&case.cand.genres));
        let set: HashSet<String> = case.cand.genres.iter().cloned().collect();
        let score =
            backend.score_pair_fast(&sp, &cp, &case.cand, &case.cand, Some(&set), 0.0, &h.w_list);
        // via truthiness mirrors Python: an EMPTY via string is not an
        // exemption (pinned by the k=15 clash candidate)
        let exempt = matches!(case.cand.via.as_deref(), Some(v) if !v.is_empty());
        if exempt {
            assert_ne!(score, 0.0, "case {}: exemption must score", case.index);
            let ulp = cine_rec_scoring::ulp_distance(score, case.score);
            assert!(ulp <= cine_rec_scoring::MAX_SCORE_ULP);
        } else {
            assert_eq!(score, 0.0, "case {}: gate must zero the score", case.index);
        }
    }
}
