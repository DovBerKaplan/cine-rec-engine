//! Loading the two sides of the parity contract and joining them into
//! per-pair golden cases.
//!
//! Two sets are loaded:
//!
//! * BASE — outputs: `<repo>/tests/golden_feature_vectors.json` (480
//!   entries, cands-major × seeds-minor); inputs:
//!   `tests/fixtures/entities_120_4.json` from
//!   `benchmarks/bench_scoring.py::build(120, 4)`.
//! * ENRICHED — both sides live in `tests/fixtures/`: the base rows
//!   carry no `via`/`tmdb_rank`/`narrative`/`pacing`/`emotional_arc`/
//!   `director`/style keywords, so six features sit constant at 0.0
//!   across the whole base goldens and the via gate-exemption is
//!   never exercised. `entities_enriched.json` adds those fields
//!   deterministically; `goldens_enriched.json` captures the reference
//!   path (`feature_vector` + `_score_candidate_vs_seed`) on them.
//!
//! All shape expectations fail LOUDLY at load time (an empty or
//! mis-shaped harness must never look green) and the pairing
//! convention is anchored by probes whose values are COMPUTED by
//! running the Python scoring functions and whose golden index is
//! FOUND by unique search over the golden file (see [`Probe`]).

use std::collections::HashMap;
use std::path::Path;

use crate::types::Entity;

/// One entry of a golden file: the captured `feature_vector` and the
/// captured reference score for one (candidate, seed) pair.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenEntry {
    pub vec: Vec<f64>,
    pub score: f64,
}

/// Pairing-convention anchor. The generator RUNS the Python scoring
/// functions on the (cand_id, seed_id) pair, records the computed
/// vector slots and fast-path score, and LOCATES `golden_index` by a
/// unique full-vector search over the golden file — no pairing formula
/// is involved on the generator side. If the goldens had been captured
/// under a different convention, the found index would disagree with
/// the harness's index for the same entity pair.
///
/// Probe pairs are chosen with non-zero scores so they discriminate
/// between cases. `score` is the COMPUTED fast-path value: the anchor
/// contract for it is ≤ [`crate::MAX_SCORE_ULP`] against the golden
/// (same as the harness rule); `vec_slots` are bit-exact.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    pub cand_id: i64,
    pub seed_id: i64,
    pub golden_index: usize,
    pub score: f64,
    pub vec_slots: HashMap<String, f64>,
}

/// The frozen input fixture (see module docs). `deny_unknown_fields`
/// pins the fixture shape — a renamed key must fail loudly here, not
/// silently deserialize as an empty list (the runtime
/// degrade-never-fail rule concerns optional DB channels, not this
/// frozen contract file). `Entity` itself stays lenient
/// (`#[serde(default)]`) to mirror Python `dict.get` semantics.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub generator: String,
    pub n_cands: usize,
    pub n_seeds: usize,
    pub w_list: Vec<f64>,
    pub feature_names: Vec<String>,
    pub pairing: String,
    #[serde(default)]
    pub probes: Vec<Probe>,
    pub cands: Vec<Entity>,
    pub seeds: Vec<Entity>,
}

/// One parity case: pair `index` of the canonical order, the two input
/// entities, and the expected outputs.
#[derive(Debug, Clone)]
pub struct GoldenCase {
    pub index: usize,
    pub cand: Entity,
    pub seed: Entity,
    pub vec: Vec<f64>,
    pub score: f64,
}

/// Everything the harness needs: inputs, weights, expected outputs,
/// joined per pair.
#[derive(Debug, Clone)]
pub struct Harness {
    pub generator: String,
    pub n_cands: usize,
    pub n_seeds: usize,
    pub w_list: Vec<f64>,
    pub feature_names: Vec<String>,
    pub cases: Vec<GoldenCase>,
    pub probes: Vec<Probe>,
}

impl Harness {
    /// Find a golden case by entity ids (used by the probe test).
    pub fn case_for(&self, cand_id: i64, seed_id: i64) -> Option<&GoldenCase> {
        self.cases
            .iter()
            .find(|c| c.cand.id == cand_id && c.seed.id == seed_id)
    }

    /// Load a golden file + fixture and validate the shape contract.
    ///
    /// `golden_path` is the OUTPUT side (the repo's
    /// `tests/golden_feature_vectors.json` for the base set, or the
    /// captured `goldens_enriched.json` for the supplementary set);
    /// `fixture_path` points at the generated entities file.
    pub fn load(golden_path: &Path, fixture_path: &Path) -> Result<Self, String> {
        let goldens: Vec<GoldenEntry> = serde_json::from_str(
            &std::fs::read_to_string(golden_path)
                .map_err(|e| format!("read {}: {e}", golden_path.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", golden_path.display()))?;
        let fixture: Fixture = serde_json::from_str(
            &std::fs::read_to_string(fixture_path)
                .map_err(|e| format!("read {}: {e}", fixture_path.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", fixture_path.display()))?;

        // ── Shape contract: fail loud, never empty-green ──
        if fixture.cands.is_empty() || fixture.seeds.is_empty() {
            return Err("fixture has no candidates or no seeds".into());
        }
        if fixture.n_cands != fixture.cands.len() {
            return Err(format!(
                "fixture declares n_cands={} but carries {} candidates",
                fixture.n_cands,
                fixture.cands.len()
            ));
        }
        if fixture.n_seeds != fixture.seeds.len() {
            return Err(format!(
                "fixture declares n_seeds={} but carries {} seeds",
                fixture.n_seeds,
                fixture.seeds.len()
            ));
        }
        if fixture.pairing.trim().is_empty() {
            return Err("fixture carries no pairing description".into());
        }
        if fixture.w_list.len() != crate::VEC_LEN {
            return Err(format!(
                "w_list has {} entries, expected {}",
                fixture.w_list.len(),
                crate::VEC_LEN
            ));
        }
        let expected_pairs = fixture.cands.len() * fixture.seeds.len();
        if goldens.len() != expected_pairs {
            return Err(format!(
                "goldens have {} entries but {} cands × {} seeds = {} pairs",
                goldens.len(),
                fixture.cands.len(),
                fixture.seeds.len(),
                expected_pairs
            ));
        }
        for (i, g) in goldens.iter().enumerate() {
            if g.vec.len() != crate::VEC_LEN {
                return Err(format!("golden {i}: vec has {} entries", g.vec.len()));
            }
            if !g.vec.iter().all(|v| v.is_finite()) || !g.score.is_finite() {
                return Err(format!("golden {i}: non-finite value (NaN/Inf)"));
            }
        }
        let all_distinct = |es: &[Entity]| {
            let set: std::collections::HashSet<_> = es.iter().map(|e| e.id).collect();
            set.len() == es.len()
        };
        if !all_distinct(&fixture.cands) || !all_distinct(&fixture.seeds) {
            return Err("duplicate entity ids in fixture (probes key on ids)".into());
        }

        // ── Pairing: golden case i == (cands[i / n_seeds], seeds[i % n_seeds]) ──
        let n_seeds = fixture.seeds.len();
        let mut cases = Vec::with_capacity(expected_pairs);
        for (i, golden) in goldens.iter().enumerate() {
            let cand = &fixture.cands[i / n_seeds];
            let seed = &fixture.seeds[i % n_seeds];
            cases.push(GoldenCase {
                index: i,
                cand: cand.clone(),
                seed: seed.clone(),
                vec: golden.vec.clone(),
                score: golden.score,
            });
        }

        Ok(Self {
            generator: fixture.generator,
            n_cands: fixture.cands.len(),
            n_seeds,
            w_list: fixture.w_list,
            feature_names: fixture.feature_names,
            cases,
            probes: fixture.probes,
        })
    }
}
