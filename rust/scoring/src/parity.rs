//! Parity comparators and the run report.
//!
//! Two rules, no approximations:
//!
//! * `feature_vector` — BIT-EXACT: every one of the 28 entries must
//!   equal the golden entry's bits. The Python computation is fully
//!   deterministic here (every set is only measured with len/bool or
//!   summed in sorted order), so bit equality is attainable and is the
//!   contract. A NaN/Inf actual is a mismatch by construction (goldens
//!   are validated finite at load time).
//! * `_score_pair_fast` — at most [`crate::MAX_SCORE_ULP`] (1) ULP from
//!   the golden score: the golden file was captured from the reference
//!   path, whose accumulation order differs from the hot loop's.

use std::fmt;

/// Monotonic key over the IEEE-754 double bit patterns, so that
/// `|key(a) - key(b)|` is the ULP distance for finite values:
/// negative floats get their bits complemented (descending →
/// ascending), positive floats get the sign bit forced on.
fn f64_key(x: f64) -> u64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    let b = x.to_bits();
    if b & SIGN != 0 {
        !b
    } else {
        b | SIGN
    }
}

/// ULP distance between two doubles.
///
/// * equal values (including `-0.0` vs `0.0`, which compare equal
///   numerically) → 0
/// * either value NaN → `u64::MAX` (the caller reports a mismatch — a
///   NaN where a finite golden sits is always wrong)
pub fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == b {
        // covers ±0.0 and any exactly-equal pair, NaN==NaN excluded by
        // IEEE semantics
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let (ka, kb) = (f64_key(a), f64_key(b));
    ka.abs_diff(kb)
}

/// First mismatch of each kind, with enough context to debug from the
/// test output alone.
#[derive(Debug, Clone)]
pub struct VecMismatch {
    pub index: usize,
    pub slot: usize,
    pub expected: f64,
    pub actual: f64,
    pub expected_bits: u64,
    pub actual_bits: u64,
}

#[derive(Debug, Clone)]
pub struct ScoreMismatch {
    pub index: usize,
    pub expected: f64,
    pub actual: f64,
    pub ulp: u64,
}

impl fmt::Display for VecMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case {} slot {}: expected {} ({:#018x}) got {} ({:#018x})",
            self.index, self.slot, self.expected, self.expected_bits, self.actual, self.actual_bits
        )
    }
}

impl fmt::Display for ScoreMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case {}: expected {} got {} ({} ULP apart, max {})",
            self.index,
            self.expected,
            self.actual,
            self.ulp,
            crate::MAX_SCORE_ULP
        )
    }
}

/// Result of driving a backend over every golden case.
#[derive(Debug, Clone, Default)]
pub struct ParityReport {
    pub n_pairs: usize,
    pub n_vec_mismatch: usize,
    pub n_score_mismatch: usize,
    pub first_vec_mismatch: Option<VecMismatch>,
    pub first_score_mismatch: Option<ScoreMismatch>,
}

impl ParityReport {
    /// Empty report over `n_pairs` cases. Exposed so external tests
    /// can exercise the comparators directly (perturbation checks).
    pub fn new(n_pairs: usize) -> Self {
        Self {
            n_pairs,
            ..Self::default()
        }
    }

    /// Compare one (actual, expected) pair and record mismatches.
    /// Exposed for the same reason as [`ParityReport::new`].
    pub fn record_pair(
        &mut self,
        index: usize,
        actual_vec: &[f64],
        expected_vec: &[f64],
        actual_score: f64,
        expected_score: f64,
    ) {
        for (slot, (actual, expected)) in actual_vec.iter().zip(expected_vec).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                self.n_vec_mismatch += 1;
                if self.first_vec_mismatch.is_none() {
                    self.first_vec_mismatch = Some(VecMismatch {
                        index,
                        slot,
                        expected: *expected,
                        actual: *actual,
                        expected_bits: expected.to_bits(),
                        actual_bits: actual.to_bits(),
                    });
                }
                break; // one report per case is enough
            }
        }
        let ulp = ulp_distance(actual_score, expected_score);
        if ulp > crate::MAX_SCORE_ULP {
            self.n_score_mismatch += 1;
            if self.first_score_mismatch.is_none() {
                self.first_score_mismatch = Some(ScoreMismatch {
                    index,
                    expected: expected_score,
                    actual: actual_score,
                    ulp,
                });
            }
        }
    }

    /// Green means: pairs were actually checked (n > 0 — a harness
    /// that silently ran zero cases must never look green) AND nothing
    /// mismatched.
    pub fn green(&self) -> bool {
        self.n_pairs > 0 && self.n_vec_mismatch == 0 && self.n_score_mismatch == 0
    }
}

impl fmt::Display for ParityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} pairs | vec mismatches: {} | score mismatches: {}",
            self.n_pairs, self.n_vec_mismatch, self.n_score_mismatch
        )?;
        if let Some(m) = &self.first_vec_mismatch {
            write!(f, "\n  first vec: {m}")?;
        }
        if let Some(m) = &self.first_score_mismatch {
            write!(f, "\n  first score: {m}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_up(x: f64) -> f64 {
        f64::from_bits(x.to_bits() + 1) // positive, non-max inputs in these tests
    }

    #[test]
    fn ulp_distance_boundaries() {
        let x = 11.888_888_888_888_89;
        assert_eq!(ulp_distance(x, x), 0);
        assert_eq!(ulp_distance(x, next_up(x)), 1);
        assert_eq!(ulp_distance(next_up(x), x), 1);
        assert_eq!(ulp_distance(x, next_up(next_up(x))), 2);
        assert_eq!(ulp_distance(0.0, -0.0), 0);
        assert_eq!(ulp_distance(x, f64::NAN), u64::MAX);
        assert_eq!(ulp_distance(f64::NAN, x), u64::MAX);
    }

    #[test]
    fn ulp_distance_negatives() {
        // negatives live on the mirrored key branch: distance is
        // symmetric around zero (d(-1,-2) == d(1,2) == 2^52)
        assert_eq!(ulp_distance(-1.0, -2.0), ulp_distance(1.0, 2.0));
        assert_eq!(ulp_distance(-1.0, -1.0 - 2.0 * f64::EPSILON), 2);
        assert_eq!(ulp_distance(-1.0, -1.0), 0);
    }
}
