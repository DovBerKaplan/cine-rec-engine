//! Service orchestration — the Rust side of `find_similar.py`'s
//! compute path (phase 5b): recall-channel fan-out, candidate-pool
//! assembly, the scoring loop (over the parity-verified scoring
//! crate), MMR and media balancing.
//!
//! The Python side stays the source of truth until gate 5; this crate
//! is wired in through a PyO3 bridge only after live parity holds.

pub mod cache;
#[cfg(feature = "pyo3")]
pub mod pybridge;
pub mod similar;

pub use cache::{NoopCache, ResultCache};
