//! Service orchestration — the Rust side of `find_similar.py`'s
//! compute path: recall-channel fan-out, candidate-pool assembly, the
//! scoring loop (over the parity-verified scoring crate), MMR and
//! media balancing.
//!
//! The Python side remains the reference implementation; production
//! reaches this crate only through the PyO3 bridge.

pub mod cache;
pub mod explain;
pub mod recommend;
#[cfg(feature = "pyo3")]
pub mod pybridge;
pub mod similar;
pub mod user_context;

pub use cache::{NoopCache, ResultCache};
