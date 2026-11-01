//! find_similar's compute path (phase 5b): seed normalization, the
//! recall-channel fan-out over `cine-rec-db`, candidate-pool assembly,
//! the scoring loop over `cine-rec-scoring`, MMR and media balancing.
//!
//! Built in stages; parity target is Python's find_similar.py compute
//! path (cache bypassed) on frozen seeds.
