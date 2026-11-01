//! The result-list cache seam. Python resolves the Redis client
//! lazily and degrades to "no cache" on any failure — the DEFAULT
//! here is behaviorally identical to a Redis-less deployment, which
//! is exactly what the parity harness runs against.

use async_trait::async_trait;

/// Cache key + payload, shaped like the Python cache strings (the
/// `v=3` format is a deployment-compatibility contract — mixed
/// Python/Rust fleets must hit the same entries).
#[async_trait]
pub trait ResultCache: Send + Sync {
    /// `None` = miss (or cache unavailable — degrade, never fail).
    async fn get(&self, key: &str) -> Option<String>;
    /// A failed store is silently dropped (Python parity).
    async fn set(&self, key: &str, payload: &str, ttl_secs: u64);
}

/// The default: no cache at all. Identical to the Python engine with
/// Redis absent — every call is a compute path.
pub struct NoopCache;

#[async_trait]
impl ResultCache for NoopCache {
    async fn get(&self, _key: &str) -> Option<String> {
        None
    }
    async fn set(&self, _key: &str, _payload: &str, _ttl_secs: u64) {}
}
