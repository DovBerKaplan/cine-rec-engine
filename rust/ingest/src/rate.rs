//! Token-bucket rate limiting — a port of `ingest/rate.py`.
//!
//! A semaphore caps CONCURRENCY, not RATE: 40 concurrent in-flight
//! requests against a fast API is a burst of hundreds per second. The
//! bucket paces the caller to at most `rate` acquisitions per second,
//! smoothly (credit refills continuously, not in whole-second batches).
//!
//! Two layers, so the pacing policy is testable without any clock:
//!
//! * [`TokenBucket`] — the pure policy, every method taking `now` as a
//!   monotonic-seconds float. All the refill/wait math lives here and
//!   is pinned bit-exact by a golden trace generated from the REAL
//!   Python `RateLimiter` (clock and sleep monkeypatched —
//!   `tests/fixtures/gen_rate_trace.py`).
//! * [`RateLimiter`] — the async wrapper, faithful to the Python
//!   original down to its locking shape: the lock is held ACROSS the
//!   sleep (`async with self._lock` spans the sleep in
//!   ingest/rate.py:34-41), so concurrent acquirers queue behind the
//!   sleeping one instead of waking together. Its tests run under
//!   tokio paused time — deterministic, never a real sleep.
//!
//! Faithfulness notes:
//! * The refill is `min(capacity, tokens + (now - updated) * rate)` in
//!   exactly that operation order — a clock that goes backwards makes
//!   `tokens` shrink, exactly as in Python (no clamping at zero).
//! * `f64::min` matches CPython's `min()` for NaN the same way: if
//!   either argument is NaN the other is returned. Combined with the
//!   constructor accepting NaN rates (Python's `rate <= 0` check is
//!   false for NaN), a NaN-rate bucket degenerates to "always full" —
//!   unlimited, never sleeping — IDENTICALLY on both sides. Pinned by
//!   a fixture scenario.
//! * The constructor rejects `rate <= 0` with Python's message.
//!   `burst` is `max(1, int(burst))` — burst 0 or negative means a
//!   capacity of 1, like Python.

use std::time::Duration;

use tokio::sync::Mutex;

/// One iteration of the Python acquire loop (ingest/rate.py:36-41).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquireStep {
    /// A token was taken — the caller may proceed.
    Acquired,
    /// No token yet; sleep this many seconds and loop. The bucket has
    /// already been refilled to `now`, mirroring the Python loop order
    /// (`_refill()` runs before the `sleep((1.0 - tokens) / rate)`).
    Wait(f64),
}

/// `ValueError` parity for bad construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateError(pub String);

impl std::fmt::Display for RateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RateError {}

/// The pure token-bucket policy (see the module docs). All methods
/// take `now` in monotonic seconds — the caller owns the clock.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    rate: f64,
    capacity: f64,
    tokens: f64,
    updated: f64,
}

impl TokenBucket {
    /// Python `RateLimiter.__init__`: `rate_per_second <= 0` raises
    /// `ValueError`; capacity is `max(1, int(burst))` and starts full.
    pub fn new(rate_per_second: f64, burst: i64) -> Result<Self, RateError> {
        // NaN <= 0.0 is false on both sides, so a NaN rate is accepted
        // here exactly like Python — see the module docs for the
        // (identical) degenerate behavior that follows.
        let positive = matches!(
            rate_per_second.partial_cmp(&0.0),
            Some(std::cmp::Ordering::Greater)
        );
        if !positive && !rate_per_second.is_nan() {
            return Err(RateError("rate_per_second must be > 0".into()));
        }
        let capacity = 1.0_f64.max(burst as f64);
        Ok(Self {
            rate: rate_per_second,
            capacity,
            tokens: capacity,
            updated: 0.0,
        })
    }

    /// `RateLimiter._refill` — continuous refill capped at capacity,
    /// in the exact Python operation order.
    pub fn refill(&mut self, now: f64) {
        // min(capacity, tokens + (now - updated) * rate) — f64::min
        // returns the non-NaN argument, same as CPython min() here.
        self.tokens = self
            .capacity
            .min(self.tokens + (now - self.updated) * self.rate);
        self.updated = now;
    }

    /// One loop iteration of `RateLimiter.acquire` at time `now`:
    /// refill, then take a token if one is available, otherwise report
    /// the sleep the Python code would do.
    pub fn acquire_step(&mut self, now: f64) -> AcquireStep {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            AcquireStep::Acquired
        } else {
            AcquireStep::Wait((1.0 - self.tokens) / self.rate)
        }
    }

    /// Current credit (post-refill only if the caller refilled).
    pub fn tokens(&self) -> f64 {
        self.tokens
    }

    /// Configured rate (per second).
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Configured capacity (`max(1, int(burst))`).
    pub fn capacity(&self) -> f64 {
        self.capacity
    }
}

/// The async wrapper — `ingest/rate.py::RateLimiter` for tokio.
#[derive(Debug)]
///
/// `acquire().await` blocks until a token is available. The mutex is
/// deliberately held across the sleep (Python holds `asyncio.Lock`
/// across `asyncio.sleep` there), which serializes concurrent
/// acquirers; see the module docs.
pub struct RateLimiter {
    bucket: Mutex<TokenBucket>,
    origin: tokio::time::Instant,
}

impl RateLimiter {
    /// Same validation as [`TokenBucket::new`]; the clock origin is
    /// captured at construction (Python stores `time.monotonic()`).
    pub fn new(rate_per_second: f64, burst: i64) -> Result<Self, RateError> {
        Ok(Self {
            bucket: Mutex::new(TokenBucket::new(rate_per_second, burst)?),
            origin: tokio::time::Instant::now(),
        })
    }

    /// Monotonic seconds since construction, the Rust analog of
    /// `time.monotonic()` deltas.
    fn now_secs(&self) -> f64 {
        (tokio::time::Instant::now() - self.origin).as_secs_f64()
    }

    /// Block until a token is available (`RateLimiter.acquire`).
    pub async fn acquire(&self) {
        let mut bucket = self.bucket.lock().await;
        loop {
            match bucket.acquire_step(self.now_secs()) {
                AcquireStep::Acquired => return,
                AcquireStep::Wait(secs) => {
                    // Branch invariant: positive and finite (tokens < 1
                    // strictly, rate > 0; NaN rates never wait).
                    debug_assert!(secs > 0.0 && secs.is_finite());
                    // Quantize to the timer resolution: a positive wait
                    // below 1 ns becomes Duration::ZERO, which under a
                    // paused/virtual clock never advances time and the
                    // loop would spin forever (the float black hole —
                    // see tests/fixtures/gen_rate_trace.py). On a real
                    // clock Python self-heals because wall time passes
                    // during the yield; a 1 ns floor reproduces that
                    // dynamic on every clock, virtual included.
                    let sleep_for = Duration::from_secs_f64(secs).max(Duration::from_nanos(1));
                    // The lock is held across this sleep on purpose —
                    // faithful to ingest/rate.py:34-41.
                    tokio::time::sleep(sleep_for).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_validation_and_capacity() {
        assert_eq!(
            TokenBucket::new(0.0, 1).unwrap_err(),
            RateError("rate_per_second must be > 0".into())
        );
        assert_eq!(
            TokenBucket::new(-1.0, 1).unwrap_err(),
            RateError("rate_per_second must be > 0".into())
        );
        // NaN is accepted exactly like Python (see module docs).
        assert!(TokenBucket::new(f64::NAN, 1).is_ok());
        // capacity = max(1, int(burst)): 0 and negatives mean 1.
        assert_eq!(TokenBucket::new(1.0, 0).unwrap().capacity(), 1.0);
        assert_eq!(TokenBucket::new(1.0, -5).unwrap().capacity(), 1.0);
        assert_eq!(TokenBucket::new(1.0, 3).unwrap().capacity(), 3.0);
    }

    #[test]
    fn burst_drains_then_paces() {
        let mut b = TokenBucket::new(10.0, 2).unwrap();
        // Capacity 2: two immediate acquires, then waits of 0.1 each.
        assert_eq!(b.acquire_step(0.0), AcquireStep::Acquired);
        assert_eq!(b.acquire_step(0.0), AcquireStep::Acquired);
        let w = match b.acquire_step(0.0) {
            AcquireStep::Wait(w) => w,
            other => panic!("expected wait, got {other:?}"),
        };
        assert!((w - 0.1).abs() < 1e-12, "{w}");
        // Sleeping exactly the wait lands the bucket at one full token.
        assert_eq!(b.acquire_step(0.1), AcquireStep::Acquired);
    }

    #[test]
    fn refill_caps_at_capacity() {
        let mut b = TokenBucket::new(1000.0, 2).unwrap();
        assert_eq!(b.acquire_step(0.0), AcquireStep::Acquired);
        assert_eq!(b.acquire_step(0.0), AcquireStep::Acquired);
        // A huge gap must not credit more than capacity tokens.
        assert_eq!(b.acquire_step(60.0), AcquireStep::Acquired);
        assert_eq!(b.acquire_step(60.0), AcquireStep::Acquired);
        assert!(matches!(b.acquire_step(60.0), AcquireStep::Wait(_)));
    }

    #[test]
    fn backwards_clock_shrinks_credit_like_python() {
        let mut b = TokenBucket::new(10.0, 2).unwrap();
        assert_eq!(b.acquire_step(100.0), AcquireStep::Acquired);
        assert_eq!(b.acquire_step(100.0), AcquireStep::Acquired);
        // now < updated: the Python refill subtracts credit; it never
        // clamps at zero, so the wait grows past 1/rate.
        let w = match b.acquire_step(99.0) {
            AcquireStep::Wait(w) => w,
            other => panic!("expected wait, got {other:?}"),
        };
        assert!(w > 0.1, "{w}");
    }

    #[test]
    fn nan_rate_is_always_full() {
        // Degenerate-but-faithful: tokens + dt*NaN is NaN, min() then
        // returns capacity — the bucket never runs dry, never sleeps.
        let mut b = TokenBucket::new(f64::NAN, 2).unwrap();
        for _ in 0..100 {
            assert_eq!(b.acquire_step(123.456), AcquireStep::Acquired);
        }
    }
}
