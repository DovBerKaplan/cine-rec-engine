//! Behavior tests for the async [`RateLimiter`] under tokio PAUSED
//! time — deterministic virtual clocks, never a real sleep.
//!
//! These pin the WRAPPER's semantics (lock held across the sleep →
//! concurrent acquirers serialize; pacing; burst drain), while
//! tests/rate_parity.rs pins the bucket math bit-exactly.

use cine_rec_ingest::rate::RateLimiter;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[tokio::test(start_paused = true)]
async fn paces_at_configured_rate() {
    let lim = RateLimiter::new(10.0, 1).unwrap();
    let start = tokio::time::Instant::now();
    let mut times = Vec::new();
    for _ in 0..5 {
        lim.acquire().await;
        times.push((tokio::time::Instant::now() - start).as_secs_f64());
    }
    assert_eq!(times.len(), 5); // never measure nothing
    assert!(times[0] < 1e-6, "first token must be immediate: {times:?}");
    // Tolerance 5 ms: tokio timers tick at 1 ms granularity, and when
    // the sub-nanosecond float-black-hole wait hits (see src/rate.rs)
    // the 1 ns recovery floor rounds up to the next tick. The exact
    // pacing MATH is pinned bit-exactly by tests/rate_parity.rs.
    for (i, t) in times.iter().enumerate().skip(1) {
        assert!(
            approx(*t, 0.1 * i as f64, 0.005),
            "acquire {i} off schedule: {times:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn burst_caps_the_initial_burst() {
    // Capacity 3: three immediate acquires, then 1 s per token.
    let lim = RateLimiter::new(1.0, 3).unwrap();
    let start = tokio::time::Instant::now();
    let mut times = Vec::new();
    for _ in 0..5 {
        lim.acquire().await;
        times.push((tokio::time::Instant::now() - start).as_secs_f64());
    }
    assert_eq!(times.len(), 5);
    for t in &times[..3] {
        assert!(*t < 1e-6, "burst tokens must be immediate: {times:?}");
    }
    assert!(approx(times[3], 1.0, 1e-6), "{times:?}");
    assert!(approx(times[4], 2.0, 1e-6), "{times:?}");
}

#[tokio::test(start_paused = true)]
async fn concurrent_acquirers_serialize() {
    // The lock is held across the sleep (faithful to ingest/rate.py),
    // so 8 concurrent acquirers finish ~one spacing apart, not in a
    // herd. Under paused time this is exact.
    const N: usize = 8;
    let lim = std::sync::Arc::new(RateLimiter::new(10.0, 1).unwrap());
    let start = tokio::time::Instant::now();
    let mut handles = Vec::new();
    for _ in 0..N {
        let lim = std::sync::Arc::clone(&lim);
        handles.push(tokio::spawn(async move {
            lim.acquire().await;
            (tokio::time::Instant::now() - start).as_secs_f64()
        }));
    }
    let mut times = Vec::new();
    for h in handles {
        times.push(h.await.unwrap());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(times.len(), N); // never measure nothing
    assert!(times[0] < 1e-6, "first token immediate: {times:?}");
    // Same 5 ms tolerance as paces_at_configured_rate (tokio's 1 ms
    // timer ticks + the 1 ns recovery floor).
    for i in 1..N {
        assert!(
            approx(times[i], 0.1 * i as f64, 0.005),
            "acquirer {i} broke serialization: {times:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn nan_rate_degenerates_to_unlimited_without_sleeping() {
    // Parity quirk: min(capacity, tokens + dt*NaN) returns capacity,
    // so the bucket never runs dry and never sleeps (see src/rate.rs).
    let lim = RateLimiter::new(f64::NAN, 1).unwrap();
    let start = tokio::time::Instant::now();
    for _ in 0..100 {
        lim.acquire().await;
    }
    let elapsed = (tokio::time::Instant::now() - start).as_secs_f64();
    assert!(elapsed < 1e-9, "NaN rate must never sleep: {elapsed}s");
}

#[test]
fn constructor_rejects_nonpositive_rates() {
    for bad in [0.0, -1.0, -0.001] {
        assert_eq!(
            RateLimiter::new(bad, 1).unwrap_err().to_string(),
            "rate_per_second must be > 0",
            "rate={bad}"
        );
    }
}
