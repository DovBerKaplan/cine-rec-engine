//! Per-item weights w_i (spec v0.2 §D) — pure, synchronous port of
//! `user_weights.py`. The SQL layer (personalization.rs) feeds it
//! aggregated rows; every branch is pinned by unit tests here plus the
//! live Python-parity fixture.

/// §D.1 drop thresholds
pub const MOVIE_DROP_RATIO: f64 = 0.15;
pub const SERIES_FIRST_EP_DROP: f64 = 0.50;

/// §D.2 movie
pub const MOVIE_COMPLETED_RATIO: f64 = 0.75;
pub const MOVIE_PAUSE_PENALTY: f64 = 0.85;
pub const MOVIE_PAUSE_MIN: i64 = 3;

/// §D.3 series depth ladder
pub const SERIES_EP_FLOOR_0: f64 = 0.30; // eps 0
pub const SERIES_EP_FLOOR_1: f64 = 0.30; // eps 1
pub const SERIES_EP_FLOOR_2: f64 = 0.45; // eps 2
pub const SERIES_HOOKED_FLOOR: f64 = 0.65; // ≥3
pub const SERIES_LONG_WATCH_SEC: i64 = 90 * 60;
pub const SERIES_LAST_EP_EXIT_BONUS: f64 = 0.10;
pub const SERIES_LAST_EP_EXIT_RATIO: f64 = 0.80;
pub const SERIES_LAST_EP_ABANDON_RATIO: f64 = 0.20;
pub const SERIES_LAST_EP_ABANDON_PENALTY: f64 = 0.90;

/// §D.4 recency
pub const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// §D.5 engagement
pub const ENGAGEMENT_REWATCH: f64 = 1.5;
pub const ENGAGEMENT_FAVORITE: f64 = 1.5;
pub const ENGAGEMENT_WATCHLIST: f64 = 1.2;
pub const WATCHLIST_ONLY_S: f64 = 0.7;

pub const W_MAX: f64 = 1.5;

/// §D.2 — completion-ratio score for a movie.
/// r < 0.15 → 0 (dropped) · 0.15 ≤ r < 0.75 → r · r ≥ 0.75 → 1.0.
/// Pause penalty applies only while unfinished.
pub fn s_movie(max_ratio: Option<f64>, pause_count_total: i64) -> f64 {
    let r = max_ratio.unwrap_or(0.0);
    if r < MOVIE_DROP_RATIO {
        return 0.0;
    }
    if r >= MOVIE_COMPLETED_RATIO {
        return 1.0;
    }
    let mut s = r;
    if pause_count_total >= MOVIE_PAUSE_MIN {
        s *= MOVIE_PAUSE_PENALTY;
    }
    s
}

/// Aggregated per-series inputs (user_title_stats columns).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeriesSignals {
    pub episodes_watched: i64,
    pub total_episodes: i64,
    /// last unit watched/duration
    pub last_ep_ratio: Option<f64>,
    /// >0 means an episode was STARTED
    pub last_ep_watched_sec: i64,
    pub total_watched_sec: i64,
}

/// §D.3 — depth ladder + last-episode exits for a series.
/// Depth dominates; ratio = episodes/total only ever lifts, never caps.
pub fn s_series(sig: &SeriesSignals) -> f64 {
    let eps = sig.episodes_watched;
    let ratio = eps as f64 / sig.total_episodes.max(1) as f64;

    let mut s = if eps <= 0 {
        // Only the first episode (partially) seen — §D.1 gate handles < 0.50
        SERIES_EP_FLOOR_0
    } else if eps == 1 {
        SERIES_EP_FLOOR_1.max(ratio)
    } else if eps == 2 {
        SERIES_EP_FLOOR_2.max(ratio)
    } else {
        SERIES_HOOKED_FLOOR.max(ratio)
    };

    // Long watch time floor (optional rule): 90+ minutes → at least 0.65
    if sig.total_watched_sec >= SERIES_LONG_WATCH_SEC {
        s = s.max(SERIES_HOOKED_FLOOR);
    }

    // Last-episode exits — only when an episode was actually started
    if sig.last_ep_watched_sec > 0 {
        if let Some(r) = sig.last_ep_ratio {
            if r >= SERIES_LAST_EP_EXIT_RATIO {
                s = (s + SERIES_LAST_EP_EXIT_BONUS).min(1.0);
            } else if r < SERIES_LAST_EP_ABANDON_RATIO {
                s *= SERIES_LAST_EP_ABANDON_PENALTY;
            }
            // finished an episode and never started the next → natural
            // stop, no penalty — falls through untouched.
        }
    }
    s
}

/// §D.1 — drop filter: weight 0, out of the sum entirely.
pub fn is_dropped(
    media_type: &str,
    max_ratio: Option<f64>,
    episodes_watched: i64,
    last_ep_ratio: Option<f64>,
) -> bool {
    if media_type == "movie" {
        max_ratio.unwrap_or(0.0) < MOVIE_DROP_RATIO
    } else {
        episodes_watched == 0 && last_ep_ratio.unwrap_or(0.0) < SERIES_FIRST_EP_DROP
    }
}

/// §D.4 — S_recency = 2^(−Δdays / T_half). `now_epoch` and
/// `last_epoch` are UTC seconds since the Unix epoch (Python computes
/// with datetime deltas of the same clocks).
pub fn recency_decay(last_epoch: Option<f64>, now_epoch: f64) -> f64 {
    let last = match last_epoch {
        Some(t) => t,
        None => return 0.0,
    };
    let days = (now_epoch - last).max(0.0) / 86400.0;
    2.0f64.powf(-days / RECENCY_HALF_LIFE_DAYS)
}

/// §D end-to-end: the stored w_item for one (user, title).
///
/// * `watchlist_only` — never watched: S = 0.7, engagement 1.2, recency
///   anchored to NOW (a watchlist entry is alive until watched).
/// * `disliked` — 0, always; hard-filtered at retrieval too.
pub fn w_item(
    media_type: &str,
    max_ratio: Option<f64>,
    pause_count_total: i64,
    series: Option<&SeriesSignals>,
    last_epoch: Option<f64>,
    now_epoch: f64,
    rewatch_count: i64,
    favorite: bool,
    watchlist_only: bool,
    disliked: bool,
) -> f64 {
    if disliked {
        return 0.0;
    }

    let (s, engagement, recency) = if watchlist_only {
        (WATCHLIST_ONLY_S, ENGAGEMENT_WATCHLIST, 1.0)
    } else {
        let default_signals = SeriesSignals::default();
        let sig = series.unwrap_or(&default_signals);
        if is_dropped(
            media_type,
            max_ratio,
            sig.episodes_watched,
            sig.last_ep_ratio,
        ) {
            return 0.0;
        }
        let s = if media_type == "movie" {
            s_movie(max_ratio, pause_count_total)
        } else {
            s_series(sig)
        };
        let mut engagement = 1.0;
        if rewatch_count >= 1 || favorite {
            engagement = ENGAGEMENT_REWATCH.max(ENGAGEMENT_FAVORITE);
        }
        let recency = recency_decay(last_epoch, now_epoch);
        (s, engagement, recency)
    };

    (s * recency * engagement).min(W_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(
        eps: i64,
        total: i64,
        last_ratio: Option<f64>,
        last_sec: i64,
        total_sec: i64,
    ) -> SeriesSignals {
        SeriesSignals {
            episodes_watched: eps,
            total_episodes: total,
            last_ep_ratio: last_ratio,
            last_ep_watched_sec: last_sec,
            total_watched_sec: total_sec,
        }
    }

    #[test]
    fn movie_branches() {
        assert_eq!(s_movie(Some(0.10), 0), 0.0); // dropped
        assert_eq!(s_movie(None, 0), 0.0); // None ratio = 0
        assert_eq!(s_movie(Some(0.75), 0), 1.0); // completed
        assert_eq!(s_movie(Some(0.90), 10), 1.0); // completed ignores pauses
        assert_eq!(s_movie(Some(0.50), 0), 0.5);
        assert_eq!(s_movie(Some(0.50), 3), 0.5 * MOVIE_PAUSE_PENALTY); // pause penalty
        assert_eq!(s_movie(Some(0.50), 2), 0.5); // below pause min
    }

    #[test]
    fn series_ladder() {
        // eps=0 → floor 0.30
        assert_eq!(s_series(&sig(0, 10, None, 0, 0)), 0.30);
        // eps=1 of 10 → max(0.30, 0.1) = 0.30; of 2 → max(0.30, 0.5)
        assert_eq!(s_series(&sig(1, 10, None, 1, 60)), 0.30);
        assert_eq!(s_series(&sig(1, 2, None, 1, 60)), 0.5);
        // eps=2 of 4 → max(0.45, 0.5)
        assert_eq!(s_series(&sig(2, 4, None, 2, 120)), 0.5);
        // eps≥3 → hooked floor lifts
        assert_eq!(s_series(&sig(3, 100, None, 3, 180)), 0.65);
        // 5/8 = 0.625 < hooked floor → the FLOOR wins
        assert_eq!(s_series(&sig(5, 8, None, 5, 300)), 0.65);
        // long watch floor: 90+ min → ≥ 0.65 even at eps 1
        assert_eq!(s_series(&sig(1, 100, None, 1, 90 * 60)), 0.65);
    }

    #[test]
    fn series_last_ep_exits() {
        // exit bonus adds to the DEPTH score (0.5 + 0.10), NOT the
        // ratio — the cap only bites when the base is already high
        assert_eq!(s_series(&sig(2, 4, Some(0.9), 40 * 60, 120)), 0.6);
        // base ≥ 0.90 + bonus hits the 1.0 cap
        assert_eq!(s_series(&sig(9, 10, Some(0.9), 40 * 60, 120)), 1.0);
        // abandon penalty
        let base = s_series(&sig(2, 4, None, 10, 120));
        assert_eq!(
            s_series(&sig(2, 4, Some(0.1), 10, 120)),
            base * SERIES_LAST_EP_ABANDON_PENALTY
        );
        // natural stop (0.2 ≤ r < 0.8) — untouched
        let base2 = s_series(&sig(2, 4, None, 30 * 60, 120));
        assert_eq!(s_series(&sig(2, 4, Some(0.5), 30 * 60, 120)), base2);
        // started-and-abandoned requires last_sec > 0
        let base3 = s_series(&sig(2, 4, None, 0, 120));
        assert_eq!(s_series(&sig(2, 4, Some(0.1), 0, 120)), base3);
    }

    #[test]
    fn drop_filter() {
        assert!(is_dropped("movie", Some(0.10), 0, None));
        assert!(!is_dropped("movie", Some(0.15), 0, None));
        assert!(is_dropped("tv", None, 0, Some(0.4)));
        assert!(!is_dropped("tv", None, 0, Some(0.5)));
        assert!(!is_dropped("tv", None, 1, Some(0.1))); // eps≥1 never drops here
    }

    #[test]
    fn recency_half_life() {
        let now = 1_800_000_000.0;
        assert_eq!(recency_decay(None, now), 0.0);
        assert_eq!(recency_decay(Some(now), now), 1.0);
        // 30 days → exactly a half
        let ago30 = now - 30.0 * 86400.0;
        assert!((recency_decay(Some(ago30), now) - 0.5).abs() < 1e-12);
        // future watch clamps at 1.0
        assert_eq!(recency_decay(Some(now + 999.0), now), 1.0);
    }

    #[test]
    fn w_item_end_to_end() {
        let now = 1_800_000_000.0;
        // dislike → 0 always
        assert_eq!(
            w_item(
                "movie",
                Some(1.0),
                0,
                None,
                Some(now),
                now,
                0,
                true,
                false,
                true
            ),
            0.0
        );
        // dropped movie → 0
        assert_eq!(
            w_item(
                "movie",
                Some(0.05),
                0,
                None,
                Some(now),
                now,
                0,
                false,
                false,
                false
            ),
            0.0
        );
        // completed now → 1.0
        assert_eq!(
            w_item(
                "movie",
                Some(0.9),
                0,
                None,
                Some(now),
                now,
                0,
                false,
                false,
                false
            ),
            1.0
        );
        // rewatch × 1.5 clips at W_MAX
        assert_eq!(
            w_item(
                "movie",
                Some(0.9),
                0,
                None,
                Some(now),
                now,
                1,
                false,
                false,
                false
            ),
            1.5
        );
        // watchlist-only: 0.7 × 1.2 = 0.84
        assert!(
            (w_item("movie", None, 0, None, None, now, 0, false, true, false) - 0.84).abs() < 1e-12
        );
        // 30-day-old completed movie: 1.0 × 0.5 × 1.0
        let ago30 = now - 30.0 * 86400.0;
        assert!(
            (w_item(
                "movie",
                Some(0.9),
                0,
                None,
                Some(ago30),
                now,
                0,
                false,
                false,
                false
            ) - 0.5)
                .abs()
                < 1e-9
        );
    }
}
