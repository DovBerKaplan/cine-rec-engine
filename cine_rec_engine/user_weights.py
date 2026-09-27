"""Per-item weights w_i (spec v0.2 §D).

    w_i = clip(S × S_recency × S_engagement, 0, 1.5)

- S is computed SEPARATELY for movies and series: a relative completion
  ratio on a series biases against long shows, so series score by DEPTH
  (episodes) with the completion ratio only as a floor.
- Drop-offs and explicit dislikes weigh 0 and never enter the user vector.
- Everything here is pure and synchronous — the SQL layer (user_stats.py)
  feeds it aggregated rows; tests pin every branch to §D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# --- §D.1 drop thresholds ---
MOVIE_DROP_RATIO = 0.15
SERIES_FIRST_EP_DROP = 0.50

# --- §D.2 movie ---
MOVIE_COMPLETED_RATIO = 0.75
MOVIE_PAUSE_PENALTY = 0.85
MOVIE_PAUSE_MIN = 3

# --- §D.3 series depth ladder ---
SERIES_EP_FLOORS = {0: 0.30, 1: 0.30, 2: 0.45}   # ≥3 → 0.65
SERIES_HOOKED_FLOOR = 0.65
SERIES_LONG_WATCH_SEC = 90 * 60                   # 90 minutes → ≥ 0.65
SERIES_LAST_EP_EXIT_BONUS = 0.10
SERIES_LAST_EP_EXIT_RATIO = 0.80
SERIES_LAST_EP_ABANDON_RATIO = 0.20
SERIES_LAST_EP_ABANDON_PENALTY = 0.90

# --- §D.4 recency ---
RECENCY_HALF_LIFE_DAYS = 30.0

# --- §D.5 engagement ---
ENGAGEMENT_REWATCH = 1.5
ENGAGEMENT_FAVORITE = 1.5
ENGAGEMENT_WATCHLIST = 1.2
WATCHLIST_ONLY_S = 0.7

W_MAX = 1.5


def s_movie(max_ratio: Optional[float], pause_count_total: int = 0) -> float:
    """§D.2 — completion-ratio score for a movie.

    r < 0.15 → 0 (dropped) · 0.15 ≤ r < 0.75 → r · r ≥ 0.75 → 1.0.
    Pause penalty applies only while unfinished.
    """
    r = max_ratio or 0.0
    if r < MOVIE_DROP_RATIO:
        return 0.0
    if r >= MOVIE_COMPLETED_RATIO:
        return 1.0
    s = r
    if pause_count_total >= MOVIE_PAUSE_MIN:
        s *= MOVIE_PAUSE_PENALTY
    return s


@dataclass
class SeriesSignals:
    """Aggregated per-series inputs (user_title_stats columns)."""

    episodes_watched: int = 0
    total_episodes: int = 0
    last_ep_ratio: Optional[float] = None    # last unit watched/duration
    last_ep_watched_sec: int = 0             # >0 means an episode was STARTED
    total_watched_sec: int = 0


def s_series(sig: SeriesSignals) -> float:
    """§D.3 — depth ladder + last-episode exits for a series.

    Depth dominates; ratio = episodes/total only ever lifts, never caps.
    """
    eps = sig.episodes_watched
    ratio = eps / max(sig.total_episodes, 1)

    if eps <= 0:
        # Only the first episode (partially) seen — §D.1 gate handles < 0.50
        s = SERIES_EP_FLOORS[0]
    elif eps == 1:
        s = max(SERIES_EP_FLOORS[1], ratio)
    elif eps == 2:
        s = max(SERIES_EP_FLOORS[2], ratio)
    else:
        s = max(SERIES_HOOKED_FLOOR, ratio)

    # Long watch time floor (optional rule): 90+ minutes → at least 0.65
    if sig.total_watched_sec >= SERIES_LONG_WATCH_SEC:
        s = max(s, SERIES_HOOKED_FLOOR)

    # Last-episode exits — only when an episode was actually started
    if sig.last_ep_watched_sec > 0 and sig.last_ep_ratio is not None:
        if sig.last_ep_ratio >= SERIES_LAST_EP_EXIT_RATIO:
            s = min(1.0, s + SERIES_LAST_EP_EXIT_BONUS)
        elif sig.last_ep_ratio < SERIES_LAST_EP_ABANDON_RATIO:
            s *= SERIES_LAST_EP_ABANDON_PENALTY
        # finished an episode and never started the next → natural stop,
        # no penalty — falls through untouched.
    return s


def is_dropped(
    media_type: str,
    max_ratio: Optional[float],
    episodes_watched: int,
    last_ep_ratio: Optional[float],
) -> bool:
    """§D.1 — drop filter: weight 0, out of the sum entirely."""
    if media_type == "movie":
        return (max_ratio or 0.0) < MOVIE_DROP_RATIO
    return episodes_watched == 0 and (last_ep_ratio or 0.0) < SERIES_FIRST_EP_DROP


def recency_decay(
    last_watched_at: Optional[datetime],
    now: Optional[datetime] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """§D.4 — S_recency = 2^(−Δt / T_half), Δt in days since last watch."""
    if last_watched_at is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    delta = now - last_watched_at
    days = max(delta.total_seconds(), 0.0) / 86400.0
    return math.pow(2.0, -days / half_life_days)


def w_item(
    media_type: str,
    *,
    max_ratio: Optional[float] = None,
    pause_count_total: int = 0,
    series: Optional[SeriesSignals] = None,
    last_watched_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
    rewatch_count: int = 0,
    favorite: bool = False,
    watchlist_only: bool = False,
    disliked: bool = False,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """§D end-to-end: the stored w_item for one (user, title).

    watchlist_only: the title was never watched — S = 0.7, engagement 1.2,
    recency anchored to NOW (a watchlist entry is alive until watched).
    disliked: 0, always — hard-filtered at retrieval too.
    """
    if disliked:
        return 0.0

    if watchlist_only:
        s = WATCHLIST_ONLY_S
        engagement = ENGAGEMENT_WATCHLIST
        recency = 1.0
    else:
        if is_dropped(
            media_type, max_ratio,
            series.episodes_watched if series else 0,
            series.last_ep_ratio if series else None,
        ):
            return 0.0
        if media_type == "movie":
            s = s_movie(max_ratio, pause_count_total)
        else:
            s = s_series(series or SeriesSignals())
        engagement = 1.0
        if rewatch_count >= 1 or favorite:
            engagement = max(ENGAGEMENT_REWATCH, ENGAGEMENT_FAVORITE)
        recency = recency_decay(last_watched_at, now=now, half_life_days=half_life_days)

    return min(s * recency * engagement, W_MAX)
