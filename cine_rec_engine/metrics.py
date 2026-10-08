"""In-process serving metrics (RFC M3 §5): recall-channel coverage,
request latency histogram, cache hit rate.

Plain counters on the serving event loop — no locks, no allocation
beyond the dicts, no I/O; a scrape reads a consistent-enough snapshot.
The engine observes cache outcomes and channel contribution; the HTTP
layer observes request latency. Degrade never fail applies: metrics
must never break a request, so every observe* is exception-safe by
construction (no external calls).
"""

from __future__ import annotations

import time
from typing import Dict

# latency bucket upper bounds in ms (last is +inf)
_LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, float("inf"))

_requests: Dict[str, dict] = {}      # path -> {count, total_ms, buckets[]}
_cache = {"hit": 0, "miss": 0, "store": 0}
_channels: Dict[str, list] = {}      # name -> [with_candidates, total]

_started = time.time()


def observe_request(path: str, elapsed_ms: float) -> None:
    s = _requests.get(path)
    if s is None:
        s = _requests[path] = {
            "count": 0, "total_ms": 0.0,
            "buckets": [0] * len(_LATENCY_BUCKETS_MS),
        }
    s["count"] += 1
    s["total_ms"] += elapsed_ms
    for i, ub in enumerate(_LATENCY_BUCKETS_MS):
        if elapsed_ms <= ub:
            s["buckets"][i] += 1
            break


def observe_cache(outcome: str) -> None:
    """outcome: 'hit' | 'miss' | 'store'."""
    if outcome in _cache:
        _cache[outcome] += 1


def observe_channel(name: str, contributed: int) -> None:
    """One recall-channel execution; contributed = candidates it
    returned (coverage = with_candidates / total)."""
    s = _channels.get(name)
    if s is None:
        s = _channels[name] = [0, 0]
    s[1] += 1
    if contributed > 0:
        s[0] += 1


def snapshot() -> dict:
    def path_stats(s):
        n = s["count"] or 1
        return {
            "count": s["count"],
            "mean_ms": round(s["total_ms"] / n, 1),
            "buckets_ms": [b for b in _LATENCY_BUCKETS_MS],
            "bucket_counts": list(s["buckets"]),
        }

    hits, misses = _cache["hit"], _cache["miss"]
    return {
        "uptime_s": round(time.time() - _started, 1),
        "requests": {p: path_stats(s) for p, s in _requests.items()},
        "cache": {
            **_cache,
            "hit_rate": round(hits / (hits + misses), 3) if hits + misses else None,
        },
        "channels": {
            n: {"coverage": round(w / t, 3) if t else None,
                "contributed": w, "total": t}
            for n, (w, t) in _channels.items()
        },
    }


def render_prometheus() -> str:
    """Minimal text/exposition format — scrape-ready without deps."""
    lines = []
    snap = snapshot()
    lines.append("# cine_rec uptime seconds")
    lines.append(f"cine_rec_uptime_seconds {snap['uptime_s']}")
    for path, s in snap["requests"].items():
        label = path.replace('"', "'")
        lines.append(f'cine_rec_requests_total{{path="{label}"}} {s["count"]}')
        lines.append(f'cine_rec_request_ms_sum{{path="{label}"}} '
                     f'{round(s["mean_ms"] * s["count"], 1)}')
        for ub, c in zip(s["buckets_ms"], s["bucket_counts"]):
            le = "+Inf" if ub == float("inf") else str(ub)
            lines.append(f'cine_rec_request_ms_bucket{{path="{label}",le="{le}"}} {c}')
    c = snap["cache"]
    lines.append(f'cine_rec_cache_hits_total {c["hit"]}')
    lines.append(f'cine_rec_cache_misses_total {c["miss"]}')
    lines.append(f'cine_rec_cache_stores_total {c["store"]}')
    hr = c["hit_rate"]
    lines.append(f'cine_rec_cache_hit_rate {hr if hr is not None else "NaN"}')
    for name, ch in snap["channels"].items():
        cov = ch["coverage"]
        lines.append(f'cine_rec_channel_contributed_total{{channel="{name}"}} {ch["contributed"]}')
        lines.append(f'cine_rec_channel_requests_total{{channel="{name}"}} {ch["total"]}')
        lines.append(f'cine_rec_channel_coverage{{channel="{name}"}} '
                     f'{cov if cov is not None else "NaN"}')
    return "\n".join(lines) + "\n"


def reset() -> None:
    """Test hook — counters are process-global."""
    _requests.clear()
    _cache.update(hit=0, miss=0, store=0)
    _channels.clear()
