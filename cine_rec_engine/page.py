"""Page-of-rows composition (RFC §1, M1 MVP).

A modern streaming UI consumes a page of themed rows, not one flat
list. compose_page() builds that page through the EXISTING pipelines —
a `because:` row lands in the same find_similar cache entry a direct
call would, top_picks IS recommend_for_user — executed concurrently,
then deduplicated first-row-wins across the page.

MVP semantics (documented limits):
- rows run concurrently at their own limit; dedup removes a title from
  LATER rows only, so a later row can come back short when an earlier
  row already used its titles.
- a failing row degrades to an omitted row, never a failed page.
- every row carries a stateless impression token (impressions.py).

Row grammar: top_picks | because:<id>[:movie|tv] | hidden_gems |
trending:<genre_id> — comma-separated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from .impressions import issue as issue_token

ROW_LIMIT_CAP = 30


@dataclass(frozen=True)
class RowSpec:
    kind: str                      # top_picks | because | hidden_gems | trending | discover
    seed: Optional[Tuple[int, Optional[str]]] = None   # (tmdb_id, media_type|None)
    genre_id: Optional[int] = None
    raw: str = ""


def parse_row_specs(raw: str) -> List[RowSpec]:
    """'top_picks,because:155:movie,hidden_gems,trending:80' → specs.

    Raises ValueError with a teachable message on bad specs.
    """
    items = [s.strip() for s in (raw or "").split(",") if s.strip()]
    if not items:
        raise ValueError("rows is empty")
    specs: List[RowSpec] = []
    for item in items:
        kind, _, arg = item.partition(":")
        if kind == "top_picks":
            if arg:
                raise ValueError(f"top_picks takes no argument: {item!r}")
            specs.append(RowSpec(kind="top_picks", raw=item))
        elif kind == "because":
            sid, sep, mt = arg.partition(":")
            try:
                tmdb_id = int(sid)
            except ValueError:
                raise ValueError(f"bad because seed: {item!r} — expected because:<id>[:movie|tv]")
            if tmdb_id <= 0:
                raise ValueError(f"bad because seed id in {item!r}")
            mt = mt.strip().lower() if sep else ""
            if mt and mt not in ("movie", "tv"):
                raise ValueError(f"bad media_type in {item!r}: movie|tv")
            specs.append(RowSpec(
                kind="because", seed=(tmdb_id, mt or None), raw=item))
        elif kind == "hidden_gems":
            if arg:
                raise ValueError(f"hidden_gems takes no argument: {item!r}")
            specs.append(RowSpec(kind="hidden_gems", raw=item))
        elif kind == "discover":
            if arg:
                raise ValueError(f"discover takes no argument: {item!r}")
            specs.append(RowSpec(kind="discover", raw=item))
        elif kind == "trending":
            try:
                genre_id = int(arg)
            except ValueError:
                raise ValueError(f"bad trending spec: {item!r} — expected trending:<genre_id>")
            if genre_id <= 0:
                raise ValueError(f"bad genre id in {item!r}")
            specs.append(RowSpec(kind="trending", genre_id=genre_id, raw=item))
        else:
            raise ValueError(
                f"unknown row kind {kind!r} — expected top_picks | because:<id>[:mt] | "
                f"hidden_gems | trending:<genre_id> | discover")
    return specs


async def compose_page(
    service,
    user_id: Optional[int] = None,
    rows: str = "top_picks",
    row_limit: int = 12,
    filters: Optional[dict] = None,
    explore: Optional[float] = None,
) -> dict:
    """Build the page. Raises ValueError for bad specs (caller → 400);
    everything else degrades — a broken row disappears, the page lives.

    The shared user context (exclusions, user vector) is computed ONCE
    and threaded through every row (RFC §4 latency follow-up) — each
    row would otherwise re-query it. `explore` is the exploration
    budget share (None = config default) for because:/top_picks rows;
    a `discover` row IS exploration (adjacent clusters via ANN).
    """
    specs = parse_row_specs(rows)
    row_limit = max(1, min(row_limit, ROW_LIMIT_CAP))
    if any(s.kind in ("top_picks", "discover") for s in specs) \
            and user_id is None:
        raise ValueError("top_picks and discover require user_id")

    from .queries import (
        generate_discovery_candidates,
        generate_hidden_gems,
        generate_trending_genre,
        top_user_genre_ids,
    )

    # Shared per-request user context — one exclusion fetch, one vector
    # ensure, reused by every row (pure optimization; identical results).
    user_ctx: Optional[dict] = None
    top_genre_ids: list = []
    if user_id is not None:
        user_ctx = {}
        try:
            user_ctx["exclusions"] = await service._merged_exclusions(
                set(), user_id)
        except Exception:
            user_ctx["exclusions"] = set()
        try:
            from .user_vector import ensure_user_vector

            user_ctx["uvec"] = await ensure_user_vector(
                service.pool, user_id, None) or None
        except Exception:
            user_ctx["uvec"] = None
        try:
            top_genre_ids = await top_user_genre_ids(service.pool, user_id)
        except Exception:
            top_genre_ids = []

    gem_genres: Optional[list] = None
    if top_genre_ids:
        gem_genres = top_genre_ids

    # Deterministic direct rows (gems/trending/discover) are TTL-cached
    # like find_similar payloads — a warm page must not re-run raw SQL
    # per request. because:/top_picks already ride the find_similar cache.
    from .queries import filters_fingerprint
    from .service import _get_cache

    cache = await _get_cache()
    flt_fp = filters_fingerprint(filters)

    async def _cached(key: str, factory):
        if cache is not None:
            try:
                hit = await cache.get(key)
                if hit is not None:
                    return hit
            except Exception:
                pass
        val = await factory()
        if cache is not None and val:
            try:
                await cache.set(key, val, ttl=600)
            except Exception:
                pass
        return val

    async def _run(spec: RowSpec) -> List[dict]:
        if spec.kind == "top_picks":
            out = await service.recommend_for_user(
                user_id, limit=row_limit, explore=explore,
                user_context=user_ctx)
            return out["results"]
        if spec.kind == "because":
            sid, mt = spec.seed
            seeds: list = [(sid, mt)] if mt else [sid]
            return await service.find_similar(
                seeds, limit=row_limit,
                user_id=user_id,
                exclude=set() if user_id is not None else None,
                filters=filters,
                explore=explore,
                user_context=user_ctx,
            )
        if spec.kind == "hidden_gems":
            # user-scoped: the genre scope comes from the user's clusters
            key = f"page:row:gems:u={user_id}:n={row_limit}:fl={flt_fp}"
            return await _cached(key, lambda: generate_hidden_gems(
                service.pool, row_limit, genre_ids=gem_genres, filters=filters))
        if spec.kind == "discover":
            uvec = (user_ctx or {}).get("uvec")

            async def _discover():
                if not uvec or not top_genre_ids:
                    return []  # no vector or no clusters → no honest novelty
                rows = await generate_discovery_candidates(
                    service.pool, uvec, top_genre_ids,
                    limit=max(60, row_limit * 4))
                return rows[:row_limit]

            key = f"page:row:discover:u={user_id}:n={row_limit}"
            return await _cached(key, _discover)
        key = f"page:row:trend:g={spec.genre_id}:n={row_limit}:fl={flt_fp}"
        return await _cached(key, lambda: generate_trending_genre(
            service.pool, spec.genre_id, row_limit, filters))

    gathered = await asyncio.gather(
        *(_run(s) for s in specs), return_exceptions=True)

    page_rows: List[dict] = []
    seen: set = set()
    for spec, res in zip(specs, gathered):
        if isinstance(res, BaseException):
            logger.debug(f"page row {spec.raw!r} failed: {res}")
            continue  # a row degrades to omitted, never fails the page
        titles = []
        for r in res:
            key = (r.get("tmdb_id"), r.get("media_type"))
            if key in seen:      # first-row-wins across the page
                continue
            seen.add(key)
            titles.append(r)
        if not titles:
            continue
        page_rows.append({
            "kind": spec.kind,
            "spec": spec.raw,
            "titles": titles,
            "count": len(titles),
            "impression": issue_token({
                "u": user_id, "r": spec.kind, "spec": spec.raw,
                # composite keys — a click must attribute the right medium
                "i": [[t.get("tmdb_id"), t.get("media_type")]
                      for t in titles],
            }),
        })

    return {
        "user_id": user_id,
        "rows": page_rows,
        "row_limit": row_limit,
        "page_token": issue_token({"u": user_id, "page": [s.raw for s in specs]}),
    }
