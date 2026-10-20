"""Standalone HTTP serving — the plug-and-play API.

    pip install "cine-rec-engine[serve]"
    DATABASE_URL=postgresql://user:pw@localhost/db \
        uvicorn cine_rec_engine.serve:app --port 8000

    GET /health                     DB readiness — 503 when degraded, so an
                                    orchestrator healthcheck can actually fail
    GET /metrics                    latency histogram, cache hit rate,
                                    recall-channel coverage (Prometheus text)
    GET /similar?seed=155           item-to-item (optionally user-filtered)
    GET /similar?seed=155:movie,1396:tv&user_id=42&model=e5e
    GET /page?user_id=42&rows=...   composed rows (top_picks, because:<id>,
                                    hidden_gems, trending:<gid>, discover)
    GET /by-text?q=space%20western  natural-language cold start
    POST /feedback                  one click/watch/skip/dislike outcome for
                                    a served title (impression-token gated)
    GET /for-user/42?include_why=1  the personalized row

Zero config beyond DATABASE_URL: the engine degrades every optional
channel, and user vectors rebuild on demand. The app factory accepts an
already-initialized RecommendationService for tests and embedding.
"""

from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable, List, Literal, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__
from . import init_db
from .config import env_int
from .impressions import consume as consume_impression
from .impressions import verify as verify_impression
from .model_spaces import REC_MODELS, normalize_model
from .page import compose_page
from .queries import normalize_filters
from .service import RecommendationService
from .text_query import by_text

# Result-size ceiling. Env-configurable (CINE_REC_MAX_LIMIT) so a caller
# with heavier pages can raise it without forking the module.
MAX_LIMIT = env_int("CINE_REC_MAX_LIMIT", 100, lo=1, hi=10_000)
DEFAULT_LIMIT = 30

# Bare paths that duplicate a /v1/ route — served as deprecated aliases
# until 1.0 (docs/stability.md).
_DEPRECATED_PATHS = frozenset({
    "/similar", "/page", "/by-text", "/feedback", "/for-user/{user_id}",
})


class FeedbackBody(BaseModel):
    """POST /feedback — one outcome for one served title (RFC §4)."""
    token: str
    outcome: Literal["click", "watch", "skip", "dislike"]
    tmdb_id: int


def _filters_from(
    year_min: Optional[int], year_max: Optional[int],
    genres: Optional[str], exclude_genres: Optional[str],
    max_runtime: Optional[int],
) -> Optional[dict]:
    """Query params → validated filter dict (400 on bad input)."""
    try:
        return normalize_filters({
            "year_min": year_min, "year_max": year_max,
            "genre_ids": genres, "exclude_genre_ids": exclude_genres,
            "max_runtime": max_runtime,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _parse_seed(raw: str) -> Union[List[int], List[Tuple[int, str]]]:
    """'155' → [155]; '155:movie,1396:tv' → [(155,'movie'),(1396,'tv')].

    Mixed forms are rejected — the engine takes either bare ids or
    composite keys per request, and guessing a media_type for half the
    seeds is how movie 155 ends up blended with tv 155.
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise HTTPException(status_code=400, detail="seed is empty")
    with_type: List[Tuple[int, str]] = []
    bare: List[int] = []
    for item in items:
        id_part, sep, mt = item.partition(":")
        try:
            tmdb_id = int(id_part)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"bad seed item {item!r}: expected <id> or <id>:<media_type>",
            )
        if tmdb_id <= 0:
            raise HTTPException(status_code=400,
                                detail=f"bad seed item {item!r}: id must be positive")
        mt = mt.strip().lower() if sep else ""
        if mt:
            if mt not in ("movie", "tv"):
                raise HTTPException(
                    status_code=400,
                    detail=f"bad seed item {item!r}: media_type must be movie|tv",
                )
            with_type.append((tmdb_id, mt))
        else:
            bare.append(tmdb_id)
    if with_type and bare:
        raise HTTPException(
            status_code=400,
            detail="mixed seed forms: give <id>:<media_type> for every seed, or none",
        )
    return with_type or bare


def _validate_model(model: Optional[str]) -> Optional[str]:
    key = normalize_model(model)
    if model and key is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model!r}: expected one of {sorted(REC_MODELS)}",
        )
    return key


def create_app(
    service: Optional[RecommendationService] = None,
    dsn: Optional[str] = None,
    auth_dependency: Optional[Callable] = None,
) -> FastAPI:
    """Build the API app. `service` injects an initialized engine (tests);
    otherwise a pool is created from DATABASE_URL on startup.
    `auth_dependency` injects a FastAPI dependency (e.g. your own
    principal extraction) that gates every data route — without it,
    CINE_REC_API_KEYS (comma-separated) enables built-in API-key auth;
    with neither set the API is open (degrade never fail)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
            app.state.owns_pool = False
        else:
            import asyncpg

            url = dsn or os.getenv("DATABASE_URL") or os.getenv("CINE_REC_DATABASE_URL")
            if not url:
                raise RuntimeError(
                    "no database: set DATABASE_URL "
                    "(or pass dsn=/service= to create_app)"
                )
            try:
                pool_size = max(1, int(os.getenv("CINE_REC_POOL_SIZE", "20")))
            except ValueError:
                pool_size = 20
            pool = await asyncpg.create_pool(
                url, min_size=min(pool_size, 10), max_size=pool_size
            )
            if os.getenv("CINE_REC_AUTO_INIT", "").strip().lower() in ("1", "true", "yes"):
                async with pool.acquire() as conn:
                    result = await init_db.apply_schema(conn, full=True)
                app.state.auto_init = result
            # An explicitly remapped table that is missing is a config
            # error — refuse to serve rather than degrade to empty rows.
            problems = await init_db.startup_verify(pool)
            if problems:
                await pool.close()
                raise RuntimeError(
                    "table map failed startup verification: "
                    + "; ".join(problems)
                )
            svc = RecommendationService()
            await svc.initialize(pool)
            app.state.service = svc
            app.state.owns_pool = True
            if not (os.getenv("CINE_REC_IMPRESSION_SECRET") or "").strip():
                from loguru import logger

                logger.warning(
                    "CINE_REC_IMPRESSION_SECRET is not set — impression "
                    "tokens are signed with a random per-process secret: "
                    "with more than one worker or replica, tokens minted by "
                    "one process fail verification in another, and all "
                    "tokens die on restart. Set a fixed secret in production."
                )
        yield
        if app.state.owns_pool:
            await app.state.service.pool.close()

    app = FastAPI(
        title="cine-rec-engine",
        version=__version__,
        lifespan=lifespan,
        description=__doc__,
    )

    # -- auth: env-driven API keys, or a caller-injected dependency ----
    # (unset + no hook = open service, same on/off idiom as CORS).
    api_keys = [k.strip() for k in (os.getenv("CINE_REC_API_KEYS") or "")
                .split(",") if k.strip()]
    if auth_dependency is None and api_keys:

        async def _env_api_key_auth(request: Request) -> None:
            supplied = request.headers.get("X-API-Key", "")
            if not supplied:
                header = request.headers.get("Authorization", "")
                if header.lower().startswith("bearer "):
                    supplied = header[7:].strip()
            if not supplied or not any(
                    hmac.compare_digest(supplied, k) for k in api_keys):
                raise HTTPException(
                    status_code=401, detail="invalid or missing API key",
                    headers={"WWW-Authenticate": 'ApiKey realm="cine-rec"'})

        auth_dependency = _env_api_key_auth
    auth_deps = [Depends(auth_dependency)] if auth_dependency is not None else None

    cors = os.getenv("CINE_REC_CORS_ORIGINS", "").strip()
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors.split(",") if o.strip()],
            # The middleware defaults are GET-only with zero allowed
            # headers — a browser preflight for POST /feedback (which
            # always carries content-type) would be rejected 400.
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        from . import metrics

        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        # Label by route template, never the raw path: /for-user/42 and
        # /for-user/43 are one series, and a scanner hitting random paths
        # cannot mint unbounded label cardinality. scope["route"] is only
        # populated once the router ran, i.e. after call_next returns.
        route = request.scope.get("route")
        route_path = getattr(route, "path", "")
        metrics.observe_request(route_path or "unmatched", elapsed_ms)
        if route_path in _DEPRECATED_PATHS:
            # Legacy alias of a /v1 route — kept until 1.0 (docs/stability.md)
            response.headers["Deprecation"] = "true"
            response.headers["X-Deprecated-Removal"] = "1.0"
        if response.status_code >= 500:
            from loguru import logger

            logger.warning(
                f"{request.method} {request.url.path} -> {response.status_code} "
                f"({elapsed_ms:.0f}ms)"
            )
        return response

    @app.get("/metrics", dependencies=auth_deps)
    async def metrics_endpoint():
        """Serving metrics (RFC M3): per-path latency histogram, cache
        hit rate, recall-channel coverage. Prometheus text format —
        scrape-ready; no dependencies, no external pushes."""
        from . import metrics

        return Response(content=metrics.render_prometheus(),
                        media_type="text/plain; version=0.0.4")

    @app.get("/health")
    async def health(request: Request):
        """Readiness probe: 200 only when the DB answers. A degraded
        service reports 503 so container healthchecks and load balancers
        actually stop routing to it — a 200-degraded body would make
        every HEALTHCHECK pass forever."""
        try:
            await request.app.state.service.pool.fetchval("SELECT 1")
            return {"status": "ok", "db": True}
        except Exception:
            return JSONResponse(
                status_code=503, content={"status": "degraded", "db": False})

    async def similar(
        request: Request,
        seed: str = Query(..., description="e.g. 155 or 155:movie,1396:tv"),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        randomness: float = Query(0.0, ge=0.0, le=1.0),
        allow_cross_media: bool = False,
        media_type: Optional[str] = Query(None, pattern="^(movie|tv)$"),
        user_id: Optional[int] = Query(None, ge=1),
        model: Optional[str] = Query(None, description="REC_MODELS key"),
        explore: Optional[float] = Query(
            None, ge=0.0, le=0.5,
            description="exploration budget share; 0 disables"),
        year_min: Optional[int] = Query(None, ge=1900, le=2100),
        year_max: Optional[int] = Query(None, ge=1900, le=2100),
        genres: Optional[str] = Query(None, description="pinned genre ids, comma-separated"),
        exclude_genres: Optional[str] = Query(None, description="excluded genre ids"),
        max_runtime: Optional[int] = Query(None, ge=1, le=1440,
                                           description="minutes; movies only"),
    ):
        svc: RecommendationService = request.app.state.service
        seeds = _parse_seed(seed)
        model_key = _validate_model(model)
        req_filters = _filters_from(year_min, year_max, genres,
                                    exclude_genres, max_runtime)
        # user_id without an explicit exclude must still arm the per-user
        # pass (watched/dislike filtering) — None would skip it entirely.
        results = await svc.find_similar(
            seeds,
            limit=limit,
            randomness=randomness,
            allow_cross_media=allow_cross_media,
            media_type=media_type,
            user_id=user_id,
            exclude=set() if user_id is not None else None,
            rec_model=model_key,
            filters=req_filters,
            explore=explore,
        )
        return {"results": results, "count": len(results)}

    async def page(
        request: Request,
        user_id: Optional[int] = Query(None, ge=1),
        rows: str = Query(
            "top_picks",
            description="top_picks,because:155:movie,hidden_gems,trending:80,discover"),
        row_limit: int = Query(12, ge=1, le=30),
        explore: Optional[float] = Query(None, ge=0.0, le=0.5),
        year_min: Optional[int] = Query(None, ge=1900, le=2100),
        year_max: Optional[int] = Query(None, ge=1900, le=2100),
        genres: Optional[str] = Query(None),
        exclude_genres: Optional[str] = Query(None),
        max_runtime: Optional[int] = Query(None, ge=1, le=1440),
    ):
        svc: RecommendationService = request.app.state.service
        page_filters = _filters_from(year_min, year_max, genres,
                                     exclude_genres, max_runtime)
        try:
            return await compose_page(
                svc, user_id=user_id, rows=rows,
                row_limit=row_limit, filters=page_filters,
                explore=explore,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def by_text_endpoint(
        request: Request,
        q: str = Query(..., min_length=1, max_length=500),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        randomness: float = Query(0.0, ge=0.0, le=1.0),
        explore: Optional[float] = Query(None, ge=0.0, le=0.5),
        year_min: Optional[int] = Query(None, ge=1900, le=2100),
        year_max: Optional[int] = Query(None, ge=1900, le=2100),
        genres: Optional[str] = Query(None),
        exclude_genres: Optional[str] = Query(None),
        max_runtime: Optional[int] = Query(None, ge=1, le=1440),
    ):
        svc: RecommendationService = request.app.state.service
        text_filters = _filters_from(year_min, year_max, genres,
                                     exclude_genres, max_runtime)
        try:
            return await by_text(svc, q, limit=limit,
                                 randomness=randomness,
                                 filters=text_filters, explore=explore)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def feedback(request: Request, body: FeedbackBody):
        """Outcome ingestion (RFC §4): verify the impression, map the
        outcome onto the existing §D/§J machinery — no trainer, no
        external queue; w_i adjusts transactionally."""
        payload = verify_impression(body.token)
        if payload is None:
            raise HTTPException(status_code=400,
                                detail="invalid or expired impression token")
        user_id = payload.get("u")
        served = payload.get("i") or []
        if not isinstance(user_id, int):
            raise HTTPException(status_code=400, detail="token carries no user")
        match = next((m for i, m in served if i == body.tmdb_id), None)
        if match is None:
            raise HTTPException(
                status_code=400,
                detail=f"tmdb_id {body.tmdb_id} was not part of this impression")

        svc: RecommendationService = request.app.state.service
        from . import user_stats

        # One-time use: record_feedback is idempotent, apply_skip_decay
        # is not — a replayed token must never re-nudge w_item.
        if not await consume_impression(body.token, payload):
            return {"status": "duplicate", "outcome": body.outcome,
                    "user_id": user_id, "tmdb_id": body.tmdb_id}

        try:
            if body.outcome == "watch":
                await user_stats.record_event(svc.pool, {
                    "user_id": user_id, "tmdb_id": body.tmdb_id,
                    "media_type": match,
                    "watched_at": datetime.now(timezone.utc),  # dt object —
                    # asyncpg encodes timestamptz; a string would not bind
                    "completed": True,
                })
            elif body.outcome == "skip":
                await user_stats.record_feedback(
                    svc.pool, user_id, body.tmdb_id, match, "skip")
                await user_stats.apply_skip_decay(
                    svc.pool, user_id, body.tmdb_id, match)
            else:  # click | dislike
                await user_stats.record_feedback(
                    svc.pool, user_id, body.tmdb_id, match, body.outcome)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "recorded", "outcome": body.outcome,
                "user_id": user_id, "tmdb_id": body.tmdb_id}

    async def for_user(
        request: Request,
        user_id: int = Path(..., ge=1),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        randomness: float = Query(0.0, ge=0.0, le=1.0),
        include_why: bool = False,
        explore: Optional[float] = Query(None, ge=0.0, le=0.5),
        model: Optional[str] = Query(None, description="REC_MODELS key"),
    ):
        svc: RecommendationService = request.app.state.service
        model_key = _validate_model(model)
        try:
            return await svc.recommend_for_user(
                user_id,
                limit=limit,
                randomness=randomness,
                include_why=include_why,
                explore=explore,
                vector_space=model_key,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # -- route registration: /v1/* is canonical, the bare paths are ----
    # deprecated aliases that keep every existing integrator working
    # until 1.0 (docs/stability.md). The SAME handler objects serve
    # both, so behavior and signatures cannot drift between them.
    for path, endpoint, methods in (
        ("/similar", similar, ["GET"]),
        ("/page", page, ["GET"]),
        ("/by-text", by_text_endpoint, ["GET"]),
        ("/feedback", feedback, ["POST"]),
        ("/for-user/{user_id}", for_user, ["GET"]),
    ):
        app.add_api_route(path, endpoint, methods=methods, deprecated=True,
                          dependencies=auth_deps)
        app.add_api_route(f"/v1{path}", endpoint, methods=methods,
                          dependencies=auth_deps)

    return app


app = create_app()


if __name__ == "__main__":  # python -m cine_rec_engine.serve
    import uvicorn

    uvicorn.run(app, host=os.getenv("CINE_REC_HOST", "0.0.0.0"),
                port=env_int("CINE_REC_PORT", 8000, lo=1, hi=65535))
