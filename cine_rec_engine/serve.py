"""Standalone HTTP serving — the plug-and-play API.

    pip install "cine-rec-engine[serve]"
    DATABASE_URL=postgresql://user:pw@localhost/db \
        uvicorn cine_rec_engine.serve:app --port 8000

    GET /health                     liveness + DB readiness
    GET /metrics                    latency histogram, cache hit rate,
                                    recall-channel coverage (Prometheus text)
    GET /similar?seed=155           item-to-item (optionally user-filtered)
    GET /similar?seed=155:movie,1396:tv&user_id=42&model=e5e
    GET /for-user/42?include_why=1  the personalized row

Zero config beyond DATABASE_URL: the engine degrades every optional
channel, and user vectors rebuild on demand. The app factory accepts an
already-initialized RecommendationService for tests and embedding.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from . import __version__
from .impressions import verify as verify_impression
from .model_spaces import REC_MODELS, normalize_model
from .page import compose_page
from .queries import normalize_filters
from .service import RecommendationService
from .text_query import by_text

MAX_LIMIT = 100
DEFAULT_LIMIT = 30


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
) -> FastAPI:
    """Build the API app. `service` injects an initialized engine (tests);
    otherwise a pool is created from DATABASE_URL on startup."""

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
            pool = await asyncpg.create_pool(url)
            svc = RecommendationService()
            await svc.initialize(pool)
            app.state.service = svc
            app.state.owns_pool = True
        yield
        if app.state.owns_pool:
            await app.state.service.pool.close()

    app = FastAPI(
        title="cine-rec-engine",
        version=__version__,
        lifespan=lifespan,
        description=__doc__,
    )

    cors = os.getenv("CINE_REC_CORS_ORIGINS", "").strip()
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors.split(",") if o.strip()],
        )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        from . import metrics

        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        metrics.observe_request(request.url.path, elapsed_ms)
        if response.status_code >= 500:
            from loguru import logger

            logger.warning(
                f"{request.method} {request.url.path} -> {response.status_code} "
                f"({elapsed_ms:.0f}ms)"
            )
        return response

    @app.get("/metrics")
    async def metrics_endpoint():
        """Serving metrics (RFC M3): per-path latency histogram, cache
        hit rate, recall-channel coverage. Prometheus text format —
        scrape-ready; no dependencies, no external pushes."""
        from . import metrics

        return Response(content=metrics.render_prometheus(),
                        media_type="text/plain; version=0.0.4")

    @app.get("/health")
    async def health(request: Request):
        try:
            await request.app.state.service.pool.fetchval("SELECT 1")
            return {"status": "ok", "db": True}
        except Exception:
            return {"status": "degraded", "db": False}

    @app.get("/similar")
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

    @app.get("/page")
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

    @app.get("/by-text")
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

    @app.post("/feedback")
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

    @app.get("/for-user/{user_id}")
    async def for_user(
        request: Request,
        user_id: int,
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

    return app


app = create_app()


if __name__ == "__main__":  # python -m cine_rec_engine.serve
    import uvicorn

    uvicorn.run(app, host=os.getenv("CINE_REC_HOST", "0.0.0.0"),
                port=int(os.getenv("CINE_REC_PORT", "8000")))
