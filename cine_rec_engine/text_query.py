"""Natural-language cold start (RFC §3 Phase B).

    GET /by-text?q=like Inception but darker

Bring-your-own encoder via CINE_REC_ENCODER="module:function" (a
callable text -> list[float] living in YOUR import path — see
models/README.md). With an encoder, the query embeds into the catalog's
space and anchors the KNN recall; without one, an honest degrade to a
popularity-floor recall — the scorer still ranks by text overlap
(bigrams/style) against the query, documented as the fallback.

The anchor rides two EXISTING seams: `seed_info_override` (a virtual
seed shaped from the query text — every seed field the scorer reads is
.get()-guarded, so a text seed is safe) and `extra_candidates` (the
ANN/popular rows feed the pool and bypass the genre gate via their
`via` tag). No watch history required — that's the point.
"""

from __future__ import annotations

import importlib
from typing import List, Optional

from loguru import logger

VIRTUAL_SEED_ID = 0  # recall excludes seed ids; TMDB ids are all > 0,
# so the virtual seed never collides with a real title

_encoder_cache = {}


def load_encoder():
    """Resolve CINE_REC_ENCODER="module:function" once per process.
    None when unset or unloadable — the popularity fallback takes over."""
    if "fn" in _encoder_cache:
        return _encoder_cache["fn"]
    import os

    spec = os.getenv("CINE_REC_ENCODER", "").strip()
    fn = None
    if spec and ":" in spec:
        mod_name, _, attr = spec.rpartition(":")
        try:
            fn = getattr(importlib.import_module(mod_name), attr)
            if not callable(fn):
                fn = None
        except Exception as e:
            logger.warning(f"CINE_REC_ENCODER {spec!r} unusable ({e}) — "
                           f"/by-text falls back to popularity recall")
    _encoder_cache["fn"] = fn
    return fn


def build_virtual_seed(query: str) -> dict:
    """A seed dict made of the query itself. The scorer's seed reads are
    all .get()-guarded (see _entity_profile) — empty collections mean
    the metadata features score 0 and text-overlap features carry it."""
    return {
        "id": VIRTUAL_SEED_ID,
        "tmdb_id": VIRTUAL_SEED_ID,
        "title": query,
        "title_en": query,
        "overview": query,
        "overview_en": query,
        "genres": [],
        "keywords": [],
        "media_type": "movie",
        "vote_average": 0.0,
        "vote_count": 0,
        "popularity": 0.0,
    }


async def by_text(
    service,
    query: str,
    limit: int = 20,
    filters: Optional[dict] = None,
    randomness: float = 0.0,
    explore: Optional[float] = None,
) -> dict:
    """Query text in, ranked titles out — the cold-start anchor."""
    query = (query or "").strip()
    if not query:
        raise ValueError("q must not be empty")
    recall_limit = max(120, limit * 4)

    anchor: Optional[list] = None
    extra: Optional[List[dict]] = None
    encoder = load_encoder()
    if encoder is not None:
        try:
            anchor = [float(x) for x in encoder(query)]
            from .queries import generate_user_vector_candidates

            extra = await generate_user_vector_candidates(
                service.pool, anchor, limit=recall_limit)
        except Exception as e:
            logger.debug(f"/by-text encoder path failed, degrading: {e}")
            anchor, extra = None, None
    if not extra:
        from .queries import generate_popular_candidates

        extra = await generate_popular_candidates(
            service.pool, limit=recall_limit, filters=filters)
        anchor = None  # no same-space vector — text features only

    seed_vectors = None
    if anchor is not None:
        from .config import EMBEDDING_COLUMN

        seed_vectors = {VIRTUAL_SEED_ID: {EMBEDDING_COLUMN: anchor}}

    results = await service.find_similar(
        [VIRTUAL_SEED_ID],
        limit=limit,
        randomness=randomness,
        seed_info_override=build_virtual_seed(query),
        seed_vectors=seed_vectors,
        extra_candidates=extra,
        filters=filters,
        explore=explore,
    )
    return {
        "query": query,
        "results": results,
        "count": len(results),
        "vector_used": anchor is not None,
    }
