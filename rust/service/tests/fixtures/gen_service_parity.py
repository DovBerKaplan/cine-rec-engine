"""Generate service_parity.json — Python-computed find_similar
COMPUTE-PATH outputs on frozen seeds, for the Rust parity tests.

Scope (phase 5b): the compute path only — cache bypassed (no Redis),
exclude=None (the per-user passes are 5c), randomness=0 (the Mersenne
stream stays Python-side). The recommend layer records
recommendation_plan's seeds + the same compute path driven with the
user's STORED vector as the uvec extras source (frozen vector →
deterministic ANN rows on both sides).

    CINE_REC_INTEGRATION_DSN=... .venv/bin/python \
        rust/service/tests/fixtures/gen_service_parity.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import asyncpg  # noqa: E402

from cine_rec_engine import service as svc_mod  # noqa: E402
from cine_rec_engine.config import (  # noqa: E402
    AUTEUR_DECAY_FACTORS,
    EMBEDDING_COLUMN,
    GENRE_MISMATCH_AUTEUR_FACTOR,
    MMR_ENABLED,
    MMR_LAMBDA,
)
from cine_rec_engine.queries import normalize_filters  # noqa: E402

DSN = os.environ.get("CINE_REC_INTEGRATION_DSN", "")
USER = 910_001  # the phase-4 personalization fixture user


async def pick_seeds(conn) -> dict:
    movie = await conn.fetchrow(
        "SELECT m.id FROM tmdb_media m WHERE m.media_type='movie' "
        "AND m.vote_count > 100 ORDER BY m.id LIMIT 1")
    movie2 = await conn.fetchrow(
        "SELECT m.id FROM tmdb_media m WHERE m.media_type='movie' "
        "AND m.vote_count > 100 ORDER BY m.id OFFSET 3 LIMIT 1")
    tv = await conn.fetchrow(
        "SELECT t.id FROM tmdb_tv t JOIN tmdb_media m ON (m.id=t.id AND m.media_type='tv') "
        "WHERE t.number_of_episodes >= 3 AND m.vote_count > 50 ORDER BY t.id LIMIT 1")
    genre = await conn.fetchrow(
        "SELECT g.id FROM tmdb_genres g "
        "WHERE EXISTS (SELECT 1 FROM tmdb_media_genres mg WHERE mg.genre_id=g.id) "
        "ORDER BY g.id LIMIT 1")
    return {"movie": movie["id"], "movie2": movie2["id"], "tv": tv["id"],
            "genre": genre["id"]}


def spec_of(case, seeds) -> dict:
    """The ComputeSpec the Rust side receives, built from the same
    module state the Python run used."""
    return {
        "seeds": case["seeds"],
        "media_type": case.get("media_type"),
        "limit": case.get("limit", 20),
        "allow_cross_media": case.get("allow_cross_media", False),
        "seed_weights": case.get("seed_weights"),
        "emb_column": EMBEDDING_COLUMN,
        "knn_ft_enabled": os.getenv("ENABLE_KNN_FINETUNED", "true").lower() != "false",
        "filters": case.get("filters") or {},
        "weights": {
            "active": list(svc_mod.ACTIVE_WEIGHTS.values()),
            "heuristic27": list(svc_mod.HEURISTIC_WEIGHTS_LIST),
        },
        "decay": {
            "auteur_factors": list(AUTEUR_DECAY_FACTORS),
            "genre_mismatch_factor": GENRE_MISMATCH_AUTEUR_FACTOR,
            "mmr_enabled": MMR_ENABLED,
            "mmr_lambda": MMR_LAMBDA,
        },
        "uvec_extra": case.get("uvec_extra"),
    }


async def main() -> None:
    assert DSN, "set CINE_REC_INTEGRATION_DSN"
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    svc = svc_mod.RecommendationService()
    await svc.initialize(pool)
    try:
        # force the NoopCache deployment (a dev Redis may be running and
        # the engine would silently use it): the cache seam is forced
        # OFF — neither reads nor writes — exactly the deployment the
        # Rust side models by default
        import cine_rec_engine.find_similar as fs_mod

        async def _no_cache():
            return None

        fs_mod._get_cache = _no_cache

        picks = await pick_seeds(pool)
        async def uvec_of(user):
            from cine_rec_engine.user_vector import load_user_vector
            return await load_user_vector(pool, user, None)

        uvec = await uvec_of(USER)
        from cine_rec_engine.user_vector import top_weighted_seeds
        weighted = await top_weighted_seeds(pool, USER, limit=20)
        plan_seeds = [(i, mt) for i, mt, _w in weighted]

        cases = [
            {"name": "single_movie", "seeds": [(picks["movie"], "movie")]},
            {"name": "single_tv", "seeds": [(picks["tv"], "tv")]},
            {"name": "multi_mixed", "seeds": [(picks["movie"], "movie"),
                                              (picks["tv"], "tv")]},
            {"name": "seed_weights",
             "seeds": [(picks["movie"], "movie"), (picks["movie2"], "movie")],
             "seed_weights": {str(picks["movie"]): 2.0, str(picks["movie2"]): 1.0}},
            {"name": "cross_media", "seeds": [(picks["movie"], "movie")],
             "allow_cross_media": True},
            {"name": "filters", "seeds": [(picks["movie"], "movie")],
             "filters": {"year_min": 2000, "genre_ids": [picks["genre"]]}},
            {"name": "recommend_910001_core", "seeds": plan_seeds,
             "limit": 30, "uvec_extra": ([f for f in uvec], 60)},
        ]

        out_cases = []
        for case in cases:
            filters = case.get("filters")
            normalized = normalize_filters(filters) if filters else None
            uvec_extra = case.get("uvec_extra")
            if uvec_extra is not None:
                from cine_rec_engine.queries import generate_user_vector_candidates
                frozen_vec, extra_limit = uvec_extra
                async def factory(v=frozen_vec, lim=extra_limit):
                    return await generate_user_vector_candidates(pool, v, lim, EMBEDDING_COLUMN)
            else:
                factory = None
            results = await svc.find_similar(
                [s for s in case["seeds"]],
                limit=case.get("limit", 20),
                randomness=0.0,
                allow_cross_media=case.get("allow_cross_media", False),
                seed_weights=({int(k): v for k, v in case["seed_weights"].items()}
                              if case.get("seed_weights") else None),
                filters=filters,
                exclude=None,
                extra_factory=factory,
                extra_limit=uvec_extra[1] if uvec_extra else 60,
            )
            assert len(results) > 0, f"case {case['name']}: EMPTY — n>0 rule"
            spec = spec_of(case, picks)
            spec["filters"] = dict(normalized) if normalized else {}
            out_cases.append({
                "name": case["name"],
                "spec": spec,
                "movie_ratio": (
                    sum(1 for s in case["seeds"] if (s[1] or "") == "movie")
                    / len(case["seeds"])
                ),
                "results": [
                    {k: r.get(k) for k in
                     ("tmdb_id", "score", "title", "title_en", "media_type",
                      "rating", "genres", "poster_path", "collection_id",
                      "release_year", "director_ids")}
                    for r in results
                ],
                "raw_per_seed": None,  # not exposed by find_similar; see below
            })
            print(f"case {case['name']}: {len(results)} results")

        payload = {
            "note": "compute-path parity; cache off; exclude=None; r=0",
            "emb_column": EMBEDDING_COLUMN,
            "plan_seeds_910001": plan_seeds,
            "cases": out_cases,
        }
    finally:
        await pool.close()

    dest = Path(__file__).parent / "service_parity.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {dest}: {len(out_cases)} cases")


if __name__ == "__main__":
    asyncio.run(main())
