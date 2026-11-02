"""Generate live_parity_cases.json — the FIELD-LEVEL expected results
of every ported recall channel, computed by the REAL Python queries
against the seeded demo DB. The Rust integration test replays each
call and must agree on ids, ORDER, and every compared field.

    CINE_REC_INTEGRATION_DSN=... .venv/bin/python \\
        rust/db/tests/fixtures/gen_live_parity.py

Deterministic seed picks: first ids by stable ORDER BY from the demo
DB (movie seed, tv seed, a director seed, a tmdb-recs seed). Regenerate
whenever the demo data changes (the file records the data fingerprint).
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

from cine_rec_engine import queries as Q  # noqa: E402

DSN = os.environ.get("CINE_REC_INTEGRATION_DSN", "")
EMB = "embedding_minilm"

FIELDS = [
    "id", "title", "title_en", "media_type", "vote_average",
    "vote_count", "popularity", "collection_id", "release_year",
    "genres", "via", "tmdb_rank", "knn_similarity",
]


def slim(rows):
    out = []
    for r in rows:
        d = {k: r.get(k) for k in FIELDS}
        if "knn_similarity" in d and d["knn_similarity"] is not None:
            d["knn_similarity"] = round(float(d["knn_similarity"]), 6)
        out.append(d)
    return out


async def main() -> None:
    assert DSN, "set CINE_REC_INTEGRATION_DSN"
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    try:
        async def one(sql):
            async with pool.acquire() as c:
                return await c.fetchrow(sql)

        movie = await one(
            "SELECT id, ARRAY(SELECT g.name FROM tmdb_media_genres mg "
            "JOIN tmdb_genres g ON g.id=mg.genre_id "
            "WHERE (mg.media_id, mg.media_type)=(m.id, m.media_type)) AS genres "
            "FROM tmdb_media m WHERE m.media_type='movie' AND m.id != 155 "
            "ORDER BY m.id LIMIT 1")
        tv = await one(
            "SELECT id, ARRAY(SELECT g.name FROM tmdb_media_genres mg "
            "JOIN tmdb_genres g ON g.id=mg.genre_id "
            "WHERE (mg.media_id, mg.media_type)=(m.id, m.media_type)) AS genres "
            "FROM tmdb_media m WHERE m.media_type='tv' ORDER BY m.id LIMIT 1")
        rec_seed = await one(
            "SELECT media_id, media_type FROM tmdb_recommendations "
            "ORDER BY media_id LIMIT 1")
        dir_seed = await one(
            "SELECT c.media_id FROM tmdb_crew c WHERE c.job='Director' "
            "AND c.media_type='movie' ORDER BY c.media_id LIMIT 1")
        emb_seed = await one(
            f"SELECT id, {EMB}::text AS v FROM tmdb_movies "
            f"WHERE {EMB} IS NOT NULL ORDER BY id LIMIT 1")

        mg = [g for g in movie["genres"]][:3]
        tg = [g for g in tv["genres"]][:3]
        vec = [float(x) for x in emb_seed["v"].strip("[]").split(",")]

        cases = {}

        cases["genre_same"] = slim(await Q.generate_candidates(
            pool, movie["id"], mg, "movie", 30, False))
        cases["genre_cross"] = slim(await Q.generate_candidates(
            pool, movie["id"], mg, "movie", 30, True))
        cases["knn_movie"] = slim(await Q.generate_knn_candidates(
            pool, movie["id"], "movie", 20, False, EMB))
        cases["knn_tv"] = slim(await Q.generate_knn_candidates(
            pool, tv["id"], "tv", 20, False, EMB))
        cases["knn_cross"] = slim(await Q.generate_knn_candidates(
            pool, movie["id"], "movie", 20, True, EMB))
        cases["tmdb_single"] = slim(await Q.generate_tmdb_rec_candidates(
            pool, rec_seed["media_id"], rec_seed["media_type"], 20))
        tmdb_b = await Q.generate_tmdb_rec_candidates_batch(
            pool, [(rec_seed["media_id"], rec_seed["media_type"])], 20)
        cases["tmdb_batch"] = slim(tmdb_b[0])
        cases["director_single"] = slim(await Q.generate_director_candidates(
            pool, dir_seed["media_id"], "movie", 15))
        dir_b = await Q.generate_director_candidates_batch(
            pool, [(dir_seed["media_id"], "movie")], 15)
        cases["director_batch"] = slim(dir_b[0])
        cases["uvec"] = slim(await Q.generate_user_vector_candidates(
            pool, vec, 10, EMB))
        cases["discovery"] = slim(await Q.generate_discovery_candidates(
            pool, vec, [1, 2], 10, EMB))
        # Coverage: tv genre seed, deep KNN (floors bite at 120),
        # gems/trending page rows, popular with a year range
        cases["genre_same_tv"] = slim(await Q.generate_candidates(
            pool, tv["id"], tg, "tv", 30, False))
        cases["knn_tv_deep"] = slim(await Q.generate_knn_candidates(
            pool, tv["id"], "tv", 120, False, EMB))
        cases["popular"] = [
            {"tmdb_id": r["tmdb_id"], "title": r["title"],
             "rating": r["rating"], "vote_count": r["vote_count"],
             "media_type": r["media_type"], "release_year": r["release_year"],
             "title_en": r["title_en"]}
            for r in await Q.generate_popular_candidates(pool, 8)
        ]
        cases["popular_years"] = [
            {"tmdb_id": r["tmdb_id"], "release_year": r["release_year"]}
            for r in await Q.generate_popular_candidates(
                pool, 8, filters={"year_min": 1990, "year_max": 2015})
        ]
        emb = await Q.fetch_embeddings_batch(
            pool, [(c["id"], c["media_type"]) for c in cases["knn_movie"][:6]], EMB)
        cases["embeddings_keys"] = sorted([f"{k[0]}|{k[1]}" for k in emb])

        gems = await Q.generate_hidden_gems(pool, 6)
        cases["gems"] = [
            {"tmdb_id": r["tmdb_id"], "title": r["title"], "rating": r["rating"],
             "vote_count": r["vote_count"], "media_type": r["media_type"],
             "release_year": r["release_year"], "title_en": r["title_en"]}
            for r in gems
        ]
        gids = None
        if gems:
            async with pool.acquire() as c:
                gids = await c.fetchval(
                    "SELECT genre_id FROM tmdb_movie_genres_map "
                    "WHERE movie_id = $1 LIMIT 1", gems[0]["tmdb_id"])
        if gids:
            trend = await Q.generate_trending_genre(pool, gids, 6)
            cases["trending"] = [
                {"tmdb_id": r["tmdb_id"], "title": r["title"], "rating": r["rating"],
                 "vote_count": r["vote_count"], "media_type": r["media_type"],
                 "release_year": r["release_year"], "title_en": r["title_en"]}
                for r in trend
            ]

        enrich = await Q.enrich_candidates_batch(
            pool, [c["id"] for c in cases["genre_same"][:25]],
            [c["media_type"] for c in cases["genre_same"][:25]])
        cases["enrich"] = {
            str(k): {"director": v["director"],
                     "director_ids": sorted(v["director_ids"]),
                     "writer_ids": sorted(v["writer_ids"]),
                     "composer_ids": sorted(v["composer_ids"]),
                     "dp_ids": sorted(v["dp_ids"]),
                     "cast_list": v["cast_list"][:3],
                     "cast_ids": v["cast_ids"][:3],
                     "keywords": sorted(v["keywords"]),
                     "companies": sorted(v["companies"]),
                     "networks": sorted(v["networks"]),
                     "collection_id": v["collection_id"],
                     "original_language": v["original_language"]}
            for k, v in sorted(enrich.items())
        }

        payload = {
            "seeds": {
                "movie_id": movie["id"], "movie_genres": mg,
                "tv_id": tv["id"], "tv_genres": tg,
                "rec_seed": [rec_seed["media_id"], rec_seed["media_type"]],
                "dir_seed": dir_seed["media_id"],
                "user_vector": vec,
            },
            "emb_column": EMB,
            "cases": cases,
        }
    finally:
        await pool.close()

    dest = Path(__file__).parent / "live_parity_cases.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {dest}: {len(cases)} cases; seeds={payload['seeds']['movie_id']}/"
          f"{payload['seeds']['tv_id']}/{payload['seeds']['dir_seed']}")


if __name__ == "__main__":
    asyncio.run(main())
