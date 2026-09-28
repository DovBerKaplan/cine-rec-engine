"""Offline evaluation: 3 scorers × (pairwise accuracy, NDCG@10).

Runs against the demo catalog (any DB with the split schema). Methods:
  tmdb_sim   — TMDB's own /recommendations graph (rank-decayed)
  cosine     — character-bigram cosine between seed and candidate overviews
  engine     — this repo's scorer (recall channels + 22/28-feature weights)

    python eval/eval.py [--dsn ...]     # prints a markdown table
"""

from __future__ import annotations

import argparse
import os
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cine_rec_engine import RecommendationService  # noqa: E402


def ndcg_at_k(ranked_relevant: list[bool], k: int = 10) -> float:
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked_relevant[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(ranked_relevant), k)))
    return dcg / ideal if ideal else 0.0


async def run(dsn: str) -> dict:
    import asyncpg

    pool = await asyncpg.create_pool(dsn)
    from pgvector.asyncpg import register_vector

    async with pool.acquire() as _c:
        await register_vector(_c)  # codec per-connection; use one conn for fetch
        rows_raw = await _c.fetch(
            """SELECT id, 'movie'::text AS media_type, title, overview,
                      embedding_minilm AS emb FROM tmdb_movies
               UNION ALL SELECT id, 'tv', name, overview, embedding_minilm FROM tmdb_tv"""
        )
    overview = {(r["id"], r["media_type"]): (r["title"] or r["name"], r["overview"] or "") for r in rows_raw}
    def _vec(v):
        if v is None:
            return None
        return v.to_list() if hasattr(v, "to_list") else list(v)

    embedding = {(r["id"], r["media_type"]): _vec(r["emb"]) for r in rows_raw}
    all_ids = list(overview.keys())
    judgments = [
        json.loads(line)
        for line in (Path(__file__).parent / "judgments.jsonl").read_text().splitlines()
    ]

    svc = RecommendationService()
    await svc.initialize(pool)

    # REAL cosine baseline: MiniLM-384 ships with the demo catalog (the
    # slot was character-bigram Jaccard mislabeled "cosine" before)

    stats = {m: {"pair_ok": 0, "pair_n": 0, "ndcg": []}
             for m in ("tmdb_sim", "minilm_cosine", "engine")}

    for j in judgments:
        seed_id, seed_type = j["seed"]
        good = {tuple(g) for g in j["good"]}
        bad = {tuple(b) for b in j["bad"]}

        # engine scores over the full pool (via the service, same path as prod)
        eng = {(r["tmdb_id"], r["media_type"]): r["score"]
               for r in await svc.find_similar(
                   seed_id, limit=None, allow_cross_media=False,
                   media_type=seed_type)}

        # TMDB behavioral graph: rank-decayed, missing → 0
        tmdb = {}
        recs = await pool.fetch(
            """SELECT rec_media_id, rec_media_type, rank FROM tmdb_recommendations
               WHERE media_id = $1 AND media_type = $2""",
            seed_id, seed_type)
        for r in recs:
            tmdb[(r["rec_media_id"], r["rec_media_type"])] = max(
                0.25, 1.0 - (r["rank"] - 1) * 0.15)


        seed_emb = embedding.get((seed_id, seed_type))

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def cos(cand):
            e1, e2 = seed_emb, embedding.get(cand)
            if not e1 or not e2:
                return 0.0
            return dot(e1, e2)  # MiniLM vectors are L2-normalized

        scores = {
            "engine": eng,
            "tmdb_sim": tmdb,
            "minilm_cosine": {c: cos(c) for c in all_ids},
        }

        for method, table in scores.items():
            for g in good:
                for b in bad:
                    sg, sb = table.get(g, 0.0), table.get(b, 0.0)
                    stats[method]["pair_n"] += 1
                    if sg > sb:
                        stats[method]["pair_ok"] += 1
            ranked = sorted(all_ids, key=lambda c: table.get(c, 0.0), reverse=True)
            stats[method]["ndcg"].append(
                ndcg_at_k([c in good for c in ranked], k=10))

    await pool.close()
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default=os.getenv("DATABASE_URL",
                                        "postgresql://demo:demo@localhost:54329/demo"))
    args = p.parse_args()

    stats = asyncio.run(run(args.dsn))
    print("| method | pairwise acc. | NDCG@10 |")
    print("|---|---|---|")
    for m in ("tmdb_sim", "minilm_cosine", "engine"):
        s = stats[m]
        acc = s["pair_ok"] / s["pair_n"] if s["pair_n"] else 0.0
        nd = sum(s["ndcg"]) / len(s["ndcg"]) if s["ndcg"] else 0.0
        print(f"| {m} | {acc:.2f} | {nd:.2f} |")


if __name__ == "__main__":
    main()
