"""Build the public MovieLens benchmark — two independent steps.

  judgments (offline, no API key): derive the public judgment set from
  MovieLens ml-latest-small. Collaborative ground truth, deliberately
  independent of this engine's own tuning rules: a seed is a
  widely-loved movie; good = other movies the same loving cohort rated
  >= 4.0; bad = movies that cohort rated <= 2.5. Deterministic — ties
  break by id, seeds order by lovers then id. TMDB ids only; no TMDB
  metadata is stored.

      python eval/build_ml_bench.py judgments \\
          --ml ml-latest-small.zip --out eval/ml_judgments.jsonl

  catalog (needs YOUR TMDB api key; ~1 request per title): bootstrap a
  database with the TMDB mirror for the benchmark universe, optionally
  computing MiniLM embeddings for the minilm_cosine baseline. The
  universe is every linked movie with at least --min-ratings ratings
  (default 5) — a few thousand titles, a couple of minutes at the
  default rate.

      python eval/build_ml_bench.py catalog \\
          --ml ml-latest-small.zip \\
          --dsn postgresql://demo:demo@localhost:5432/mlbench \\
          --api-key $TMDB_API_KEY --embed

Then score:  python eval/eval.py --dsn ... --judgments eval/ml_judgments.jsonl

Licensing: derived id/label files are committed with attribution to
MovieLens (F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens
Datasets: History and Context. ACM TiiS — grouplens.org). TMDb metadata
fetched by `catalog` stays in YOUR database: this product uses the TMDB
API but is not endorsed or certified by TMDb.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def read_movielens(ml_zip: str):
    """(ratings rows, movieId -> tmdbId) from an ml-latest-small zip."""
    ratings = []  # (userId, movieId, rating)
    links: dict[int, int] = {}
    with zipfile.ZipFile(ml_zip) as z:
        with z.open([n for n in z.namelist()
                     if n.endswith("ratings.csv")][0]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                ratings.append((int(row["userId"]), int(row["movieId"]),
                                float(row["rating"])))
        with z.open([n for n in z.namelist()
                     if n.endswith("links.csv")][0]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                tmdb = (row.get("tmdbId") or "").strip()
                if tmdb:
                    links[int(row["movieId"])] = int(float(tmdb))
    return ratings, links


def derive_judgments(ratings, links, seed_lovers_min=100, agree_min=5,
                     max_seeds=100, per_side=10, min_side=3) -> list[dict]:
    loved: dict[int, set] = defaultdict(set)
    by_user: dict[int, list] = defaultdict(list)
    for u, m, r in ratings:
        by_user[u].append((m, r))
        if r >= 4.0:
            loved[m].add(u)
    seed_ids = sorted(
        (m for m, users in loved.items()
         if len(users) >= seed_lovers_min and m in links),
        key=lambda m: (-len(loved[m]), m),
    )[:max_seeds]

    def top(c: Counter) -> list[int]:
        return [links[m] for m in sorted(
            (m for m, a in c.items() if a >= agree_min and m in links),
            key=lambda m: (-c[m], m))[:per_side]]

    out = []
    for s in seed_ids:
        cohort = loved[s]
        good: Counter = Counter()
        bad: Counter = Counter()
        for u in cohort:
            for m, r in by_user[u]:
                if m == s:
                    continue
                if r >= 4.0:
                    good[m] += 1
                elif r <= 2.5:
                    bad[m] += 1
        g, b = top(good), top(bad)
        if len(g) >= min_side and len(b) >= min_side:
            out.append({"seed": [links[s], "movie"],
                        "good": [[i, "movie"] for i in g],
                        "bad": [[i, "movie"] for i in b]})
    return out


def cmd_judgments(args) -> None:
    ratings, links = read_movielens(args.ml)
    js = derive_judgments(ratings, links,
                          seed_lovers_min=args.seed_lovers_min,
                          max_seeds=args.max_seeds)
    n_good = sum(len(j["good"]) for j in js)
    n_bad = sum(len(j["bad"]) for j in js)
    with open(args.out, "w") as f:
        for j in js:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    print(f"{len(js)} seeds | {n_good} good | {n_bad} bad "
          f"-> {args.out} (pairs: {n_good * n_bad})")


async def _catalog(args) -> None:
    import asyncpg

    from cine_rec_engine import init_db
    from ingest.loader import TmdbIngest

    ratings, links = read_movielens(args.ml)
    counts: Counter = Counter(m for _u, m, _r in ratings)
    universe = sorted(links[m] for m, c in counts.items()
                      if c >= args.min_ratings and m in links)

    pool = await asyncpg.create_pool(args.dsn)
    async with pool.acquire() as conn:
        await init_db.apply_schema(conn, full=True)
    print(f"universe: {len(universe)} movies — bootstrapping TMDB mirror")

    async def entries():
        for tmdb_id in universe:
            yield tmdb_id, "movie"

    ingest = TmdbIngest(pool, api_key=args.api_key,
                        rate_per_second=args.rate)
    await ingest.bootstrap(entries())
    print("mirror bootstrapped")

    if args.embed:
        from pgvector.asyncpg import register_vector

        from cine_rec_engine import db, tables

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        async with pool.acquire() as conn:
            await register_vector(conn)  # vector codec is per-connection
            rows = await db.fetch(conn, "SELECT id, title, overview "
                                        "FROM {t_tmdb_movies}")
            texts = [f"{r['title']}. {r['overview'] or ''}" for r in rows]
            vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=True)
            # eval reads BOTH fact tables in one UNION — the empty tv
            # table needs the column too or the fetch itself fails
            for table in (tables.name("tmdb_movies"), tables.name("tmdb_tv")):
                await db.execute(conn,
                                 f"ALTER TABLE {table} ADD COLUMN IF NOT "
                                 f"EXISTS embedding_minilm vector(384)")
            table = tables.name("tmdb_movies")
            for r, v in zip(rows, vecs):
                await db.execute(conn,
                                 f"UPDATE {table} SET embedding_minilm = $1 "
                                 f"WHERE id = $2", v.tolist(), r["id"])
        print(f"embedded {len(rows)} overviews (MiniLM-L6-v2)")
    await pool.close()
    print("catalog ready — now run: python eval/eval.py --dsn ... "
          "--judgments eval/ml_judgments.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pj = sub.add_parser("judgments", help="offline: ratings -> judgments")
    pj.add_argument("--ml", required=True, help="path to ml-latest-small.zip")
    pj.add_argument("--out", default="eval/ml_judgments.jsonl")
    pj.add_argument("--seed-lovers-min", type=int, default=100,
                    help="a seed needs at least this many >=4.0 ratings")
    pj.add_argument("--max-seeds", type=int, default=100)

    pc = sub.add_parser("catalog", help="bootstrap the TMDB mirror (key!)")
    pc.add_argument("--ml", required=True)
    pc.add_argument("--dsn", required=True)
    pc.add_argument("--api-key", required=True)
    pc.add_argument("--min-ratings", type=int, default=5,
                    help="universe = linked movies with >= N ratings")
    pc.add_argument("--rate", type=float, default=40.0)
    pc.add_argument("--embed", action="store_true",
                    help="compute MiniLM embeddings (needs the `bench` extra)")

    args = p.parse_args()
    if args.cmd == "judgments":
        cmd_judgments(args)
    else:
        asyncio.run(_catalog(args))


if __name__ == "__main__":
    main()
