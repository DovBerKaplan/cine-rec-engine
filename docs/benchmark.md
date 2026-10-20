# The public MovieLens benchmark

An external, reproducible quality check that does not depend on this
repo's own tuning: [MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/)
ratings converted into preference judgments, scored by this engine and
four simple baselines on the same catalog.

## What the judgments are

`eval/ml_judgments.jsonl` — 150 seeds, 10 good + 10 bad each
(2.25M good×bad pairs), derived offline and deterministically by
`eval/build_ml_bench.py judgments`:

- a **seed** is a movie at least 50 MovieLens users rated ≥ 4.0;
- **good** = the ten other movies most agreed-upon (≥ 5 agreements,
  ties by id) among users who rated the seed ≥ 4.0;
- **bad** = the ten movies that same loving cohort most agreed it
  rated ≤ 2.5.

This is *collaborative* ground truth — what people who loved the seed
also loved vs. disliked — independent of any content features, which is
exactly what a content-based engine should be measured against.

## Reproducing

```bash
# 1. judgments (offline, no key) — already committed, this regenerates:
python eval/build_ml_bench.py judgments --ml ml-latest-small.zip \
    --seed-lovers-min 50 --max-seeds 150 --out eval/ml_judgments.jsonl

# 2. catalog — bootstraps a DB with the TMDB mirror for the benchmark
#    universe (~every linked movie with ≥5 ratings; one API call per
#    title, a couple of minutes) + MiniLM embeddings:
pip install -e ".[pg,bench]"
python eval/build_ml_bench.py catalog --ml ml-latest-small.zip \
    --dsn postgresql://demo:demo@localhost:5432/mlbench \
    --api-key $TMDB_API_KEY --embed

# 3. score — five methods, one table:
python eval/eval.py --dsn postgresql://demo:demo@localhost:5432/mlbench \
    --judgments eval/ml_judgments.jsonl
```

Methods: `engine`, `tmdb_sim` (TMDB's own recommendations graph),
`minilm_cosine` (overview embeddings), `popularity` (vote_count),
`keywords` (keyword-set Jaccard).

## Attribution

- MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The
  MovieLens Datasets: History and Context. ACM Transactions on
  Interactive Intelligent Systems (TiiS). The committed judgments file
  contains TMDB ids and good/bad labels derived from that data — no
  MovieLens content and no TMDb metadata is redistributed.
- The catalog step fetches from the TMDB API into YOUR database: this
  product uses the TMDB API but is not endorsed or certified by TMDb.
