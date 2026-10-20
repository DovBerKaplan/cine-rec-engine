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
`keywords` (keyword-set Jaccard). The scoring run needs
`CINE_REC_EMBEDDING=minilm` so the engine reads the benchmark's
embedding column.

## Results (2026-10-21, catalog of 3,636 titles)

| method | pairwise acc. | NDCG@10 | coverage@10 | novelty@10 |
|---|---|---|---|---|
| tmdb_sim | 0.02 | 0.06 | 0.18 | 11.4 |
| minilm_cosine | 0.60 | 0.03 | 0.28 | 12.1 |
| popularity | **0.96** | **0.16** | 0.00 | 8.5 |
| keywords | 0.33 | 0.03 | 0.27 | 11.8 |
| engine | 0.45 | 0.05 | 0.26 | 12.1 |

How to read this — the ground truth is **mass agreement**: movies the
seed's loving cohort also loved, vs. movies that cohort disliked.

- **Popularity wins by construction.** Cohort-agreement picks famous
  titles (every good/bad id here passes a 500-vote floor), so ranking
  by vote_count is nearly the answer key. Its coverage@10 of 0.00 is
  the tell: the *same* ten titles for every seed — precisely the
  block-repeat failure mode list diversity exists to avoid.
- **TMDB's own graph scores 0.02** — below random: famous-but-disliked
  titles sit in its "people also liked" head, so it actively prefers
  the bad side.
- **The engine measures a different thing.** It optimizes taste
  similarity (auteur signals, keywords, plot structure) with diversity
  reranking — its home ground is the taste-judgment eval in the README,
  where the single-signal baselines win their own construction instead.
  Pure plot-cosine (0.60) aligns with mass agreement better than the
  engine's taste mix; that is a real property of content-based taste
  ranking, reported as-is.
- NDCG@10 discriminates weakly here (10 goods in a 3,636-title pool —
  even an oracle caps low); pairwise and coverage carry the signal.

## Attribution

- MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The
  MovieLens Datasets: History and Context. ACM Transactions on
  Interactive Intelligent Systems (TiiS). The committed judgments file
  contains TMDB ids and good/bad labels derived from that data — no
  MovieLens content and no TMDb metadata is redistributed.
- The catalog step fetches from the TMDB API into YOUR database: this
  product uses the TMDB API but is not endorsed or certified by TMDb.
