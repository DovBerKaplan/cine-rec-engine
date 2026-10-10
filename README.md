# Cine Rec Engine

> Extracted from a production self-hosted system — the same scorer serves real users daily. [Where this came from →](ORIGIN.md)

[![CI](https://github.com/DovBerKaplan/cine-rec-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DovBerKaplan/cine-rec-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Recommendations from your own database. No black-box API, no rented taste.**

**PostgreSQL in · ranked titles out · zero external calls on the hot path.**

**🖥️ [Try it in your browser — no install](https://dovberkaplan.github.io/cine-rec-engine/)** — every recommendation precomputed by the real engine, each row with its *why*, plus two user personas producing entirely different lists. Recommendations are picked from the bundled **830-title demo catalog**, not from all of TMDB — point the engine at your own mirror (ingest included) for the full catalog.

![demo](docs/demo.gif)

## Try it in 60 seconds — no API key

```bash
git clone https://github.com/DovBerKaplan/cine-rec-engine && cd cine-rec-engine/demo
docker compose up          # Postgres + 830 real titles + recommendations
```

That runs the full stack against a bundled 830-title demo catalog
(TMDB top-rated + their recommendation graphs, en-US, attribution
below) and prints, for The Dark Knight and Breaking Bad:

```
Because you watched  The Dark Knight  (2008)
  The Dark Knight Rises (2012)   why: same saga · same director
  The Batman (2022)              why: same style tags · shared keywords
  Memento (2000)                 why: same director · year window
```

A content-based recommendation engine for movies & series: give it one title
(or a user's whole watch history) and it returns similar titles, ranked by a
22-feature scorer whose weights were learned — not guessed.

```
   your data (TMDB mirror + user events)        what you get back
  ┌───────────────────────────────┐            ┌───────────────────────┐
  │ PostgreSQL                          │            │ ranked similar titles     │
  │  · tmdb_media + satellites          │            │  · score + why            │
  │  · (optional) embeddings            │──engine──►│  · movies/series mix      │
  │  · (optional) TMDB rec cache        │            │  · per-user filtering     │
  │  · your watch/rating events         │            │  · saga advancement       │
  └───────────────────────────────┘            └───────────────────────┘
```

## Quick start

```bash
pip install cine-rec-engine               # PyPI (or: pip install -e ".[pg,redis]")
psql -d yourdb -f docs/schema.sql         # the tables it expects
# no psql? same thing, idempotent, user layer included:
cine-rec init --dsn postgresql://user:pw@localhost/yourdb
```

```python
import asyncpg
from cine_rec_engine import RecommendationService

pool = await asyncpg.create_pool("postgresql://user:pw@localhost/yourdb")
rec = RecommendationService()
await rec.initialize(pool)

# Similar to one title
results = await rec.find_similar(155)  # The Dark Knight → [Batman Begins, …]

# Or the titles a user loved (movies AND series, auto-balanced)
results = await rec.find_similar(
    tmdb_id=[155, 27205, 1396],   # Dark Knight, Inception, Breaking Bad
    limit=60,
    user_id=42,                   # filters watched titles, advances sagas
)
```

Each result carries its evidence — score, matched features, media type —
so your UI can explain *why* it recommended something.

## Run it as a service

```bash
pip install "cine-rec-engine[serve]"
DATABASE_URL=postgresql://user:pw@localhost/yourdb \
    uvicorn cine_rec_engine.serve:app --port 8000
# or equivalently: cine-rec serve --port 8000
```

Or as a container (`docker build -t cine-rec-engine .`):

```bash
docker run -p 8000:8000 -e DATABASE_URL=postgresql://user:pw@host/db cine-rec-engine
```

Booting against an empty database? One env var applies the schema
first (idempotent — same files `cine-rec init` uses):

```bash
docker run -p 8000:8000 \
    -e DATABASE_URL=postgresql://user:pw@host/db \
    -e CINE_REC_AUTO_INIT=1 cine-rec-engine
```

For a self-host with compose (pool sizing, persistent feedback secret,
optional table-name map — all commented in place):

```bash
cd deploy && DATABASE_URL=postgresql://user:pw@host/db docker compose up -d
```

```bash
curl 'localhost:8000/similar?seed=155:movie,1396:tv&limit=10&user_id=42&year_min=1995&genres=80'
curl 'localhost:8000/for-user/42?include_why=true&limit=12&explore=0.15'
curl 'localhost:8000/page?user_id=42&rows=top_picks,because:155,discover,hidden_gems'
curl 'localhost:8000/by-text?q=like+Inception+but+darker'
curl -X POST localhost:8000/feedback -d '{"token":"<impression>","outcome":"watch","tmdb_id":111}'
curl 'localhost:8000/health'
curl 'localhost:8000/metrics'   # latency histogram, cache hit rate, channel coverage (Prometheus text)
```

`/similar` is item-to-item — `user_id` optionally filters watched
titles, advances sagas, and gently tilts near-ties toward the user
(boost-only; the seed stays primary). Session filters (`year_min/max`,
`genres`, `exclude_genres`, `max_runtime`) apply at the recall level and
hash into the cache key. `/for-user` is the personalized row;
`explore=` reserves a deterministic discovery slice (novel AND relevant
— never noise). `/page` composes themed rows in one call — concurrent
execution, first-row-wins dedup, a stateless impression token per row,
and a `discover` row of adjacent-cluster titles. `/by-text` is the
no-history cold start (bring your own encoder via
`CINE_REC_ENCODER="module:function"`; falls back honestly otherwise).
`/feedback` closes the loop: verified impression token + outcome
(click/watch/skip/dislike) adjusts the user's weights transactionally —
no trainer, no cron. User vectors rebuild on demand behind every
endpoint. `model=e5e` runs a whole request in one embedding space.

## Configuration

Everything is environment-driven — no code edits, no config class to
instantiate. Vars marked ⚙ are read once at process start.

| Variable | Default | What it controls |
|---|---|---|
| `DATABASE_URL` (or `CINE_REC_DATABASE_URL`) | — | the PostgreSQL DSN (required) |
| `CINE_REC_SCHEMA_MAP` | off | path to a JSON **table-name map** (below) |
| `CINE_REC_TABLE_<NAME>` | off | per-table override, e.g. `CINE_REC_TABLE_TMDB_MEDIA=app_media` (wins over the file) |
| `CINE_REC_AUTO_INIT` | off | `1` = apply the schema on serve boot, idempotent |
| `CINE_REC_POOL_SIZE` | 20 | asyncpg pool `max_size` (small pools dominate cold latency) |
| `CINE_REC_PORT` / `CINE_REC_HOST` | 8000 / 0.0.0.0 | for `cine-rec serve` and the Docker image |
| ⚙ `CINE_REC_EMBEDDING` | `original` | embedding column/space (`CINE_REC_EMBEDDING=minilm` → `embedding_minilm`) |
| ⚙ `CINE_REC_COSINE_BLEND` | — | blend several embedding columns (`col:weight,col:weight`) |
| ⚙ `CINE_REC_SCORER` | heuristic | `learned` = fitted weights (bundled, or `CINE_REC_WEIGHTS`) |
| ⚙ `CINE_REC_WEIGHTS` | bundled | path to your own fitted artifact (`eval/tune_weights.py`) |
| `CINE_REC_ENCODER` | — | `module:function` providing `/by-text` embeddings |
| `CINE_REC_REDIS_URL` | off | Redis result/history cache (falls back to in-process) |
| `CINE_REC_IMPRESSION_SECRET` | random | persistent feedback tokens across restarts |
| `CINE_REC_CORS_ORIGINS` | off | comma-separated allow-list |
| `TMDB_API_KEY` | — | behavioral-graph sync (`ingest/`, `tmdb_recs.py`) |
| ⚙ `CINE_REC_USER_TILT_ALPHA` / `CINE_REC_EXPLORE_SHARE` / `CINE_REC_SKIP_DECAY` / `CINE_REC_USER_VECTOR_MAX_AGE_HOURS` | 0.15 / 0.12 / 0.9 / 24 | personalization tuning |

### Bring your own tables

Every table and view the engine (and the ingest loader) touches has one
logical name — all 41 are listed in `cine_rec_engine/tables.py`
(`LOGICAL_TABLES`). Point them at your own names with a JSON map:

```json
{ "tmdb_media": "app_media",
  "user_watches": "app_schema.user_watches",
  "user_watch_events": "app_events" }
```

```bash
CINE_REC_SCHEMA_MAP=/etc/cine-rec/tables.json cine-rec serve
```

Rules: physical names are lower-case identifiers, optionally
schema-qualified; unknown keys are rejected with the valid list; and an
explicitly mapped table that doesn't exist **fails startup loudly** — a
misconfigured map must never look like "no recommendations" (missing
default-named tables keep the usual graceful degradation instead).

Names map; **columns stay the contract**. If your existing table has
different column names too, write one thin view with aliases — the
§6 compatibility views in `docs/schema.sql` are exactly this pattern
and the reference for the shape each logical name expects.

## Why another recommender

| | Hosted rec APIs | Collaborative-filtering stacks | **Cine Rec Engine** |
|---|---|---|---|
| Works on day one (no user base) | ✅ | ❌ cold start | ✅ content-based |
| Your data stays yours | ❌ | ✅ | ✅ |
| Needs users' rating matrix | ❌ | ✅ | ❌ — metadata + your event table |
| Explainable per-title | partial | partial | ✅ 22 named features |
| Runs offline / on-prem | ❌ | ✅ | ✅ one Postgres |
| Embedding models included | — | — | ❌ **bring your own** (see below) |

The engine separates **recall** (SQL: genres, keywords, cast, crew,
companies, networks, collections, cached TMDB behavioral recs, optional
pgvector KNN) from **ranking** (a logistic scorer over 22 features with
published, fitted weights) from **models** (your embedding encoders —
deliberately not shipped).

## Measured against baselines

On the bundled 830-title demo catalog with a 160-pair consensus
judgment file (`eval/judgments.jsonl`, regenerated anytime by
`eval/build_judgments.py`, scored by `eval/eval.py`; **a regression
gate, not a power claim** — every pair is included only when two
independent methods already agree it is obvious: a structured-metadata
rule AND the embedding space):

| method | pairwise acc. | NDCG@10 |
|---|---|---|
| TMDB similar (behavioral graph) | 0.88 | **0.78** |
| MiniLM cosine over overviews | **0.99** | 0.26 |
| **this engine (learned 22-feature scorer)** | 0.88 | 0.34 |

Full honesty: single-signal baselines score high here BY CONSTRUCTION —
half the inclusion rule is cosine agreement and the goods sit in the
TMDB graph's head, so each corresponding baseline aces its own half.
The file exists to gate regressions: CI fails if the engine drops
below pairwise 0.75 / NDCG@10 0.28 (`--min-pairwise --min-ndcg`),
floors set measured-minus-margin. On the old 10 hand pairs the spread
was similar; the expanded file trades discriminator power for
regression coverage across genres, decades and both media.

Honest caveats: the pool is small (830 titles, recommendation-closed,
with bundled MiniLM embeddings — `demo/data/`), recall runs in the same
same-medium mode the bot uses, and the judgments are mechanically
generated consensus, not human taste. Bring your own judgments file —
the harness is in the repo. The same gates run on a laptop via
`make test-integration` (throwaway pgvector + the live CI-class suite).

## Feature highlights

- **Multi-seed blending** — one title or a thousand; per-seed weights;
  movie/series ratio follows the seed mix (90/10 cap so a minority is
  never silenced).
- **Saga-aware** — collection members chain: watched *Rocky I* → recommends
  *Rocky II*, not *Rocky I* again. Whole-saga watchers graduate out.
- **Per-user personalization** — `recommend_for_user()`: raw watch
  events → per-title weights (completion, series depth, recency,
  engagement) → a normalized user vector (rebuilt automatically when
  stale — no cron needed) → an ANN recall channel through the
  same LTR scorer, with watched/rated/disliked hard-filtered.
  See `docs/personalization.md`.
- **Auteur recall** — director/writer/composer/DP channels with decay
  (someone's 8th film matters less than their 2nd).
- **Popularity guardrails** — vote floors per media type kill
  "high rating, 12 votes" noise; an action-popularity leak term keeps
  Marvel out of every list.
- **MMR diversification** — optional re-rank so one franchise doesn't
  take five consecutive slots.
- **Deterministic or noisy** — `randomness=0.0` is reproducible; dial it
  up for exploration without burying the top-3.
- **Degrades gracefully** — no embeddings? no TMDB rec cache? no Redis?
  Those channels switch off; the rest of the engine keeps working.

## The published weights

`cine_rec_engine/weights.json` — 22 features fitted with pairwise
logistic regression (6,650 training pairs). Opt in with
`CINE_REC_SCORER=learned`; features the artifact doesn't cover keep
their heuristic coefficients (partial application). A few, to set the
scale:

| Feature | Weight |
|---|---|
| `tmdb_rec_decay` (behavioral signal, rank-decayed) | 13.74 |
| `composer_match` | 5.54 |
| `cosine_sim` (overview embedding similarity) | 5.13 |
| `keyword_sim` | 4.35 |
| `writer_match` | 4.08 |
| `director_match` | 3.50 |
| `shared_collection` | 2.51 |
| `medium_mismatch` (movie↔tv penalty) | −0.52 |

Also in the repo: the evaluation harness and the demo judgments. Not
included: the larger labeled training sets the weights were fitted on and
the optional narrative-tag enrichment — the fitted coefficients
themselves are published in full.

### Fit your own

The same schema is a self-service contract. Collect preference judgments
in the public format — one line per seed with the titles that should
rank above and below (shape: `eval/judgments.jsonl`) — then:

```bash
python eval/tune_weights.py --dsn postgresql://... \
    --judgments mine.jsonl --out my-weights.json
python eval/eval.py --dsn postgresql://... --judgments mine.jsonl   # sanity gate
CINE_REC_SCORER=learned CINE_REC_WEIGHTS=/abs/my-weights.json cine-rec serve
```

The fit is deterministic pairwise logistic regression anchored to the
heuristic prior (L2 on the distance from it), the holdout split is whole
seeds, and the report compares against the heuristic baseline — you see
whether your judgments actually earned each weight move. Small sets
overfit; treat big moves with suspicion. (The bundled demo judgments are
mechanical consensus, not taste: the heuristic already aces them, and
the honest fitted result there is +0.000.)

## Feeding it data

Bring a local TMDB mirror — the built-in `ingest/` loader builds it from
TMDB's official daily ID exports (`python -m ingest.cli bootstrap`, then a
daily `refresh` off `/changes`; `en-US` only, adult-filtered, one API call
per title, upserts by key). The catalog schema splits movies and TV into
two fact tables with independent id spaces — exactly like TMDB — with
compatibility views serving the engine unchanged. `docs/data.md` has the
full contract. For user data, feed `user_watch_events` from your player
(`docs/personalization.md`). Already recording events in your own
tables? Rename, don't re-plumb — the [table map](#bring-your-own-tables)
above points the engine at your names with zero code changes.

## Repo layout

```
cine_rec_engine/    the engine (recall · scoring · ranking · weights)
ingest/             built-in TMDB mirror loader (bootstrap + daily refresh)
demo/               one-command demo: 830 bundled titles, no API key
deploy/             self-host compose: env knobs + optional table map
eval/               pairwise/NDCG harness + the demo judgments
models/             embedding sidecar — YOUR encoders plug in here
docs/schema.sql     canonical split schema (movies | tv) + engine views
docs/user_data.sql  user layer: events, feedback, stats, vectors
docs/data.md        one-call ingest, filters, rate limits, acceptance rule
docs/personalization.md  w_i formulas, user vectors, recommend_for_user
examples/           runnable snippets
tests/              offline unit tests (no DB needed)
```

## Requirements & support

- Python ≥ 3.10, PostgreSQL ≥ 14, asyncpg
- Optional extras: `pip install ".[pg]"` (pgvector — KNN recall),
  `".[redis]"` (result + history caching), `TMDB_API_KEY` (behavioral
  rec sync), sentence-embedding encoders (KNN + cosine features)

## Performance

Measured on the synthetic 25k-title benchmark (`benchmarks/bench_e2e.py`,
PostgreSQL 16, 2-core container): **cold single-seed ≈ 45 ms · warm
(cache) 0.4 ms · 5-seed 322 ms**. The scorer runs ~37k (seed, candidate)
pairs/second/core with outputs within 1 float ULP of the reference
implementation (golden-vector tests). `benchmarks/bench_scoring.py`
reproduces the hot-loop number without any database.

Personalized rows (`recommend_for_user`, 15–18 weighted seeds, same
catalog, `benchmarks/bench_personalization.py`): **first request ≈ 1.2 s**
(includes the inline vector rebuild; once per freshness window) ·
**repeat request ≈ 0.2–0.3 s** (result cache hit; the ANN query is
skipped when the list is cached; ~9 small per-user queries remain —
seeds, freshness probe, watched/dislike exclusions, saga filters) ·
**cold, other users ≈ 0.9 s median**. A cold request fires ~4 recall
queries per seed (the two KNN channels share one query; semantic
similarity is one batched cross-join for all seeds) — size the
connection pool `max_size ≥ 20` or the queue dominates: at
`max_size=10` cold requests measured ~3× slower than at 25. The vector
recalls are ANN-indexed (hnsw/ivfflat on your embedding column); without
an index they degrade to a table scan but still work. Keep the
`tmdb_movies_collection_idx` partial index from `docs/schema.sql` in
place — the saga filter pass needs it to stay off a catalog scan.

## Roadmap

See [ROADMAP.md](ROADMAP.md) — personalization shipped in 0.3; v0.7 ships on-demand user vectors + the HTTP API/Docker service; next: controlled personalized item-to-item.

## License

[MIT](LICENSE) — engine code and the fitted weights.
TMDB metadata itself is © TMDb — the engine never redistributes it.
