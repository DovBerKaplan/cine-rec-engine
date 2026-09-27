# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [0.4.0] — 2026-09-27

### Added — 60-second demo, evaluation, packaging
- `demo/` — one-command demo with NO TMDB key: `docker compose up` starts
  Postgres + pgvector, seeds 400 real top-rated titles (bundled 5MB dump,
  fetched by this repo's own ingest), and prints recommendations with the
  WHY (top contributing features per result). Verified from a clean
  machine.
- `docs/demo.gif` — the demo, as a terminal recording.
- `eval/` — offline harness (pairwise accuracy + NDCG@10) with three
  methods on hand-curated judgments; engine 0.83/0.36 vs cosine 0.63/0.04
  and the TMDB graph 0.09/0.13 on the demo pool.
- README: demo up top, measured-baselines table, PyPI install path,
  friendlier framing of what ships and what doesn't.

## [0.3.0] — 2026-09-27

### Added — personalization end-to-end (spec v0.2 §B–§J)
- `docs/user_data.sql` — the user layer: raw `user_watch_events`,
  `user_feedback`, derived `user_title_stats` / `user_stats` /
  `user_genre_stats` / `user_vectors` (multi-persona storage), and the
  legacy `user_watches` contract as a DERIVED view (schema.sql §7
  slimmed accordingly).
- `cine_rec_engine/user_weights.py` — the §D formulas: movie completion
  scoring with pause penalty, series depth ladder (3-episode rule,
  floors, last-episode exits), drop-off filters, 30-day recency
  half-life, rewatch/favorite/watchlist/dislike engagement, clip at 1.5.
  28 offline tests pin every branch.
- `cine_rec_engine/user_stats.py` — `record_event` (single transaction:
  event + title stats + w_i + last_event_at, §J), `record_feedback`,
  `refresh_user_stats` (full §C profile + genre distribution),
  `recompute_weights`, `nightly_recompute` (§G.2).
- `cine_rec_engine/user_vector.py` — weighted aggregation + L2 (§E.1),
  no fake zero vectors, storage with `vector_updated_at`/`persona_count`.
- Engine §F: `recommend_for_user()` — top-w_i seeds + a user-vector ANN
  recall channel merged through the existing LTR scorer; explicit
  dislikes hard-filtered alongside watched/rated.
- `benchmarks/smoke_personalization.py` — the live end-to-end smoke
  (events → weights → stats → vector → filtered recommendations),
  verified against PostgreSQL 16 + pgvector.

### Fixed (found by the live smoke)
- pgvector extension is named `vector`, not `pgvector`.
- Aggregation SQL: column ambiguity, LATERAL last-event join, FILTER
  clauses require WHERE, aggregates are illegal inside FILTER.
- rewatch semantics: a re-view of the SAME unit — episodes after the
  first are progress, not rewatch (series scores were inflated ~1.5×).
- `dropped` stays §D.1-pure (a disliked-but-completed movie is completed,
  not dropped).
- pgvector codec registration on every connection that reads vectors.

## [0.2.1] — 2026-09-27

### Fixed (ingest — review findings)
- TV `append_to_response` lost the separator before `external_ids`
  (`…recommendationsexternal_ids,`) — tv `imdb_id` was never fetched. The
  list is now built programmatically (separator bug structurally impossible).
- Bridges MIRROR the payload instead of accumulating: rows removed from
  TMDB (recast actors, dropped keywords/networks/companies) are deleted;
  cast/crew changes upsert (character name, department). Behavioral recs
  prune ranks beyond the fresh page.
- `refresh` is the full daily loop: export diff for ids that crossed the
  popularity threshold after debut (Changes never lists them) → then
  `/changes` pages through the SAME adult + popularity gate as bootstrap.
- `bootstrap` skips ids already in the DB (resumable; `--force` to
  refetch) and ingests concurrently (default 8 queue workers) under the
  token-bucket rate limiter — the serial per-id await made bulk loads
  crawl regardless of the rate budget.

## [0.2.0] — 2026-09-27

### Performance (measured on the 25k-title synthetic bench, PostgreSQL 16)

| scenario | before | after |
|---|---|---|
| cold single-seed `find_similar` (median) | 2,426 ms | **45 ms** (54×) |
| 5-seed `find_similar` | 2,058 ms | **322 ms** (6.4×) |
| warm (cache hit) | 0.4 ms | 0.4 ms |
| scoring loop | 98.2 µs/pair | **26.7 µs/pair** (3.7×) |

- **Scoring loop**: per-entity profiles built once instead of per (candidate,
  seed) pair; lazy imports hoisted; single accumulation pass. Outputs
  within 1 float ULP of the original (golden-vector tests; set-iteration
  order makes float sums hash-seed dependent, so exact bits are not
  portable — sums are now deterministic via sorted iteration).
- **MMR**: incremental max-similarity — n·k pair computations instead of
  k²·n/2, pick order provably identical (30/30 random-trial parity).
- **TMDB sync latency bomb**: 4xx (≠429) responses are non-retryable — an
  absent/invalid API key used to burn ~2s of retry sleeps per cold seed;
  no-key now skips HTTP entirely.
- **KNN channels** latch off after the first missing-column/extension error
  instead of re-running the failing SQL per seed.
- Benchmarks shipped: `benchmarks/bench_scoring.py` (pure-Python hot loop,
  no DB) and `benchmarks/bench_e2e.py` (seeds + drops a synthetic 25k-title
  catalog in any Postgres).

## [0.1.3] — 2026-09-27

### Fixed
- `find_similar(user_id=...)` — the documented per-user filtering is now a
  real parameter: loads the user's watched ∪ rated pairs from
  `user_watches`/`title_ratings` and merges them into `exclude`
  (degrades to the explicit `exclude` on lookup failure). Previously
  `user_id` was documented in the README but absent from the signature.
- Learned weights now apply **partially**: `CINE_REC_SCORER=learned`
  blends the artifact's features over the heuristic baseline instead of
  rejecting the shipped 22/28-feature `weights.json` outright — the
  published artifact is usable, missing features keep heuristic values.
- `pgvector` import guarded inside the KNN recall path — the channel now
  degrades to empty when the optional package is absent instead of
  raising ImportError.
- Redis cache: probed once per process (no ping per call); a dead Redis
  falls through to the in-process TTL cache.
- Export download streams gzip lines instead of buffering the full file
  (the movie export decompresses to hundreds of MB).
- Ingest pacing: real token-bucket rate limiter (≤ req/s), not a
  concurrency semaphore; people rows upsert in one statement.

### Fixed (docs)
- Stale `engine/...` paths in README/models/schema comments → the real
  `cine_rec_engine/` package layout; pgvector import in the cosine
  channel guarded like the KNN channel.

### Changed
- `--genres-first` → `--skip-genres` (the old flag was always-true).
- cron example no longer relies on shell `$VARS` cron never expands.

## [0.1.2] — 2026-09-27

### Fixed
- `feature_vector` produced a 27-entry vector against 28 feature names:
  `narrative_match` was computed but never emitted, so from index 22 every
  mood feature was multiplied by its neighbor's weight (`tone_compat` at
  narrative's 1.5 instead of its own 5.0, etc.). The vector now carries
  all 28 features; weight/feature alignment is pinned by tests.
  Synced from the upstream production deployment of this engine.

## [0.1.1] — 2026-09-27

### Added
- Canonical split mirror schema: movies and TV as two fact tables with
  independent TMDB id spaces, shared dimensions, per-medium genre lists
  and bridges, per-medium behavioral recommendation tables — plus
  compatibility views that serve the engine unchanged (reads through
  views; the engine's upsert target stays a physical table).
- Built-in ingest (`ingest/`): daily-export discovery with adult +
  popularity filters, one-call-per-title loading via
  `append_to_response`, cast top-5 / crew job whitelist, genre-list sync,
  ≤40 req/s with Retry-After backoff, upsert-by-PK writes, the acceptance
  rule, and a `refresh` pass over `/movie/changes` + `/tv/changes`.
- 14 offline tests for the ingest logic (54 total).

## [0.1.0] — 2026-09-27

First public release: the similar-titles engine, extracted as-is from a
production system that serves it daily.

### Added
- `RecommendationService.find_similar()` — single-seed and multi-seed
  (up to 1,000) similar-title retrieval with per-seed weights,
  movie/series auto-balancing, MMR diversification, and noise control.
- SQL recall channels: genre+popularity, keywords, cast, crew (auteur with
  decay), companies, networks, collections, cached TMDB behavioral
  recommendations, pgvector KNN (optional).
- 22-feature learned scorer; fitted weights published in
  `cine_rec_engine/weights.json` (pairwise logistic regression, 6,650 pairs).
- Per-user filtering: watched/rated exclusions and saga advancement over
  the schema's `user_watches` / `title_ratings` tables (swap in your own
  event tables by editing one SQL constant).
- Graceful degradation for every optional input: embeddings, Redis,
  behavioral cache, cinematic tags — missing pieces disable features,
  never queries.
- In-process TTL cache fallback when Redis is absent.
- `docs/schema.sql` (full expected schema), `docs/data.md` (building a
  daily-refreshed local TMDB mirror), offline test suite (40 tests).
