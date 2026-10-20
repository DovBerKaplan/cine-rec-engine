---
name: cine-rec-engine-guide
description: Use this skill to understand or change the recommendation engine — find_similar flow, scoring, weights, user personalization, and the invariants that must not break
---

# cine-rec-engine — Engine Guide

## The one request → what happens

```python
rec = RecommendationService(); await rec.initialize(pool)
results = await rec.find_similar(155, limit=30)          # similar titles
results = await rec.recommend_for_user(user_id)           # personalized
```

`find_similar` pipeline (service.py):
1. Normalize seeds (int | [int] | [(id, media_type)], ≤1000).
2. Seed info: `get_movie_info_batch` (+ optional cinematic tags — degrade
   off if the table is missing).
3. On-demand TMDB sync for cold seeds (tmdb_recs.py; skipped with no
   `TMDB_API_KEY`, 4xx is non-retryable by design).
4. Recall, all channels gathered in parallel (queries.py): genre+
   popularity, keywords, cast, director (auteur), TMDB behavioral graph,
   KNN (needs pgvector + an embedding column), fine-tuned KNN.
   Missing infrastructure latches the channel off for the process.
5. Batch enrichment for candidates (one query per 3 fields).
6. **Scoring loop** — the hot path: per-entity `_entity_profile` built
   ONCE (canonical genres, keyword/cast/company sets, bigrams, scalars),
   then `_score_pair_fast` per (candidate, seed): single accumulation
   pass, no intermediate list. ~27µs/pair.
7. `_apply_user_filters` — watched/rated/disliked exclusion + **saga
   advancement** (watched Rocky I → recommend II), then media-type
   balancing (movie/tv ratio follows seeds, 10% minority floor), MMR
   diversification (incremental max-sim), noise if requested.

`recommend_for_user` (service.py): top-w_i seeds (+watchlist for
intent-only users) + user-vector ANN channel
(`generate_user_vector_candidates`) → the same `find_similar` path with
`user_id=` for hard filters.

## Weights: two layers

- `config.HEURISTIC_WEIGHTS` — hand-set coefficients over 28 named
  features (`FEATURE_NAMES`); the always-on baseline.
- `cine_rec_engine/weights.json` — 22 fitted coefficients; opt in with
  `CINE_REC_SCORER=learned`. Partial application: learned values override
  the heuristic per feature; missing features keep heuristic values.
- `ACTIVE_WEIGHTS = LEARNED or HEURISTIC` is the request default; a solo
  embedding model (`rec_model=`) overrides the cosine weight.

## Invariants (breaking these breaks the product)

1. **Golden parity**: `tests/golden_feature_vectors.json` pins
   `feature_vector` exactly and `_score_pair_fast` to ≤1 ULP. Changing
   the feature space ⇒ regenerate golden IN THE SAME COMMIT.
2. Feature order == `FEATURE_NAMES` == `HEURISTIC_WEIGHTS` key order ==
   `w_list` positions. `_score_pair_fast` indexes weights positionally.
3. Float sums over sets iterate **sorted** (hash-seed determinism).
4. Every optional channel degrades to empty — never raises.
5. (id, media_type) everywhere; bare ids collide across movie/tv.
6. Performance numbers in README are measured — re-measure before
   changing them (`benchmarks/bench_scoring.py`, `bench_e2e.py`).

## Where to touch what

- New recall channel → `queries.py` (return rows with `via` tag +
  `knn_similarity` if semantic) + merge in `find_similar` step 4.
- New feature → `FEATURE_NAMES` + `HEURISTIC_WEIGHTS` + both vector paths
  (`_vec_from_profiles` AND `_score_pair_fast`) + golden regen. Weights
  start at 0.0 in the heuristic; values come from fitting, not guesswork
  (see the privacy policy in AGENTS.md before documenting tuning).
- Per-user logic → `user_weights.py` (pure, spec §D — every branch is
  tested) or `watched.py` (exclusions), never inside the scoring loop.
