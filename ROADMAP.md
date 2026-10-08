# Roadmap

What's shipping next, in rough order. Dates are intentions, not promises —
the engine is production-hardened, the packaging is young.

## v0.2 — history-based personalization — ✅ SHIPPED (0.3.0/0.4.0)

`recommend_for_user(user_id)` is live: raw watch events → per-title
weights (completion, series depth ladder, recency half-life, engagement)
→ a normalized user vector → an ANN recall channel through the same LTR
scorer, with dislikes hard-filtered. See `docs/personalization.md`.

Still open from the original v0.2 sketch:
- **Exploration budget** — X% of every list reserved for adjacent-genre
  discovery, so profiles never ossify into one cluster.

## v0.7 — out-of-the-box consumption + serving — core ✅ SHIPPED (0.10.0)

- ✅ **Plug-and-play "Recommended" row** — `recommend_for_user` rebuilds
  a stale user vector inline (`ensure_user_vector`, bounded by
  `USER_VECTOR_MAX_AGE_HOURS`); watched/disliked titles never surface
  (regression-pinned).
- ✅ **Per-model user vectors, on demand** — one model space per request
  end to end: the vector is ensured, ANN-recalled, and cosine-scored in
  the same `model_spaces.REC_MODELS` column. No nightly batch anywhere
  in the architecture — `nightly_recompute` remains optional fleet
  tooling.
- ✅ **Lightweight serving** — `pip install ".[serve]"` → FastAPI
  `/similar`, `/for-user`, `/health` over the existing cache, plus a
  single-container Docker image (`Dockerfile`).

Still open in this milestone:

- ✅ **Personalized item-to-item ("More like this"), controlled** —
  `find_similar(seed, user_id=…)` now applies a gentle taste tilt after
  the hard filters: boost-only, capped at `USER_TILT_ALPHA` (default
  0.15), per-request and post-cache (the shared payload stays
  user-agnostic). The seed stays primary — near-ties break toward the
  user, the candidate pool never changes. Verified live against the
  Docker container + demo Postgres.

What comes next lives in [docs/rfc-next-evolution.md](docs/rfc-next-evolution.md)
— page-of-rows delivery (`/page`), a deterministic exploration budget,
session shaping (filter params + natural-language cold start),
impression tokens & outcome feedback through the existing §D weights,
and regression hardening (bigger judgments, local integration harness).

## Later — see [docs/rfc-next-evolution.md](docs/rfc-next-evolution.md) for the full design

- M1 (v0.8) ✅ shipped (0.10.0): `/page` row composition · session filter params · impression tokens
- M2 (v0.9) ✅ shipped (0.10.0): exploration budget + `discover` row · `/feedback` outcome ingestion (bounded skip decay, no trainer) · `/by-text` NL cold start · shared page context (warm /page ~2× faster)
- M3 (v0.10) ✅ shipped (0.10.0): 160-pair judgments + raised CI gates · local integration harness · metrics
- **Schema migrator** — versions the expected schema so upgrades are
  `engine.migrate(pool)`.

## Dropped — public learning loop

The weight-fitting harness (trainer CLI over graded pairs) is out of
scope. The fitted `weights.json` stays the published artifact; a public
trainer over our feature space is deliberately not something we ship.

## Parking lot (research)

- Hybrid collaborative filtering once user bases get big enough.
- Cross-lingual seeds (Hebrew title in → English-catalog results out).
- Fairness/serendipity audits over the popularity guardrails.
