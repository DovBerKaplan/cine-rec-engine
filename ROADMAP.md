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

## v0.3 — session & context

- **Session-mode recommendation** — one request, multiple filters,
  conversation-shaped (genre pins, year windows, "more like this but
  funnier").
- **Cold-start seeds from a single text query** — natural-language
  "something like Inception but sadder" → embedding-space anchor →
  standard recall/rank.

## v0.5 — learning loop

- **Weight fitting harness** — the pairwise-logistic trainer as a
  documented CLI: bring your own graded pairs, fit your own
  `weights.json`.
- **Online feedback signals** — recommendation→click outcomes feed the
  next fit as additional pairs.

## v1.0 — serving

- **FastAPI microservice** — `/similar`, `/for-user`, `/explain` with
  batching and caching, Docker image.
- **Health & metrics** — recall-channel coverage, latency histograms,
  cache hit rates.
- **Schema migrator** — versions the expected schema so upgrades are
  `engine.migrate(pool)`.

## Parking lot (research)

- Hybrid collaborative filtering once user bases get big enough.
- Cross-lingual seeds (Hebrew title in → English-catalog results out).
- Fairness/serendipity audits over the popularity guardrails.
