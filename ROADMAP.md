# Roadmap

What's shipping next, in rough order. Dates are intentions, not promises —
the engine is production-hardened, the packaging is young.

## v0.2 — history-based personalization (next)

The engine already accepts a user's seeds; v0.2 turns it into a real
`recommend_for_user(user_id)`:

- **Taste profile from events** — `user_watches` + `title_ratings` folded
  into seed weights automatically: loved ≠ watched; a 10/10 outweighs a
  click. Negative signals (DNF'd series, low ratings) push clusters away.
- **Recency decay** — what a user watched last month says more than last
  year; exponential half-life per event.
- **Exploration budget** — X% of every list reserved for adjacent-genre
  discovery, so profiles never ossify into one cluster.

## v0.3 — session & context

- **Session-mode recommendation** — one request, multiple filters,
  conversation-shaped (genre pins, year windows, "more like this but
  funnier").
- **Cold-start seeds from a single text query** — natural-language
  "something like Inception but sadder" → embedding-space anchor →
  standard recall/rank.

## v0.4 — learning loop

- **Weight fitting harness** — the pairwise-logistic trainer as a
  documented CLI: bring your own graded pairs, fit your own
  `weights.json`. (Our datasets stay private; the *method* opens up.)
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
