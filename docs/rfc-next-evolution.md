# RFC — Next evolution: page delivery, exploration, session control, feedback loop

Status: draft (2026-10-08) · scope: v0.8–v0.10 · supersedes the "Later"
section of ROADMAP.md (kept there as a summary).
**M1 shipped (0.10.0): /page, session filters, impression tokens.**
**M2 shipped (0.10.0): exploration budget + discover row, /feedback
outcome ingestion, /by-text cold start, shared page context — all
verified live against the Docker container + demo catalog.**
**M3 shipped (0.10.0): 160-pair consensus judgments + raised CI
gates (pairwise ≥ 0.75, NDCG@10 ≥ 0.28 — see the measured note in §5),
`make test-integration` / `pytest -m integration` local harness, /metrics
(channel coverage, latency histogram, cache hit rate), and the six
reserved cinematic features documented in FEATURE_NAMES + the
weights.json schema note.**

Standing principles every phase inherits:
- **Zero-cron** — every derived artifact self-heals on demand; batch jobs
  are optional fleet tooling, never correctness dependencies.
- **Seed-primary personalization** — user signals tilt and filter; they
  never replace the seed's context (the item-to-item tilt is the model:
  boost-only, capped, post-cache).
- **User-agnostic shared cache** — anything user-specific happens after
  the cache layer, on the request's copy.
- **Degrade, never fail** — each new optional input (encoder, feedback
  table) switches its channel off, never 500s.
- **No public trainer** — feedback adjusts the §D affinity weights in
  the operator's own DB; nothing re-fits the published scorer weights.

## 1. Slate & row composition — `/page` (M1)

A streaming UI consumes a page of rows, not a flat list. N rows must
not mean N recall round-trips.

- **Contract**: `GET /page?user_id=…&rows=top_picks,because:155,hidden_gems,trending:80`
  (each row spec = kind + params). Response: ordered rows, each with its
  titles, why, and an impression token (§4).
- **Row kinds v1**: `top_picks` (the recommend_for_user core), `because:{seed}`
  (find_similar + user filters + tilt), `hidden_gems` (quality floor +
  bottom-half popularity, ranked by score), `trending:{genre_id}`.
- **Shared recall**: one recall phase per page — union the row seed sets,
  recall once (channels already batch), then each row is an in-memory
  slice/re-rank. Row payloads that exist in cache are reused (a
  `because:` row is just the existing find_similar cache key).
- **Cross-row dedup**: first-row-wins with an exclusion mask; a title
  appears once per page unless a row opts in (saga rows may repeat).
- **Failures**: a failing row degrades to an omitted row, never a failed
  page.

## 2. Exploration budget & serendipity (M2)

Filter bubbles are an ossification failure, and noise is not serendipity.
The budget targets **novel AND relevant** only.

- **EXPLORE_SHARE** (default 0.12, env `CINE_REC_EXPLORE_SHARE`,
  per-request `explore=` override) of every personalized list/page.
- **Eligibility** (deterministic): genre set disjoint from the user's
  top-3 genre clusters (novel) AND cosine-to-user-vector ≥ the median
  affinity of the main list (relevant). No RNG — same input, same slice.
- **Placement**: fixed downstream slots (every k-th tail position, top-3
  pinned) so the main list keeps its shape; on `/page` this becomes one
  dedicated discovery row + an in-row slice on top_picks.
- **Measurement**: eval harness gains coverage/novelty columns; the CI
  gate becomes "no pairwise/NDCG regression AND coverage does not drop".

## 3. Session shaping & dynamic context (M1 filters, M2 anchor)

- **Phase A — ad-hoc constraints** on `/similar` and `/for-user`:
  `year_min`/`year_max`, `genre_include`/`genre_exclude` (comma genre
  ids), `max_runtime`. Applied as recall-level SQL predicates (not
  post-hoc slicing — limit semantics stay honest) and folded into the
  cache-key fingerprint.
- **Phase B — natural-language cold start**: `GET /by-text?q=like
  Inception but darker`. The encoder stays bring-your-own (models/
  sidecar contract: one `encode(text) -> vector` entry point); the
  anchor vector feeds the existing KNN recall and the scorer through
  the **seed_info_override virtual-seed seam** — no watch history
  required. No encoder configured → honest degrade to bigram-cosine
  fallback, documented as such.

## 4. Exposure logging & outcome feedback (M2 tokens, M3 tuning)

Closing the loop without a trainer: outcomes flow into the EXISTING
§D/§J weight machinery — the same transactional w_i updates watch events
already use.

- **Impression tokens**: each served row/page carries an HMAC-signed,
  stateless token (user, row kind, seeds, served ids+scores, ts, ttl).
  No server-side session store; verification is pure.
- **Ingestion**: `POST /feedback {token, outcome, tmdb_id?}` with
  outcomes `click | watch | skip | dislike` mapped to:
  click → `record_feedback(kind="click")`; watch/completion →
  `record_event` (w_i adjusts in-transaction); skip/dislike →
  `record_feedback` (skip applies a bounded recency-decay, never an
  instant bury).
- **Latency follow-up from M1**: fetch the shared user context
  (exclusions ∪ dislikes, user vector, tilt embeddings) ONCE per page
  and thread it through every row, instead of per-row queries — the
  measured because: row spends ~90ms of its ~104ms on that context.
- **Why this is enough**: engagement/avoidance signals are exactly what
  §D already encodes; the operator's DB accumulates impression→outcome
  history for their own analysis while the public scorer stays frozen.
- **Privacy**: tokens carry no PII beyond the opaque user id.

## 5. Engine hygiene & regression hardening (M3)

- **Judgments 10 → ~150 pairs**: stratified by genre × decade × medium;
  inclusion rule = two-agreement consensus; `eval/` gains batch tooling
  (the harness already accepts any judgments file). CI gate raises to
  pairwise ≥ 0.75 and adds an NDCG@10 floor.
  *Measured note (M3): the engine scores 0.88 pairwise / 0.34 NDCG@10
  on the shipped 160-pair file — the original ≥ 0.40 NDCG target was
  set before measurement and is unreachable on the 830-title pool even
  for a pure-cosine oracle (~0.35–0.46); the enforced floor is 0.28,
  measured-minus-margin. Generator: `eval/build_judgments.py`.*
- **The 6 narrative features stay** — pruning breaks golden parity and
  buys nothing (zero-cost names); instead they are documented as
  "reserved: filled by the private tagger, always 0 in public builds"
  in FEATURE_NAMES and the weights.json schema note.
- **Local integration harness**: `pytest -m integration` +
  `make test-integration` spins a throwaway pgvector container, applies
  schema.sql + user_data.sql, and runs exactly the CI demo-eval class of
  tests (recall smoke, eval gate, personalization end-to-end, tilt) —
  one suite, two entry points, so the 0.5.1/0.5.3 bug class is catchable
  on a laptop.
- **Metrics**: `/metrics` (or /health extension) exposes recall-channel
  coverage, latency histogram, cache hit rate — the counters fall out of
  the middleware and cache paths added in 0.7.

## Milestones

| phase | ships | gate |
|---|---|---|
| M1 (v0.8) | /page MVP, session filter params, impression tokens (emit) | offline suite + CI demo-eval green; /page ≤ 2× a comparable single row — measured 160ms warm ≈ 1.1× the slowest row (for-user 147ms): composition adds ~15ms; the remaining cost is per-row user-context queries (exclusions, vector probe, tilt fetch), a shared-fetch cut is noted for M2 |
| M2 (v0.9) | exploration budget, /feedback ingestion, NL cold-start anchor | coverage column in eval; no accuracy regression |
| M3 (v0.10) | 150-pair judgments + raised gates, integration harness, metrics, narrative-feature docs | gates enforced in CI (`--min-pairwise 0.75 --min-ndcg 0.28`); make test-integration works offline-first — SHIPPED |

Sequencing rationale: tokens are designed with /page (M1) because the
page is the natural impression unit; the exploration budget waits for
/page so the budget is row-aware instead of bolted onto flat lists.
