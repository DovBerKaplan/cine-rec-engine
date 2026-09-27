# Personalization — user stats & recommendations (spec v0.2)

Three write-disjoint layers: the catalog (schema.sql) never records who
watched what; the app writes raw user events; the engine derives
everything else. `docs/user_data.sql` creates the user layer.

```
 record_event() ──► user_watch_events (raw truth)
                        │ same transaction (§J)
                        ▼
                user_title_stats ── w_i (§D formulas) ──► user_vectors (§E)
                        │                                      │
                        ▼                                      ▼
        user_stats + user_genre_stats (§C)      recommend_for_user (§F)
```

## The item weight `w_i` (§D)

`w = clip(S × S_recency × S_engagement, 0, 1.5)` — computed separately
for movies (completion ratio) and series (depth ladder: the 3-episode
rule, never a bare episodes/total), with:

- drop-off filter (movie < 15%, first-episode < 50%) → weight 0
- recency half-life 30 days (`2^(−Δdays/30)`)
- rewatch/favorite × 1.5, watchlist-only 0.7 × 1.2, dislike → 0
- explicit dislike is additionally hard-filtered from every result list

Every branch is pinned by `tests/test_personalization.py` (28 tests) and
the end-to-end smoke (`benchmarks/smoke_personalization.py`).

## The user vector (§E)

`u = normalize(Σ w_i · v_i)` over the embedding column of weighted
titles (bring-your-own encoder, `models/README.md`). No weighted items →
**no vector** (never a fake zero vector); items lacking embeddings skip
the vector but stay in SQL recall (§G.3). Multi-persona storage
(`user_vectors.persona_id`, up to 3) is in place; K-Means clustering
waits for real smearing data, per the spec.

## Recommendations (§F)

`RecommendationService.recommend_for_user(user_id, limit)`:

1. Seeds = the user's top-w_i titles (or watchlist entries for
   intent-only users).
2. An extra ANN channel recalls titles nearest the user vector and
   feeds them through the same LTR scorer (cosine reuses the distance).
3. Hard filters: watched (derived `user_watches` view), rated,
   disliked, plus saga advancement — all via the existing
   `find_similar(user_id=...)` path.

## Lifecycle (§G)

- `record_event()` — one transaction: raw event + title stats + w_i +
  `last_event_at` (§J). `record_feedback()` refreshes the touched title.
- `nightly_recompute()` — recency refresh for active users (default
  90-day window), stats rebuild, vector rebuild when older than 24h.
- Vector rebuilds are async by design — never on the player hot path.

## Quick start

```bash
psql -d yourdb -f docs/schema.sql      # catalog (split + views)
psql -d yourdb -f docs/user_data.sql   # events + stats + vectors + legacy view
```

```python
import asyncio, asyncpg
from cine_rec_engine import user_stats, user_vector, RecommendationService

async def main():
    pool = await asyncpg.create_pool("postgresql://user:pw@localhost/db")
    await user_stats.record_event(pool, {
        "user_id": 1, "tmdb_id": 155, "media_type": "movie",
        "watched_at": "2026-09-27T20:00:00+00:00",
        "watched_sec": 8000, "duration_sec": 8000, "completed": True,
    })
    await user_stats.refresh_user_stats(pool, 1)
    await user_vector.build_user_vector(pool, 1)
    rec = RecommendationService(); await rec.initialize(pool)
    out = await rec.recommend_for_user(1, limit=10, include_why=True)
    print(out["reason"], out["seeds"])          # personalized / watchlist / cold_start
    for r in out["results"]:
        print(r["title_en"], "why:", out["why"].get(str(r["tmdb_id"])))

asyncio.run(main())
```

Nightly (cron, after the catalog refresh):

```python
from cine_rec_engine.user_stats import nightly_recompute
await nightly_recompute(pool)          # weights + stats + stale vectors
```
## The event contract (§A — what MUST be written)

| Field | Required | Notes |
|---|---|---|
| `user_id`, `tmdb_id`, `media_type`, `watched_at` | ✅ | ValueError without them |
| `watched_sec`, `duration_sec` | recommended | ratio → completion score; missing duration ⇒ ratio unknown ⇒ title can complete only via `completed=true` |
| `season`, `episode` | tv only | episode counting + rewatch detection |
| `completed` | recommended | the ≥50% threshold fallback |
| `pause_count`, `last_position_sec` | optional | pause penalty / resume |

Degradation rules (all by design, all tested):
- No episodes recorded → series score by watch time floors only.
- No embedding column → no user vector; SQL recall + seeds still work
  (`vector_used: false` in the response).
- Fewer than 3 weighted titles → watchlist blending, not a persona;
  nothing at all → `reason=cold_start` with EMPTY results (never a
  silent blockbuster list).
- Dislike → w=0 AND hard-filtered from every list.
- Saga: a watched title never returns, and its saga ADVANCES (watched
  part 1 ⇒ part 2 is the recommendation, part 1 is not).

## Feeding users from outside (§D — one source, done)

**Letterboxd CSV** (ratings export):

```bash
python examples/import_letterboxd.py ratings.csv --user 1 --dsn $DSN
# rating ≥ 3.5 → favorite seed · ≤ 2.0 → dislike (hard-filter) ·
# 2.5–3.0 → no signal (watched-but-unremarkable) · unmatched titles
# are REPORTED, never fuzzy-guessed
```

Jellyfin/Tautulli and manual sources feed the same `record_event`
contract — one connector at a time, per the spec.

## Nightly (§C)

```cron
# 03:00 — recency moves w_i; stats rebuild; stale vectors (>24h)
0 3 * * * deploy python -c "import asyncio,asyncpg;from cine_rec_engine.user_stats import nightly_recompute;   asyncio.run(nightly_recompute(asyncio.run(asyncpg.create_pool('$DSN'))))"
```
Hot path stays hot: during playback only `record_event` runs — no
vector builds, no LTR, nothing else.
