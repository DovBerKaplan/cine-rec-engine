# Feeding the engine: a local TMDB mirror

The engine reads **your** PostgreSQL and nothing else. `docs/schema.sql`
defines the canonical mirror — movies and TV as **two fact tables** with
independent id spaces (movie 155 ≠ tv 155), shared dimensions, per-medium
bridges, and compatibility views so the engine works unchanged.

## The built-in ingest (recommended)

`ingest/` implements the whole contract — discovery, filters, rate limits,
upserts, refresh:

```bash
# 1. schema
psql -d yourdb -f docs/schema.sql

# 2. full load from TMDB's daily exports (adult=false, popularity ≥ 1)
python -m ingest.cli bootstrap \
    --dsn postgresql://user:pw@localhost/yourdb \
    --api-key $TMDB_API_KEY \
    --popularity-min 1.0

# 3. daily refresh (cron, after 08:00 UTC — TMDB publishes changes ~05:00)
#    full loop: export diff (new threshold-crossers) THEN /changes (updates)
python -m ingest.cli refresh --dsn ... --api-key $TMDB_API_KEY --popularity-min 1.0
```

What it does per title — **one API call**:

```
GET /3/movie/{id}?language=en-US&append_to_response=credits,keywords,recommendations
GET /3/tv/{id}?language=en-US&append_to_response=credits,keywords,recommendations,external_ids
```

- **Cast:** top-5 by TMDB `order`; same person can hold two roles.
- **Crew:** only Director / Creator (tv) / Writer / Original Music
  Composer / Director of Photography — the jobs the scorer uses.
- **Genres:** synced once per run from `/genre/movie/list` and
  `/genre/tv/list` (the lists differ — tv has 10759 "Action & Adventure").
- **Recommendations:** page 1 lands in the behavioral graph; TMDB recs
  stay within their medium, like the API.
- **Rate:** paced to ≤ 40 requests/second (token-bucket rate limiter —
  not just a concurrency cap); `Retry-After` honored with backoff.
- **Writes:** upsert by PK. Never delete-insert of the catalog.
- **Acceptance:** a title is *complete* with details + genres (if TMDB
  returned any) + credits (even empty) + keywords (even empty) +
  recommendations page 1 (even empty). Missing sections still ingest —
  that recall channel just returns 0 for that title.

**Discovery:** TMDB's official daily ID exports
(`files.tmdb.org/p/exports/…`), job runs daily ~07:00 UTC — the loader
takes yesterday's file, applies `adult=false` + the popularity floor,
skips ids already in the DB (resumable; `--force` refetches), and ingests
with bounded concurrency (default 8 workers) paced by the rate limiter.

**Refresh (the full daily loop):**
1. *Export diff* — filter yesterday's export, keep only ids missing from
   the DB. This catches titles that crossed your popularity threshold
   after their TMDB debut — the Changes API never lists them (a title
   only "changes" when its row is touched).
2. *Changes* — `/movie/changes` + `/tv/changes`, paged; re-ingested
   through the same adult + popularity gate, so the catalog shape stays
   what bootstrap chose.

**Bridge mirroring:** every ingest mirrors the bridge rows to the fresh
payload — recast actors, dropped keywords and removed networks are
deleted, not just accumulated (`ON CONFLICT DO NOTHING` alone would keep
them forever).

## Alternatives (bring your own loader)

Any loader that fills the schema works:

- [populate-movies](https://github.com/transitive-bullshit/populate-movies)
  — downloads the same daily exports and enriches into PostgreSQL
  (Node/Prisma; MIT). Map its output onto `docs/schema.sql`.
- [tmdb-caching-api](https://github.com/tm1981/tmdb-caching-api) — a
  Next.js service that lazily caches TMDB data into PostgreSQL/MySQL/
  MariaDB behind a local API, with an admin dashboard.

## What the engine needs at minimum

| Recall channel | Tables | Required? |
|---|---|---|
| genre + popularity | `tmdb_movies`/`tmdb_tv` + genre maps | ✅ core |
| keywords / cast / crew | keyword + cast + crew bridges, `tmdb_people` | ✅ core |
| companies / networks | company bridges, `tmdb_tv_networks_map` | recommended |
| TMDB behavioral recs | `tmdb_recommendations` (auto-synced by the engine too, with `TMDB_API_KEY`) | recommended — strongest single feature |
| vector KNN + cosine | `embedding_*` columns (see `models/README.md`) | optional — add your own to the view |

Everything optional degrades silently: missing pieces turn features off,
they never fail a query.

## Daily cron

```cron
# /etc/cron.d/tmdb-mirror — 09:00 UTC: after the export (07:00) and the
# changes list (05:00) have both landed. NOTE: cron does NOT expand
# $VARS from your shell — set them here or use an EnvironmentFile wrapper.
DATABASE_URL=postgresql://user:pw@localhost/yourdb
TMDB_API_KEY=YourKey
0 9 * * * deploy cd /srv/cine-rec-engine && \
  /usr/bin/env DATABASE_URL=$DATABASE_URL TMDB_API_KEY=$TMDB_API_KEY \
  python -m ingest.cli refresh --dsn $DATABASE_URL --api-key $TMDB_API_KEY \
  >> /var/log/tmdb-mirror.log 2>&1
```

Not for the mirror: images (paths only), non-`en-US` translations,
seasons/episodes, reviews, videos, watch providers, and anything not
returned by TMDB.
