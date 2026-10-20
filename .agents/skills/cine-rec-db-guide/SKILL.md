---
name: cine-rec-db-guide
description: Use this skill for anything database-related — the three write-disjoint layers, the split catalog schema, the user-data tables, compatibility views, and how to spin a local/throwaway Postgres for testing
---

# cine-rec-engine — Database Guide

## Three layers that never mix writes

1. **Catalog** (`docs/schema.sql`) — the TMDB mirror. Written ONLY by
   `ingest/`. Movies and TV are separate fact tables (ids collide across
   media: movie 155 ≠ tv 155).
2. **User raw** (`docs/user_data.sql`) — `user_watch_events`,
   `user_feedback`, `user_searches`, `title_ratings`. Written ONLY by the
   application.
3. **User derived** — `user_title_stats`, `user_stats`,
   `user_genre_stats`, `user_vectors`. Written ONLY by
   `user_stats.py` / `user_vector.py` (recomputable from layer 2).

## Catalog quick map (17 tables + views)

- Facts: `tmdb_movies`, `tmdb_tv` (tv has `name`, no collection_id).
- Dimensions: `tmdb_people`, `tmdb_keywords`,
  `tmdb_production_companies`, `tmdb_networks`, per-medium genre lists
  (`tmdb_movie_genres` / `tmdb_tv_genres` — TMDB's lists differ).
- Bridges per medium: genres_map, keywords, companies, cast (top-5,
  PK includes cast_order), crew (whitelist: Director/Creator/Writer/
  Original Music Composer/Director of Photography), tv networks.
- Behavioral graph: `tmdb_movie_recommendations`, `tmdb_tv_recommendations`
  + the shared physical `tmdb_recommendations` (the engine upserts into
  it on demand — a view can't take ON CONFLICT).
- **Compatibility views** (the engine reads these, not the fact tables):
  `tmdb_media`, `tmdb_genres`, `tmdb_media_genres`, `tmdb_cast`,
  `tmdb_crew`, `tmdb_media_keywords`, `tmdb_media_companies`,
  `tmdb_media_networks`.
- Embeddings are OPTIONAL columns on the fact tables (§2 of the schema is
  commented out until you add them; see `models/README.md`).

## User layer quick map

- `user_watch_events` — one row per viewing session/unit (watched_sec,
  duration_sec, pauses, completed, season/episode, last_position).
- `user_title_stats` — per (user, title): sessions, ratios, episodes,
  rewatch (same-unit re-views!), `w_item`, `dropped`.
- `user_stats` / `user_genre_stats` — the §C profile (shares, medians,
  hooked/abandoned counts; genre distribution for explanations).
- `user_vectors` — normalized user embedding, `space` column matching
  your encoder, up to 3 personas (K-Means deferred until real smearing).
- `user_watches` is a **VIEW** over user_title_stats (w>0 or completed
  movie) — the legacy engine contract. Don't create it as a table.

## Spinning a database

Local demo DB (bundled 400 titles, no API key):
```bash
cd demo && docker compose up --abort-on-container-exit
# DB stays on localhost:54329 (demo/demo/demo)
```

Throwaway test DB anywhere:
```bash
docker run -d --name tmp_pg -p 55432:5432 \
  -e POSTGRES_USER=t -e POSTGRES_PASSWORD=t -e POSTGRES_DB=t \
  pgvector/pgvector:pg16
psql ... -f docs/schema.sql -f docs/user_data.sql
# teardown: docker rm -f tmp_pg
```
Note: the pgvector EXTENSION is named `vector`
(`CREATE EXTENSION vector`), and the codec must be registered on every
asyncpg connection that reads/writes `vector` columns
(`from pgvector.asyncpg import register_vector`).

## Ingest writes (how data gets in)

`ingest/cli.py bootstrap` — daily-export discovery → filter
(adult/popularity) → skip existing → 8 concurrent workers under the
token-bucket (40 req/s), one API call per title
(`append_to_response` built as a list — the TV comma bug was real).
`refresh` = export diff (threshold-crossers) THEN /changes (updates),
both through the same gate. Bridges MIRROR the payload (delete stale
rows, upsert changes) — never append-only.
