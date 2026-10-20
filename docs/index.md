# cine-rec-engine

> **Recommendations from your own database. No black-box API, no rented
> taste.** PostgreSQL in · ranked titles out · zero external calls on
> the hot path.

A content-based movie/series recommendation engine over your own
PostgreSQL TMDB mirror — a 28-feature scorer whose weights were learned
(not guessed), item-to-item and personalized rows, an HTTP API with a
quantitative *why* on every recommendation, and feedback that adjusts
per-user weights transactionally. Extracted from a production
self-hosted system.

## Start here

- **[Architecture](architecture.md)** — recall → scoring → rerank, the
  three write-disjoint layers, and what is cached where.
- **[HTTP API](api.md)** — interactive OpenAPI reference for `/v1/*`.
- **[Data contract](data.md)** — what the ingest writes, exactly.
- **[Personalization](personalization.md)** — the §B–§J machinery:
  weights, user vectors, exploration, feedback.
- **[Public benchmark](benchmark.md)** — MovieLens-derived ground truth,
  reproducible end to end.
- **[Stability policy](stability.md)** — what you can depend on pre-1.0.
- **[Live demo](/demo/)** — precomputed by the real engine, no install.

## 60-second local run

```bash
git clone https://github.com/DovBerKaplan/cine-rec-engine \
  && cd cine-rec-engine/demo
docker compose up   # Postgres + 830 real titles + recommendations
```

```python
pip install "cine-rec-engine[serve]"
```

```bash
DATABASE_URL=postgresql://user:pw@host/db uvicorn cine_rec_engine.serve:app
curl 'localhost:8000/v1/similar?seed=155:movie&limit=5&why=1'
```
