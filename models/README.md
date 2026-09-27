# Models — the optional embedding sidecar

The engine is deliberately split in two:

```
┌─────────────────────────┐          ┌──────────────────────────────┐
│     cine_rec_engine/    │          │           models/            │
│  recall · scoring ·     │ ◄──────  │  YOUR sentence encoders,     │
│  ranking · weights.json │  vectors │  your embedding pipeline,    │
│  (published, MIT)       │  in DB   │  your training data (yours)  │
└─────────────────────────┘          └──────────────────────────────┘
```

**The engine ships with zero model weights.** Its strongest optional
input is a set of overview embeddings — dense vectors describing each
title's plot. You produce them however you like, store them as
`vector` columns on `tmdb_media`, and register the columns in
`cine_rec_engine/model_spaces.py`.

## Adding a space (any encoder, any language)

```python
# 1. encode overviews with YOUR model (example: sentence-transformers)
from sentence_transformers import SentenceTransformer
import asyncpg

model = SentenceTransformer("intfloat/multilingual-e5-base")  # yours
pool = await asyncpg.create_pool(dsn)
await pool.execute("ALTER TABLE tmdb_media ADD COLUMN embedding_e5e vector(768)")

rows = await pool.fetch("SELECT id, media_type, overview_en FROM tmdb_media")
for row in rows:
    vec = model.encode(row["overview_en"] or "")
    await pool.execute(
        "UPDATE tmdb_media SET embedding_e5e = $1 WHERE id = $2 AND media_type = $3",
        vec.tolist(), row["id"], row["media_type"],
    )

# 2. register the column in cine_rec_engine/model_spaces.py REC_MODELS
# 3. done — KNN recall and the cosine feature now use your space
```

## What is published vs. kept private

| Artifact | In this repo? |
|---|---|
| Engine code (recall, scoring, ranking) | ✅ MIT |
| Learned feature **weights** (`cine_rec_engine/weights.json`) | ✅ published |
| Embedding model weights / our vectors | ❌ yours to make |
| Training sets, graded evaluation datasets, tuning pipelines | ❌ not published |
| Narrative/cinematic tag enrichment process | ❌ not published |

The weights tell you *how much* each feature matters once candidates are
recalled; they do not include the labeled data they were fitted on.
