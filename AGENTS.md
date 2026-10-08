# AGENTS.md — cine-rec-engine

Content-based movie/series recommendation engine over the user's own
PostgreSQL. Python ≥3.10, asyncpg, optional pgvector/Redis. MIT.

## The one rule above all: WHAT IS PUBLIC vs PRIVATE

This repo publishes the **fitted weights** (`cine_rec_engine/weights.json`)
but NEVER the labeling, training data, or tuning methodology behind them.

- ❌ Never commit: graded/training datasets, tuning scripts, holdout
  scores, sweep results, or comments explaining HOW a coefficient was
  tuned ("blind holdout 19.15→26.27…", "the round-5 sweep picked…").
- ✅ Comments describe WHAT a constant does and WHY the behavior matters,
  not the measurement history that produced its value.
- ❌ Never reintroduce the removed tuning-methodology comments. The git
  history was squashed to a single commit on purpose (2026-09-27) to
  purge them — do not restore old content from anywhere.
- The `eval/judgments.jsonl` (12 obvious demo pairs) and
  `tests/golden_feature_vectors.json` (numbers only) are fine.

Before pushing anything ask: *does this teach a competitor to reproduce
our accuracy?* If yes, it stays out.

## Layout map

| Path | What it is |
|---|---|
| `cine_rec_engine/` | the engine: `service.py` (find_similar, recommend_for_user, scoring loop), `queries.py` (SQL recall incl. user-vector ANN), `scoring.py` (pure similarity fns), `config.py` (constants/weights), `tables.py` (logical table-name registry — `CINE_REC_SCHEMA_MAP`), `db.py` (fetch/execute helpers resolving `{t_<name>}` placeholders), `user_weights.py` (§D item weights), `user_stats.py` (events→stats pipeline), `user_vector.py` (§E user vectors), `watched.py` (per-user exclusions), `tmdb_recs.py` (behavioral-graph sync), `serve.py` (HTTP API), `init_db.py` + `sql/` (schema apply + startup map verification), `cli.py` (`cine-rec init/check/serve`), `weights.json` (published coefficients) |
| `ingest/` | TMDB mirror loader: `loader.py` (upserts + bridge mirroring — writes through the same table registry), `exports.py` (daily ID exports, streaming), `rate.py` (token bucket), `cli.py` (`bootstrap`/`refresh`/`genres`) |
| `demo/` | `docker compose up` → Postgres + bundled titles + recommendations with WHY. No API key. |
| `deploy/` | self-host `docker-compose.yml` (env knobs, optional table-map mount) |
| `eval/` | pairwise/NDCG harness + demo judgments |
| `docs/` | `schema.sql` (split catalog + compatibility views), `user_data.sql` (user layer; `user_watches` is a DERIVED view), `data.md` (ingest contract), `personalization.md` (§B–§J) |
| `benchmarks/` | `bench_scoring.py` (offline hot loop), `bench_e2e.py` (synthetic 25k catalog), `smoke_personalization.py` (live end-to-end) |

## Non-negotiable invariants

1. **Scoring parity**: `feature_vector` and `_score_pair_fast` must stay
   output-identical to `tests/golden_feature_vectors.json` (exact for
   feature_vector, ≤1 ULP for the fast path). If you change the feature
   space, regenerate the golden file IN THE SAME COMMIT and say so.
2. **Degrade, never fail**: every optional input (embeddings, Redis,
   pgvector, cinematic tags, TMDB key) must switch its channel off, not
   raise. A query path that 500s on a missing optional table is a bug.
3. **Three write-disjoint layers**: catalog tables never store user data;
   user tables only get raw events from the app; everything derived is
   recomputed by `user_stats.py`/`user_vector.py`.
4. **TMDB ids collide across media**: movie 155 ≠ tv 155. Every key is
   (id, media_type).
5. `record_event` is one transaction (event + title stats + w_i +
   last_event_at) — acceptance rule §J.

## Workflow

```bash
pip install -e ".[dev]"          # or: make install
pytest -q                        # 104 offline tests, no DB needed
ruff check cine_rec_engine ingest tests benchmarks eval demo
make build                       # wheel; twine step in PUBLISH.md
```

- Commits: `user.name=DovBerKaplan`,
  `user.email=216677395+DovBerKaplan@users.noreply.github.com`.
- CI (3.10–3.12) must stay green on main.
- Performance work: measure with `benchmarks/` before and after; the
  README's numbers are real measurements — update them only with new
  measurements.
- Demo data (`demo/data/`) comes from TMDB via our own ingest; keep the
  attribution note in `demo/data/README.md`.

## Skills

Deeper guides live in `.zcode/skills/` (any agent can read them):
- `cine-rec-engine-guide` — how the engine works end to end
- `cine-rec-db-guide` — schemas, layers, spinning up a DB
- `cine-rec-release` — versioning, changelog, PyPI, privacy checklist
