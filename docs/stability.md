# Stability policy

cine-rec-engine moves fast but predictably. This page defines what you
can depend on, how a change is announced, and for how long old behavior
stays available. In effect from 0.14.0.

## What is public API

- **HTTP** — the `/v1/*` routes (`/v1/similar`, `/v1/page`,
  `/v1/by-text`, `POST /v1/feedback`, `/v1/for-user/{user_id}`) plus
  `/health` and `/metrics`, with their documented query/body parameters
  and response fields. Additive response fields and optional parameters
  are not breaking changes.
- **Python** — `RecommendationService` (`initialize`, `find_similar`,
  `recommend_for_user`), `create_app`, `record_event`,
  `record_feedback`, the table registry (`cine_rec_engine.tables`), and
  every name exported by `cine_rec_engine.__init__`.
- **CLI** — `cine-rec init | check | serve` and their flags.
- **Artifacts** — the `weights.json` schema: artifacts produced by an
  older `eval/tune_weights.py` keep loading in later minors.
- **Environment** — every `CINE_REC_*` variable in the README table.
  Removing or repurposing a documented variable is a breaking change.

Everything else — private modules and helper functions, SQL text, the
demo dataset, internals of the ingest — may change in any release.

## Deprecation

1. A feature slated for removal is marked deprecated in a **minor**
   release: a `Deprecated` section in the CHANGELOG, the `deprecated`
   flag in OpenAPI, and `Deprecation: true` +
   `X-Deprecated-Removal: 1.0` HTTP headers on affected routes.
2. Deprecated behavior keeps working for **at least two minor
   releases**, and in any case until 1.0.
3. Live example: the bare data routes (`/similar`, `/page`, `/by-text`,
   `/feedback`, `/for-user/{id}`) are deprecated aliases of their
   `/v1/*` counterparts since 0.14.0 and will be removed in 1.0. The
   same handler objects serve both, so aliases never drift.

## Breaking changes

Pre-1.0 a breaking change may land in a minor bump (0.x SemVer allows
it), but it must: bump the **minor** (never disguise one as a patch),
document the migration path under a `Breaking` heading in the
CHANGELOG, and keep a deprecation shim where technically possible.

## Supported versions

The latest minor receives all fixes. The previous minor receives
security fixes until the next minor after it. Security reports:
SECURITY.md.
