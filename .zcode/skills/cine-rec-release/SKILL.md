---
name: cine-rec-release
description: Use this skill when cutting a release, touching packaging/PyPI, refreshing the demo dataset, or reviewing changes for the public-vs-private policy (weights are public, tuning methodology is not)
---

# cine-rec-engine — Release & Privacy Guide

## Release checklist

1. Tests green: `pytest -q` (offline, no DB). Lint: `ruff check
   cine_rec_engine ingest tests benchmarks eval demo`.
2. Bump version in BOTH `cine_rec_engine/__init__.py` and
   `pyproject.toml` (keep them in sync).
3. `CHANGELOG.md` — new `## [x.y.z] — date` section, Keep-a-Changelog
   style, concrete (numbers, not adjectives).
4. Update docs the change touches: README numbers are MEASURED — refresh
   only with new benchmark output. ROADMAP: mark shipped sections ✅.
5. Build: `python -m build` (wheel must install + import in a clean
   venv — verify). PyPI upload is manual (`PUBLISH.md`).
6. Commit as `DovBerKaplan <216677395+DovBerKaplan@users.noreply.github.com>`.
   CI must be green on main.

## The privacy gate (run on EVERY change)

Public: engine code, fitted `weights.json` (the coefficients),
demo-scale judgments (`eval/judgments.jsonl`), numeric fixtures.

Private — must NEVER land:
- Training/graded datasets (any size).
- Tuning methodology: holdout scores, sweep grids/results, round
  histories, per-coefficient "how we picked 26.0" narratives.
- Production moderation/user data, real user ids.

Check before committing:
```bash
grep -rniE "holdout|sweep|scorecard|round-[0-9]|tune_weights|blind" \
  --include="*.py" cine_rec_engine ingest
# must return nothing
```
Comments must say WHAT a constant does; the measurement history behind
its value is company knowledge. The history was squashed once
(2026-09-27) to purge this — never restore removed comments or old
file contents from memory/notes/backups.

Also scan for accidental secrets: API keys, DSNs with passwords.

## Refreshing the demo dataset

`demo/data/titles.jsonl.gz` = 400 top-rated titles fetched with OUR OWN
ingest client (top_rated pages 1–10, en-US). To refresh:
```bash
# any machine with a TMDB key:
python - <<'PY'
# adapt ingest/exports + /3/{movie,tv}/top_rated; dump payload JSONL gz
PY
```
Keep: the attribution note (`demo/data/README.md`), size ≤ ~6MB, and
re-verify `docker compose up` prints sensible WHY lines for seeds 155
and 1396 before pushing.

## Adding a golden-breaking change

Feature-space changes regenerate `tests/golden_feature_vectors.json`
(`benchmarks/bench_scoring.py::build` + `feature_vector` dump) in the
SAME commit, and the commit message says the golden file was regenerated
and why.
