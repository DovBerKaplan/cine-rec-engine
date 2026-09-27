# Contributing

Thanks for considering a contribution.

## Getting started

```bash
git clone https://github.com/DovBerKaplan/cine-rec-engine
cd cine-rec-engine
pip install -e ".[dev]"
pytest                    # offline suite — no database needed
```

## Ground rules

1. **Tests are offline by default.** Anything in `tests/` must pass without
   PostgreSQL, Redis, or a TMDB key. DB-dependent checks live behind an
   opt-in marker (`-m db`) — add one only with a fixtures story.
2. **Degrade, never fail.** New recall channels / features must switch
   themselves off when their table/column/extension is missing. An engine
   that 500s because an optional index is absent is a bug.
3. **Weights are frozen per release.** Changes to feature definitions or
   the scorer invalidate `weights.json` — bump the version and note the
   refit in the PR.
4. **No user data, no proprietary datasets.** Everything committed must be
   safe to publish under MIT. Graded/evaluation datasets stay out.
5. **Match the code style**: type hints on public surfaces, Google-style
   docstrings, ≤100 columns. `ruff check .` must be clean.

## Pull requests

- One logical change per PR; describe the *why* in the first paragraph.
- New features need: tests, a README/ROADMAP note, and graceful-degradation
  behavior spelled out.
- Benchmarks (before/after latency or recall@k on your mirror) are welcome
  evidence but not required.

## Reporting bugs

Open an issue with: engine version, Postgres version, which optional
tables exist (`pgvector`? embeddings? `tmdb_recommendations`?), and the
smallest seed/parameter combo that reproduces it.

## Security

See [SECURITY.md](SECURITY.md) — do not open public issues for anything
involving credentials.
