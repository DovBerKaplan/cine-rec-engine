# cine-rec-scoring — scoring core port (rust-port, phases 1–2)

Phase 1 pinned the parity contract as an executable harness; phase 2
translated the scoring core itself (`real.rs`, proven bit-exact by the
same harness) and exposed it to Python through the optional pyo3
bridge (`pybridge.rs`, feature `pyo3`, module `cine_rec_engine._scoring_rs`).

## The contract

| Path | Rule |
|---|---|
| `feature_vector` (28 entries) | bit-exact vs the golden files, **length included** — empty/truncated/padded vectors are red |
| `_score_pair_fast` | ≤ 1 ULP vs the golden score |

The goldens were captured from the reference path; the hot loop
accumulates in a different order, hence the 1-ULP tolerance. On the
base set the fast path's live distribution is 472/480 exact, 8/480 at
exactly 1 ULP. Signed zero is a documented decision: `±0.0` compare
equal for scores; the tolerance is asymmetric across zero (stricter on
the negative side); inside a vector `-0.0` vs `0.0` is a mismatch.

ULP distances depend on **operation order**. The phase-2 port must
replicate the Python computation order one-to-one; no fast-math-style
reordering anywhere (Rust does not reorder float ops by default — the
port's advantage, not to be given up).

## Two golden sets

| Set | Inputs | Outputs | Why |
|---|---|---|---|
| BASE | `tests/fixtures/entities_120_4.json` (frozen `build(120, 4)` rows) | repo `tests/golden_feature_vectors.json` (480 pairs) | the exact inputs/outputs of the Python golden test |
| ENRICHED | `tests/fixtures/entities_enriched.json` (16×4 rows with `via`/`tmdb_rank`/`narrative`/`pacing`/`emotional_arc`/`director`/style keywords, incl. deliberate genre clashes ± via) | `tests/fixtures/goldens_enriched.json` (64 pairs) | the base rows leave six features constant-zero (`director_channel`, `tmdb_rec_decay`, `style_match`, `narrative_match`, `emotional_arc_match`, `pacing_match`) and never exercise the via gate-exemption — a port bug there would pass the base set alone |

Inputs are pinned, not regenerated — no Python RNG semantics enter the
parity question.

Pairing convention: golden case `i` is `(cands[i / n_seeds],
seeds[i % n_seeds])`. It is anchored by probes inside each fixture:
the generator RUNS the Python scoring functions on a probe pair,
records the computed values, and LOCATES the golden index by unique
full-vector search — the pairing formula is never used to derive a
recorded index (its only generator-side role is filtering probe pairs
by their computed score), so a wrongly-captured convention cannot
agree with the harness. Probe pairs carry non-zero computed scores so
they discriminate.

Note on provenance: BOTH golden sets — the repo's
`tests/golden_feature_vectors.json` and `goldens_enriched.json` — are
regression pins of the CURRENT Python implementation, not external
truth (Python is the source of truth). Any change to Python scoring
behavior requires re-capturing the goldens and regenerating the
fixtures in the same change.

## Run

```bash
cargo test                                  # both sets, red-green proof
cargo test -- --ignored demo_red_end_to_end # expected to FAIL — the red demo
```

## Red-green proof (phase gate 1)

- **Green**: `OracleBackend` serves the golden answers through the
  real driver path (fixture → pairing → backend → comparators) on
  BOTH sets → zero mismatches — while validating every driver argument
  (weights, sides, genres, overview_sim), so driver bugs fail here
  rather than in phase 2 disguised as port bugs.
- **Red**: `ZeroBackend` is flagged on exactly the cases that differ
  (counts derived from the goldens themselves), empty/27/29-entry
  vectors are caught (comparator-level AND through the full path),
  1-bit vec drift is caught, 2-ULP score drift is caught, 1-ULP up AND
  down around a non-zero golden pass (the exact contract boundary),
  NaN is always caught, index drift is caught, and the oracle refuses
  misaligned entities and wrong driver arguments.

## Fixture regeneration

```bash
python rust/scoring/tests/fixtures/gen_entities.py
```

Regenerates `entities_120_4.json`, `entities_enriched.json` and
`goldens_enriched.json` (the base set's outputs stay in
`tests/golden_feature_vectors.json` — untouched here). Regenerate ONLY
together with a re-capture of the golden files, under the same clean
environment the goldens were captured in (no `REC_ENSEMBLE_CONFIG` /
`CINE_REC_SCORER` overrides) — fixtures and goldens are one contract.


## Phase 2 — the real port + pyo3 bridge

* `src/real.rs` — the translated scoring core; `tests/real_backend_parity.rs`
  drives it through the SAME harness: **bit-exact vectors, ≤1-ULP fast
  scores, on both golden sets (480 + 64 pairs)**.
* `src/pybridge.rs` — optional pyo3 extension (feature `pyo3`, abi3):
  opaque `PyProfile` handles + a `PairScorer` binding the weight lists
  once; the per-pair call releases the GIL, so the warm-path thread
  pool parallelizes for real. Build: `cargo build --features pyo3
  --release`, then copy `target/release/libcine_rec_scoring.so` to
  `cine_rec_engine/_scoring_rs.so` (wheels bundle it at release time —
  phase 8).
* Python side: `cine_rec_engine/_scoring_accel.py` rebinds the service
  dispatchers when the extension imports; `CINE_REC_SCORING_RS=0`
  forces the pure path. A missing/broken extension never breaks the
  engine.

### Portability discoveries worth remembering

1. **serde_json needs `float_roundtrip`**: the default parser can land
   1 ULP off the correctly-rounded value on some literals — invisible
   while both sides parse identically, fatal against computed bits.
2. **CPython 3.12 `sum()` is Neumaier-compensated**: every ported
   Python `sum()` over floats must use the same compensated fold
   (`real.rs::py_sum`), or slot 27 lands 1 ULP off.
3. **Python truthiness is part of the contract**: `via=""` is NOT a
   gate exemption, `tmdb_rank=0` reads as rank 1, year 0 reads as
   missing, empty genre sets read as `None`, `pacing=""` scores no
   pacing match, `collection_id=0` is falsy everywhere (including the
   `data or cand` fallback), and an EMPTY `entity_data` dict — the
   regular production input for non-enriched candidates — reads empty
   for the enriched-source fields but falls back to the entity for
   narrative/pacing/emotional-arc. Several of these were caught by the
   live Python suite and the phase-2 differential fuzz, not by the
   goldens — run the full suite against the extension.
4. **The recall SQL's `cast_list` is actor NAMES**, `cast_ids` is ids;
   Python `set()` keeps them distinct — hence `CastKey::Id | Name`.

### Bench (this machine, 2500 pairs, best of 3 × 2 runs)

| Path | Pure Python | Rust | Speedup |
|---|---|---|---|
| reference scorer (`_score_candidate_vs_seed`) | 14.1–14.3 µs/pair | 9.6–9.7 µs/pair | ~1.5× |
| hot loop (profiles + `_score_pair_fast`) | 6.1–6.2 µs/pair | 3.8–3.9 µs/pair | ~1.6× |

The remaining per-pair cost is dominated by the Python↔Rust boundary
(wrapper dispatch, per-pair genre-set conversion); batching the
per-candidate scoring into one call is the natural next lever — an
A/B decision deliberately left for the release gate.

### Deployment guards (phase-2 review round 2)

* The extension installs ONLY on CPython >= 3.12 — `sum()` over floats
  became Neumaier-compensated there, and the port replicates THAT
  fold; on 3.10/3.11 it would differ from the pure path in the last
  ULP and stays off (`python-fallback`).
* A locatable extension file below ~400 KB is skipped as truncated —
  a cut-off `.so` can SIGBUS the process at import, below any Python
  `except`. The guaranteed escape hatch stays `CINE_REC_SCORING_RS=0`.
* Per-call fallback: conversion errors on exotic field types degrade
  THAT call to pure Python; profile pairs dispatch by kind (dict =
  pure, opaque = Rust), and a MIXED pair raises loudly rather than
  guessing.
* Known-open (phase 8, per the port plan): CI does not yet build the
  extension, so the bridge tests skip there; a cargo job in the CI
  matrix is part of the packaging phase.
