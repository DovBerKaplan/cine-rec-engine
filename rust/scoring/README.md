# cine-rec-scoring — golden parity harness (rust-port, phase 1)

Rust-side contract for porting the engine's scoring core. The Python
package stays the source of truth; this crate pins the parity contract
in executable form **before** any scoring code is ported.

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
full-vector search — no pairing formula is used on the generator side,
so a wrongly-captured convention cannot agree with the harness.
Probe pairs carry non-zero scores so they discriminate.

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
