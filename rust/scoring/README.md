# cine-rec-scoring — golden parity harness (rust-port, phase 1)

Rust-side contract for porting the engine's scoring core. The Python
package stays the source of truth; this crate pins the parity contract
in executable form **before** any scoring code is ported.

## The contract

| Path | Rule |
|---|---|
| `feature_vector` (28 entries) | bit-exact vs `tests/golden_feature_vectors.json` |
| `_score_pair_fast` | ≤ 1 ULP vs the golden score |

The golden file was captured from the reference path; the hot loop
accumulates in a different order, hence the 1-ULP tolerance. The fast
path's live distribution against the goldens: 472/480 exact, 8/480 at
exactly 1 ULP.

ULP distances depend on **operation order**. The phase-2 port must
replicate the Python computation order one-to-one; no fast-math-style
reordering anywhere (Rust does not reorder float ops by default — the
port's advantage, not to be given up).

## Inputs are pinned, not regenerated

`tests/fixtures/entities_120_4.json` freezes the exact entity rows and
weight list from `benchmarks/bench_scoring.py::build(120, 4)` — the
same deterministic generator the Python golden test uses. No Python
RNG / `random.sample` semantics ever enter the parity question.

Pairing convention: golden case `i` is
`(cands[i / 4], seeds[i % 4])`. The convention is anchored by
Python-computed probes inside the fixture (`probes` array) — a wrong
convention cannot pass by agreeing with itself.

## Run

```bash
cargo test                    # harness green + red-green proof
cargo test -- --ignored demo_red_end_to_end   # expected to FAIL — the red demo
```

## Red-green proof (phase gate 1)

- **Green**: `OracleBackend` serves the golden answers through the real
  driver path (fixture → pairing → backend → comparators) → zero
  mismatches over all 480 pairs.
- **Red**: `ZeroBackend` is flagged on every case whose golden differs
  (counts derived from the goldens themselves), a 1-bit vec drift is
  caught, 2-ULP score drift is caught, 1-ULP is accepted (the exact
  contract boundary), NaN is always caught, index drift is caught, and
  the oracle refuses misaligned entities.

## Fixture regeneration

```bash
python rust/scoring/tests/fixtures/gen_entities.py
```

Regenerate ONLY together with a re-capture of the golden file, under
the same clean environment the goldens were captured in (no
`REC_ENSEMBLE_CONFIG` / `CINE_REC_SCORER` overrides) — the two files
are one contract.
