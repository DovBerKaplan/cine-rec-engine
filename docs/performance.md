# Performance & the optional compiled extension

The engine ships in two flavors: **pure Python** (the canonical PyPI
wheel — zero compiled dependencies) and an **optional Rust extension**
that accelerates the scoring hot loop and the DB recall layer. Both
produce identical results (verified by a bit-exactness test harness
over frozen golden vectors; see `rust/` for the parity suite).

## Measured numbers

All measurements: 2-core container, PostgreSQL 16, 25k-title synthetic
catalog, cache disabled (forced), Python 3.12, single-flight.

| scenario | pure Python | with Rust ext | delta |
|---|---|---|---|
| cold single-seed | 48 ms | 47 ms | within noise |
| warm (cache hit) | 0.3 ms | 0.2 ms | same |
| 5-seed multi | 89 ms | 74 ms | −17% |
| HTTP overhead (p50/p99 @c16) | — | −3%/12% | skin cost, not core |

The full methodology and raw runs are reproducible via
`benchmarks/bench_e2e.py` (synthetic catalog) and
`benchmarks/bench_scoring.py` (pair-scoring hot loop; ~×1.5 on the
pair path).

## HTTP + serialization overhead

The Python HTTP layer (FastAPI/uvicorn) adds ≤15% to request latency
at both p50 and p99 under concurrency 16. This is below the
pre-registered threshold for translating the HTTP layer to Rust;
the Python "skin" (serve.py, CLI, init) stays.

## Wheel policy

| platform | wheel | notes |
|---|---|---|
| Linux x86_64 (manylinux) | **Rust ext** (cp310-abi3) | one wheel covers Python 3.10+ |
| Linux aarch64 | **Rust ext** (cp310-abi3) | |
| macOS arm64 + x86_64 | **Rust ext** (cp310-abi3) | |
| macOS x86_64 (older) | **pure Python** | Rust wheel may not load on older macOS |
| Windows | **Rust ext** (cp310-abi3) | |
| any (sdist) | **pure Python** + rust/ source | self-build with: `maturin build --release` |

The **pure-Python wheel is always available on PyPI** and is the
default `pip install cine-rec-engine` artifact. The Rust-enhanced
wheels are published alongside; `pip` resolves the best match for
your platform automatically.

Forcing pure Python: `CINE_REC_SCORING_RS=0` in the environment.
A missing or broken extension degrades silently — the engine always
works, the extension just makes it faster.

## Self-building from sdist

```bash
pip install cine-rec-engine --no-binary :all:  # if you have Rust
# or explicitly:
pip download cine-rec-engine --no-binary :all: --no-deps
tar xzf cine_rec_engine-*.tar.gz && cd cine_rec_engine-*
maturin build --release && pip install dist/*.whl
```

Requires Rust 1.91+ and `maturin` (`pip install maturin`).
