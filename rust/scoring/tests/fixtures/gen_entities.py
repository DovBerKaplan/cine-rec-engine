"""Regenerate entities_120_4.json — the frozen INPUT side of the Rust
parity harness.

The 480-entry tests/golden_feature_vectors.json pins the OUTPUTS of
feature_vector/_score_pair_fast; this fixture pins the INPUTS (the exact
entity dicts and the weight list the Python golden test feeds them).
Together they are the full parity contract — the Rust crate never
regenerates entities itself, so no RNG/`random.sample` semantics ever
enter the parity question.

Provenance: benchmarks/bench_scoring.py build(120, 4) — the same
deterministic generator the Python golden test uses (seed 42). Regenerate
ONLY with the same environment the goldens were captured under (clean
env: no REC_ENSEMBLE_CONFIG / CINE_REC_SCORER overrides) and the same
CPython family; if the golden file is ever re-captured, regenerate this
fixture in the same change and re-run `cargo test` in rust/scoring/.

    python rust/scoring/tests/fixtures/gen_entities.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from bench_scoring import build  # noqa: E402

from cine_rec_engine.service import HEURISTIC_WEIGHTS  # noqa: E402

N_CANDS = 120
N_SEEDS = 4

# Pairing-convention anchors: the Rust harness derives golden case i as
# (cands[i // n_seeds], seeds[i % n_seeds]). These probes pin that
# convention (and the JSON float round-trip) to Python's own enumerate
# order — computed here, independent of any Rust iteration logic, so a
# wrong pairing convention cannot pass the harness by agreeing with
# itself. Spread across the space: first/last pair, seed-wrap boundary,
# middle, end.
PROBE_PAIRS = [
    (0, 10000),
    (0, 10003),
    (1, 10000),  # crosses the seed-wrap boundary (idx 3 -> 4)
    (59, 10002),
    (119, 10003),
]
# vec slots worth anchoring: 0 cosine, 3 genre_priority_sum (sorted-sum
# order matters), 27 heuristic_score (accumulation order matters).
PROBE_SLOTS = (0, 3, 27)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "entities_120_4.json"),
    )
    args = p.parse_args()

    cands, seeds = build(N_CANDS, N_SEEDS)
    cand_ix = {c["id"]: i for i, c in enumerate(cands)}
    seed_ix = {s["id"]: i for i, s in enumerate(seeds)}

    golden = json.loads(
        (REPO_ROOT / "tests" / "golden_feature_vectors.json").read_text()
    )
    probes = []
    for cand_id, seed_id in PROBE_PAIRS:
        i = cand_ix[cand_id] * N_SEEDS + seed_ix[seed_id]
        vec = golden[i]["vec"]
        probes.append(
            {
                "cand_id": cand_id,
                "seed_id": seed_id,
                "golden_index": i,
                "score": golden[i]["score"],
                "vec_slots": {str(s): vec[s] for s in PROBE_SLOTS},
            }
        )

    payload = {
        "generator": "benchmarks/bench_scoring.py::build(120, 4), seed 42",
        "n_cands": N_CANDS,
        "n_seeds": N_SEEDS,
        # list(HEURISTIC_WEIGHTS.values()) — what the Python golden test
        # passes as W_LIST to _score_pair_fast (all 28 entries, residual
        # weight last).
        "w_list": list(HEURISTIC_WEIGHTS.values()),
        "feature_names": list(HEURISTIC_WEIGHTS.keys()),
        "pairing": "golden case i == (cands[i // n_seeds], seeds[i % n_seeds])",
        "probes": probes,
        "cands": cands,
        "seeds": seeds,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(
        f"wrote {args.out}: {len(cands)} cands × {len(seeds)} seeds = "
        f"{len(cands) * len(seeds)} pairs, w_list len {len(payload['w_list'])}"
    )


if __name__ == "__main__":
    main()
