"""Regenerate the parity-harness fixtures — the frozen INPUT side of
the Rust contract.

Files written next to this script:

* ``entities_120_4.json`` — the base set: ``build(120, 4)`` entity rows
  + the weight list + probes (see PROBES below). Its OUTPUT side is the
  repo's ``tests/golden_feature_vectors.json`` (not written here).
* ``entities_enriched.json`` + ``goldens_enriched.json`` — the
  supplementary set. The base bench rows lack ``via`` / ``tmdb_rank`` /
  ``narrative`` / ``pacing`` / ``emotional_arc`` / ``director`` (name)
  / style-triggering keywords, so six features sit constant at 0.0
  across the whole base goldens and the via gate-exemption path is
  never exercised. The enriched set adds those fields deterministically
  and captures its goldens from the CURRENT reference implementation
  (``feature_vector`` + ``_score_candidate_vs_seed``).

Probes anchor the pairing convention WITHOUT circularity: for a probe
pair the generator RUNS the Python scoring functions, then LOCATES the
result in the golden file by unique full-vector search and records the
FOUND index. If the goldens had been captured under a different
pairing convention, the found index would disagree with the harness's
index for the same entity pair and the probe test fails. Probe pairs
are chosen with non-zero scores so they discriminate.

Regenerate ONLY with the same environment the goldens were captured
under (clean env: no REC_ENSEMBLE_CONFIG / CINE_REC_SCORER overrides)
and the same CPython family; if the golden files are ever re-captured,
regenerate the fixtures in the same change and re-run ``cargo test``
in rust/scoring/.

    python rust/scoring/tests/fixtures/gen_entities.py [--outdir PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from bench_scoring import build  # noqa: E402

from cine_rec_engine.service import (  # noqa: E402
    HEURISTIC_WEIGHTS,
    _entity_profile,
    _score_candidate_vs_seed,
    _score_pair_fast,
    feature_vector,
)

N_CANDS = 120
N_SEEDS = 4
N_PROBES = 5

# Keywords guaranteed to map to style labels (config.KEYWORDS_STYLE) —
# the base bench keywords never do, so style_match is dead there.
STYLE_KW = ["dream", "simulation", "cyberpunk", "dystopia", "memory loss"]
# Genre outside the bench GENRES pool — disjoint from every seed genre.
DISJOINT_GENRE = "War"


def pair_score(cand: dict, seed: dict, w_list: list) -> float:
    """The exact call shape the Rust harness drives (fast path)."""
    cp = _entity_profile(cand, cand, genres_override=cand["genres"])
    sp = _entity_profile(seed, None)
    return _score_pair_fast(
        sp, cp, cand, cand, set(cand["genres"]) or None, 0.0, w_list
    )


def make_probes(
    cands: list, seeds: list, goldens: list, w_list: list, label: str
) -> list:
    """Computed + searched probes (see module docstring).

    A probe records: the entity pair, the index FOUND by unique search
    over the goldens, the COMPUTED vector slots, and the COMPUTED fast
    score. Non-zero-score pairs only (filtered by the COMPUTED score),
    spread across the case space. The pairing formula plays no role
    here: pairs enter only by their entities, and the recorded index
    always comes from the search.
    """
    pairs = [(c, s) for c in cands for s in seeds]
    pool = []
    for i, (c, s) in enumerate(pairs):
        computed_score = pair_score(c, s, w_list)
        if computed_score != 0.0:
            pool.append((i, c, s, computed_score))
    assert len(pool) >= N_PROBES, f"{label}: not enough non-zero pairs for probes"

    step = len(pool) / N_PROBES
    picked = [pool[min(int(k * step), len(pool) - 1)] for k in range(N_PROBES)]
    # dedupe by case index while keeping order (near-uniform picks can
    # collide when the non-zero pool is small)
    seen: set[int] = set()
    chosen = [p for p in picked if not (p[0] in seen or seen.add(p[0]))]

    probes = []
    for i, c, s, computed_score in sorted(chosen):
        computed_vec = feature_vector(c, c, c["genres"], s)
        # LOCATE by unique full-vector search — no pairing formula here
        matches = [
            j for j, g in enumerate(goldens) if g["vec"] == computed_vec
        ]
        assert len(matches) == 1, (
            f"{label}: computed vec for ({c['id']}, {s['id']}) matches "
            f"{len(matches)} goldens — probe must be unique"
        )
        j = matches[0]
        ulp = ulp_distance(computed_score, goldens[j]["score"])
        assert ulp <= 1, f"{label}: computed score {ulp} ULP from golden {j}"
        probes.append(
            {
                "cand_id": c["id"],
                "seed_id": s["id"],
                "golden_index": j,
                "score": computed_score,
                "vec_slots": {str(k): computed_vec[k] for k in (0, 3, 27)},
            }
        )
    return probes


def ulp_distance(a: float, b: float) -> int:
    """Bit-monotonic ULP distance (mirrors the Rust comparator)."""
    import struct

    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b):
        return 2**64 - 1

    def k(x: float) -> int:
        b = struct.unpack("<Q", struct.pack("<d", x))[0]
        sign = 1 << 63
        return (~b & (2**64 - 1)) if b & sign else (b | sign)

    ka, kb = k(a), k(b)
    return abs(ka - kb)


def build_enriched() -> tuple[list, list]:
    """Deterministic enriched entities over the base generator rows.

    Adds exactly the fields the base bench rows lack, so the features
    that are constant-zero on the base set get exercised: via channels
    (+ tmdb_rank), narrative/pacing/emotional_arc, the director-name
    fallback, style keywords, and deliberate genre clashes (with and
    without a via tag → gate-exemption coverage).
    """
    cands_all, seeds_all = build(N_CANDS, N_SEEDS)
    vias = [None, "knn", "tmdb", "director", "writer"]
    narratives = ["linear", "anthology", None]
    pacings = ["contemplative", "propulsive", None]
    arcs = [["feel-good", "uplifting"], ["dark", "tense"], None]

    cands = []
    for k in range(16):
        c = dict(cands_all[k])
        c["via"] = vias[k % len(vias)]
        c["narrative"] = narratives[k % len(narratives)]
        c["pacing"] = pacings[k % len(pacings)]
        c["emotional_arc"] = arcs[k % len(arcs)]
        if c["via"] == "tmdb":
            c["tmdb_rank"] = (k % 5) + 1
        # style keywords on half the candidates (seed side too, below)
        if k % 2 == 0:
            c["keywords"] = list(c["keywords"]) + STYLE_KW[: 1 + k % len(STYLE_KW)]
        # deliberate genre clash block: candidates 8–11 get genres
        # disjoint from every seed. Via tags land as k%5: 8=director,
        # 9=writer, 10=None, 11=knn — so the block covers exempt-and-
        # scored (8, 9, 11) AND gated-to-zero (10, no via) against the
        # same seeds.
        if 8 <= k <= 11:
            c["genres"] = [DISJOINT_GENRE]
        # a couple of named directors to exercise the name fallback
        if k % 4 == 0:
            c["director"] = f"Director Name {k // 4}"
        # EMPTY via is falsy in Python — NOT gate-exempt. Pin that: a
        # clash candidate with via="" must gate to 0.0 like a no-via one
        # (caught live by test_gate_exemptions/hypothesis; anchored
        # here as a permanent pin).
        if k == 15:
            c["via"] = ""
            c["genres"] = [DISJOINT_GENRE]
        cands.append(c)

    seeds = []
    for k in range(4):
        s = dict(seeds_all[k])
        s["narrative"] = narratives[k % len(narratives)]
        s["pacing"] = pacings[(k + 1) % len(pacings)]
        s["emotional_arc"] = arcs[(k + 1) % len(arcs)]
        if k == 2:
            s["keywords"] = list(s["keywords"]) + STYLE_KW[:2]
        if k == 3:
            # animation + Japanese origin → modality "anime" (medium
            # mismatch coverage against the live-action candidates)
            s["genres"] = ["Animation", "Adventure"]
            s["original_language"] = "ja"
        if k == 0:
            # matches candidates 0/4/8/12 named directors (id fallback
            # is separate; this drives the NAME fallback when ids
            # differ)
            s["director"] = "Director Name 0"
        seeds.append(s)
    return cands, seeds


def capture_goldens(cands: list, seeds: list) -> list:
    """Reference-path capture: feature_vector for the vec,
    _score_candidate_vs_seed for the score (same capture path as the
    original golden file)."""
    out = []
    for c in cands:
        for s in seeds:
            vec = feature_vector(c, c, c["genres"], s)
            score = _score_candidate_vs_seed(
                c, c, c["genres"], s, overview_sim=0.0, weights=HEURISTIC_WEIGHTS
            )
            out.append({"vec": vec, "score": score})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(Path(__file__).parent))
    args = p.parse_args()
    outdir = Path(args.outdir)

    w_list = list(HEURISTIC_WEIGHTS.values())
    base_goldens = json.loads(
        (REPO_ROOT / "tests" / "golden_feature_vectors.json").read_text()
    )

    # ── base set: inputs frozen, outputs live in the repo tests dir ──
    cands, seeds = build(N_CANDS, N_SEEDS)
    base_payload = {
        "generator": "benchmarks/bench_scoring.py::build(120, 4), seed 42",
        "n_cands": N_CANDS,
        "n_seeds": N_SEEDS,
        "w_list": w_list,
        "feature_names": list(HEURISTIC_WEIGHTS.keys()),
        "pairing": "golden case i == (cands[i // n_seeds], seeds[i % n_seeds])",
        "probes": make_probes(cands, seeds, base_goldens, w_list, "base"),
        "cands": cands,
        "seeds": seeds,
    }
    _write(outdir / "entities_120_4.json", base_payload)

    # ── enriched set: inputs + captured goldens, one contract ──
    e_cands, e_seeds = build_enriched()
    e_goldens = capture_goldens(e_cands, e_seeds)
    e_payload = {
        "generator": "gen_entities.py::build_enriched() over build(120,4) rows",
        "n_cands": len(e_cands),
        "n_seeds": len(e_seeds),
        "w_list": w_list,
        "feature_names": list(HEURISTIC_WEIGHTS.keys()),
        "pairing": "golden case i == (cands[i // n_seeds], seeds[i % n_seeds])",
        "probes": make_probes(e_cands, e_seeds, e_goldens, w_list, "enriched"),
        "cands": e_cands,
        "seeds": e_seeds,
    }
    _write(outdir / "entities_enriched.json", e_payload)
    _write(outdir / "goldens_enriched.json", e_goldens)

    print(
        f"wrote 3 fixtures: base {len(cands)}×{len(seeds)} "
        f"(+{len(base_payload['probes'])} probes), "
        f"enriched {len(e_cands)}×{len(e_seeds)} = {len(e_goldens)} goldens "
        f"(+{len(e_payload['probes'])} probes)"
    )


def _write(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
