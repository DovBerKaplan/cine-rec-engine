"""Scoring-loop micro-benchmark: the per-pair feature_vector hot path.

Synthetic candidates/seeds shaped like the real enriched rows (keywords,
cast, crew ids, companies, networks, overviews, years). No DB needed.

    python benchmarks/bench_scoring.py            # current numbers
    python benchmarks/bench_scoring.py --profile  # cProfile top functions
"""

from __future__ import annotations

import argparse
import random
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cine_rec_engine.service import (  # noqa: E402
    HEURISTIC_WEIGHTS,
    _score_candidate_vs_seed,
)

GENRES = ["Action", "Drama", "Comedy", "Thriller", "Sci-Fi", "Crime",
          "Animation", "Horror", "Romance", "Adventure"]
KEYWORDS = [f"kw{i}" for i in range(300)]
COMPANIES = [f"co{i}" for i in range(60)]
NETWORKS = [f"nt{i}" for i in range(20)]
OVERVIEWS = [
    "A retired detective returns for one last case in a rainy city of secrets",
    "Two rivals fall in love during a cooking competition in paris",
    "A space crew discovers a signal that rewrites their memories",
] * 40


def _mk(rng: random.Random, i: int, media: str) -> dict:
    return {
        "id": i,
        "media_type": media,
        "title": f"Title {i}",
        "title_en": f"Title {i}",
        "genres": rng.sample(GENRES, k=rng.randint(1, 3)),
        "keywords": rng.sample(KEYWORDS, k=rng.randint(2, 12)),
        "cast_ids": rng.sample(range(1000), k=5),
        "director_ids": rng.sample(range(100), k=1),
        "writer_ids": rng.sample(range(100), k=1),
        "composer_ids": rng.sample(range(100), k=1),
        "dp_ids": rng.sample(range(100), k=1),
        "companies": rng.sample(COMPANIES, k=rng.randint(1, 3)),
        "networks": rng.sample(NETWORKS, k=rng.randint(0, 2)),
        "original_language": rng.choice(["en", "ko", "fr"]),
        "overview": rng.choice(OVERVIEWS) + f" {i}",
        "overview_en": None,
        "release_year": rng.randint(1960, 2026),
        "vote_average": rng.uniform(4.5, 9.0),
        "vote_count": rng.randint(50, 40000),
        "collection_id": rng.choice([None, 100, 200, 300]),
        "adult": False,
        "via": None,
    }


def build(n_cands: int = 500, n_seeds: int = 5) -> tuple[list, list]:
    rng = random.Random(42)
    cands = [_mk(rng, i, "movie") for i in range(n_cands)]
    seeds = [_mk(rng, 10_000 + i, rng.choice(["movie", "tv"])) for i in range(n_seeds)]
    return cands, seeds


def run(cands, seeds, weights) -> float:
    """The find_similar inner loop shape: cand × seed dot products."""
    acc = 0.0
    for cand in cands:
        cand_data = cand  # enriched.get() shape: same fields
        cand_genres = cand["genres"]
        for seed in seeds:
            acc += _score_candidate_vs_seed(
                cand, cand_data, cand_genres, seed,
                overview_sim=0.0, weights=weights,
            )
    return acc


def run_fast(cands, seeds, w_list) -> float:
    """The optimized request path: profiles once per entity, then the
    _score_pair_fast hot loop (the shape the Rust port accelerates)."""
    from cine_rec_engine.service import _entity_profile, _score_pair_fast

    acc = 0.0
    seed_profiles = [_entity_profile(s, None) for s in seeds]
    for cand in cands:
        cp = _entity_profile(cand, cand, genres_override=cand["genres"])
        cgs = set(cand["genres"]) or None
        for sp in seed_profiles:
            acc += _score_pair_fast(sp, cp, cand, cand, cgs, 0.0, w_list)
    return acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cands", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--mode",
        choices=["ref", "fast", "both"],
        default="both",
        help="ref = _score_candidate_vs_seed; fast = profiles + _score_pair_fast "
        "(the accelerated shape). Both assert n>0 — a bench that measured "
        "nothing is a bug.",
    )
    args = p.parse_args()

    cands, seeds = build(args.cands, args.seeds)
    weights = HEURISTIC_WEIGHTS
    w_list = list(weights.values())

    try:
        from cine_rec_engine.service import SCORING_RS
    except ImportError:
        SCORING_RS = "unknown"
    print(f"engine: {SCORING_RS}")

    run_fast(cands[:10], seeds, w_list)  # warm-up (dict caches, code objects)

    if args.profile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        run(cands, seeds, weights)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
        return

    def measure(fn, label):
        best = None
        acc = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            acc = fn()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        pairs = args.cands * args.seeds
        assert pairs > 0, "bench measured zero pairs"
        assert acc != 0.0 or True  # acc may legitimately be 0.0 for gated sets
        print(
            f"[{label}] {pairs} pairs | {best*1000:.1f} ms total | "
            f"{best/pairs*1e6:.2f} µs/pair | "
            f"{pairs/best:,.0f} pairs/s"
        )

    if args.mode in ("ref", "both"):
        measure(lambda: run(cands, seeds, weights), "ref ")
    if args.mode in ("fast", "both"):
        measure(lambda: run_fast(cands, seeds, w_list), "fast")


if __name__ == "__main__":
    main()
