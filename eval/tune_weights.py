"""Fit the 22-feature scorer to YOUR OWN preference judgments.

The judgments file uses the public eval format (eval/judgments.jsonl):

    {"seed": [tmdb_id, media_type],
     "good": [[id, media_type], ...],   # should rank ABOVE the bad ones
     "bad":  [[id, media_type], ...]}

Every (good x bad) pair for a seed is one preference triplet. Pipeline:
connect to your catalog DB -> materialize the engine's own features for
every (seed, candidate) pair via the public chain (get_movie_info_batch
+ enrich_candidates_batch + embedding cosine when available) -> fit
deterministic pairwise-logistic SGD with L2, initialized from the
heuristic weights -> report holdout accuracy -> write a weights.json
schema artifact.

    python eval/tune_weights.py --dsn postgresql://... \\
        --judgments mine.jsonl --out my-weights.json

    # verify against the same judgments end-to-end:
    python eval/eval.py --dsn ... --judgments mine.jsonl

    # serve the fitted weights:
    CINE_REC_SCORER=learned CINE_REC_WEIGHTS=/abs/my-weights.json cine-rec serve

Honesty notes: holdout split is by WHOLE SEEDS (no leakage); metrics
reported are yours, on your data, with the heuristic baseline shown for
comparison; small judgment sets overfit — the L2 (alpha) keeps the fit
near the heuristic until your data earns a move. This tool is a generic
textbook fit over user-supplied judgments; it contains no labeled
corpora and nothing about how the published weights were produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg  # noqa: E402

from cine_rec_engine import queries, scoring  # noqa: E402
from cine_rec_engine.service import (  # noqa: E402
    FEATURE_NAMES,
    HEURISTIC_WEIGHTS,
    feature_vector,
)

# The public artifact schema: exactly the features the bundled
# weights.json carries (22; the 6 reserved cinematic slots stay
# heuristic in public builds).
_BUNDLED = json.loads(
    (Path(__file__).parent.parent / "cine_rec_engine" / "weights.json").read_text()
)
PUBLIC_FEATURES = [n for n in FEATURE_NAMES if n in set(_BUNDLED["feature_names"])]
_PUBLIC_IDX = [FEATURE_NAMES.index(n) for n in PUBLIC_FEATURES]

VALID_MEDIA = ("movie", "tv")


# ---------------------------------------------------------------- judgments

def load_judgments(path: str) -> list:
    out = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{lineno}: bad JSON ({e})")
        problems = []
        for key in ("seed", "good", "bad"):
            if key not in rec:
                problems.append(f"missing {key!r}")
        seed = rec.get("seed")
        if not (isinstance(seed, list) and len(seed) == 2
                and seed[1] in VALID_MEDIA):
            problems.append("seed must be [tmdb_id, 'movie'|'tv']")
        for key in ("good", "bad"):
            val = rec.get(key)
            if not (isinstance(val, list) and val
                    and all(isinstance(p, list) and len(p) == 2
                            and p[1] in VALID_MEDIA for p in val)):
                problems.append(f"{key} must be a non-empty [[id, media_type], ...]")
        if problems:
            raise SystemExit(f"{path}:{lineno}: " + "; ".join(problems))
        out.append(rec)
    if not out:
        raise SystemExit(f"{path}: no judgment lines")
    return out


def expand_triplets(judgments: list) -> list:
    """(seed_key, good_key, bad_key) for every good x bad pair per seed."""
    out = []
    for j in judgments:
        seed = tuple(j["seed"])
        for g in j["good"]:
            for b in j["bad"]:
                out.append((seed, tuple(g), tuple(b)))
    return out


# ------------------------------------------------------------ features

async def materialize(pool, judgments: list, use_embeddings: bool = True) -> dict:
    """Everything needed to build feature rows: info / enrichment / cosine."""
    seeds_by_type: dict = {}
    cands_by_type: dict = {}
    for j in judgments:
        sid, smt = j["seed"]
        seeds_by_type.setdefault(smt, [])
        if sid not in seeds_by_type[smt]:
            seeds_by_type[smt].append(sid)
        for cid, cmt in j["good"] + j["bad"]:
            cands_by_type.setdefault(cmt, [])
            if cid not in cands_by_type[cmt]:
                cands_by_type[cmt].append(cid)

    # per medium: ids collide across movie/tv, so never mix the keys
    info: dict = {}
    for mt, ids in {**seeds_by_type, **cands_by_type}.items():
        todo = [i for i in ids if (i, mt) not in info]
        if not todo:
            continue
        batch = await queries.get_movie_info_batch(pool, todo, media_type=mt)
        for i, d in batch.items():
            info[(i, mt)] = d

    enriched: dict = {}
    for mt, ids in cands_by_type.items():
        batch = await queries.enrich_candidates_batch(
            pool, ids, media_types=[mt] * len(ids))
        for i, d in batch.items():
            enriched[(i, mt)] = d

    sim: dict = {}
    if use_embeddings:
        try:
            for mt, sids in seeds_by_type.items():
                cids = cands_by_type.get(mt, [])
                if not cids:
                    continue
                by_seed = await scoring.semantic_overview_similarity_batch(
                    pool, sids, cids)
                for sid, cmap in by_seed.items():
                    for cid, s in cmap.items():
                        sim[((sid, mt), (cid, mt))] = s
        except Exception as e:  # optional channel — degrade like the engine
            print(f"embedding cosine unavailable ({e}) — "
                  "cosine_sim falls back to text overlap")
    return {"info": info, "enriched": enriched, "sim": sim}


def build_rows(triplets: list, mat: dict) -> tuple:
    """(good_vec, bad_vec) 22-dim rows; ids missing from the catalog drop."""
    cache: dict = {}

    def vec(seed_key, cand_key):
        if (seed_key, cand_key) in cache:
            return cache[(seed_key, cand_key)]
        seed = mat["info"].get(seed_key)
        cand = mat["info"].get(cand_key)
        row = None
        if seed is not None and cand is not None:
            merged = {**cand, **{k: v for k, v in
                                 mat["enriched"].get(cand_key, {}).items()
                                 if v is not None}}
            full = feature_vector(
                merged, merged, merged.get("genres") or [], seed,
                mat["sim"].get((seed_key, cand_key), 0.0))
            row = [full[i] for i in _PUBLIC_IDX]
        cache[(seed_key, cand_key)] = row
        return row

    rows, dropped = [], 0
    for s_key, g_key, b_key in triplets:
        g, b = vec(s_key, g_key), vec(s_key, b_key)
        if g is None or b is None:
            dropped += 1
            continue
        rows.append((g, b))
    return rows, dropped


# ---------------------------------------------------------------- fit

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z)) if z < 35 else 1.0
    e = math.exp(z) if z > -35 else 0.0
    return e / (1.0 + e)


def fit(rows: list, alpha: float = 1e-4, epochs: int = 300, lr: float = 0.05,
        init: list | None = None) -> list:
    """Deterministic full-batch pairwise logistic gradient descent.

    Objective: mean -log sigmoid(w.g - w.b) + (alpha/2)|w - init|^2 —
    the L2 penalty anchors the DISTANCE FROM THE STARTING POINT (the
    heuristic prior), so a small judgment set cannot drag the scorer
    away without evidence. With a zero start this is classic ridge."""
    dim = len(PUBLIC_FEATURES)
    start = list(init) if init else [0.0] * dim
    w = list(start)
    n = len(rows) or 1
    for _ in range(epochs):
        grad = [0.0] * dim
        for g, b in rows:
            d = [gi - bi for gi, bi in zip(g, b)]
            coeff = _sigmoid(sum(wi * di for wi, di in zip(w, d))) - 1.0
            for j, dj in enumerate(d):
                grad[j] += coeff * dj
        for j in range(dim):
            w[j] -= lr * (grad[j] / n + alpha * (w[j] - start[j]))
    return w


def pairwise_accuracy(weights: list, rows: list) -> float:
    if not rows:
        return 0.0
    ok = sum(1 for g, b in rows
             if sum(wi * gi for wi, gi in zip(weights, g))
             > sum(wi * bi for wi, bi in zip(weights, b)))
    return ok / len(rows)


def split_by_seed(triplets: list, rows: list, holdout: float, seed: int) -> tuple:
    """Hold out WHOLE seeds (not individual pairs) — no leakage of a
    seed's own preference into the training side."""
    seed_keys = sorted({t[0] for t in triplets})
    rng = random.Random(seed)
    rng.shuffle(seed_keys)
    n_hold = max(1, round(len(seed_keys) * holdout)) if seed_keys else 0
    hold_seeds = set(seed_keys[:n_hold])
    train = [r for t, r in zip(triplets, rows) if t[0] not in hold_seeds]
    test = [r for t, r in zip(triplets, rows) if t[0] in hold_seeds]
    return train, test, n_hold


# ---------------------------------------------------------------- cli

async def run(args) -> int:
    judgments = load_judgments(args.judgments)
    triplets = expand_triplets(judgments)
    if len(triplets) < args.min_pairs:
        print(f"only {len(triplets)} triplets from {len(judgments)} judgments "
              f"(minimum {args.min_pairs}) — collect more judgments or lower "
              "--min-pairs deliberately", file=sys.stderr)
        return 2
    if len(triplets) < 200:
        print(f"note: {len(triplets)} triplets is a small sample — the L2 "
              "prior keeps the fit near the heuristic; treat big weight "
              "moves with suspicion")

    pool = await asyncpg.create_pool(args.dsn)
    try:
        mat = await materialize(pool, judgments, use_embeddings=not args.no_embeddings)
    finally:
        await pool.close()
    rows, dropped = build_rows(triplets, mat)
    if dropped:
        print(f"dropped {dropped} triplets whose ids are missing from the catalog")
    if len(rows) < args.min_pairs:
        print(f"only {len(rows)} usable triplets after catalog lookup "
              f"(minimum {args.min_pairs})", file=sys.stderr)
        return 2

    init = [float(HEURISTIC_WEIGHTS[n]) for n in PUBLIC_FEATURES]
    train, test, n_hold = split_by_seed(triplets, rows, args.holdout, args.seed)
    fitted = fit(train, alpha=args.alpha, epochs=args.epochs, init=init)

    base_acc = pairwise_accuracy(init, test)
    fit_acc = pairwise_accuracy(fitted, test)
    print(f"triplets: {len(rows)} (train {len(train)} / holdout {len(test)}; "
          f"holdout = {n_hold} whole seeds)")
    print(f"heuristic baseline holdout accuracy: {base_acc:.3f}")
    print(f"fitted holdout accuracy:          {fit_acc:.3f}"
          f"   ({fit_acc - base_acc:+.3f})")
    deltas = sorted(
        ((n, w - h) for n, w, h in zip(PUBLIC_FEATURES, fitted, init)),
        key=lambda kv: -abs(kv[1]))[:5]
    print("largest moves vs heuristic: "
          + ", ".join(f"{n} {d:+.2f}" for n, d in deltas))

    final = fit(rows, alpha=args.alpha, epochs=args.epochs, init=init)
    artifact = {
        "model": "pairwise-logistic-sgd",
        "alpha": args.alpha,
        "triplets": len(rows),
        "_schema_note": {
            "fitted_by_owner": True,
            "note": ("fitted by the artifact owner from their own preference"
                     " judgments via eval/tune_weights.py (public tool); not"
                     " the published weights"),
        },
        "feature_names": PUBLIC_FEATURES,
        "weights": {n: round(w, 6) for n, w in zip(PUBLIC_FEATURES, final)},
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n",
                              encoding="utf-8")
    print(f"artifact: {args.out} (fitted on all {len(rows)} triplets; "
          "holdout metrics above came from the train-split model)")
    print(f"serve with: CINE_REC_SCORER=learned "
          f"CINE_REC_WEIGHTS={Path(args.out).resolve()}")
    if fit_acc < base_acc - 0.02:
        print("WARNING: fitted model is WORSE than the heuristic on holdout — "
              "do not deploy this artifact", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="tune-weights",
        description="Fit the scorer weights to your own preference judgments.")
    p.add_argument("--dsn", default=os.getenv("DATABASE_URL", ""))
    p.add_argument("--judgments",
                   default=str(Path(__file__).parent / "judgments.jsonl"))
    p.add_argument("--out", default="my-weights.json")
    p.add_argument("--alpha", type=float, default=1e-4,
                   help="L2 strength — larger keeps the fit nearer the heuristic")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--holdout", type=float, default=0.2,
                   help="fraction of SEEDS held out for validation")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-pairs", type=int, default=50)
    p.add_argument("--no-embeddings", action="store_true",
                   help="skip the embedding cosine (text-overlap fallback)")
    args = p.parse_args()
    if not args.dsn:
        p.error("no database: pass --dsn or set DATABASE_URL")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
