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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg  # noqa: E402

from cine_rec_engine import db, queries, scoring  # noqa: E402
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

async def materialize(pool, judgments: list, use_embeddings: bool = True,
                      blend_cols: list | None = None) -> dict:
    """Everything needed to build feature rows: info / enrichment / cosine.

    With blend_cols: additionally fetch RAW per-column cosines (one batch
    call per column with weights=[1.0] — the public single-column path),
    keyed sim_cols[col][(seed_key, cand_key)]. The default blended sim is
    still materialized for the baseline comparison."""
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

    async def fetch(columns, weights):
        out = {}
        for mt, sids in seeds_by_type.items():
            cids = cands_by_type.get(mt, [])
            if not cids:
                continue
            by_seed = await scoring.semantic_overview_similarity_batch(
                pool, sids, cids, columns=columns, weights=weights)
            for sid, cmap in by_seed.items():
                for cid, s in cmap.items():
                    out[((sid, mt), (cid, mt))] = s
        return out

    sim: dict = {}
    if use_embeddings:
        try:
            sim = await fetch(None, None)  # the engine's current blend
        except Exception as e:  # optional channel — degrade like the engine
            print(f"embedding cosine unavailable ({e}) — "
                  "cosine_sim falls back to text overlap")

    sim_cols: dict = {}
    seed_has_col: dict = {}
    if blend_cols:
        for col in blend_cols:
            try:
                col_sim = await fetch([col], [1.0])
            except Exception as e:
                print(f"blend column {col} unavailable ({e}) — excluded")
                continue
            if col_sim:
                sim_cols[col] = col_sim
        # exact serve-time semantics need to know which columns each SEED
        # holds (a candidate missing a seed-held column falls back)
        from cine_rec_engine import tables as _tables

        for col in sim_cols:
            holders: set = set()
            async with pool.acquire() as conn:
                for mt, ids in seeds_by_type.items():
                    logical = "tmdb_movies" if mt == "movie" else "tmdb_tv"
                    parts = _tables.name(logical).rsplit(".", 1)
                    schema, table = ((parts[0], parts[1]) if len(parts) == 2
                                     else ("public", parts[0]))
                    rows = await db.fetch(
                        conn,
                        f"SELECT id FROM {schema + '.' if schema else ''}{table} "
                        f"WHERE id = ANY($1::bigint[]) AND {col} IS NOT NULL",
                        ids)
                    holders.update((r["id"], mt) for r in rows)
            seed_has_col[col] = holders
    return {"info": info, "enriched": enriched, "sim": sim, "sim_cols": sim_cols,
            "seed_has_col": seed_has_col}


async def discover_blend_columns(pool, override: str | None = None) -> list:
    """Vector-typed embedding_* columns on the physical fact tables.

    The registry (model_spaces) can't see unregistered columns — real
    DBs have them (the demo's embedding_minilm, for one) — so discovery
    goes to information_schema, through the table-name remap."""
    if override:
        cols = [c.strip() for c in override.split(",") if c.strip()]
        for c in cols:
            if not re.match(r"^[a-z_][a-z0-9_]*$", c):
                raise SystemExit(f"invalid blend column name {c!r}")
        return cols
    from cine_rec_engine import tables

    found: set = set()
    async with pool.acquire() as conn:
        for logical in ("tmdb_movies", "tmdb_tv"):
            parts = tables.name(logical).rsplit(".", 1)
            schema, table = ((parts[0], parts[1]) if len(parts) == 2
                             else ("public", parts[0]))
            rows = await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = $1 AND table_name = $2
                     AND column_name LIKE 'embedding%'
                     AND udt_name = 'vector'""",
                schema or "public", table)
            found.update(r["column_name"] for r in rows)
    return sorted(found)


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
        init: list | None = None, dim: int | None = None) -> list:
    """Deterministic full-batch pairwise logistic gradient descent.

    Objective: mean -log sigmoid(w.g - w.b) + (alpha/2)|w - init|^2 —
    the L2 penalty anchors the DISTANCE FROM THE STARTING POINT (the
    heuristic prior), so a small judgment set cannot drag the scorer
    away without evidence. With a zero start this is classic ridge.
    dim defaults to the public feature count; blend tuning passes an
    expanded dim (features minus cosine_sim, plus one per column)."""
    dim = dim or len(PUBLIC_FEATURES)
    if init is not None and len(init) != dim:
        raise ValueError(f"init has {len(init)} entries, dim is {dim}")
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


# ------------------------------------------------------------ blend tuning

_COS_POS = PUBLIC_FEATURES.index("cosine_sim")


def build_blend_records(triplets: list, mat: dict, blend_cols: list) -> tuple:
    """Per-triplet records with the cosine slot split per column.

    rec = {"feats21": [...], "sims": {col: s}, "fallback": float} —
    feats21 = the public features without cosine_sim, fallback = what
    feature_vector's cosine slot holds with NO embedding similarity
    (the text-overlap path), used when the pair misses a column."""
    base_cache: dict = {}
    cos_cols = [c for c in blend_cols if c in mat["sim_cols"]]

    def rec(seed_key, cand_key):
        if (seed_key, cand_key) in base_cache:
            return base_cache[(seed_key, cand_key)]
        out = None
        seed = mat["info"].get(seed_key)
        cand = mat["info"].get(cand_key)
        if seed is not None and cand is not None:
            merged = {**cand, **{k: v for k, v in
                                 mat["enriched"].get(cand_key, {}).items()
                                 if v is not None}}
            full = feature_vector(merged, merged,
                                  merged.get("genres") or [], seed, 0.0)
            feats = [v for i, v in zip(_PUBLIC_IDX, full)]
            out = {
                "key": cand_key,
                "feats21": feats[:_COS_POS] + feats[_COS_POS + 1:],
                "fallback": feats[_COS_POS],
                "sims": {c: mat["sim_cols"][c].get((seed_key, cand_key))
                         for c in cos_cols},
            }
        base_cache[(seed_key, cand_key)] = out
        return out

    records, dropped = [], 0
    for s_key, g_key, b_key in triplets:
        g, b = rec(s_key, g_key), rec(s_key, b_key)
        if g is None or b is None:
            dropped += 1
            continue
        records.append((s_key, g, b))
    return records, dropped, cos_cols


def expand_rows(records: list, cos_cols: list) -> list:
    """Linear expansion: 21 features + one raw sim per column (missing=0).

    Substituting gamma_b = w_cos * beta_b / sum(beta) into the serve-time
    renormalized average makes the whole model linear in these rows —
    one logistic pass fits scorer weights and blend together."""
    return [
        (g["feats21"] + [g["sims"].get(c) or 0.0 for c in cos_cols],
         b["feats21"] + [b["sims"].get(c) or 0.0 for c in cos_cols])
        for _s, g, b in records
    ]


def decompose(gamma_cols: list, cos_cols: list) -> tuple:
    """Expanded per-column coefficients -> (beta, w_cos).

    beta_b = max(gamma_b, 0) (serve requires positive blend weights);
    w_cos = sum(gamma) — then w_cos * (sum(beta*s)/sum(beta)) equals
    sum(gamma*s) exactly, so the deployed model IS the fitted one."""
    beta = {c: max(g, 0.0) for c, g in zip(cos_cols, gamma_cols) if max(g, 0.0) > 0}
    return beta, sum(gamma_cols)


def deployable_cosine(rec: dict, seed_key, beta: dict, all_cols: list,
                      seed_has_col: dict) -> float:
    """cosine_sim exactly as the engine serves it under `beta`:
    renormalized weighted average over the columns the SEED holds, and
    a candidate missing any of them falls back to text overlap."""
    active = [c for c in all_cols if seed_key in seed_has_col.get(c, ())]
    if not beta or not active:
        return rec["fallback"]
    if any(rec["sims"].get(c) is None for c in active):
        return rec["fallback"]
    w_sum = sum(beta.get(c, 0.0) for c in active)
    if w_sum <= 0:
        return rec["fallback"]
    return sum(beta[c] * rec["sims"][c] for c in active) / w_sum


def blend_model_accuracy(records: list, beta: dict, cos_cols: list,
                         w21: list, w_cos: float, seed_has_col: dict) -> float:
    """Pairwise accuracy of the DEPLOYABLE model: 21 fitted features +
    reconstructed serve-time cosine. No leakage of the linearization."""
    ok = 0
    for s_key, g, b in records:
        gs = sum(w * v for w, v in zip(w21, g["feats21"])) \
            + w_cos * deployable_cosine(g, s_key, beta, cos_cols, seed_has_col)
        bs = sum(w * v for w, v in zip(w21, b["feats21"])) \
            + w_cos * deployable_cosine(b, s_key, beta, cos_cols, seed_has_col)
        ok += gs > bs
    return ok / len(records) if records else 0.0


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
        blend_cols: list = []
        if args.tune_blend:
            blend_cols = await discover_blend_columns(pool, args.blend_columns)
            if len(blend_cols) < 2:
                print(f"--tune-blend: {len(blend_cols)} embedding column(s) "
                      f"found ({', '.join(blend_cols) or 'none'}) — blend "
                      "tuning needs at least two; fitting scorer weights only")
                blend_cols = []
            else:
                print(f"blend tuning over: {', '.join(blend_cols)}")
        mat = await materialize(pool, judgments,
                                use_embeddings=not args.no_embeddings,
                                blend_cols=blend_cols)
    finally:
        await pool.close()

    records, dropped, cos_cols = ([], 0, [])
    if blend_cols:
        records, dropped, cos_cols = build_blend_records(triplets, mat, blend_cols)

    rows, dropped_plain = build_rows(triplets, mat)
    dropped = max(dropped, dropped_plain)
    if dropped:
        print(f"dropped {dropped} triplets whose ids are missing from the catalog")
    if len(records or rows) < args.min_pairs:
        print(f"only {len(records or rows)} usable triplets after catalog lookup "
              f"(minimum {args.min_pairs})", file=sys.stderr)
        return 2

    init = [float(HEURISTIC_WEIGHTS[n]) for n in PUBLIC_FEATURES]

    if records and len(cos_cols) >= 2:
        return _run_blend(args, triplets, mat, records, cos_cols, init)

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
    _write_artifact(args, artifact, rows, fit_acc, base_acc)
    return 0


def _weights_dict(w21: list, w_cos: float) -> dict:
    out = {}
    j = 0
    for name in PUBLIC_FEATURES:
        if name == "cosine_sim":
            out[name] = round(w_cos, 6)
        else:
            out[name] = round(w21[j], 6)
            j += 1
    return out


def _heuristic21() -> list:
    return [float(HEURISTIC_WEIGHTS[n]) for n in PUBLIC_FEATURES
            if n != "cosine_sim"]


def _baseline_accuracy(records: list, mat: dict) -> float:
    """Heuristic weights + the engine's CURRENT blend behavior."""
    h21 = _heuristic21()
    h_cos = float(HEURISTIC_WEIGHTS["cosine_sim"])
    ok = 0
    for s_key, g, b in records:
        def cos_for(rec):
            return mat["sim"].get((s_key, rec["key"]), rec["fallback"])
        gs = sum(w * v for w, v in zip(h21, g["feats21"])) \
            + h_cos * cos_for(g)
        bs = sum(w * v for w, v in zip(h21, b["feats21"])) \
            + h_cos * cos_for(b)
        ok += gs > bs
    return ok / len(records) if records else 0.0


def _run_blend(args, triplets, mat, records, cos_cols, init22) -> int:
    seed_keys = sorted({r[0] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(seed_keys)
    n_hold = max(1, round(len(seed_keys) * args.holdout))
    hold_seeds = set(seed_keys[:n_hold])
    train = [r for r in records if r[0] not in hold_seeds]
    test = [r for r in records if r[0] in hold_seeds]

    h21 = _heuristic21()
    init_exp = h21 + [1.0] * len(cos_cols)
    exp_train = expand_rows(train, cos_cols)
    fitted = fit(exp_train, alpha=args.alpha, epochs=args.epochs,
                 init=init_exp, dim=len(init_exp))
    w21, gamma = fitted[:len(h21)], fitted[len(h21):]
    beta, w_cos = decompose(gamma, cos_cols)

    base_acc = _baseline_accuracy(test, mat)
    fit_acc = blend_model_accuracy(test, beta, cos_cols, w21, w_cos,
                                   mat["seed_has_col"])
    print(f"triplets: {len(records)} (train {len(train)} / holdout {len(test)}; "
          f"holdout = {n_hold} whole seeds)")
    print(f"columns: {', '.join(cos_cols)}")
    print(f"heuristic + current blend, holdout accuracy: {base_acc:.3f}")
    print(f"fitted scorer + fitted blend:            {fit_acc:.3f}"
          f"   ({fit_acc - base_acc:+.3f})")
    b_total = sum(beta.values()) or 1.0
    print("fitted blend (normalized): "
          + ", ".join(f"{c} {beta.get(c, 0.0) / b_total:.3f}" for c in cos_cols))
    print(f"fitted cosine_sim weight: {w_cos:.2f} "
          f"(heuristic {HEURISTIC_WEIGHTS['cosine_sim']:.2f})")

    exp_all = expand_rows(records, cos_cols)
    final = fit(exp_all, alpha=args.alpha, epochs=args.epochs,
                init=init_exp, dim=len(init_exp))
    f21, fgamma = final[:len(h21)], final[len(h21):]
    fbeta, fw_cos = decompose(fgamma, cos_cols)
    if not fbeta:
        print("fitted blend came out empty — artifact keeps scorer weights "
              "only (no cosine_blend section)")
    artifact = {
        "model": "pairwise-logistic-sgd",
        "alpha": args.alpha,
        "triplets": len(records),
        "_schema_note": {
            "fitted_by_owner": True,
            "note": ("fitted by the artifact owner from their own preference"
                     " judgments via eval/tune_weights.py (public tool); not"
                     " the published weights"),
        },
        "feature_names": PUBLIC_FEATURES,
        "weights": _weights_dict(f21, fw_cos),
    }
    if fbeta:
        bsum = sum(fbeta.values())
        artifact["cosine_blend"] = {c: round(w / bsum, 6)
                                    for c, w in sorted(fbeta.items())}
    _write_artifact(args, artifact, records, fit_acc, base_acc)
    if artifact.get("cosine_blend"):
        env = "CINE_REC_COSINE_BLEND=" + ",".join(
            f"{c}:{w}" for c, w in artifact["cosine_blend"].items())
        print(f"env-only equivalent: {env}")
    return 0


def _write_artifact(args, artifact, rows, fit_acc, base_acc) -> None:
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n",
                              encoding="utf-8")
    print(f"artifact: {args.out} (fitted on all {len(rows)} triplets; "
          "holdout metrics above came from the train-split model)")
    print(f"serve with: CINE_REC_SCORER=learned "
          f"CINE_REC_WEIGHTS={Path(args.out).resolve()}")
    if fit_acc < base_acc - 0.02:
        print("WARNING: fitted model is WORSE than the heuristic on holdout — "
              "do not deploy this artifact", file=sys.stderr)


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
    p.add_argument("--tune-blend", action="store_true",
                   help="also fit per-embedding-column blend weights "
                        "(needs >=2 embedding columns in the catalog)")
    p.add_argument("--blend-columns", default=None,
                   help="comma-separated embedding columns to blend-tune "
                        "(default: discovered vector-typed embedding_* "
                        "columns on the fact tables)")
    args = p.parse_args()
    if not args.dsn:
        p.error("no database: pass --dsn or set DATABASE_URL")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
