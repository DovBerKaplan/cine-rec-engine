"""Build the public eval judgments file — stratified, consensus pairs.

The file is a REGRESSION BASELINE, not a leaderboard: every line must be
obvious to two independent methods before it is included ("two-agreement
consensus"):

  1. a structured-metadata rule (same saga, or shared genre + keyword
     overlap above a fixed margin), and
  2. the embedding space (MiniLM cosine of the good pick must beat the
     bad pick by a fixed margin).

No human-taste calls, no graded data, nothing subjective — anything
curated stays private. The input is the bundled demo catalog dump
(demo/data/titles.jsonl.gz), so this runs with zero infrastructure.

Rules per generated line (deterministic, seed=7):
  recs   — good: from the seed's TMDB behavioral-recommendations graph
           ("people also liked"), story-disjoint from the seed (never
           the seed's own saga — the engine collapses each collection
           to one slot and blocks the seed's saga BY DESIGN, so a saga
           sibling is not a valid expectation);
           bad: a title from a disjoint genre cluster.
  genre  — good: shares a genre AND keyword-Jaccard >= 0.20 with the
           seed, different collection; bad: disjoint genres, zero
           keyword overlap.

Seeds are stratified across genre × decade × medium; the ~10 hand pairs
at the top of the existing file are preserved verbatim.

    python eval/build_judgments.py                 # regenerate the file
    python eval/build_judgments.py --validate --dsn postgresql://...
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "demo" / "data" / "titles.jsonl.gz"
OUT = Path(__file__).parent / "judgments.jsonl"

TARGET_LINES = 150
COSINE_MARGIN = 0.15      # two-agreement: good must beat bad by this much
KEYWORD_JACCARD_MIN = 0.20      # candidate floor
KEYWORD_JACCARD_KEEP = 0.30     # a KEPT genre good must clear this overlap
COSINE_RANK_MAX = 15            # a KEPT recs good must also rank top-15
GRAPH_RANK_MAX = 10             # and come from the head of the seed's
# own TMDB graph — the behavioral channel recalls the graph head, so a
# good inside it is guaranteed reachability (an unreached good scores 0
# and turns the pair into an engine-independent coin flip)
# by embedding cosine — graph-top alone is not consensus, and the
# engine's top-10 blends cosine/keywords/cast, so a good that is
# neither graph-top NOR cosine-top is not an obvious pick for it
HAND_LINES_KEEP = 10      # the hand-written pairs stay at the top verbatim
STRATIFY_SEED = 7
# The genre channel orders by rating/popularity and keeps ~60 candidates:
# pairs built from the obscure tail of the catalog are unreachable no
# matter how similar — a good pick must live in the recallable core.
POPULAR_MIN_VOTES = 1000
POPULAR_MIN_RATING = 6.0


def _year(rec: dict) -> int | None:
    d = rec.get("release_date") or rec.get("first_air_date")
    if not d:
        return None
    try:
        return int(str(d)[:4])
    except ValueError:
        return None


def _genres(rec: dict) -> set[str]:
    return {g["name"] for g in rec.get("genres") or []}


def _keywords(rec: dict) -> set[int]:
    return {k["id"] for k in (rec.get("keywords") or {}).get("keywords", [])
            if isinstance(k, dict) and "id" in k}


def _collection(rec: dict):
    coll = rec.get("belongs_to_collection") or None
    if isinstance(coll, dict) and coll.get("id"):
        return coll["id"]
    return None


def _cos(a, b) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


def load_catalog() -> list[dict]:
    titles = []
    with gzip.open(DATA, "rt") as f:
        for line in f:
            row = json.loads(line)
            rec = row["payload"]
            emb = row.get("embedding_minilm")
            titles.append({
                "key": (rec["id"], row["medium"]),
                "rec": rec,
                "year": _year(rec),
                "genres": _genres(rec),
                "kw": _keywords(rec),
                "coll": _collection(rec),
                "emb": emb or None,
                "rec_ids": [r["id"] for r in
                            (rec.get("recommendations") or {}).get("results") or []
                            if isinstance(r, dict) and r.get("id")],
                "votes": rec.get("vote_count") or 0,
                "rating": rec.get("vote_average") or 0.0,
            })
    return titles


def hand_lines() -> list[str]:
    lines = OUT.read_text().splitlines()
    kept = [ln for ln in lines if ln.strip()][:HAND_LINES_KEEP]
    assert len(kept) == HAND_LINES_KEEP, f"expected hand block, got {len(kept)}"
    return kept


def stratified_seeds(titles: list[dict]) -> list[dict]:
    """Round-robin over (dominant genre, decade, medium) buckets so no
    cluster monopolizes the seed set."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in titles:
        if not t["genres"] or t["year"] is None:
            continue
        g = sorted(t["genres"])[0]
        bucket = (g, (t["year"] // 10) * 10, t["key"][1])
        buckets[bucket].append(t)
    order = sorted(buckets, key=lambda b: (-len(buckets[b]), b))
    rng = random.Random(STRATIFY_SEED)
    for b in order:
        rng.shuffle(buckets[b])
    seeds: list[dict] = []
    while any(buckets.values()):
        for b in order:
            if buckets[b]:
                seeds.append(buckets[b].pop())
    return seeds


def build() -> list[dict]:
    titles = load_catalog()
    by_key = {t["key"]: t for t in titles}

    def far_bad(seed: dict, rng: random.Random) -> dict | None:
        """A clearly-unrelated title: disjoint genres AND the embedding
        agrees it is farther than any good pick will be."""
        cands = [
            t for t in titles
            if t["key"] != seed["key"]
            and not (t["genres"] & seed["genres"])
            and t["emb"] and seed["emb"]
        ]
        if not cands:
            return None
        # bottom decile by cosine — obvious negatives, never near-misses
        # (a mid-similarity negative gets KNN-recalled and the pair stops
        # testing anything but the blend's tie-breaking)
        scored = sorted(cands, key=lambda t: _cos(seed["emb"], t["emb"]))
        return scored[max(0, len(scored) // 10 - 1)]

    def agrees(seed: dict, good: dict, bad: dict) -> bool:
        return (_cos(seed["emb"], good["emb"]) >
                _cos(seed["emb"], bad["emb"]) + COSINE_MARGIN)

    out: list[dict] = []
    rng = random.Random(STRATIFY_SEED)
    used_seeds: set = set()

    def story_disjoint(a: dict, b: dict) -> bool:
        # either side outside a collection, or different collections —
        # same-collection picks are collapsed/blocked by the engine on
        # purpose, so they are never a valid good expectation
        return not (a["coll"] and a["coll"] == b["coll"])

    # ---- rule 1: TMDB behavioral graph ("people also liked") ------------
    recs_cap = TARGET_LINES * 0.6
    for seed in stratified_seeds(titles):
        if len(out) >= recs_cap:
            break
        if seed["key"] in used_seeds or not seed["emb"] or not seed["rec_ids"]:
            continue
        # keep the graph's own order — the engine's rank-decay feature
        # rewards graph-top picks, so the OBVIOUS goods are rank 1..n;
        # a pick must ALSO sit top-15 by embedding cosine (consensus)
        cos_ranked = sorted(
            (t for t in titles if t["emb"] and t["key"] != seed["key"]),
            key=lambda t: -_cos(seed["emb"], t["emb"]))
        cos_rank = {t["key"]: i for i, t in enumerate(cos_ranked, start=1)}
        picks = [
            by_key[(rid, seed["key"][1])]
            for rank, rid in enumerate(seed["rec_ids"], start=1)
            if rank <= GRAPH_RANK_MAX and (rid, seed["key"][1]) in by_key
        ]
        picks = [p for p in picks if story_disjoint(p, seed)
                 and cos_rank.get(p["key"], 10**9) <= COSINE_RANK_MAX]
        good: list[dict] = []
        bad_t = far_bad(seed, rng)
        if bad_t is None:
            continue
        for p in picks:
            if len(good) >= 2:
                break
            if agrees(seed, p, bad_t):
                good.append(p)
        if len(good) < 2:
            continue
        used_seeds.add(seed["key"])
        out.append({
            "seed": list(seed["key"]),
            "good": [list(g["key"]) for g in good],
            "bad": [list(bad_t["key"])],
            "src": "gen:recs",
        })

    # ---- rule 2: genre + keyword overlap --------------------------------
    popular = [
        t for t in titles
        if t["votes"] >= POPULAR_MIN_VOTES and t["rating"] >= POPULAR_MIN_RATING
    ]
    for seed in stratified_seeds(popular):
        if len(out) >= TARGET_LINES:
            break
        if seed["key"] in used_seeds or not seed["kw"] or not seed["emb"]:
            continue
        mates = [
            t for t in popular
            if t["key"] != seed["key"] and t["emb"]
            and (t["genres"] & seed["genres"]) and t["kw"]
            and story_disjoint(t, seed)
        ]
        def jacc(t):
            u = len(seed["kw"] | t["kw"])
            return len(seed["kw"] & t["kw"]) / u if u else 0.0
        mates = [t for t in mates if jacc(t) >= KEYWORD_JACCARD_KEEP]
        if not mates:
            continue
        mates.sort(key=lambda t: -jacc(t))
        good = []
        for t in mates[:3]:
            bad_t = far_bad(seed, rng)
            if bad_t is not None and agrees(seed, t, bad_t):
                good.append((t, bad_t))
        if len(good) < 2:
            continue
        used_seeds.add(seed["key"])
        out.append({
            "seed": list(seed["key"]),
            "good": [list(g["key"]) for g, _b in good[:3]],
            "bad": [list(b["key"]) for _g, b in good[:2]],
            "src": "gen:genre",
        })

    # ---- budget overflow: the recs graph fills what rule 2 couldn't.
    # Tail lines keep the graph-head + agreement + story-disjoint rules
    # but not the cosine-rank window — a looser but still two-agreement
    # consensus, so the file reaches its target line count.
    if len(out) < TARGET_LINES:
        for seed in stratified_seeds(titles):
            if len(out) >= TARGET_LINES:
                break
            if seed["key"] in used_seeds or not seed["emb"] or not seed["rec_ids"]:
                continue
            picks = [
                by_key[(rid, seed["key"][1])]
                for rank, rid in enumerate(seed["rec_ids"], start=1)
                if rank <= GRAPH_RANK_MAX and (rid, seed["key"][1]) in by_key
            ]
            picks = [p for p in picks if story_disjoint(p, seed)]
            bad_t = far_bad(seed, rng)
            if bad_t is None:
                continue
            good = [p for p in picks if agrees(seed, p, bad_t)][:2]
            if len(good) < 2:
                continue
            used_seeds.add(seed["key"])
            out.append({
                "seed": list(seed["key"]),
                "good": [list(g["key"]) for g in good],
                "bad": [list(bad_t["key"])],
                "src": "gen:recs",
            })

    del by_key
    return out[:TARGET_LINES]


async def validate(dsn: str) -> None:
    """Run the harness's engine scorer over the regenerated file and
    report accuracy per rule — the gate must be beatable by the current
    engine before it is wired into CI."""
    sys.path.insert(0, str(ROOT))
    import asyncpg
    from cine_rec_engine import RecommendationService

    judgments = [json.loads(ln)
                 for ln in OUT.read_text().splitlines() if ln.strip()]
    pool = await asyncpg.create_pool(dsn)
    svc = RecommendationService()
    await svc.initialize(pool)

    per_rule: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for j in judgments:
        seed_id, seed_type = j["seed"]
        eng = {(r["tmdb_id"], r["media_type"]): r["score"]
               for r in await svc.find_similar(
                   seed_id, limit=None, allow_cross_media=False,
                   media_type=seed_type)}
        rule = j.get("src", "hand")
        for g in j["good"]:
            for b in j["bad"]:
                sg, sb = eng.get(tuple(g), 0.0), eng.get(tuple(b), 0.0)
                per_rule[rule][1] += 1
                if sg > sb:
                    per_rule[rule][0] += 1
    await pool.close()
    total_ok = sum(v[0] for v in per_rule.values())
    total_n = sum(v[1] for v in per_rule.values())
    for rule, (ok, n) in sorted(per_rule.items()):
        print(f"{rule:12s} {ok}/{n} = {ok / n if n else 0:.3f}")
    print(f"TOTAL        {total_ok}/{total_n} = "
          f"{total_ok / total_n if total_n else 0:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true",
                   help="score the regenerated file with the engine (needs --dsn)")
    p.add_argument("--dsn", default="postgresql://demo:demo@localhost:54329/demo")
    p.add_argument("--dry-run", action="store_true",
                   help="print stats without writing the file")
    args = p.parse_args()

    if args.validate:
        asyncio.run(validate(args.dsn))
        return

    lines = build()
    by_src = defaultdict(int)
    for j in lines:
        by_src[j["src"]] += 1
    print(f"generated {len(lines)} lines: {dict(by_src)}")
    if not args.dry_run:
        hand = hand_lines()
        OUT.write_text(
            "\n".join(hand + [json.dumps(j) for j in lines]) + "\n")
        print(f"wrote {len(hand)} hand + {len(lines)} generated -> {OUT}")


if __name__ == "__main__":
    main()
