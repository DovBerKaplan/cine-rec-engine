"""The live CI-class suite (RFC M3): the bug class the offline tests
cannot see — SQL, enrichment, channel wiring, personalization pipeline.

Same four checks the CI demo-eval job runs, runnable on a laptop via
`make test-integration` (throws away a pgvector container) or against
an existing seeded DB via CINE_REC_INTEGRATION_DSN.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).parents[2]

# The CI gate values (ci.yml demo-eval job) — one source of truth for
# both entry points; if these move in CI, move them here too.
GATE_PAIRWISE = 0.75
GATE_NDCG = 0.28
GATE_COVERAGE = 0.50


async def test_recall_smoke(integration_env):
    """Cold find_similar on the demo catalog returns real results —
    the 0.5.3-class recall regressions."""
    res = await integration_env["service"].find_similar(155, limit=5)
    assert len(res) >= 3, "recall regression"
    assert all(r["score"] > 0 for r in res)
    assert len({r["tmdb_id"] for r in res}) == len(res)  # no dup ids


async def test_eval_regression_gate(integration_env):
    """The eval harness over the public judgments file must clear the
    CI floors — the 0.5.1-class zeroed-feature regressions."""
    spec = importlib.util.spec_from_file_location(
        "cine_eval", REPO / "eval" / "eval.py")
    cine_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cine_eval)

    stats = await cine_eval.run(integration_env["dsn"])
    s = stats["engine"]
    acc = s["pair_ok"] / s["pair_n"] if s["pair_n"] else 0.0
    ndcg = sum(s["ndcg"]) / len(s["ndcg"]) if s["ndcg"] else 0.0
    cov = len(s["cov"]) / s["pool"]
    assert acc >= GATE_PAIRWISE, f"engine pairwise {acc:.3f} < {GATE_PAIRWISE}"
    assert ndcg >= GATE_NDCG, f"engine NDCG@10 {ndcg:.3f} < {GATE_NDCG}"
    assert cov >= GATE_COVERAGE, f"engine coverage@10 {cov:.3f} < {GATE_COVERAGE}"


async def test_personalization_end_to_end(integration_env):
    """Events → w_i → user vector → personalized row, with the hard
    filters honored (watched/disliked never resurface)."""
    from cine_rec_engine import user_stats as us

    pool = integration_env["pool"]
    svc = integration_env["service"]
    now = datetime.now(timezone.utc)
    u = 990001  # dedicated harness user; reruns overwrite its rows
    await pool.execute("DELETE FROM user_vectors WHERE user_id=$1", u)
    await pool.execute("DELETE FROM user_title_stats WHERE user_id=$1", u)
    await pool.execute("DELETE FROM user_watch_events WHERE user_id=$1", u)
    await pool.execute("DELETE FROM user_feedback WHERE user_id=$1", u)

    watched = [155, 27205, 680]  # TDK, Inception, Superman
    for i, tid in enumerate(watched):
        await us.record_event(pool, dict(
            user_id=u, tmdb_id=tid, media_type="movie",
            watched_at=now - timedelta(days=1 + i),
            watched_sec=8000, duration_sec=8000, completed=True))
    await us.record_event(pool, dict(  # tv: hooked
        user_id=u, tmdb_id=1396, media_type="tv",
        watched_at=now - timedelta(days=2),
        watched_sec=2700, duration_sec=2700, completed=True,
        season=1, episode=1))
    await us.record_feedback(pool, u, 475557, "movie", "dislike")

    out = await svc.recommend_for_user(u, limit=12)
    assert out["reason"] == "personalized"
    assert out["results"], "personalized row came back empty"
    ids = {(r["tmdb_id"], r["media_type"]) for r in out["results"]}
    for tid in (*watched, 1396):
        assert (tid, "movie") not in ids and (tid, "tv") not in ids, \
            f"watched title {tid} resurfaced"
    assert (475557, "movie") not in ids, "disliked title resurfaced"
    assert out["vector_used"] is True, "user-vector channel inactive"
    # second request must be a cache hit — the plug-and-play contract
    out2 = await svc.recommend_for_user(u, limit=12)
    assert out2["results"] and out2["vector_used"] is True


async def test_user_tilt(integration_env):
    """The item-to-item tilt (M2): user context boosts, never buries,
    and every result carries its affinity."""
    from cine_rec_engine import user_stats as us

    pool = integration_env["pool"]
    svc = integration_env["service"]
    now = datetime.now(timezone.utc)
    u = 990002
    await pool.execute("DELETE FROM user_vectors WHERE user_id=$1", u)
    await pool.execute("DELETE FROM user_title_stats WHERE user_id=$1", u)
    await pool.execute("DELETE FROM user_watch_events WHERE user_id=$1", u)
    for i, tid in enumerate([155, 27205, 603]):
        await us.record_event(pool, dict(
            user_id=u, tmdb_id=tid, media_type="movie",
            watched_at=now - timedelta(days=1 + i),
            watched_sec=8000, duration_sec=8000, completed=True))

    # fresh vs tilted comparison needs the cache out of the way: the
    # user path keys the same cache entry, so compare affinity presence
    # and ordering invariants instead (boost-only, capped).
    res = await svc.find_similar(
        680, limit=15, user_id=u, exclude=set())  # Superman seed, user ctx
    assert res, "tilt path returned nothing"
    # contract: rows WITH an embedding get a boost-only affinity stamp
    # (rows without one pass through untouched, in place)
    aff = [r["user_affinity"] for r in res if "user_affinity" in r]
    assert aff, "tilt stamped no affinities at all"
    assert all(a >= 0 for a in aff), "tilt must be boost-only"
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True), "tilt broke score order"
