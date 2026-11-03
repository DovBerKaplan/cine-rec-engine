"""Offline coverage for the on-demand TMDB sync modes (CINE_REC_SYNC_MODE).

find_similar's step 1.5 routes through _dispatch_seed_sync; these tests
pin the routing contract — blocking awaits inline (the historical
behavior), background never blocks the request — plus the tmdb_recs
fire-and-forget primitives. No database, no network: every sync seam
is monkeypatched with gated coroutines so "did the caller wait?" is
observable as "is the gate still closed?".
"""

from __future__ import annotations

import asyncio

import pytest


# --- config.env_choice ------------------------------------------------------


def test_env_choice_contract(monkeypatch):
    from cine_rec_engine.config import env_choice

    monkeypatch.delenv("CINE_REC_TEST_CHOICE", raising=False)
    assert (
        env_choice("CINE_REC_TEST_CHOICE", "blocking", ("blocking", "background"))
        == "blocking"
    )
    monkeypatch.setenv("CINE_REC_TEST_CHOICE", "  background ")  # stripped
    assert (
        env_choice("CINE_REC_TEST_CHOICE", "blocking", ("blocking", "background"))
        == "background"
    )
    monkeypatch.setenv("CINE_REC_TEST_CHOICE", "sideways")
    with pytest.raises(ValueError, match="CINE_REC_TEST_CHOICE"):
        env_choice("CINE_REC_TEST_CHOICE", "blocking", ("blocking", "background"))


def test_sync_mode_knob_is_well_formed():
    from cine_rec_engine.config import SYNC_MODE

    assert SYNC_MODE in ("blocking", "background")


# --- tmdb_recs.start_background_sync / drain_background_syncs ----------------


def test_start_background_sync_no_key_is_a_noop(monkeypatch):
    from cine_rec_engine import tmdb_recs

    async def _spy(*args, **kwargs):
        raise AssertionError("must not sync without a TMDB key")

    monkeypatch.setattr(tmdb_recs, "TMDB_API_KEY", "")
    monkeypatch.setattr(tmdb_recs, "ensure_seeds_synced", _spy)
    assert tmdb_recs.start_background_sync(object(), [(1, "movie")]) is None
    assert not tmdb_recs._BACKGROUND_SYNC_TASKS


async def test_start_background_sync_is_fire_and_forget(monkeypatch):
    from cine_rec_engine import tmdb_recs

    gate = asyncio.Event()
    seen = {}

    async def _gated(pool, seeds, session=None):
        seen["seeds"] = list(seeds)
        await gate.wait()
        seen["done"] = True

    monkeypatch.setattr(tmdb_recs, "TMDB_API_KEY", "test-key")
    monkeypatch.setattr(tmdb_recs, "ensure_seeds_synced", _gated)

    task = tmdb_recs.start_background_sync(object(), [(155, "movie"), (155, "tv")])
    assert task is not None
    await asyncio.sleep(0)  # let the task start and park on the gate
    # The sync is running but still parked — spawn returned without it.
    assert seen["seeds"] == [(155, "movie"), (155, "tv")]
    assert "done" not in seen
    assert task in tmdb_recs._BACKGROUND_SYNC_TASKS

    gate.set()
    assert await tmdb_recs.drain_background_syncs(timeout=5) == 1
    assert seen["done"] is True
    # The task deregistered itself on completion.
    assert task not in tmdb_recs._BACKGROUND_SYNC_TASKS


async def test_start_background_sync_swallows_failures(monkeypatch):
    from cine_rec_engine import tmdb_recs

    async def _boom(pool, seeds, session=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(tmdb_recs, "TMDB_API_KEY", "test-key")
    monkeypatch.setattr(tmdb_recs, "ensure_seeds_synced", _boom)
    task = tmdb_recs.start_background_sync(object(), [(1, "movie")])
    # Degrade, never fail: the task completes (exception swallowed),
    # nothing stays registered, nothing propagates to the caller.
    await asyncio.wait_for(task, timeout=5)
    assert not tmdb_recs._BACKGROUND_SYNC_TASKS


async def test_drain_with_nothing_pending():
    from cine_rec_engine import tmdb_recs

    assert await tmdb_recs.drain_background_syncs() == 0


# --- find_similar._dispatch_seed_sync (step 1.5) -----------------------------


async def test_dispatch_blocking_awaits_the_sync(monkeypatch):
    from cine_rec_engine import find_similar as fs

    gate = asyncio.Event()
    calls = []

    async def _inline(pool, seeds):
        calls.append((pool, list(seeds)))
        await gate.wait()

    monkeypatch.setattr(fs, "ensure_seeds_synced", _inline)
    pool = object()
    task = asyncio.ensure_future(
        fs._dispatch_seed_sync(pool, [(1, "movie")], "blocking"))
    await asyncio.sleep(0)  # let the coroutine run to its first await
    assert calls == [(pool, [(1, "movie")])]
    assert not task.done()  # the caller waits — the historical inline await
    gate.set()
    await asyncio.wait_for(task, timeout=5)


async def test_dispatch_background_never_awaits(monkeypatch):
    from cine_rec_engine import find_similar as fs

    spawned = []
    hang = asyncio.get_running_loop().create_future()

    def _spawn(pool, seeds):
        spawned.append((pool, list(seeds)))
        return hang  # a sync that never completes

    monkeypatch.setattr(fs, "start_background_sync", _spawn)
    # Completes while the spawned sync hangs — that is the whole point.
    await asyncio.wait_for(
        fs._dispatch_seed_sync(object(), [(7, "tv")], "background"), timeout=1)
    assert len(spawned) == 1


async def test_dispatch_blocking_swallows_sync_failures(monkeypatch):
    from cine_rec_engine import find_similar as fs

    async def _boom(pool, seeds):
        raise RuntimeError("tmdb down")

    monkeypatch.setattr(fs, "ensure_seeds_synced", _boom)
    await fs._dispatch_seed_sync(object(), [(1, "movie")], "blocking")


# --- find_similar parameter + service delegate -------------------------------


async def test_find_similar_rejects_bad_sync_mode():
    from cine_rec_engine import find_similar as fs

    class _Svc:
        async def _ensure_initialized(self):
            pass

    with pytest.raises(ValueError, match="sync_mode"):
        await fs.find_similar(_Svc(), 155, sync_mode="sometimes")


async def test_service_delegate_forwards_sync_mode(monkeypatch):
    from cine_rec_engine import find_similar as fs
    from cine_rec_engine.service import RecommendationService

    captured = {}

    async def _fake(svc, tmdb_id, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(fs, "find_similar", _fake)
    svc = RecommendationService.__new__(RecommendationService)
    await svc.find_similar(155, sync_mode="background")
    assert captured["sync_mode"] == "background"
