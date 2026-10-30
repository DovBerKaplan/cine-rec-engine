"""Offline coverage for RecommendationService._apply_user_filters.

The merged per-user pass (watched filtering + saga advancement +
seed-saga blocking) was previously pinned only by live-DB smoke runs —
these tests drive it against a fake pool so the ordering rules in the
docstring stay enforceable without a database.
"""

from __future__ import annotations

import pytest


class _SagaPool:
    """Answers the two collection queries _apply_user_filters issues.

    The blocked-saga query selects seed collections' members
    (`s.id = ANY($1::bigint[])`); the members query fetches saga rows
    for the RESULT list (`collection_id = ANY($1::int[])`).
    """

    def __init__(self, blocked_rows=(), member_rows=(), fail=()):
        self.blocked_rows = list(blocked_rows)
        self.member_rows = list(member_rows)
        self.fail = set(fail)
        self.calls: list[str] = []

    async def fetch(self, sql, *args):
        is_blocked_query = "::bigint[]" in sql
        if ("blocked" in self.fail and is_blocked_query) or (
            "members" in self.fail and not is_blocked_query
        ):
            raise RuntimeError("simulated fetch failure")
        self.calls.append(sql)
        return self.blocked_rows if is_blocked_query else self.member_rows


def _service(pool):
    from cine_rec_engine.service import RecommendationService

    rec = RecommendationService.__new__(RecommendationService)
    rec.pool = pool
    return rec


def _res(tmdb_id, media_type="movie", collection_id=None, **extra):
    return {
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "collection_id": collection_id,
        "score": 0.9,
        **extra,
    }


def _member(coll, mid, title=None, year=2000, rating=7.0):
    return {
        "collection_id": coll,
        "id": mid,
        "title": title or f"Part {mid}",
        "title_en": title or f"Part {mid}",
        "vote_average": rating,
        "poster_path": None,
        "release_year": year,
    }


class TestWatchedFiltering:
    async def test_watched_non_saga_title_dropped(self):
        rec = _service(_SagaPool())
        out = await rec._apply_user_filters(
            [_res(10), _res(11)], {(10, "movie")}, seed_ids=set()
        )
        assert [r["tmdb_id"] for r in out] == [11]

    async def test_media_type_matters_in_watch_check(self):
        # movie 155 watched does not block tv 155 (composite keys)
        rec = _service(_SagaPool())
        out = await rec._apply_user_filters(
            [_res(155, "tv")], {(155, "movie")}, seed_ids=set()
        )
        assert [r["tmdb_id"] for r in out] == [155]

    async def test_bare_seed_ids_never_return(self):
        rec = _service(_SagaPool())
        out = await rec._apply_user_filters(
            [_res(10), _res(11)], set(), seed_ids={10}
        )
        assert [r["tmdb_id"] for r in out] == [11]


class TestSagaAdvancement:
    async def test_saga_slot_advances_to_first_unwatched(self):
        # collection 100 = movies 1,2,3; the user watched 1 → slot shows 2
        rec = _service(_SagaPool(member_rows=[
            _member(100, 1, "Saga I", 2001),
            _member(100, 2, "Saga II", 2002),
            _member(100, 3, "Saga III", 2003),
        ]))
        out = await rec._apply_user_filters(
            [_res(3, collection_id=100, score=0.7)],
            {(1, "movie")},
            seed_ids=set(),
        )
        assert len(out) == 1
        assert out[0]["tmdb_id"] == 2
        assert out[0]["title"] == "Saga II"
        assert out[0]["release_year"] == 2002
        assert out[0]["score"] == 0.7  # the slot keeps its score

    async def test_slot_keeps_own_movie_when_unwatched(self):
        # the saga's first unwatched member IS the slot's movie → no swap
        rec = _service(_SagaPool(member_rows=[
            _member(100, 1, "Saga I", 2001),
            _member(100, 2, "Saga II", 2002),
        ]))
        out = await rec._apply_user_filters(
            [_res(1, collection_id=100)], set(), seed_ids=set()
        )
        assert [r["tmdb_id"] for r in out] == [1]

    async def test_whole_saga_watched_drops_slot(self):
        rec = _service(_SagaPool(member_rows=[
            _member(100, 1), _member(100, 2),
        ]))
        out = await rec._apply_user_filters(
            [_res(2, collection_id=100)],
            {(1, "movie"), (2, "movie")},
            seed_ids=set(),
        )
        assert out == []

    async def test_advancement_never_lands_on_blocked_member(self):
        # movie 2 belongs to a seed's saga (blocked) → skip to 3, even
        # though 2 is the first unwatched member
        rec = _service(_SagaPool(
            blocked_rows=[{"id": 2}],
            member_rows=[_member(100, 1), _member(100, 2), _member(100, 3)],
        ))
        out = await rec._apply_user_filters(
            [_res(1, collection_id=100)], {(1, "movie")}, seed_ids={999}
        )
        assert [r["tmdb_id"] for r in out] == [3]

    async def test_swap_collisions_dedupe(self):
        # two saga slots can only advance to the same first-unwatched
        # member — the second swap is a duplicate and must vanish
        rec = _service(_SagaPool(member_rows=[
            _member(100, 1), _member(100, 2),
        ]))
        out = await rec._apply_user_filters(
            [_res(1, collection_id=100), _res(2, collection_id=100)],
            {(1, "movie")},
            seed_ids=set(),
        )
        assert [r["tmdb_id"] for r in out] == [2]


class TestSeedSagaBlocking:
    async def test_seed_saga_members_blocked_from_output(self):
        # seed 1 belongs to collection 100 → ALL its members are blocked,
        # even when a result slot belongs to that same collection
        rec = _service(_SagaPool(
            blocked_rows=[{"id": 1}, {"id": 2}, {"id": 3}],
            member_rows=[_member(100, 1), _member(100, 2), _member(100, 3)],
        ))
        out = await rec._apply_user_filters(
            [_res(3, collection_id=100), _res(11)], set(), seed_ids={1}
        )
        # the whole saga slot drops (every member blocked), 11 survives
        assert [r["tmdb_id"] for r in out] == [11]

    async def test_tv_results_have_no_saga_semantics(self):
        # tv rows carry NULL collection_id → plain watched filtering only
        rec = _service(_SagaPool())
        out = await rec._apply_user_filters(
            [_res(7, "tv", collection_id=None)], set(), seed_ids=set()
        )
        assert [r["tmdb_id"] for r in out] == [7]


class TestDegradation:
    async def test_members_fetch_failure_degrades_to_plain_filter(self):
        # the members query fails → saga slots behave like plain rows:
        # watched filtering still applies, nothing raises
        rec = _service(_SagaPool(member_rows=[], fail=("members",)))
        out = await rec._apply_user_filters(
            [_res(5, collection_id=100), _res(6)], {(5, "movie")}, seed_ids=set()
        )
        assert [r["tmdb_id"] for r in out] == [6]

    async def test_blocked_fetch_failure_keeps_seed_guard(self):
        # the blocked-saga query fails → blocked stays == seeds only
        rec = _service(_SagaPool(fail=("blocked",)))
        out = await rec._apply_user_filters(
            [_res(10), _res(11)], set(), seed_ids={10}
        )
        assert [r["tmdb_id"] for r in out] == [11]


@pytest.mark.parametrize("seed_ids", [set(), {1}])
async def test_empty_results_stay_empty(seed_ids):
    rec = _service(_SagaPool())
    assert await rec._apply_user_filters([], set(), seed_ids) == []
