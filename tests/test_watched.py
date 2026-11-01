"""Offline locks for watched-layer bug fixes (no DB).

The fake pool emulates the one asyncpg behavior the bugs hinged on: a
str bind for a ::bigint-cast parameter is rejected (a DataError
stand-in), which the module's broad except then swallowed.
"""

from cine_rec_engine.watched import get_user_title_progress


class _FakePool:
    """Records binds; returns a canned row; rejects str binds."""

    def __init__(self, row=None):
        self.row = row
        self.calls = []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        for i, arg in enumerate(args, start=1):
            if isinstance(arg, str):
                raise TypeError(
                    f"invalid input for query argument ${i}: {arg!r} "
                    "(expected int, got str)"
                )
        return self.row


class TestProgressBindsInt:
    async def test_movie_id_binds_as_int_not_str(self):
        pool = _FakePool(row={"watched": True})
        out = await get_user_title_progress(pool, 7, 155, "movie")
        # the old code bound str(tmdb_id) against $2::bigint — asyncpg
        # rejected it, the except swallowed, and watched was ALWAYS False
        assert out == {"watched": True}
        sql, args = pool.calls[0]
        assert args == (7, 155)
        assert isinstance(args[1], int)


class TestReachedLastNeedsCatalogEnd:
    async def test_known_end_reached(self):
        pool = _FakePool(row={
            "max_s": 2, "max_e": 5, "seen_n": 3,
            "cat_s": 2, "cat_e": 5, "cat_n": 10,
        })
        out = await get_user_title_progress(pool, 7, 155, "tv")
        assert out["watched"] is True
        assert out["reached_last"] is True
        assert out["watched_all"] is False  # 3 of 10 episodes seen

    async def test_missing_end_is_watched_but_not_reached_last(self):
        # episodes table empty/missing: the catalog end is NULL. The old
        # code compared (int, int) >= (None, None) — a TypeError the
        # broad except swallowed, so episodes that WERE seen came back
        # as watched=False. An unknown end must not fabricate completion.
        pool = _FakePool(row={
            "max_s": 1, "max_e": 3, "seen_n": 3,
            "cat_s": None, "cat_e": None, "cat_n": 0,
        })
        out = await get_user_title_progress(pool, 7, 155, "tv")
        assert out["watched"] is True
        assert out["reached_last"] is False
