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
