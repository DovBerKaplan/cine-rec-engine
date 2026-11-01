"""Generate personalization_parity.json — Python-computed expected
state after a scripted §A event sequence, for the Rust replay test.

Uses DEDICATED user ids (910_000+) so it can run against any DB seeded
with schema + demo data (the catalog join only needs real titles).

    CINE_REC_INTEGRATION_DSN=... .venv/bin/python \
        rust/db/tests/fixtures/gen_personalization_parity.py

The Rust test replays the SAME events (ids offset by +1000) and
compares: user_title_stats rows (w_item within recency tolerance,
everything else exact), user_stats §C counters, genre stats, and the
user vector.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import asyncpg  # noqa: E402

from cine_rec_engine import user_stats, user_vector  # noqa: E402
from cine_rec_engine import user_weights as uw  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


DSN = os.environ.get("CINE_REC_INTEGRATION_DSN", "")
EMB = "embedding_minilm"
U = 910_001  # fixture user; Rust replay uses U + 1000


async def catalog_titles(conn) -> dict:
    """Deterministic catalog picks: a movie, a long tv series, a short one."""
    movie = await conn.fetchrow(
        "SELECT m.id FROM tmdb_media m WHERE m.media_type='movie' "
        "AND m.vote_count > 100 ORDER BY m.id LIMIT 1")
    tv = await conn.fetchrow(
        "SELECT t.id FROM tmdb_tv t JOIN tmdb_media m ON (m.id=t.id AND m.media_type='tv') "
        "WHERE t.number_of_episodes >= 3 AND m.vote_count > 50 ORDER BY t.id LIMIT 1")
    movie2 = await conn.fetchrow(
        "SELECT m.id FROM tmdb_media m WHERE m.media_type='movie' "
        "AND m.vote_count > 100 ORDER BY m.id OFFSET 5 LIMIT 1")
    return {"movie": movie["id"], "tv": tv["id"], "movie2": movie2["id"]}


def EVENTS(t):
    return [
    # completed movie
    {"tmdb_id": t["movie"], "media_type": "movie",
     "watched_at": _ts("2026-10-20T20:00:00+00:00"),
     "watched_sec": 7000, "duration_sec": 7000, "completed": True},
    # same movie AGAIN (a rewatch: previous completed, same unit)
    {"tmdb_id": t["movie"], "media_type": "movie",
     "watched_at": _ts("2026-10-22T21:00:00+00:00"),
     "watched_sec": 7000, "duration_sec": 7000, "completed": True},
    # dropped movie (10%)
    {"tmdb_id": t["movie2"], "media_type": "movie",
     "watched_at": _ts("2026-10-18T20:00:00+00:00"),
     "watched_sec": 600, "duration_sec": 6000, "completed": False,
     "pause_count": 4},
    # series: eps 1+2 watched ≥50%, ep 3 barely started (abandon)
    {"tmdb_id": t["tv"], "media_type": "tv",
     "watched_at": _ts("2026-10-15T20:00:00+00:00"),
     "watched_sec": 2400, "duration_sec": 2400, "completed": True,
     "season": 1, "episode": 1},
    {"tmdb_id": t["tv"], "media_type": "tv",
     "watched_at": _ts("2026-10-16T20:00:00+00:00"),
     "watched_sec": 2400, "duration_sec": 2400, "completed": True,
     "season": 1, "episode": 2},
    {"tmdb_id": t["tv"], "media_type": "tv",
     "watched_at": _ts("2026-10-17T20:00:00+00:00"),
     "watched_sec": 200, "duration_sec": 2400, "completed": False,
     "season": 1, "episode": 3},
    ]

FEEDBACK = [
    # favorite the completed movie (engagement ×1.5)
    (910_001, "movie-fav", "movie", "favorite"),
]


async def main() -> None:
    assert DSN, "set CINE_REC_INTEGRATION_DSN"
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            titles = dict(await catalog_titles(conn))
        # clean slate for the fixture user
        async with pool.acquire() as conn:
            for tbl in ("user_watch_events", "user_title_stats", "user_feedback"):
                await conn.execute(f"DELETE FROM {tbl} WHERE user_id = $1", U)
            await conn.execute("DELETE FROM user_stats WHERE user_id = $1", U)
            await conn.execute("DELETE FROM user_genre_stats WHERE user_id = $1", U)
            await conn.execute("DELETE FROM user_vectors WHERE user_id = $1", U)

        for ev in EVENTS(titles):
            await user_stats.record_event(pool, {"user_id": U, **ev})
        await user_stats.record_feedback(pool, U, titles["movie"], "movie", "favorite")
        await user_stats.refresh_user_stats(pool, U)
        vec = await user_vector.build_user_vector(pool, U, None, EMB)

        async with pool.acquire() as conn:
            uts = await conn.fetch(
                "SELECT tmdb_id, media_type, sessions, total_watched_sec, "
                "max_ratio, pause_count_total, rewatch_count, episodes_watched, "
                "last_season, last_episode, last_ep_watched_sec, "
                "last_ep_duration_sec, dropped, w_item, last_watched_at "
                "FROM user_title_stats WHERE user_id = $1 ORDER BY tmdb_id", U)
            stats = await conn.fetchrow(
                "SELECT titles_touched, titles_weighted, movies_completed, "
                "movies_dropped, series_started, series_hooked, "
                "series_abandoned_mid_ep, rewatch_titles, explicit_favorites, "
                "total_watch_sec_all FROM user_stats WHERE user_id = $1", U)
            genres = await conn.fetch(
                "SELECT media_type, genre_id, watch_sec, weighted_sum, "
                "title_count FROM user_genre_stats WHERE user_id = $1 "
                "ORDER BY genre_id, media_type", U)
            n_eps = await conn.fetchval(
                "SELECT number_of_episodes FROM tmdb_tv WHERE id = $1", titles["tv"])

        # H-2: recompute w at ONE recorded instant (py_now) through
        # the Python formula — the DB-stored w used record_event's own
        # now (a different moment), so it can never compare exactly.
        # w_at_py_now + last_epoch + py_now give the Rust side
        # identical inputs; the comparison is time-independent.
        py_now = datetime.now(timezone.utc)
        s_side = {}
        for r in uts:
            if r["media_type"] == "movie":
                s = uw.s_movie(r["max_ratio"], r["pause_count_total"])
            else:
                sig = uw.SeriesSignals(
                    episodes_watched=r["episodes_watched"],
                    total_episodes=n_eps if r["tmdb_id"] == titles["tv"] else 0,
                    last_ep_ratio=(
                        (r["last_ep_watched_sec"] or 0) / (r["last_ep_duration_sec"] or 1)
                        if (r["last_ep_watched_sec"] or 0) > 0
                        and (r["last_ep_duration_sec"] or 0) > 0 else None),
                    last_ep_watched_sec=r["last_ep_watched_sec"] or 0,
                    total_watched_sec=r["total_watched_sec"],
                )
                s = uw.s_series(sig)
            eng = 1.5 if (r["rewatch_count"] >= 1 or r["tmdb_id"] == titles["movie"]) else 1.0
            last = r["last_watched_at"]
            fav = r["tmdb_id"] == titles["movie"]
            w_at_py_now = uw.w_item(
                r["media_type"],
                max_ratio=r["max_ratio"],
                pause_count_total=r["pause_count_total"],
                series=(uw.SeriesSignals(
                    episodes_watched=r["episodes_watched"],
                    total_episodes=n_eps if r["tmdb_id"] == titles["tv"] else 0,
                    last_ep_ratio=(
                        (r["last_ep_watched_sec"] or 0) / (r["last_ep_duration_sec"] or 1)
                        if (r["last_ep_watched_sec"] or 0) > 0
                        and (r["last_ep_duration_sec"] or 0) > 0 else None),
                    last_ep_watched_sec=r["last_ep_watched_sec"] or 0,
                    total_watched_sec=r["total_watched_sec"],
                ) if r["media_type"] == "tv" else None),
                last_watched_at=last,
                now=py_now,
                rewatch_count=r["rewatch_count"],
                favorite=fav,
            )
            s_side[str(r["tmdb_id"])] = {
                "s": s, "engagement": eng,
                "w_at_py_now": w_at_py_now,
                "last_epoch": last.timestamp() if last else None,
            }

        payload = {
            "user": U,
            "python_now_epoch": py_now.timestamp(),
            "s_side": s_side,
            "emb_column": EMB,
            "titles": titles,
            "events": [
                {"user_id": U, **{**ev, "watched_at": ev["watched_at"].isoformat()}}
                for ev in EVENTS(titles)
            ],
            "feedback": {"tmdb_id": titles["movie"], "media_type": "movie",
                         "kind": "favorite"},
            "tv_total_episodes": n_eps,
            "title_stats": [
                {k: (r[k].timestamp() if k == "last_watched_at"
                     else float(r[k]) if isinstance(r[k], float)
                     else r[k]) for k in r.keys()}
                for r in uts
            ],
            "user_stats": {k: stats[k] for k in stats.keys()},
            "genre_stats": [
                {k: r[k] for k in r.keys()} for r in genres
            ],
            "vector_dim": len(vec) if vec else 0,
            # M-2: full vector (f32 bits, hex) — direction parity, not
            # just "some 384-dim unit vector"
            "vector_bits": ([f.hex() for f in vec] if vec else None),
            # a unit vector — the direction pins parity (component-wise
            # compare happens in Rust against its OWN build)
            "vector_norm": (
                round(sum(x * x for x in vec) ** 0.5, 6) if vec else None
            ),
        }
    finally:
        await pool.close()

    dest = Path(__file__).parent / "personalization_parity.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {dest}: {len(payload['title_stats'])} title rows, "
          f"{len(payload['genre_stats'])} genre rows, vec dim {payload['vector_dim']}")


if __name__ == "__main__":
    asyncio.run(main())
