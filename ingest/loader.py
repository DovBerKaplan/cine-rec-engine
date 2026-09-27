"""Per-title ingest: one API call, upserts, acceptance accounting."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional, Tuple

import aiohttp
import asyncpg
from loguru import logger

from cine_rec_engine.tmdb_client import fetch_with_retry

from .rate import RateLimiter

API_BASE = "https://api.themoviedb.org/3"

# Only these crew jobs survive ingest (spec §4).
CREW_JOBS = {
    "Director",
    "Creator",          # tv
    "Writer",
    "Original Music Composer",
    "Director of Photography",
}

CAST_TOP_N = 5  # by TMDB `order`


@dataclass
class IngestStats:
    attempted: int = 0
    ingested: int = 0
    filtered_adult: int = 0
    filtered_popularity: int = 0
    failed: int = 0
    complete: int = 0  # acceptance rule satisfied (details+genres+credits+keywords+recs)

    def summary(self) -> str:
        return (
            f"attempted={self.attempted} ingested={self.ingested} "
            f"complete={self.complete} adult_filtered={self.filtered_adult} "
            f"popularity_filtered={self.filtered_popularity} failed={self.failed}"
        )


def keep_cast(cast: Iterable[dict]) -> List[Tuple[int, str, int]]:
    """Top-N billing by `order` → (person_id, character, cast_order)."""
    rows = [
        (int(c["id"]), str(c.get("character") or ""), int(c["order"]))
        for c in cast
        if c.get("id") is not None and c.get("order") is not None
    ]
    rows.sort(key=lambda r: r[2])
    return rows[:CAST_TOP_N]


def keep_crew(crew: Iterable[dict]) -> List[Tuple[int, str, Optional[str]]]:
    """Whitelisted jobs only → (person_id, job, department)."""
    return [
        (int(c["id"]), str(c["job"]), c.get("department"))
        for c in crew
        if c.get("job") in CREW_JOBS and c.get("id") is not None
    ]


def gmap_table(movie: bool) -> str:
    return "tmdb_movie_genres_map" if movie else "tmdb_tv_genres_map"


def kw_table(movie: bool) -> str:
    return "tmdb_movie_keywords" if movie else "tmdb_tv_keywords"


def company_table(movie: bool) -> str:
    return "tmdb_movie_companies" if movie else "tmdb_tv_companies"


def cast_table(movie: bool) -> str:
    return "tmdb_movie_cast" if movie else "tmdb_tv_cast"


def crew_table(movie: bool) -> str:
    return "tmdb_movie_crew" if movie else "tmdb_tv_crew"


def append_to_response(medium: str) -> str:
    """The single-call append list per medium (spec §6).

    tv additionally pulls external_ids (imdb_id); built as a list so a
    missing separator is structurally impossible.
    """
    parts = ["credits", "keywords", "recommendations"]
    if medium == "tv":
        parts.append("external_ids")
    return ",".join(parts)


def should_ingest(payload: dict, min_popularity: float = 0.0,
                  include_adult: bool = False) -> bool:
    """Post-fetch gate shared by bootstrap and refresh (spec filters)."""
    if payload is None or payload.get("success") is False:
        return False
    if payload.get("adult") and not include_adult:
        return False
    if float(payload.get("popularity") or 0.0) < min_popularity:
        return False
    return True


def _as_date(value):
    """TMDB returns dates as ISO strings; asyncpg DATE params need date."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def title_row_from_payload(payload: dict) -> Tuple:
    """Movie details row, column order matching UPSERT_MOVIE."""
    return (
        int(payload["id"]),
        payload.get("title") or "",
        payload.get("original_title"),
        payload.get("original_language"),
        payload.get("overview"),
        _as_date(payload.get("release_date")),
        payload.get("runtime"),
        payload.get("status"),
        bool(payload.get("adult", False)),
        payload.get("vote_average"),
        payload.get("vote_count"),
        payload.get("popularity"),
        payload.get("poster_path"),
        payload.get("backdrop_path"),
        (payload.get("belongs_to_collection") or {}).get("id"),
        payload.get("imdb_id"),
        payload.get("budget"),
        payload.get("revenue"),
    )


def tv_row_from_payload(payload: dict) -> Tuple:
    """TV details row, column order matching UPSERT_TV (name, not title)."""
    return (
        int(payload["id"]),
        payload.get("name") or "",
        payload.get("original_name"),
        payload.get("original_language"),
        payload.get("overview"),
        _as_date(payload.get("first_air_date")),
        _as_date(payload.get("last_air_date")),
        payload.get("status"),
        payload.get("in_production"),
        payload.get("number_of_seasons"),
        payload.get("number_of_episodes"),
        bool(payload.get("adult", False)),
        payload.get("vote_average"),
        payload.get("vote_count"),
        payload.get("popularity"),
        payload.get("poster_path"),
        payload.get("backdrop_path"),
        payload.get("external_ids", {}).get("imdb_id")
        if isinstance(payload.get("external_ids"), dict)
        else None,
    )


def rec_rows_from_payload(payload: dict) -> List[Tuple[int, str, int, str, int, float]]:
    """(media_id, media_type, rec_id, rec_type, rank, popularity) — page 1.

    TMDB recs stay within their medium (movie→movie, tv→tv) exactly like
    the API, so rec_media_type mirrors media_type.
    """
    media_type = "movie" if "title" in payload else "tv"
    results = (payload.get("recommendations") or {}).get("results") or []
    return [
        (
            int(payload["id"]),
            media_type,
            int(r["id"]),
            media_type,
            rank,
            float(r.get("popularity") or 0.0),
        )
        for rank, r in enumerate(results, start=1)
    ]


def is_complete(payload: dict) -> bool:
    """Acceptance rule (spec §9): details + genres (if TMDB returned any)
    + credits + keywords + recommendations page 1 — present as keys."""
    return all(
        key in payload
        for key in ("genres", "credits", "keywords", "recommendations")
    )


async def _sync_bridge(conn, table: str, media_col: str, media_id: int,
                       other_col: str, current_ids: List[int],
                       insert_sql: Optional[str] = None,
                       rows: Optional[list] = None) -> None:
    """Mirror one bridge table to the fresh payload (spec: upsert, not append).

    Rows whose other-side id vanished from TMDB are deleted; current rows
    are upserted. Without the delete, a recast actor or a removed keyword
    would linger forever under ON CONFLICT DO NOTHING.
    """
    if current_ids:
        await conn.execute(
            f"DELETE FROM {table} WHERE {media_col} = $1 "
            f"AND {other_col} <> ALL($2::bigint[])",
            media_id, current_ids,
        )
    else:
        await conn.execute(f"DELETE FROM {table} WHERE {media_col} = $1", media_id)
    if rows:
        await conn.executemany(insert_sql, rows)


UPSERT_MOVIE = """
    INSERT INTO tmdb_movies (id, title, original_title, original_language,
        overview, release_date, runtime, status, adult, vote_average,
        vote_count, popularity, poster_path, backdrop_path, collection_id,
        imdb_id, budget, revenue, updated_at)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18, now())
    ON CONFLICT (id) DO UPDATE SET
        title=EXCLUDED.title, original_title=EXCLUDED.original_title,
        original_language=EXCLUDED.original_language, overview=EXCLUDED.overview,
        release_date=EXCLUDED.release_date, runtime=EXCLUDED.runtime,
        status=EXCLUDED.status, adult=EXCLUDED.adult,
        vote_average=EXCLUDED.vote_average, vote_count=EXCLUDED.vote_count,
        popularity=EXCLUDED.popularity, poster_path=EXCLUDED.poster_path,
        backdrop_path=EXCLUDED.backdrop_path, collection_id=EXCLUDED.collection_id,
        imdb_id=EXCLUDED.imdb_id, budget=EXCLUDED.budget, revenue=EXCLUDED.revenue,
        updated_at=now()
"""

UPSERT_TV = """
    INSERT INTO tmdb_tv (id, name, original_name, original_language,
        overview, first_air_date, last_air_date, status, in_production,
        number_of_seasons, number_of_episodes, adult, vote_average,
        vote_count, popularity, poster_path, backdrop_path, imdb_id, updated_at)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18, now())
    ON CONFLICT (id) DO UPDATE SET
        name=EXCLUDED.name, original_name=EXCLUDED.original_name,
        original_language=EXCLUDED.original_language, overview=EXCLUDED.overview,
        first_air_date=EXCLUDED.first_air_date, last_air_date=EXCLUDED.last_air_date,
        status=EXCLUDED.status, in_production=EXCLUDED.in_production,
        number_of_seasons=EXCLUDED.number_of_seasons,
        number_of_episodes=EXCLUDED.number_of_episodes, adult=EXCLUDED.adult,
        vote_average=EXCLUDED.vote_average, vote_count=EXCLUDED.vote_count,
        popularity=EXCLUDED.popularity, poster_path=EXCLUDED.poster_path,
        backdrop_path=EXCLUDED.backdrop_path, imdb_id=EXCLUDED.imdb_id,
        updated_at=now()
"""


class TmdbIngest:
    """Loads the mirror into Postgres. One class, two entry points:

    - ``bootstrap(ids)`` — ingest a set of ids (usually from the daily
      exports, already filtered).
    - ``refresh(changes_days)`` — /movie/changes + /tv/changes for daily
      updates (call after 08:00 UTC; TMDB publishes changes ~05:00 UTC).
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        api_key: Optional[str] = None,
        rate_per_second: int = 40,
    ) -> None:
        self.pool = pool
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "")
        if not self.api_key:
            raise ValueError("TMDB_API_KEY missing (param or env)")
        # Concurrency keeps the connection pool busy; the token bucket
        # enforces the actual API budget (≤ rate_per_second requests/sec).
        self._sem = asyncio.Semaphore(min(rate_per_second, 64))
        self._limiter = RateLimiter(rate_per_second)
        self.stats = IngestStats()

    async def _fetch_title(
        self, session: aiohttp.ClientSession, media_id: int, medium: str
    ) -> Optional[dict]:
        url = f"{API_BASE}/{medium}/{media_id}"
        params = {
            "api_key": self.api_key,
            "language": "en-US",
            "append_to_response": append_to_response(medium),
        }
        async with self._sem:
            await self._limiter.acquire()  # pace to the requests/sec budget
            data = await fetch_with_retry(session, url, params)
        return data if "error" not in data else None

    async def _upsert_title(self, payload: dict, medium: str) -> None:
        """Upsert one title + MIRROR its bridges to the fresh payload.

        Bridge rows removed from TMDB are deleted (recast actors, dropped
        keywords); current rows are upserted. Dimension tables (people,
        keywords, genres, companies, networks) upsert by name.
        """
        media_id = int(payload["id"])
        movie = medium == "movie"
        col = "movie_id" if movie else "tv_id"

        if movie:
            await self.pool.execute(UPSERT_MOVIE, *title_row_from_payload(payload))
        else:
            await self.pool.execute(UPSERT_TV, *tv_row_from_payload(payload))

        genres = [(g["id"], g["name"]) for g in payload.get("genres", [])]
        kw = (payload.get("keywords", {}) or {}).get("keywords", [])
        if not kw:
            kw = (payload.get("keywords", {}) or {}).get("results", [])
        credits = payload.get("credits") or {}
        cast = keep_cast(credits.get("cast") or [])
        crew = keep_crew(credits.get("crew") or [])
        people = {p for p, *_ in cast} | {p for p, *_ in crew}
        companies = payload.get("production_companies") or []
        networks = (payload.get("networks") or []) if not movie else []

        async with self.pool.acquire() as conn:
            if people:
                names = {
                    int(c["id"]): c.get("name") or ""
                    for c in (credits.get("cast") or []) + (credits.get("crew") or [])
                }
                await conn.executemany(
                    "INSERT INTO tmdb_people (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                    list(names.items()),
                )

            if genres:
                genre_table = "tmdb_movie_genres" if movie else "tmdb_tv_genres"
                await conn.executemany(
                    f"INSERT INTO {genre_table} (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                    genres,
                )
            await _sync_bridge(
                conn, gmap_table(movie), col, media_id, "genre_id",
                [gid for gid, _ in genres],
                f"INSERT INTO {gmap_table(movie)} ({col}, genre_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                [(media_id, gid) for gid, _ in genres],
            )

            if kw:
                await conn.executemany(
                    "INSERT INTO tmdb_keywords (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                    [(int(k["id"]), k["name"]) for k in kw],
                )
            await _sync_bridge(
                conn, kw_table(movie), col, media_id, "keyword_id",
                [int(k["id"]) for k in kw],
                f"INSERT INTO {kw_table(movie)} ({col}, keyword_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                [(media_id, int(k["id"])) for k in kw],
            )

            if companies:
                await conn.executemany(
                    "INSERT INTO tmdb_production_companies (id, name) "
                    "VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name",
                    [(int(c["id"]), c["name"]) for c in companies],
                )
            await _sync_bridge(
                conn, company_table(movie), col, media_id, "company_id",
                [int(c["id"]) for c in companies],
                f"INSERT INTO {company_table(movie)} ({col}, company_id) "
                "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                [(media_id, int(c["id"])) for c in companies],
            )

            if networks:
                await conn.executemany(
                    "INSERT INTO tmdb_networks (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name",
                    [(int(n["id"]), n["name"]) for n in networks],
                )
            await _sync_bridge(
                conn, "tmdb_tv_networks_map", "tv_id", media_id, "network_id",
                [int(n["id"]) for n in networks],
                "INSERT INTO tmdb_tv_networks_map (tv_id, network_id) "
                "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                [(media_id, int(n["id"])) for n in networks],
            )

            await _sync_bridge(
                conn, cast_table(movie), col, media_id, "person_id",
                sorted({pid for pid, _, _ in cast}),
                f"INSERT INTO {cast_table(movie)} ({col}, person_id, character, "
                "cast_order) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT ({}, person_id, cast_order) DO UPDATE "
                "SET character = EXCLUDED.character".format(col),
                [(media_id, pid, ch, order) for pid, ch, order in cast],
            )
            await _sync_bridge(
                conn, crew_table(movie), col, media_id, "person_id",
                sorted({pid for pid, *_ in crew}),
                f"INSERT INTO {crew_table(movie)} ({col}, person_id, job, "
                "department) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT ({}, person_id, job) DO UPDATE "
                "SET department = EXCLUDED.department".format(col),
                [(media_id, pid, job, dept) for pid, job, dept in crew],
            )

            recs = rec_rows_from_payload(payload)
            await conn.execute(
                "DELETE FROM tmdb_recommendations WHERE media_id = $1 "
                "AND media_type = $2 AND rank > $3",
                media_id, medium, len(recs),
            )
            if recs:
                await conn.executemany(
                    "INSERT INTO tmdb_recommendations (media_id, media_type, "
                    "rec_media_id, rec_media_type, rank, popularity) "
                    "VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT "
                    "(media_id, media_type, rec_media_id, rec_media_type) DO UPDATE "
                    "SET rank=EXCLUDED.rank, popularity=EXCLUDED.popularity, "
                    "synced_at=now()",
                    recs,
                )

    async def ingest_one(
        self, session: aiohttp.ClientSession, media_id: int, medium: str,
        min_popularity: float = 0.0, include_adult: bool = False,
    ) -> bool:
        """Fetch + upsert one title through the shared gate. True when ingested."""
        self.stats.attempted += 1
        try:
            payload = await self._fetch_title(session, media_id, medium)
            if not should_ingest(payload, min_popularity, include_adult):
                if payload and payload.get("adult"):
                    self.stats.filtered_adult += 1
                elif payload is not None:
                    self.stats.filtered_popularity += 1
                else:
                    self.stats.failed += 1
                return False
            await self._upsert_title(payload, medium)
            self.stats.ingested += 1
            if is_complete(payload):
                self.stats.complete += 1
            return True
        except Exception as e:
            self.stats.failed += 1
            logger.warning(f"ingest {medium}/{media_id} failed: {e}")
            return False

    async def existing_ids(self) -> dict:
        """id sets per medium — the skip-existing index for bootstrap/refresh."""
        movies = await self.pool.fetch("SELECT id FROM tmdb_movies")
        tv = await self.pool.fetch("SELECT id FROM tmdb_tv")
        return {"movie": {r["id"] for r in movies}, "tv": {r["id"] for r in tv}}

    async def bootstrap(
        self,
        entries: "asyncio.AsyncIterator | object",
        workers: int = 8,
        skip_existing: bool = True,
    ) -> IngestStats:
        """Concurrent ingest of discovered ids.

        Workers pull from an async queue (bounded, so memory stays flat);
        the shared RateLimiter paces the whole fleet to the API budget.
        skip_existing=True resumes a partial bootstrap instead of
        re-fetching the whole catalog.
        """
        known = await self.existing_ids() if skip_existing else \
            {"movie": set(), "tv": set()}
        queue: asyncio.Queue = asyncio.Queue(maxsize=workers * 4)

        async def producer():
            async for media_id, medium in entries:
                if media_id in known[medium]:
                    continue
                await queue.put((media_id, medium))
            for _ in range(workers):
                await queue.put(None)

        async def worker(session: aiohttp.ClientSession):
            while True:
                item = await queue.get()
                if item is None:
                    return
                await self.ingest_one(session, item[0], item[1])

        prod = asyncio.create_task(producer())
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(worker(session) for _ in range(workers)))
        await prod
        logger.info(f"bootstrap done: {self.stats.summary()}")
        return self.stats

    async def refresh(
        self,
        min_popularity: float = 0.0,
        include_adult: bool = False,
        changes_days: int = 1,
    ) -> IngestStats:
        """The FULL daily loop (call after 08:00 UTC):

        1. Export diff — yesterday's daily export, filtered, minus what's
           already in the DB: catches titles that crossed the popularity
           threshold AFTER their TMDB debut (the Changes API never lists
           them — a title only "changes" when its row is touched).
        2. Changes — /movie/changes + /tv/changes, paged: rows TMDB
           actually touched in the window; re-ingested through the same
           gate (adult + popularity), so refresh respects the catalog
           shape bootstrap chose.
        """
        from .exports import iter_export_ids

        known = await self.existing_ids()

        async def fresh_from_exports(session):
            for medium in ("movie", "tv"):
                async for entry in iter_export_ids(
                    medium, session,
                    min_popularity=min_popularity,
                    include_adult=include_adult,
                ):
                    if entry.id not in known[medium]:
                        yield entry.id, medium

        async with aiohttp.ClientSession() as session:
            logger.info("refresh[1/2]: export diff (new threshold-crossers)")
            async for media_id, medium in fresh_from_exports(session):
                await self.ingest_one(
                    session, media_id, medium,
                    min_popularity=min_popularity, include_adult=include_adult,
                )

            logger.info("refresh[2/2]: changes window")
            from datetime import date, timedelta

            end_day = date.today()
            start_day = end_day - timedelta(days=changes_days)
            for medium in ("movie", "tv"):
                page = 1
                while True:
                    data = await fetch_with_retry(
                        session,
                        f"{API_BASE}/{medium}/changes",
                        {
                            "api_key": self.api_key,
                            "start_date": start_day.isoformat(),
                            "end_date": end_day.isoformat(),
                            "page": page,
                        },
                    )
                    if "error" in data or not data.get("results"):
                        break
                    for item in data["results"]:
                        if item.get("adult") and not include_adult:
                            continue
                        await self.ingest_one(
                            session, int(item["id"]), medium,
                            min_popularity=min_popularity,
                            include_adult=include_adult,
                        )
                    if page >= int(data.get("total_pages") or 1):
                        break
                    page += 1
        logger.info(f"refresh done: {self.stats.summary()}")
        return self.stats
