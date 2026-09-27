"""TMDB daily ID exports — discovery source.

TMDB publishes the full id lists daily (~07:00 UTC) as gzipped JSON:
    http://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz
    http://files.tmdb.org/p/exports/tv_series_ids_MM_DD_YYYY.json.gz
Each line: {"adult": bool, "id": int, "original_title"/"original_name": str,
            "popularity": float, ...}
"""

from __future__ import annotations

import gzip
import io
import zlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import AsyncIterator, Iterator, Optional

import aiohttp

EXPORT_BASE = "http://files.tmdb.org/p/exports"
EXPORT_NAMES = {"movie": "movie_ids", "tv": "tv_series_ids"}


@dataclass
class ExportEntry:
    id: int
    adult: bool
    popularity: float


def export_url_for(day: date, medium: str) -> str:
    if medium not in EXPORT_NAMES:
        raise ValueError(f"medium must be 'movie' or 'tv', got {medium!r}")
    return f"{EXPORT_BASE}/{EXPORT_NAMES[medium]}_{day:%m_%d_%Y}.json.gz"


def latest_export_url(medium: str, today: Optional[date] = None) -> str:
    """Most recent export that should exist: yesterday's file.

    The export job runs ~07:00 UTC; before that, today's file is absent.
    Yesterday is always safe.
    """
    day = (today or date.today()) - timedelta(days=1)
    return export_url_for(day, medium)


def parse_export(gz_bytes: bytes) -> Iterator[ExportEntry]:
    """Parse a downloaded export file — one JSON object per line."""
    with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as f:
        for raw in f:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            yield ExportEntry(
                id=int(obj["id"]),
                adult=bool(obj.get("adult", False)),
                popularity=float(obj.get("popularity") or 0.0),
            )


async def iter_export_ids(
    medium: str,
    session: aiohttp.ClientSession,
    url: Optional[str] = None,
    min_popularity: float = 0.0,
    include_adult: bool = False,
) -> AsyncIterator[ExportEntry]:
    """Stream the daily export, applying the starter filters.

    Streams gzip-decompressed lines as they arrive — the full movie export
    decompresses to hundreds of MB and must never sit in RAM whole.
    adult=false + popularity floor are the default catalog shape; both
    configurable per the spec (ingest config, not a schema concern).
    """
    url = url or latest_export_url(medium)
    async with session.get(url) as resp:
        resp.raise_for_status()
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        pending = b""
        async for chunk in resp.content.iter_chunked(1 << 16):
            pending += decompressor.decompress(chunk)
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                entry = _parse_line(line)
                if entry is None:
                    continue
                if entry.adult and not include_adult:
                    continue
                if entry.popularity < min_popularity:
                    continue
                yield entry


def _parse_line(line: bytes) -> Optional[ExportEntry]:
    text = line.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return ExportEntry(
            id=int(obj["id"]),
            adult=bool(obj.get("adult", False)),
            popularity=float(obj.get("popularity") or 0.0),
        )
    except (KeyError, ValueError):
        return None
