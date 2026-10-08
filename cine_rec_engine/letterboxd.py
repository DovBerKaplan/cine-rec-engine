"""Letterboxd CSV import — the one external event source (spec §D).

Feeds the §A contract: matched titles become `favorite` feedback (strong
engagement, works without watch durations); ratings ≤ 2 become
`dislike` (hard-filtered downstream). Unmatched rows are reported, never
guessed.

    python -m examples.import_letterboxd ratings.csv --user 1 \
        --dsn postgresql://demo:demo@localhost:54329/demo

CSV format (Letterboxd export): Date,Name,Year,Letterboxd URI,Rating
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple
from . import db


@dataclass
class LetterboxdRow:
    name: str
    year: Optional[int]
    rating: float          # Letterboxd 0.5–5.0 scale


def parse_letterboxd_csv(text: str) -> List[LetterboxdRow]:
    """Parse a Letterboxd ratings export (pure — unit tested).

    Tolerates BOM, column order, and missing years/ratings.
    """
    rows: List[LetterboxdRow] = []
    BOM = chr(0xFEFF)
    if text.startswith(BOM):
        text = text[1:]
    reader = csv.DictReader(io.StringIO(text))
    for raw in reader:
        name = (raw.get("Name") or "").strip()
        if not name:
            continue
        year_raw = (raw.get("Year") or "").strip()
        rating_raw = (raw.get("Rating") or "").strip()
        try:
            year = int(year_raw) if year_raw else None
        except ValueError:
            year = None
        try:
            rating = float(rating_raw) if rating_raw else 0.0
        except ValueError:
            rating = 0.0
        rows.append(LetterboxdRow(name=name, year=year, rating=rating))
    return rows


def rating_to_kind(rating: float) -> Optional[str]:
    """Map a Letterboxd rating to a feedback kind (pure).

    ≥ 3.5 → favorite (seed-worthy taste) · ≤ 2.0 → dislike · else None
    (watched-but-unremarkable: no signal, no noise).
    """
    if rating >= 3.5:
        return "favorite"
    if rating <= 2.0:
        return "dislike"
    return None


async def match_title(pool, name: str, year: Optional[int]) -> Optional[Tuple[int, str]]:
    """Find (tmdb_id, 'movie') by title (+year when present). None = no match.

    Never fuzzy-forces: a confident miss stays a miss.
    """
    if year:
        row = await db.fetchrow(pool,
            """SELECT id FROM {t_tmdb_movies}
               WHERE lower(title) = lower($1)
                 AND EXTRACT(YEAR FROM release_date) = $2
               LIMIT 1""",
            name, year,
        )
        if row:
            return row["id"], "movie"
    row = await db.fetchrow(pool,
        """SELECT id, EXTRACT(YEAR FROM release_date) AS y FROM {t_tmdb_movies}
           WHERE lower(title) = lower($1) ORDER BY popularity DESC LIMIT 1""",
        name,
    )
    if row and (year is None or abs((row["y"] or 0) - year) <= 1):
        return row["id"], "movie"
    return None


async def import_letterboxd(pool, user_id: int, csv_text: str) -> dict:
    """Import a Letterboxd export for one user. Returns a report."""
    from .user_stats import record_feedback, refresh_user_stats

    matched, kinds = [], {"favorite": 0, "dislike": 0}
    unmatched = []
    for row in parse_letterboxd_csv(csv_text):
        hit = await match_title(pool, row.name, row.year)
        if not hit:
            unmatched.append(f"{row.name} ({row.year or '?'})")
            continue
        kind = rating_to_kind(row.rating)
        if not kind:
            continue
        await record_feedback(pool, user_id, hit[0], hit[1], kind)
        matched.append((hit[0], hit[1], kind))
        kinds[kind] += 1
    await refresh_user_stats(pool, user_id)
    return {"matched": matched, "unmatched": unmatched, "kinds": kinds}
