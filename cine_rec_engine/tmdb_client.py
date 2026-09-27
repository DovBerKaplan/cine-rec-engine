"""Self-contained TMDB API client (fetch with retry/backoff).

The engine is fully local-first — the only optional outbound call is
``tmdb_recs`` behavioral-recommendation sync (``/{movie,tv}/{id}/recommendations``),
which needs YOUR TMDB API key (``TMDB_API_KEY``). This module isolates
that dependency so the engine never imports a larger HTTP stack.
"""

from __future__ import annotations

import asyncio

import aiohttp
from loguru import logger


async def fetch_with_retry(
    session: aiohttp.ClientSession, url: str, params: dict, max_retries: int = 2
) -> dict:
    """Fetch from the TMDB API with retry + exponential backoff.

    Returns ``{"error": ...}`` on exhaustion — never raises, so callers
    can treat failure as "no data" and degrade silently.
    """
    backoff = 2
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status in (429, 500, 502, 503, 504):
                    retry_after = int(response.headers.get("Retry-After", backoff))
                    logger.warning(f"Rate limited/server error, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue

                if 400 <= response.status < 500 and response.status != 429:
                    # Auth/permission/not-found NEVER improves on retry —
                    # return immediately (retrying burned multi-second
                    # sleeps per seed when the key was absent/invalid).
                    logger.warning(f"API rejected request: HTTP {response.status}")
                    return {"error": f"HTTP {response.status}"}

                response.raise_for_status()
                return await response.json()  # type: ignore[no-any-return]

        except asyncio.TimeoutError:
            logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return {"error": "Timeout"}
            await asyncio.sleep(backoff)
            backoff *= 2

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"API error: {e}")
                return {"error": str(e)}
            await asyncio.sleep(backoff)
            backoff *= 2

    return {"error": "Max retries exceeded"}
