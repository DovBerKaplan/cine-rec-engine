"""Built-in TMDB mirror ingest — docs/schema.sql loader.

Implements the ingest contract from docs/data.md §ingest:
- daily ID exports for discovery, /changes for daily refresh
- one API call per title via append_to_response
- adult=false + popularity floor filters (configurable)
- ≤ 40 req/s, Retry-After honored
- upserts by PK — never delete-insert of the catalog
- acceptance rule: a title is complete with details + genres (if returned)
  + credits (even empty) + keywords (even empty) + recommendations page 1
  (even empty). Missing sections still ingest; that channel just returns 0.
"""

from .exports import iter_export_ids, latest_export_url
from .loader import TmdbIngest

__all__ = ["TmdbIngest", "iter_export_ids", "latest_export_url", "title_rows_from_payload"]
