# Demo catalog

`titles.jsonl.gz` — 830 real TMDB titles (top-rated pages + two hops
pages, `language=en-US`), fetched with the repo's own ingest client
(`append_to_response=credits,keywords,recommendations`) plus precomputed
MiniLM-384 embeddings for every title. ~12 MB gz.

This product uses the TMDB API but is not endorsed or certified by TMDB.
Data © TMDb — for anything beyond trying this demo, load your own mirror
with `ingest/` (docs/data.md) under your own API key and TMDB's terms.
