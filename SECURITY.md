# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | ✅ |

## Reporting a vulnerability

Email the owner via the GitHub security advisory route:
**Report a vulnerability** on the Security tab of this repository.
Please do not open a public issue for anything involving credentials,
injection payloads, or private data.

Include: engine version, Postgres version, a minimal reproduction, and
what data is at risk. You'll get an acknowledgement within 72 hours.

## Scope notes

- The engine executes SQL only from its own query modules — it never
  interpolates user input into SQL strings (everything is parameterized
  via asyncpg `$n` bind points). Report anything that breaks that promise.
- The optional TMDB sync (`tmdb_recs`) makes outbound HTTPS calls to
  api.themoviedb.org only; the API key is read from `TMDB_API_KEY`.
- The cache layer is expected to sit on a trusted network segment; tokens
  or personal data should not be placed in cache keys.
