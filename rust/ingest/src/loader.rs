//! Per-title ingest — the port of `ingest/loader.py` (wave D).
//!
//! The DB-facing half is complete and parity-pinned: row builders
//! (pure, offline-pinned against the real Python functions), the
//! catalog upserts and the bridge MIRRORING semantics. The HTTP half
//! is a seam ([`TitleFetcher`]) so tests feed fixture payloads; the
//! real fetcher (aiohttp→HTTP + retry + semaphore + rate limiting)
//! lands in wave E together with bootstrap/refresh orchestration.
//!
//! Invariants kept (AGENTS.md):
//! * **(id, media_type) keys everywhere** — movie 155 and tv 155 are
//!   different titles; the recommendation rows key on
//!   (media_id, media_type, rec_media_id, rec_media_type).
//! * **Bridges MIRROR the payload — never append-only**: every sync
//!   deletes rows whose other-side id vanished from TMDB before
//!   upserting the current ones ([`sync_bridge`]).
//! * **One registry**: every statement resolves `{t_*}` through
//!   `cine_rec_db::SchemaMap` (via `Db::sql`), matching the Python
//!   loader resolving through `tables.py`.
//!
//! Granularity parity: the Python loader runs the title upsert on the
//! pool and the whole bridge batch on ONE acquired connection, with
//! no wrapping transaction; each `executemany` is atomic on its own
//! (an asyncpg guarantee) — mirrored here statement-for-statement,
//! batch-for-batch.
//!
//! Field expressions follow PYTHON semantics (truthiness, `or`
//! defaults, strict KeyError on missing required fields) — see
//! [`crate::pyval`]; the row-builder fixture pins them exactly.

use std::collections::HashSet;

use chrono::NaiveDate;
use serde_json::Value;
use sqlx::postgres::PgConnection;

use cine_rec_db::{Db, DbError};

use crate::pyval::{is_truthy, py_float, PyConvError};

pub const API_BASE: &str = "https://api.themoviedb.org/3";

/// Only these crew jobs survive ingest (spec §4) — medium-agnostic in
/// Python (a movie credit with job "Creator" is kept too); kept as-is.
pub const CREW_JOBS: &[&str] = &[
    "Director",
    "Creator", // tv
    "Writer",
    "Original Music Composer",
    "Director of Photography",
];

/// Top-N billing by TMDB `order` (spec §4).
pub const CAST_TOP_N: usize = 5;

pub const UPSERT_MOVIE: &str = "
    INSERT INTO {t_tmdb_movies} (id, title, original_title, original_language,
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
";

pub const UPSERT_TV: &str = "
    INSERT INTO {t_tmdb_tv} (id, name, original_name, original_language,
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
";

const UPSERT_PERSON: &str = "INSERT INTO {t_tmdb_people} (id, name) VALUES ($1, $2) \
     ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name";
const UPSERT_GENRE: &str = "INSERT INTO {t_tmdb_movie_genres} (id, name) VALUES ($1, $2) \
     ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name";
const UPSERT_TV_GENRE: &str = "INSERT INTO {t_tmdb_tv_genres} (id, name) VALUES ($1, $2) \
     ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name";
const UPSERT_KEYWORD: &str = "INSERT INTO {t_tmdb_keywords} (id, name) VALUES ($1, $2) \
     ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name";
const UPSERT_COMPANY: &str = "INSERT INTO {t_tmdb_production_companies} (id, name) \
     VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name";
const UPSERT_NETWORK: &str = "INSERT INTO {t_tmdb_networks} (id, name) VALUES ($1, $2) \
     ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name";
const DELETE_RECS_TAIL: &str = "DELETE FROM {t_tmdb_recommendations} WHERE media_id = $1 \
     AND media_type = $2 AND rank > $3";
const UPSERT_REC: &str = "INSERT INTO {t_tmdb_recommendations} (media_id, media_type, \
     rec_media_id, rec_media_type, rank, popularity) \
     VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT \
     (media_id, media_type, rec_media_id, rec_media_type) DO UPDATE \
     SET rank=EXCLUDED.rank, popularity=EXCLUDED.popularity, synced_at=now()";

/// Loader failures. Everything here maps to the Python `except
/// Exception` in `ingest_one`: the title is counted `failed` and the
/// run continues — the VARIANTS exist so tests can pin where a
/// payload dies, not to change control flow.
#[derive(Debug)]
pub enum LoaderError {
    /// `{t_*}` resolution failed (unknown logical name / bad map) —
    /// Python's `tables.TableMapError`.
    Map(cine_rec_db::SchemaMapError),
    /// A database error (SQL failure, or an asyncpg-style ENCODING
    /// failure — e.g. a string `popularity` cannot bind to numeric).
    Db(DbError),
    /// A payload field failed a Python-conversion rule (KeyError on a
    /// required field, unconvertible type).
    Conv(PyConvError),
}

impl From<cine_rec_db::SchemaMapError> for LoaderError {
    fn from(e: cine_rec_db::SchemaMapError) -> Self {
        LoaderError::Map(e)
    }
}

impl From<DbError> for LoaderError {
    fn from(e: DbError) -> Self {
        LoaderError::Db(e)
    }
}

impl From<PyConvError> for LoaderError {
    fn from(e: PyConvError) -> Self {
        LoaderError::Conv(e)
    }
}

impl From<sqlx::Error> for LoaderError {
    fn from(e: sqlx::Error) -> Self {
        LoaderError::Db(DbError::Sqlx(e))
    }
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::Map(e) => write!(f, "schema map: {e}"),
            LoaderError::Db(e) => write!(f, "db: {e}"),
            LoaderError::Conv(e) => write!(f, "payload: {e}"),
        }
    }
}

impl std::error::Error for LoaderError {}

/// One cast row after the top-N filter: (person_id, character,
/// cast_order). `character` is the EMPTY STRING when the payload's
/// value is falsy — Python binds `str(c.get("character") or "")`, so
/// the column receives `''`, never NULL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastRow {
    pub person_id: i64,
    pub character: String,
    /// The full `int(order)` value — the INT4 range check happens at
    /// the BIND (asyncpg parity; review M-2).
    pub cast_order: i64,
}

/// One whitelisted crew row: (person_id, job, department) —
/// `department` stays a raw value (bind-checked later, like asyncpg).
#[derive(Debug, Clone, PartialEq)]
pub struct CrewRow {
    pub person_id: i64,
    pub job: String,
    pub department: Option<Value>,
}

/// A recommendations page-1 row.
#[derive(Debug, Clone, PartialEq)]
pub struct RecRow {
    pub media_id: i64,
    pub media_type: &'static str,
    pub rec_media_id: i64,
    pub rec_media_type: &'static str,
    pub rank: i32,
    pub popularity: f64,
}

/// The movie details row, column order matching [`UPSERT_MOVIE`].
///
/// FAITHFULNESS RULE (the whole design of these structs): a field is
/// TYPED exactly where the Python builder CONVERTS it (`int()`,
/// `str(x or "")`, `_as_date`, `bool(...)`, the two `.get()`
/// extractions) and stays a RAW JSON [`Value`] where Python passes
/// the payload value through untouched — those raw values fail at
/// BIND time in Python (asyncpg's encoder), so they fail at bind time
/// here too, never in the builder. A string `popularity`, for
/// example, builds a row on both sides and only dies at the database.
#[derive(Debug, Clone, PartialEq)]
pub struct MovieRow {
    pub id: i64,
    /// `payload.get("title") or ""` — falsy becomes `""`, a truthy
    /// value passes through RAW (bind enforces text).
    pub title: Value,
    pub original_title: Option<Value>,
    pub original_language: Option<Value>,
    pub overview: Option<Value>,
    pub release_date: Option<NaiveDate>,
    pub runtime: Option<Value>,
    pub status: Option<Value>,
    pub adult: bool,
    pub vote_average: Option<Value>,
    pub vote_count: Option<Value>,
    pub popularity: Option<Value>,
    pub poster_path: Option<Value>,
    pub backdrop_path: Option<Value>,
    /// `(payload.get("belongs_to_collection") or {}).get("id")` — the
    /// extracted id passes through raw.
    pub collection_id: Option<Value>,
    pub imdb_id: Option<Value>,
    pub budget: Option<Value>,
    pub revenue: Option<Value>,
}

/// The TV details row, column order matching [`UPSERT_TV`] — same
/// conversion rule as [`MovieRow`].
#[derive(Debug, Clone, PartialEq)]
pub struct TvRow {
    pub id: i64,
    pub name: Value,
    pub original_name: Option<Value>,
    pub original_language: Option<Value>,
    pub overview: Option<Value>,
    pub first_air_date: Option<NaiveDate>,
    pub last_air_date: Option<NaiveDate>,
    pub status: Option<Value>,
    pub in_production: Option<Value>,
    pub number_of_seasons: Option<Value>,
    pub number_of_episodes: Option<Value>,
    pub adult: bool,
    pub vote_average: Option<Value>,
    pub vote_count: Option<Value>,
    pub popularity: Option<Value>,
    pub poster_path: Option<Value>,
    pub backdrop_path: Option<Value>,
    pub imdb_id: Option<Value>,
}

// ── field conversion helpers (Python expression parity) ───────────

/// `payload.get(k)` — None when absent or JSON null.
fn get<'a>(payload: &'a Value, k: &str) -> Option<&'a Value> {
    match payload.get(k) {
        None | Some(Value::Null) => None,
        some => some,
    }
}

/// `int(x)` over a required payload slot (KeyError parity) — the
/// full `pyval::py_int` semantics: truncating floats, numeric
/// strings, bools.
fn req_int(obj: &Value, k: &str) -> Result<i64, PyConvError> {
    let v = obj
        .get(k)
        .ok_or_else(|| PyConvError::Key(format!("'{k}'")))?;
    crate::pyval::py_int(v)
}

/// A required slot passed through RAW (`g["name"]` — KeyError when
/// absent; the value itself is bind-checked later, like asyncpg).
fn req_raw<'a>(obj: &'a Value, k: &str) -> Result<&'a Value, PyConvError> {
    obj.get(k).ok_or_else(|| PyConvError::Key(format!("'{k}'")))
}

/// `str(x or "")` — Python's `str()` accepts anything: null/absent or
/// falsy → `""`, strings pass through, numbers/bools render as their
/// Python `str()` (bools are `True`/`False`).
fn py_str_or_empty(v: Option<&Value>) -> String {
    match v {
        None => String::new(),
        Some(v) if !is_truthy(v) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".into(),
        Some(Value::Bool(false)) => "False".into(),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                // Python str(float) keeps a '.0' (5.0 → "5.0") — Rust's
                // Debug does too, Display does not. Exponent FORMS
                // still differ (python "1e+16"/"1e-07" vs Rust
                // "1e16"/"1e-7") — documented divergence.
                n.as_f64().map(|f| format!("{f:?}")).unwrap_or_default()
            }
        }
        // str(list/dict) reprs are pathological for these fields; the
        // Python str() of a dict differs from JSON — pinned as a
        // documented divergence rather than replicated.
        _ => String::new(),
    }
}

/// `_as_date`: None/"" → None; ISO "YYYY-MM-DD" → date. Out-of-range
/// components raise the DATE constructor's messages (Python's
/// `fromisoformat` builds a `date()` — pinned by the fixture:
/// "2024-13-45" → "month must be in 1..12"); structurally bad strings
/// raise "Invalid isoformat string". Only the dashed form is
/// supported (3.11+ also accepts compact/week forms — documented
/// divergence; TMDB always sends dashed).
fn as_date(value: Option<&Value>) -> Result<Option<NaiveDate>, PyConvError> {
    let Some(v) = value else { return Ok(None) };
    match v {
        Value::String(s) if s.is_empty() => Ok(None),
        Value::String(s) => parse_iso_date(s).map(Some).map_err(PyConvError::Value),
        other => Err(PyConvError::Type(format!(
            "_as_date: expected a string, got {}",
            crate::pyval::py_type_name(other)
        ))),
    }
}

fn parse_iso_date(s: &str) -> Result<NaiveDate, String> {
    let bad = || format!("Invalid isoformat string: '{s}'");
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return Err(bad());
    }
    let part = |r: std::ops::Range<usize>| -> Result<i32, String> {
        let slice = &s[r];
        if slice.bytes().all(|c| c.is_ascii_digit()) {
            Ok(slice.parse().unwrap())
        } else {
            Err(bad())
        }
    };
    let (y, m, d) = (part(0..4)?, part(5..7)?, part(8..10)?);
    // Component validation with Python's date() messages — the exact
    // texts from crate::date.
    crate::date::CivilDate::new(y, m, d).map_err(|e| e.0)?;
    NaiveDate::from_ymd_opt(y, m as u32, d as u32).ok_or_else(bad)
}

// ── bind-time conversions (asyncpg encoder parity) ────────────────
// Raw Values convert to typed binds HERE — the same place Python's
// asyncpg encoder accepts or rejects them, after the builder.

/// An extracted `x or []` list: falsy → empty, array → items, truthy
/// non-array → carried as [`MaybeBad::Bad`] so the failure fires at
/// the STATEMENT Python evaluates it at (inside the batch), not at
/// extraction time (review M-2).
#[derive(Debug, Clone)]
enum MaybeBad {
    List(Vec<Value>),
    Bad(Value),
}

fn extract_list(v: Option<&Value>) -> MaybeBad {
    match v {
        None | Some(Value::Null) => MaybeBad::List(Vec::new()),
        Some(Value::Array(a)) => MaybeBad::List(a.clone()),
        Some(v) if !is_truthy(v) => MaybeBad::List(Vec::new()),
        Some(v) => MaybeBad::Bad(v.clone()),
    }
}

impl MaybeBad {
    fn bad(&self, ctx: &str) -> Option<PyConvError> {
        match self {
            MaybeBad::Bad(v) => Some(PyConvError::Type(format!(
                "{ctx}: iterating {} is a TypeError in Python",
                crate::pyval::py_type_name(v)
            ))),
            MaybeBad::List(_) => None,
        }
    }
}

fn bind_text(v: &Option<Value>, col: &str) -> Result<LBind, PyConvError> {
    match v {
        None => Ok(LBind::OptStr(None)),
        Some(Value::String(s)) => Ok(LBind::OptStr(Some(s.clone()))),
        Some(other) => Err(PyConvError::Type(format!(
            "{col}: asyncpg cannot encode {} as text",
            crate::pyval::py_type_name(other)
        ))),
    }
}

fn bind_text_req(v: &Value, col: &str) -> Result<LBind, PyConvError> {
    match v {
        Value::String(s) => Ok(LBind::Str(s.clone())),
        other => Err(PyConvError::Type(format!(
            "{col}: asyncpg cannot encode {} as text",
            crate::pyval::py_type_name(other)
        ))),
    }
}

/// The ASYNCPG INT ENCODER, not Python's `int()` (review M-1, all
/// behaviors verified live): int passes; float TRUNCATES toward zero
/// (120.5→120, -0.7→0); bool → 0/1; a STRING is a DataError
/// ("'str' object cannot be interpreted as an integer") — unlike
/// `int("120")` which succeeds at builder sites. bigint[] params
/// encode element-wise with the same rules.
fn ap_int(v: &Value, col: &str) -> Result<i64, PyConvError> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(f) = n.as_f64() {
                // truncation toward zero, like asyncpg's int(value)
                if f.trunc() >= 9_223_372_036_854_775_808.0
                    || f.trunc() < -9_223_372_036_854_775_808.0
                {
                    Err(PyConvError::Value(format!(
                        "{col}: integer out of range for int8: {f}"
                    )))
                } else {
                    Ok(f.trunc() as i64)
                }
            } else {
                Err(PyConvError::Value(format!("{col}: unbindable number {n}")))
            }
        }
        Value::Bool(b) => Ok(i64::from(*b)),
        other => Err(PyConvError::Type(format!(
            "{col}: asyncpg cannot encode {} as an integer",
            crate::pyval::py_type_name(other)
        ))),
    }
}

fn bind_i64(v: &Option<Value>, col: &str) -> Result<LBind, PyConvError> {
    match v {
        None => Ok(LBind::OptI64(None)),
        Some(v) => Ok(LBind::OptI64(Some(ap_int(v, col)?))),
    }
}

fn bind_i32(v: &Option<Value>, col: &str) -> Result<LBind, PyConvError> {
    match v {
        None => Ok(LBind::OptI32(None)),
        Some(v) => {
            let i = ap_int(v, col)?;
            i32::try_from(i)
                .map(|i| LBind::OptI32(Some(i)))
                .map_err(|_| PyConvError::Value(format!("{col}: integer {i} out of INT4 range")))
        }
    }
}

/// NUMERIC columns (vote_average, popularity) — the asyncpg codec,
/// every branch verified live: int and float pass (float as
/// `Decimal(float)`, the EXACT decimal expansion — [`decimal_exact`]);
/// bool → 0/1; a STRING parses as `Decimal(str)` — trimmed whitespace,
/// underscores between digits, exponents, ARBITRARY precision kept
/// (never routed through f64); list/dict are DataErrors.
///
/// DOCUMENTED DIVERGENCE: `Decimal('NaN')`/`Decimal('Infinity')` bind
/// fine in asyncpg (PG ≥14 numeric has NaN/Infinity), but BigDecimal
/// cannot represent them and sqlx has no numeric-NaN encoding — here
/// those two literals fail the title LOUDLY instead of storing
/// numeric NaN/Infinity. Pathological for TMDB input; pinned by a
/// unit test, listed in the README.
fn bind_numeric(v: &Option<Value>, col: &str) -> Result<LBind, PyConvError> {
    match v {
        None => Ok(LBind::OptNumeric(None)),
        Some(Value::Bool(b)) => Ok(LBind::OptNumericInt(i64::from(*b))),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Ok(LBind::OptNumericInt(i))
            } else {
                Ok(LBind::OptNumeric(n.as_f64()))
            }
        }
        Some(Value::String(s)) => decimal_from_str(s)
            .map(LBind::OptNumericBig)
            .map_err(|e| annotate(e, col)),
        Some(other) => Err(PyConvError::Type(format!(
            "{col}: asyncpg cannot encode {} as numeric",
            crate::pyval::py_type_name(other)
        ))),
    }
}

/// `Decimal(str)` acceptance minus the non-finite forms: surrounding
/// whitespace, underscores between digits, decimal/exponent syntax —
/// parsed to an EXACT BigDecimal (no f64 round-trip, so a 25-digit
/// string keeps all 25 digits like asyncpg's Decimal).
fn decimal_from_str(s: &str) -> Result<bigdecimal::BigDecimal, PyConvError> {
    let trimmed = crate::pyval::py_strip(s);
    let cleaned: String = trimmed.chars().filter(|&c| c != '_').collect();
    if cleaned.is_empty() {
        return Err(PyConvError::Value(format!(
            "could not convert string to numeric: {s:?}"
        )));
    }
    let lower = cleaned.to_ascii_lowercase();
    if lower.contains("nan") || lower.contains("inf") {
        // asyncpg stores numeric NaN/Infinity from these strings; see
        // the documented divergence above.
        return Err(PyConvError::Value(format!(
            "{s:?}: numeric NaN/Infinity literals are a documented divergence (loud failure)"
        )));
    }
    cleaned
        .parse::<bigdecimal::BigDecimal>()
        .map_err(|_| PyConvError::Value(format!("could not convert string to numeric: {s:?}")))
}

/// The EXACT decimal expansion of an f64, as asyncpg's numeric codec
/// produces via `Decimal(float)`: decompose m×2^e; for e ≥ 0 the
/// integer m≪e; for e < 0 the digits m×5^(-e) at scale -e (because
/// m/2^(-e) = (m·5^(-e))/10^(-e)). PG's float8→numeric cast is NOT
/// this (it rounds differently) — the difference is visible in stored
/// rows and pinned by the live parity test.
fn decimal_exact(f: f64) -> Result<bigdecimal::BigDecimal, PyConvError> {
    use num_bigint::BigInt;
    if !f.is_finite() {
        // Decimal(inf) reaches asyncpg's numeric encoder and fails at
        // the protocol level (PG numeric has no infinity).
        return Err(PyConvError::Value(format!("cannot convert {f} to numeric")));
    }
    if f == 0.0 {
        return Ok(bigdecimal::BigDecimal::new(BigInt::from(0), 0));
    }
    let bits = f.to_bits();
    let neg = bits >> 63 == 1;
    let raw_exp = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & 0xf_ffff_ffff_ffff; // low 52 bits
    let (mantissa, exp) = if raw_exp == 0 {
        // subnormal: no implicit bit, fixed exponent
        (frac as i64, -1074_i64)
    } else {
        ((frac | (1_u64 << 52)) as i64, raw_exp - 1075)
    };
    let m_abs = BigInt::from(mantissa);
    let big = if exp >= 0 {
        bigdecimal::BigDecimal::new(m_abs << (exp as usize), 0)
    } else {
        let scale = -exp;
        bigdecimal::BigDecimal::new(
            m_abs * BigInt::from(5_u32).pow(u32::try_from(scale).unwrap()),
            scale,
        )
    };
    Ok(if neg { -big } else { big })
}

fn bind_bool(v: &Option<Value>, col: &str) -> Result<LBind, PyConvError> {
    match v {
        None => Ok(LBind::OptBool(None)),
        Some(Value::Bool(b)) => Ok(LBind::OptBool(Some(*b))),
        Some(other) => Err(PyConvError::Type(format!(
            "{col}: asyncpg cannot encode {} as bool",
            crate::pyval::py_type_name(other)
        ))),
    }
}

fn annotate(e: PyConvError, col: &str) -> PyConvError {
    match e {
        PyConvError::Value(m) => PyConvError::Value(format!("{col}: {m}")),
        other => other,
    }
}

// ── pure builders (parity-pinned offline) ─────────────────────────

/// `keep_cast` — top-N billing by `order`. Entries missing `id` or
/// `order` are dropped first (`is not None` — a null id/order drops,
/// a non-null non-number fails `int()`); the sort is stable, so
/// `order` ties keep payload order (Python's sort is stable too).
/// `int()` truncates and `str(x or "")` converts — full Python
/// semantics at builder time, exactly like the original.
pub fn keep_cast(cast: &[Value]) -> Result<Vec<CastRow>, PyConvError> {
    let mut rows: Vec<CastRow> = Vec::new();
    for c in cast {
        let id_v = c.get("id").filter(|v| !matches!(v, Value::Null));
        let order_v = c.get("order").filter(|v| !matches!(v, Value::Null));
        let (Some(id_v), Some(order_v)) = (id_v, order_v) else {
            continue;
        };
        let id = crate::pyval::py_int(id_v)?;
        // Python's int() is arbitrary-precision: a huge order builds a
        // row fine and only the INT4 BIND fails later (after the cast
        // DELETE — pinned by review M-2). Keep the full i64 here; the
        // range check lives at the bind site.
        let order = crate::pyval::py_int(order_v)?;
        let character = py_str_or_empty(c.get("character"));
        rows.push(CastRow {
            person_id: id,
            character,
            cast_order: order,
        });
    }
    rows.sort_by_key(|r| r.cast_order);
    rows.truncate(CAST_TOP_N);
    Ok(rows)
}

/// `keep_crew` — whitelisted jobs only; `int(c["id"])` and
/// `str(c["job"])` convert at builder time, `department` passes
/// through raw (bind-checked later).
pub fn keep_crew(crew: &[Value]) -> Result<Vec<CrewRow>, PyConvError> {
    let mut rows = Vec::new();
    for c in crew {
        // `c.get("job") in CREW_JOBS`: hashable non-string jobs
        // (numbers, bools, None) simply don't match — dropped. An
        // UNHASHABLE job (list/dict) makes Python's set membership
        // raise TypeError → the title fails; loud here too (review
        // L-1).
        let job = match c.get("job") {
            Some(Value::String(job)) => job,
            Some(v @ (Value::Array(_) | Value::Object(_))) => {
                return Err(PyConvError::Type(format!(
                    "unhashable type: '{}'",
                    crate::pyval::py_type_name(v)
                )))
            }
            _ => continue,
        };
        if !CREW_JOBS.contains(&job.as_str()) {
            continue;
        }
        let Some(id_v) = c.get("id").filter(|v| !matches!(v, Value::Null)) else {
            continue;
        };
        let id = crate::pyval::py_int(id_v)?;
        let department = get(c, "department").cloned();
        rows.push(CrewRow {
            person_id: id,
            job: job.clone(),
            department,
        });
    }
    Ok(rows)
}

/// `title_row_from_payload` (see [`MovieRow`] for column order).
pub fn title_row_from_payload(payload: &Value) -> Result<MovieRow, PyConvError> {
    let id = req_int(payload, "id")?;
    // `(payload.get("belongs_to_collection") or {}).get("id")`: a FALSY
    // value (null, false, "", 0, []) reads as {} → None — the title
    // ingests; a TRUTHY non-dict fails `.get` (AttributeError →
    // failed title). Pinned by review M-3.
    let collection_id = match payload.get("belongs_to_collection") {
        None | Some(Value::Null) => None,
        Some(v) if !is_truthy(v) => None,
        Some(Value::Object(obj)) => match obj.get("id") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.clone()),
        },
        Some(other) => {
            return Err(PyConvError::Type(format!(
                "belongs_to_collection: '{}' object has no attribute 'get'",
                crate::pyval::py_type_name(other)
            )))
        }
    };
    Ok(MovieRow {
        id,
        title: match get(payload, "title") {
            Some(v) if is_truthy(v) => v.clone(),
            _ => Value::String(String::new()),
        },
        original_title: get(payload, "original_title").cloned(),
        original_language: get(payload, "original_language").cloned(),
        overview: get(payload, "overview").cloned(),
        release_date: as_date(payload.get("release_date"))?,
        runtime: get(payload, "runtime").cloned(),
        status: get(payload, "status").cloned(),
        adult: is_truthy(payload.get("adult").unwrap_or(&Value::Bool(false))),
        vote_average: get(payload, "vote_average").cloned(),
        vote_count: get(payload, "vote_count").cloned(),
        popularity: get(payload, "popularity").cloned(),
        poster_path: get(payload, "poster_path").cloned(),
        backdrop_path: get(payload, "backdrop_path").cloned(),
        collection_id,
        imdb_id: get(payload, "imdb_id").cloned(),
        budget: get(payload, "budget").cloned(),
        revenue: get(payload, "revenue").cloned(),
    })
}

/// `tv_row_from_payload` (see [`TvRow`] for column order).
pub fn tv_row_from_payload(payload: &Value) -> Result<TvRow, PyConvError> {
    let id = req_int(payload, "id")?;
    // `payload.get("external_ids", {}).get("imdb_id") if
    // isinstance(..., dict) else None` — the extracted id is raw.
    let imdb_id = match payload.get("external_ids") {
        Some(Value::Object(obj)) => match obj.get("imdb_id") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.clone()),
        },
        _ => None,
    };
    Ok(TvRow {
        id,
        name: match get(payload, "name") {
            Some(v) if is_truthy(v) => v.clone(),
            _ => Value::String(String::new()),
        },
        original_name: get(payload, "original_name").cloned(),
        original_language: get(payload, "original_language").cloned(),
        overview: get(payload, "overview").cloned(),
        first_air_date: as_date(payload.get("first_air_date"))?,
        last_air_date: as_date(payload.get("last_air_date"))?,
        status: get(payload, "status").cloned(),
        in_production: get(payload, "in_production").cloned(),
        number_of_seasons: get(payload, "number_of_seasons").cloned(),
        number_of_episodes: get(payload, "number_of_episodes").cloned(),
        adult: is_truthy(payload.get("adult").unwrap_or(&Value::Bool(false))),
        vote_average: get(payload, "vote_average").cloned(),
        vote_count: get(payload, "vote_count").cloned(),
        popularity: get(payload, "popularity").cloned(),
        poster_path: get(payload, "poster_path").cloned(),
        backdrop_path: get(payload, "backdrop_path").cloned(),
        imdb_id,
    })
}

/// `rec_rows_from_payload` — page 1 rows. The medium is derived from
/// KEY PRESENCE (`"title" in payload`), not from the caller's
/// `medium` — faithful to Python; `float(r.get("popularity") or 0.0)`
/// accepts numeric strings, matching Python's `float()`.
pub fn rec_rows_from_payload(payload: &Value) -> Result<Vec<RecRow>, PyConvError> {
    let media_type: &'static str = if payload.get("title").is_some() {
        "movie"
    } else {
        "tv"
    };
    // Lazy like the Python comprehension: `payload["id"]` is only
    // touched when there IS a row to build — an id-less payload with
    // no recommendations returns [] without error (pinned by the
    // fixture). Container shapes follow Python (review L-1): a truthy
    // non-dict recommendations / non-list results raises where the
    // `.get`/iteration would.
    let empty = Vec::new();
    let results: &[Value] = match get(payload, "recommendations") {
        None => &empty,
        Some(v) if !is_truthy(v) => &empty,
        Some(Value::Object(obj)) => match obj.get("results") {
            None | Some(Value::Null) => &empty,
            Some(v) if !is_truthy(v) => &empty,
            Some(Value::Array(a)) => a,
            Some(other) => {
                return Err(PyConvError::Type(format!(
                    "recommendations.results: iterating {} is a TypeError in Python",
                    crate::pyval::py_type_name(other)
                )))
            }
        },
        Some(other) => {
            // Python raises AttributeError on the .get — exact text
            // (pinned by the offline fixture).
            return Err(PyConvError::Attr(format!(
                "'{}' object has no attribute 'get'",
                crate::pyval::py_type_name(other)
            )));
        }
    };
    if results.is_empty() {
        return Ok(Vec::new());
    }
    let id = req_int(payload, "id")?;
    let mut rows = Vec::with_capacity(results.len());
    for (i, r) in results.iter().enumerate() {
        let rec_id = req_int(r, "id")?;
        let popularity = match r.get("popularity") {
            Some(v) if is_truthy(v) => py_float(v)?,
            _ => 0.0,
        };
        rows.push(RecRow {
            media_id: id,
            media_type,
            rec_media_id: rec_id,
            rec_media_type: media_type,
            rank: (i + 1) as i32,
            popularity,
        });
    }
    Ok(rows)
}

/// `should_ingest` — the post-fetch gate. `success` is compared by
/// IDENTITY to False in Python (`payload.get("success") is False`),
/// which for JSON means exactly the boolean `false`; `adult` is
/// truthiness; popularity is `float(x or 0.0)` with NaN never below
/// the floor. An unconvertible popularity (`float("abc")`) RAISES in
/// Python — caught by `ingest_one`'s except → `failed` — so this
/// returns Err, never a silent 0.0 (review M-3).
pub fn should_ingest(
    payload: Option<&Value>,
    min_popularity: f64,
    include_adult: bool,
) -> Result<bool, PyConvError> {
    let Some(payload) = payload else {
        return Ok(false);
    };
    if payload.get("success") == Some(&Value::Bool(false)) {
        return Ok(false);
    }
    let adult = is_truthy(payload.get("adult").unwrap_or(&Value::Bool(false)));
    if adult && !include_adult {
        return Ok(false);
    }
    let pop = match get(payload, "popularity") {
        Some(v) if is_truthy(v) => py_float(v)?,
        _ => 0.0,
    };
    Ok(!matches!(
        pop.partial_cmp(&min_popularity),
        Some(std::cmp::Ordering::Less)
    ))
}

/// `is_complete` — the acceptance rule (spec §9): all four appended
/// response KEYS present (presence, not truthiness).
pub fn is_complete(payload: &Value) -> bool {
    ["genres", "credits", "keywords", "recommendations"]
        .iter()
        .all(|k| payload.get(k).is_some())
}

/// `append_to_response` — the single-call append list per medium.
pub fn append_to_response(medium: &str) -> String {
    let mut parts = vec!["credits", "keywords", "recommendations"];
    if medium == "tv" {
        parts.push("external_ids");
    }
    parts.join(",")
}

// ── stats ─────────────────────────────────────────────────────────

/// Acceptance accounting, field-for-field `ingest.loader.IngestStats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestStats {
    pub attempted: i64,
    pub ingested: i64,
    pub filtered_adult: i64,
    pub filtered_popularity: i64,
    pub failed: i64,
    pub complete: i64,
}

impl IngestStats {
    /// Python's `summary()` format, byte-identical.
    pub fn summary(&self) -> String {
        format!(
            "attempted={} ingested={} complete={} adult_filtered={} \
             popularity_filtered={} failed={}",
            self.attempted,
            self.ingested,
            self.complete,
            self.filtered_adult,
            self.filtered_popularity,
            self.failed
        )
    }
}

// ── binds ─────────────────────────────────────────────────────────

/// A loader bind parameter. `cine_rec_db::Bind` covers the read
/// paths; the loader needs dates, i32s and bools the recall channels
/// never bind, so this enum sits here (wrapping the db seam rather
/// than touching session-1's crate).
#[derive(Debug, Clone)]
pub enum LBind {
    I64(i64),
    OptI64(Option<i64>),
    I32(i32),
    OptI32(Option<i32>),
    /// A DOUBLE PRECISION param (the recs graph) — bits identical on
    /// both sides.
    F64(f64),
    /// A NUMERIC param from an f64 via [`decimal_exact`] — the asyncpg
    /// `Decimal(float)` value.
    OptNumeric(Option<f64>),
    /// A NUMERIC param that was a JSON int (exact, no f64 detour) or a
    /// bool (0/1) — asyncpg accepts both.
    OptNumericInt(i64),
    /// A NUMERIC param parsed EXACTLY from a string (`Decimal(str)` —
    /// arbitrary precision kept).
    OptNumericBig(bigdecimal::BigDecimal),
    Bool(bool),
    OptBool(Option<bool>),
    Str(String),
    OptStr(Option<String>),
    OptDate(Option<NaiveDate>),
    I64Array(Vec<i64>),
}

impl LBind {
    fn apply<'q>(
        &'q self,
        query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    ) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
        match self {
            LBind::I64(v) => query.bind(v),
            LBind::OptI64(v) => query.bind(v),
            LBind::I32(v) => query.bind(v),
            LBind::OptI32(v) => query.bind(v),
            LBind::F64(v) => query.bind(v),
            // A non-finite f64 fails asyncpg's numeric ENCODER in
            // Python; here the same title dies one step later, at the
            // server: a text expression can never cast into a numeric
            // column. (Non-finite f64s cannot come from JSON.)
            LBind::OptNumeric(v) => match v.map(decimal_exact) {
                Some(Err(e)) => query.bind(e.to_string()),
                Some(Ok(d)) => query.bind(d),
                None => query.bind(Option::<bigdecimal::BigDecimal>::None),
            },
            LBind::OptNumericInt(i) => {
                query.bind(bigdecimal::BigDecimal::new(num_bigint::BigInt::from(*i), 0))
            }
            LBind::OptNumericBig(d) => query.bind(d.clone()),
            LBind::Bool(v) => query.bind(v),
            LBind::OptBool(v) => query.bind(v),
            LBind::Str(v) => query.bind(v),
            LBind::OptStr(v) => query.bind(v),
            LBind::OptDate(v) => query.bind(v),
            LBind::I64Array(v) => query.bind(v),
        }
    }
}

type Query<'q> = sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>;

fn bound<'q>(sql: &'q str, binds: &'q [LBind]) -> Query<'q> {
    let mut q = sqlx::query(sql);
    for b in binds {
        q = b.apply(q);
    }
    q
}

/// asyncpg `executemany` parity: all rows in ONE transaction (its
/// documented atomicity), a no-op for an empty row list.
async fn executemany(
    conn: &mut PgConnection,
    sql: &str,
    rows: &[Vec<LBind>],
) -> Result<(), DbError> {
    if rows.is_empty() {
        return Ok(());
    }
    use sqlx::Connection as _;
    let mut tx = conn.begin().await?;
    for row in rows {
        bound(sql, row).execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(())
}

/// `_sync_bridge` — MIRROR one bridge table to the fresh payload:
/// rows whose other-side id vanished from TMDB are deleted; current
/// rows are upserted. Without the delete, a recast actor or a removed
/// keyword would linger forever under ON CONFLICT DO NOTHING.
///
/// `table`/`media_col`/`other_col` are already-RESOLVED pieces (the
/// Python code f-strings the resolved name in); `insert_sql` must be
/// resolved too. `current_ids` stays RAW — genres pass their payload
/// ids untouched in Python, and the id→bigint conversion fails at the
/// DELETE bind exactly where asyncpg fails it.
// The arity mirrors `_sync_bridge` in loader.py argument-for-argument;
// collapsing parameters would obscure the 1:1 port.
#[allow(clippy::too_many_arguments)]
async fn sync_bridge(
    conn: &mut PgConnection,
    table: &str,
    media_col: &str,
    media_id: i64,
    other_col: &str,
    current_ids: &[Value],
    insert_sql: Option<&str>,
    rows: &[Vec<LBind>],
) -> Result<(), LoaderError> {
    if !current_ids.is_empty() {
        // bigint[] encoding is element-wise asyncpg: strings rejected,
        // floats truncated (review M-1).
        let ids: Vec<i64> = current_ids
            .iter()
            .map(|v| ap_int(v, "bridge current_ids"))
            .collect::<Result<Vec<_>, _>>()?;
        sqlx::query(&format!(
            "DELETE FROM {table} WHERE {media_col} = $1 AND {other_col} <> ALL($2::bigint[])"
        ))
        .bind(media_id)
        .bind(ids)
        .execute(&mut *conn)
        .await?;
    } else {
        sqlx::query(&format!("DELETE FROM {table} WHERE {media_col} = $1"))
            .bind(media_id)
            .execute(&mut *conn)
            .await?;
    }
    if !rows.is_empty() {
        let sql = insert_sql.expect("rows require insert_sql (Python asserts the same)");
        executemany(conn, sql, rows).await?;
    }
    Ok(())
}

// ── the fetcher seam ──────────────────────────────────────────────

/// The `_fetch_title` seam: fetch one title's appended payload.
/// Returns `None` on a fetch error (Python: `data if "error" not in
/// data else None`). The HTTP implementation (retry + semaphore +
/// RateLimiter pacing) is wave E; tests inject payload maps.
pub trait TitleFetcher {
    fn fetch_title(
        &mut self,
        media_id: i64,
        medium: &str,
    ) -> impl std::future::Future<Output = Option<Value>> + Send;
}

// ── the ingest ────────────────────────────────────────────────────

/// Loads the mirror into Postgres — the `TmdbIngest` port (fetching
/// via the [`TitleFetcher`] seam until wave E).
pub struct TmdbIngest {
    db: Db,
    api_key: String,
    /// Python holds the limiter + semaphore inside the ingest and
    /// paces every fetch through them; wave E's HTTP fetcher takes
    /// handles from here.
    limiter: crate::rate::RateLimiter,
    semaphore: tokio::sync::Semaphore,
    stats: std::sync::Mutex<IngestStats>,
}

impl TmdbIngest {
    /// `TmdbIngest(pool, api_key=None, rate_per_second=40)` — the key
    /// falls back to `TMDB_API_KEY` and a missing key is the same
    /// loud ValueError.
    pub fn new(db: Db, api_key: Option<&str>, rate_per_second: i64) -> Result<Self, String> {
        let api_key = api_key
            .map(str::to_string)
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var("TMDB_API_KEY").ok().filter(|k| !k.is_empty()))
            .ok_or_else(|| "TMDB_API_KEY missing (param or env)".to_string())?;
        let limiter =
            crate::rate::RateLimiter::new(rate_per_second as f64, 1).map_err(|e| e.to_string())?;
        Ok(Self {
            db,
            api_key,
            limiter,
            // min(rate, 64) like Python's Semaphore(min(rate, 64)).
            semaphore: tokio::sync::Semaphore::new(rate_per_second.clamp(1, 64) as usize),
            stats: std::sync::Mutex::new(IngestStats::default()),
        })
    }

    /// The configured API key (wave E's fetcher needs it).
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// The pacer every fetch goes through (wave E's fetcher needs it).
    pub fn limiter(&self) -> &crate::rate::RateLimiter {
        &self.limiter
    }

    /// Concurrency cap for fetches (wave E's fetcher needs it).
    pub fn semaphore(&self) -> &tokio::sync::Semaphore {
        &self.semaphore
    }

    /// Live counters (the Python `ingest.stats` object).
    pub fn stats(&self) -> IngestStats {
        self.stats.lock().unwrap().clone()
    }

    fn bump(&self, f: impl FnOnce(&mut IngestStats)) {
        let mut s = self.stats.lock().unwrap();
        f(&mut s);
    }

    /// `existing_ids` — the skip-existing index per medium.
    pub async fn existing_ids(
        &self,
    ) -> Result<std::collections::BTreeMap<&'static str, HashSet<i64>>, LoaderError> {
        use sqlx::Row;
        let mut out = std::collections::BTreeMap::new();
        for (logical, medium) in [("tmdb_movies", "movie"), ("tmdb_tv", "tv")] {
            let mut ids = HashSet::new();
            for row in self
                .db
                .fetch(&format!("SELECT id FROM {{t_{logical}}}"), &[])
                .await?
            {
                ids.insert(row.try_get::<i64, _>("id")?);
            }
            out.insert(medium, ids);
        }
        Ok(out)
    }

    /// `_upsert_title` — upsert one title + MIRROR its bridges to the
    /// fresh payload. Statement-for-statement the Python sequence:
    /// the title upsert goes to the POOL, the whole bridge batch runs
    /// on ONE acquired connection (no wrapping transaction — same
    /// granularity as Python; each executemany is atomic on its own).
    pub async fn upsert_title(&self, payload: &Value, medium: &str) -> Result<(), LoaderError> {
        let movie = medium == "movie";
        let media_id = req_int(payload, "id")?;
        let col = if movie { "movie_id" } else { "tv_id" };

        // 1. the title row (pool-level execute in Python). Raw fields
        // convert to binds HERE — asyncpg encoder parity.
        if movie {
            let row = title_row_from_payload(payload)?;
            let binds = vec![
                LBind::I64(row.id),
                bind_text_req(&row.title, "title")?,
                bind_text(&row.original_title, "original_title")?,
                bind_text(&row.original_language, "original_language")?,
                bind_text(&row.overview, "overview")?,
                LBind::OptDate(row.release_date),
                bind_i32(&row.runtime, "runtime")?,
                bind_text(&row.status, "status")?,
                LBind::Bool(row.adult),
                bind_numeric(&row.vote_average, "vote_average")?,
                bind_i32(&row.vote_count, "vote_count")?,
                bind_numeric(&row.popularity, "popularity")?,
                bind_text(&row.poster_path, "poster_path")?,
                bind_text(&row.backdrop_path, "backdrop_path")?,
                bind_i64(&row.collection_id, "collection_id")?,
                bind_text(&row.imdb_id, "imdb_id")?,
                bind_i64(&row.budget, "budget")?,
                bind_i64(&row.revenue, "revenue")?,
            ];
            bound(&self.db.sql(UPSERT_MOVIE)?, &binds)
                .execute(self.db.pool())
                .await?;
        } else {
            let row = tv_row_from_payload(payload)?;
            let binds = vec![
                LBind::I64(row.id),
                bind_text_req(&row.name, "name")?,
                bind_text(&row.original_name, "original_name")?,
                bind_text(&row.original_language, "original_language")?,
                bind_text(&row.overview, "overview")?,
                LBind::OptDate(row.first_air_date),
                LBind::OptDate(row.last_air_date),
                bind_text(&row.status, "status")?,
                bind_bool(&row.in_production, "in_production")?,
                bind_i32(&row.number_of_seasons, "number_of_seasons")?,
                bind_i32(&row.number_of_episodes, "number_of_episodes")?,
                LBind::Bool(row.adult),
                bind_numeric(&row.vote_average, "vote_average")?,
                bind_i32(&row.vote_count, "vote_count")?,
                bind_numeric(&row.popularity, "popularity")?,
                bind_text(&row.poster_path, "poster_path")?,
                bind_text(&row.backdrop_path, "backdrop_path")?,
                bind_text(&row.imdb_id, "imdb_id")?,
            ];
            bound(&self.db.sql(UPSERT_TV)?, &binds)
                .execute(self.db.pool())
                .await?;
        }

        // 2. the bridge batch, one connection. The extraction ORDER is
        // Python's order (lines 322-331, all BEFORE the acquire):
        // shape errors here fail the title with only the fact row
        // written; keywords/companies/networks CONVERSIONS happen at
        // their statements INSIDE the batch (review M-2) — a bad
        // shape there is carried as [`MaybeBad`] and raised at the
        // exact statement Python raises at.

        // genres: `[(g["id"], g["name"]) for g in
        // payload.get("genres", [])]` — eager, both slots raw. A
        // PRESENT null / truthy non-list is a TypeError in Python
        // (iterating None / a scalar); absent → [] (review M-3).
        let genres: Vec<(Value, Value)> = match payload.get("genres") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|g| Ok((req_raw(g, "id")?.clone(), req_raw(g, "name")?.clone())))
                .collect::<Result<Vec<_>, PyConvError>>()?,
            Some(other) => {
                return Err(PyConvError::Type(format!(
                    "'{}' object is not iterable",
                    crate::pyval::py_type_name(other)
                ))
                .into())
            }
        };

        // keywords container: `(payload.get("keywords", {}) or {})` —
        // falsy → {}, dict → itself, truthy non-dict → AttributeError
        // (eager — Python evaluates the .get chain immediately).
        let kw_container: Option<&serde_json::Map<String, Value>> = match payload.get("keywords") {
            None | Some(Value::Null) => None,
            Some(v) if !is_truthy(v) => None,
            Some(v @ Value::Object(_)) => Some(v.as_object().unwrap()),
            Some(other) => {
                return Err(PyConvError::Type(format!(
                    "keywords: '{}' object has no attribute 'get'",
                    crate::pyval::py_type_name(other)
                ))
                .into())
            }
        };
        // The keyword LIST is extracted eagerly (with the results
        // fallback), but its id/name conversion is DEFERRED to the
        // keywords statement inside the batch.
        let kw_list: MaybeBad = extract_list(kw_container.and_then(|c| c.get("keywords")));
        let kw_list = match &kw_list {
            MaybeBad::List(items) if items.is_empty() => {
                extract_list(kw_container.and_then(|c| c.get("results")))
            }
            other => other.clone(),
        };

        // credits: `payload.get("credits") or {}` — falsy → {},
        // truthy non-dict → AttributeError (eager).
        let credits_obj: serde_json::Map<String, Value> = match payload.get("credits") {
            None | Some(Value::Null) => Default::default(),
            Some(v) if !is_truthy(v) => Default::default(),
            Some(v @ Value::Object(_)) => v.as_object().unwrap().clone(),
            Some(other) => {
                return Err(PyConvError::Type(format!(
                    "credits: '{}' object has no attribute 'get'",
                    crate::pyval::py_type_name(other)
                ))
                .into())
            }
        };
        let cast_list: Vec<Value> = match credits_obj.get("cast") {
            Some(v) if is_truthy(v) => match v {
                Value::Array(a) => a.clone(),
                other => {
                    return Err(PyConvError::Type(format!(
                        "cast: iterating {} is a TypeError in Python",
                        crate::pyval::py_type_name(other)
                    ))
                    .into())
                }
            },
            _ => Vec::new(),
        };
        let crew_list: Vec<Value> = match credits_obj.get("crew") {
            Some(v) if is_truthy(v) => match v {
                Value::Array(a) => a.clone(),
                other => {
                    return Err(PyConvError::Type(format!(
                        "crew: iterating {} is a TypeError in Python",
                        crate::pyval::py_type_name(other)
                    ))
                    .into())
                }
            },
            _ => Vec::new(),
        };
        let cast = keep_cast(&cast_list)?;
        let crew = keep_crew(&crew_list)?;
        let people: HashSet<i64> = cast
            .iter()
            .map(|c| c.person_id)
            .chain(crew.iter().map(|c| c.person_id))
            .collect();

        // companies/networks: extraction only here (Python lines
        // 330-331); int()/name conversions run at their statements
        // inside the batch.
        let companies_m = extract_list(payload.get("production_companies"));
        let networks_m = if movie {
            MaybeBad::List(Vec::new())
        } else {
            extract_list(payload.get("networks"))
        };

        let mut conn = self.db.pool().acquire().await?;

        // People: gated on KEPT people being non-empty, but the rows
        // cover ALL credited people (last name wins per id, first
        // position kept — Python dict semantics). Names are
        // `c.get("name") or ""` — raw-or-empty.
        if !people.is_empty() {
            let mut names: Vec<(i64, Value)> = Vec::new();
            for c in cast_list.iter().chain(crew_list.iter()) {
                let id = req_int(c, "id")?;
                let name = match get(c, "name") {
                    Some(v) if is_truthy(v) => v.clone(),
                    _ => Value::String(String::new()),
                };
                match names.iter_mut().find(|(pid, _)| *pid == id) {
                    Some(slot) => slot.1 = name,
                    None => names.push((id, name)),
                }
            }
            let rows: Vec<Vec<LBind>> = names
                .iter()
                .map(|(id, name)| Ok(vec![LBind::I64(*id), bind_text_req(name, "people.name")?]))
                .collect::<Result<Vec<_>, PyConvError>>()?;
            executemany(&mut conn, &self.db.sql(UPSERT_PERSON)?, &rows).await?;
        }

        if !genres.is_empty() {
            let stmt = self
                .db
                .sql(if movie { UPSERT_GENRE } else { UPSERT_TV_GENRE })?;
            let rows: Vec<Vec<LBind>> = genres
                .iter()
                .map(|(id, name)| {
                    Ok(vec![
                        bind_i32(&Some(id.clone()), "genre.id")?,
                        bind_text_req(name, "genre.name")?,
                    ])
                })
                .collect::<Result<Vec<_>, PyConvError>>()?;
            executemany(&mut conn, &stmt, &rows).await?;
        }
        let gmap = self.db.sql(if movie {
            "{t_tmdb_movie_genres_map}"
        } else {
            "{t_tmdb_tv_genres_map}"
        })?;
        sync_bridge(
            &mut conn,
            &gmap,
            col,
            media_id,
            "genre_id",
            &genres
                .iter()
                .map(|(gid, _)| gid.clone())
                .collect::<Vec<_>>(),
            Some(&format!(
                "INSERT INTO {gmap} ({col}, genre_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
            )),
            &genres
                .iter()
                .map(|(gid, _)| {
                    Ok(vec![
                        LBind::I64(media_id),
                        bind_i32(&Some(gid.clone()), "genre_id")?,
                    ])
                })
                .collect::<Result<Vec<_>, PyConvError>>()?,
        )
        .await?;

        // keywords: the id/name conversion runs HERE (Python's
        // `[(int(k["id"]), k["name"]) for k in kw]` inside the batch)
        // — after people and genres have been written (review M-2).
        let kw_ids_names: Vec<(i64, Value)> = match &kw_list {
            MaybeBad::Bad(_) => {
                return Err(kw_list.bad("keywords").unwrap().into());
            }
            MaybeBad::List(items) => items
                .iter()
                .map(|k| Ok((req_int(k, "id")?, req_raw(k, "name")?.clone())))
                .collect::<Result<Vec<_>, PyConvError>>()?,
        };
        if !kw_ids_names.is_empty() {
            let rows: Vec<Vec<LBind>> = kw_ids_names
                .iter()
                .map(|(id, name)| Ok(vec![LBind::I64(*id), bind_text_req(name, "keyword.name")?]))
                .collect::<Result<Vec<_>, PyConvError>>()?;
            executemany(&mut conn, &self.db.sql(UPSERT_KEYWORD)?, &rows).await?;
        }
        let kw_tbl = self.db.sql(if movie {
            "{t_tmdb_movie_keywords}"
        } else {
            "{t_tmdb_tv_keywords}"
        })?;
        let kw_ids: Vec<i64> = kw_ids_names.iter().map(|(id, _)| *id).collect();
        sync_bridge(
            &mut conn,
            &kw_tbl,
            col,
            media_id,
            "keyword_id",
            &kw_ids
                .iter()
                .map(|i| serde_json::json!(i))
                .collect::<Vec<_>>(),
            Some(&format!(
                "INSERT INTO {kw_tbl} ({col}, keyword_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
            )),
            &kw_ids
                .iter()
                .map(|id| vec![LBind::I64(media_id), LBind::I64(*id)])
                .collect::<Vec<_>>(),
        )
        .await?;

        // companies: conversion HERE too (Python converts inside the
        // batch, after the keywords statements).
        let companies: Vec<(i64, Value)> = match &companies_m {
            MaybeBad::Bad(_) => {
                return Err(companies_m.bad("production_companies").unwrap().into());
            }
            MaybeBad::List(items) => items
                .iter()
                .map(|c| Ok((req_int(c, "id")?, req_raw(c, "name")?.clone())))
                .collect::<Result<Vec<_>, PyConvError>>()?,
        };
        if !companies.is_empty() {
            let rows: Vec<Vec<LBind>> = companies
                .iter()
                .map(|(id, name)| Ok(vec![LBind::I64(*id), bind_text_req(name, "company.name")?]))
                .collect::<Result<Vec<_>, PyConvError>>()?;
            executemany(&mut conn, &self.db.sql(UPSERT_COMPANY)?, &rows).await?;
        }
        let comp_tbl = self.db.sql(if movie {
            "{t_tmdb_movie_companies}"
        } else {
            "{t_tmdb_tv_companies}"
        })?;
        let comp_ids: Vec<i64> = companies.iter().map(|(id, _)| *id).collect();
        sync_bridge(
            &mut conn,
            &comp_tbl,
            col,
            media_id,
            "company_id",
            &comp_ids
                .iter()
                .map(|i| serde_json::json!(i))
                .collect::<Vec<_>>(),
            Some(&format!(
                "INSERT INTO {comp_tbl} ({col}, company_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
            )),
            &comp_ids
                .iter()
                .map(|id| vec![LBind::I64(media_id), LBind::I64(*id)])
                .collect::<Vec<_>>(),
        )
        .await?;

        // networks: tv-ONLY bridge (H-1): a movie sharing the id with
        // a series must never mirror — let alone EMPTY — the series'
        // networks. TMDB ids collide across media (AGENTS.md inv. 4);
        // fixed on BOTH sides in the same change.
        if !movie {
            let networks: Vec<(i64, Value)> = match &networks_m {
                MaybeBad::Bad(_) => {
                    return Err(networks_m.bad("networks").unwrap().into());
                }
                MaybeBad::List(items) => items
                    .iter()
                    .map(|n| Ok((req_int(n, "id")?, req_raw(n, "name")?.clone())))
                    .collect::<Result<Vec<_>, PyConvError>>()?,
            };
            if !networks.is_empty() {
                let rows: Vec<Vec<LBind>> = networks
                    .iter()
                    .map(|(id, name)| {
                        Ok(vec![LBind::I64(*id), bind_text_req(name, "network.name")?])
                    })
                    .collect::<Result<Vec<_>, PyConvError>>()?;
                executemany(&mut conn, &self.db.sql(UPSERT_NETWORK)?, &rows).await?;
            }
            let net_tbl = self.db.sql("{t_tmdb_tv_networks_map}")?;
            let net_ids: Vec<i64> = networks.iter().map(|(id, _)| *id).collect();
            sync_bridge(
                &mut conn,
                &net_tbl,
                "tv_id",
                media_id,
                "network_id",
                &net_ids
                    .iter()
                    .map(|i| serde_json::json!(i))
                    .collect::<Vec<_>>(),
                Some(&format!(
                    "INSERT INTO {net_tbl} (tv_id, network_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
                )),
                &net_ids
                    .iter()
                    .map(|id| vec![LBind::I64(media_id), LBind::I64(*id)])
                    .collect::<Vec<_>>(),
            )
            .await?;
        }

        let cast_tbl = self.db.sql(if movie {
            "{t_tmdb_movie_cast}"
        } else {
            "{t_tmdb_tv_cast}"
        })?;
        let cast_ids: Vec<i64> = {
            let mut ids: Vec<i64> = cast.iter().map(|c| c.person_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        sync_bridge(
            &mut conn,
            &cast_tbl,
            col,
            media_id,
            "person_id",
            &cast_ids
                .iter()
                .map(|i| serde_json::json!(i))
                .collect::<Vec<_>>(),
            Some(&format!(
                "INSERT INTO {cast_tbl} ({col}, person_id, character, cast_order) \
                 VALUES ($1, $2, $3, $4) ON CONFLICT ({col}, person_id, cast_order) \
                 DO UPDATE SET character = EXCLUDED.character"
            )),
            &cast
                .iter()
                .map(|c| {
                    Ok(vec![
                        LBind::I64(media_id),
                        LBind::I64(c.person_id),
                        LBind::Str(c.character.clone()),
                        // INT4 range checked at the BIND (asyncpg
                        // parity; the builder keeps the full int).
                        bind_i32(
                            &Some(Value::Number(serde_json::Number::from(c.cast_order))),
                            "cast_order",
                        )?,
                    ])
                })
                .collect::<Result<Vec<_>, PyConvError>>()?,
        )
        .await?;

        let crew_tbl = self.db.sql(if movie {
            "{t_tmdb_movie_crew}"
        } else {
            "{t_tmdb_tv_crew}"
        })?;
        let crew_ids: Vec<i64> = {
            let mut ids: Vec<i64> = crew.iter().map(|c| c.person_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        sync_bridge(
            &mut conn,
            &crew_tbl,
            col,
            media_id,
            "person_id",
            &crew_ids
                .iter()
                .map(|i| serde_json::json!(i))
                .collect::<Vec<_>>(),
            Some(&format!(
                "INSERT INTO {crew_tbl} ({col}, person_id, job, department) \
                 VALUES ($1, $2, $3, $4) ON CONFLICT ({col}, person_id, job) \
                 DO UPDATE SET department = EXCLUDED.department"
            )),
            &crew
                .iter()
                .map(|c| {
                    Ok(vec![
                        LBind::I64(media_id),
                        LBind::I64(c.person_id),
                        LBind::Str(c.job.clone()),
                        bind_text(&c.department, "department")?,
                    ])
                })
                .collect::<Result<Vec<_>, PyConvError>>()?,
        )
        .await?;

        let recs = rec_rows_from_payload(payload)?;
        sqlx::query(&self.db.sql(DELETE_RECS_TAIL)?)
            .bind(media_id)
            .bind(medium)
            .bind(recs.len() as i32)
            .execute(&mut *conn)
            .await?;
        if !recs.is_empty() {
            let rows: Vec<Vec<LBind>> = recs
                .iter()
                .map(|r| {
                    vec![
                        LBind::I64(r.media_id),
                        LBind::Str(r.media_type.to_string()),
                        LBind::I64(r.rec_media_id),
                        LBind::Str(r.rec_media_type.to_string()),
                        LBind::I32(r.rank),
                        LBind::F64(r.popularity),
                    ]
                })
                .collect::<Vec<_>>();
            executemany(&mut conn, &self.db.sql(UPSERT_REC)?, &rows).await?;
        }
        Ok(())
    }
    /// `ingest_one` — fetch + gate + upsert one title with the exact
    /// stats accounting. True when ingested; every failure path just
    /// counts `failed` and returns false (the Python catch-all — the
    /// gate itself raising, e.g. `float("abc")` in should_ingest,
    /// lands in `failed` exactly like Python's except).
    pub async fn ingest_one<F: TitleFetcher>(
        &self,
        fetcher: &mut F,
        media_id: i64,
        medium: &str,
        min_popularity: f64,
        include_adult: bool,
    ) -> bool {
        self.bump(|s| s.attempted += 1);
        let payload = fetcher.fetch_title(media_id, medium).await;
        let gate = should_ingest(payload.as_ref(), min_popularity, include_adult);
        let allowed = match gate {
            Ok(allowed) => allowed,
            Err(e) => {
                // Python: should_ingest raises inside the try → except
                // → failed (review M-3).
                self.bump(|s| s.failed += 1);
                eprintln!("ingest {medium}/{media_id} failed: {e}");
                return false;
            }
        };
        if !allowed {
            let adult = payload
                .as_ref()
                .map(|p| is_truthy(p.get("adult").unwrap_or(&Value::Bool(false))))
                .unwrap_or(false);
            if payload.is_some() && adult {
                self.bump(|s| s.filtered_adult += 1);
            } else if payload.is_some() {
                self.bump(|s| s.filtered_popularity += 1);
            } else {
                self.bump(|s| s.failed += 1);
            }
            return false;
        }
        let payload = payload.expect("should_ingest(true) implies a payload");
        match self.upsert_title(&payload, medium).await {
            Ok(()) => {
                self.bump(|s| s.ingested += 1);
                if is_complete(&payload) {
                    self.bump(|s| s.complete += 1);
                }
                true
            }
            Err(e) => {
                // logger.warning in Python; the run continues.
                self.bump(|s| s.failed += 1);
                eprintln!("ingest {medium}/{media_id} failed: {e}");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_to_response_shape() {
        assert_eq!(
            append_to_response("movie"),
            "credits,keywords,recommendations"
        );
        assert_eq!(
            append_to_response("tv"),
            "credits,keywords,recommendations,external_ids"
        );
        assert_eq!(
            append_to_response("other"),
            "credits,keywords,recommendations"
        );
    }

    #[test]
    fn keep_cast_top5_with_ties_and_drops() {
        let payload = serde_json::json!([
            {"id": 1, "character": "A", "order": 3},
            {"id": 2, "character": null, "order": 1},          // character null → ""
            {"id": 3, "order": 0},                              // missing character
            {"id": 4, "character": "D", "order": 9},
            {"id": 5, "character": null},                       // no order → dropped
            {"id": 6, "character": "F"},                        // no order → dropped
            {"id": 7, "character": "G", "order": 1},            // TIE with 2, after it
            {"id": 8, "character": "H", "order": 2},
            {"id": 9, "character": "I", "order": 2},            // tie with 8
            {"id": 10, "order": 7}                              // missing character
        ]);
        let rows = keep_cast(payload.as_array().unwrap()).unwrap();
        // Stable sort by order: [0:3, 1:2, 1:7, 2:8, 2:9] → top-5 cuts here.
        assert_eq!(
            rows,
            vec![
                CastRow {
                    person_id: 3,
                    character: "".into(),
                    cast_order: 0
                },
                CastRow {
                    person_id: 2,
                    character: "".into(),
                    cast_order: 1
                },
                CastRow {
                    person_id: 7,
                    character: "G".into(),
                    cast_order: 1
                },
                CastRow {
                    person_id: 8,
                    character: "H".into(),
                    cast_order: 2
                },
                CastRow {
                    person_id: 9,
                    character: "I".into(),
                    cast_order: 2
                },
            ]
        );
    }

    #[test]
    fn keep_crew_whitelist_and_medium_agnostic() {
        let payload = serde_json::json!([
            {"id": 1, "job": "Director", "department": "Directing"},
            {"id": 2, "job": "Sound"},                            // not whitelisted
            {"id": 3, "job": "Writer", "department": null},
            {"id": 4, "job": "Creator"},                          // kept even for movies
            {"id": 5, "department": "Writing"},                   // no job → dropped
            {"job": "Director", "id": 6},
        ]);
        let rows = keep_crew(payload.as_array().unwrap()).unwrap();
        assert_eq!(
            rows,
            vec![
                CrewRow {
                    person_id: 1,
                    job: "Director".into(),
                    department: Some("Directing".into())
                },
                CrewRow {
                    person_id: 3,
                    job: "Writer".into(),
                    department: None
                },
                CrewRow {
                    person_id: 4,
                    job: "Creator".into(),
                    department: None
                },
                CrewRow {
                    person_id: 6,
                    job: "Director".into(),
                    department: None
                },
            ]
        );
    }

    #[test]
    fn should_ingest_gate_matrix() {
        let ok = |r: Result<bool, PyConvError>| r.unwrap();
        let movie = serde_json::json!({"id": 1, "adult": false, "popularity": 5.0});
        assert!(ok(should_ingest(Some(&movie), 0.0, false)));
        assert!(!ok(should_ingest(Some(&movie), 10.0, false)));
        assert!(!ok(should_ingest(None, 0.0, false)));
        // success=false is identity-False in Python — the exact bool.
        let rejected = serde_json::json!({"success": false});
        assert!(!ok(should_ingest(Some(&rejected), 0.0, false)));
        // adult truthiness — a "false" STRING is truthy in Python.
        let adult_str = serde_json::json!({"id": 2, "adult": "false", "popularity": 1.0});
        assert!(!ok(should_ingest(Some(&adult_str), 0.0, false)));
        assert!(ok(should_ingest(Some(&adult_str), 0.0, true)));
        // popularity as a numeric string goes through float().
        let pop_str = serde_json::json!({"id": 3, "popularity": "7.5"});
        assert!(ok(should_ingest(Some(&pop_str), 7.0, false)));
        assert!(!ok(should_ingest(Some(&pop_str), 8.0, false)));
        // An unconvertible popularity RAISES (float("abc")) — the
        // caller counts the title failed, never filtered (M-3).
        let pop_bad = serde_json::json!({"id": 4, "popularity": "abc"});
        assert!(should_ingest(Some(&pop_bad), 0.0, false).is_err());
    }

    #[test]
    fn stats_summary_is_byte_identical() {
        let s = IngestStats {
            attempted: 10,
            ingested: 7,
            filtered_adult: 1,
            filtered_popularity: 1,
            failed: 1,
            complete: 6,
        };
        assert_eq!(
            s.summary(),
            "attempted=10 ingested=7 complete=6 adult_filtered=1 popularity_filtered=1 failed=1"
        );
    }

    #[test]
    fn rec_rows_derive_medium_from_key_presence() {
        // "title" present → movie, even when null.
        let movie_null_title = serde_json::json!({
            "id": 5, "title": null,
            "recommendations": {"results": [{"id": 9, "popularity": "12.5"}]}
        });
        let rows = rec_rows_from_payload(&movie_null_title).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].media_type, "movie");
        assert_eq!(rows[0].rec_media_type, "movie");
        assert_eq!(rows[0].rank, 1);
        assert_eq!(rows[0].popularity, 12.5); // numeric string via float()
                                              // tv shape: results via keywords.results is separate; recs absent → []
        let tv = serde_json::json!({"id": 5, "name": "X"});
        assert!(rec_rows_from_payload(&tv).unwrap().is_empty());
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use serde_json::json;

    /// The asyncpg encoders (review M-1) — every branch also proven by
    /// the live parity plan; these pin them offline.
    #[test]
    fn asyncpg_int_encoder_semantics() {
        assert!(
            ap_int(&json!("120"), "t").is_err(),
            "strings are a DataError"
        );
        assert_eq!(ap_int(&json!(120.5), "t").unwrap(), 120, "float truncates");
        assert_eq!(ap_int(&json!(-0.7), "t").unwrap(), 0, "toward zero");
        assert_eq!(ap_int(&json!(true), "t").unwrap(), 1);
        assert_eq!(ap_int(&json!(false), "t").unwrap(), 0);
    }

    #[test]
    fn asyncpg_numeric_encoder_semantics() {
        // bool → 0/1; strings parse as Decimal (arbitrary precision,
        // whitespace/underscore/exponent forms); NaN/Infinity strings
        // are the DOCUMENTED divergence — loud failure instead of
        // numeric NaN/∞ storage; containers rejected.
        assert!(bind_numeric(&Some(json!(true)), "v").is_ok());
        assert!(bind_numeric(&Some(json!("1.1000000000000000000001")), "v").is_ok());
        assert!(bind_numeric(&Some(json!(" 7.5 ")), "v").is_ok());
        assert!(bind_numeric(&Some(json!("1_000")), "v").is_ok());
        assert!(bind_numeric(&Some(json!("1e5")), "v").is_ok());
        assert!(bind_numeric(&Some(json!("NaN")), "v").is_err());
        assert!(bind_numeric(&Some(json!("-inf")), "v").is_err());
        assert!(bind_numeric(&Some(json!([1])), "v").is_err());
        // Exact string precision: the BigDecimal keeps all 23 digits
        // (an f64 detour would not).
        match bind_numeric(&Some(json!("1.1000000000000000000001")), "v") {
            Ok(LBind::OptNumericBig(d)) => {
                assert_eq!(d.to_string(), "1.1000000000000000000001")
            }
            other => panic!("expected exact BigDecimal, got {other:?}"),
        }
    }
}
