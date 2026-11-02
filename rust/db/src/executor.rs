//! SQL execution over a live Postgres pool, with logical-name
//! resolution — the Rust side of `db.py` + the asyncpg pool.
//!
//! All SQL goes through [`SchemaMap::resolve`] before execution
//! (`{t_<logical>}` placeholders); SQL without placeholders passes
//! through unchanged. Queries are built DYNAMICICALLY (sqlx without
//! compile-time verification): table names are runtime data through
//! the registry, which is the whole point of the schema map.

use std::sync::Arc;

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::degrade;
use crate::schema_map::{SchemaMap, SchemaMapError};

/// Errors from the DB layer. [`DbError::ChannelOff`] is the DEGRADE
/// signal — an optional channel missing its column/extension/table —
/// callers switch the channel off and continue; it is never a failure
/// of the request.
#[derive(Debug)]
pub enum DbError {
    Map(SchemaMapError),
    Sqlx(sqlx::Error),
    /// An optional channel's dependency is missing (undefined column /
    /// object / function / table). `channel` names the recall channel
    /// for logging.
    ChannelOff {
        channel: &'static str,
        reason: String,
    },
    /// Invalid caller input (the port of Python's ValueError paths —
    /// §A required fields, feedback kinds, skip-decay bounds).
    Invalid(String),
}

impl From<SchemaMapError> for DbError {
    fn from(e: SchemaMapError) -> Self {
        DbError::Map(e)
    }
}

impl From<sqlx::Error> for DbError {
    fn from(e: sqlx::Error) -> Self {
        DbError::Sqlx(e)
    }
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::Map(e) => write!(f, "schema map: {e}"),
            DbError::Sqlx(e) => write!(f, "sql: {e}"),
            DbError::ChannelOff { channel, reason } => {
                write!(f, "channel '{channel}' off: {reason}")
            }
            DbError::Invalid(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for DbError {}

impl DbError {
    /// Classify a Postgres error as a degrade signal for `channel`.
    /// Mirrors the Python excepts: asyncpg `UndefinedColumnError` /
    /// `UndefinedObjectError` / `UndefinedTableError` and the missing
    /// `<=>` operator (undefined function) all mean "channel off".
    pub fn degrade(channel: &'static str, err: sqlx::Error) -> Self {
        if let Some(code) = degrade::pg_code(&err) {
            if degrade::is_missing_dependency(code) {
                return DbError::ChannelOff {
                    channel,
                    reason: degrade::reason(code, &err),
                };
            }
        }
        DbError::Sqlx(err)
    }

    /// Re-classify a wrapped sqlx error as a degrade signal for
    /// `channel` (keeps non-degrade errors intact).
    pub fn into_degraded(self, channel: &'static str) -> Self {
        match self {
            DbError::Sqlx(ref err) => {
                let code = degrade::pg_code(err);
                if code.map(degrade::is_missing_dependency).unwrap_or(false) {
                    DbError::ChannelOff {
                        channel,
                        reason: degrade::reason(code.unwrap(), err),
                    }
                } else {
                    self
                }
            }
            other => other,
        }
    }
}

/// Pool + schema map: the single entry point for engine SQL.
#[derive(Clone)]
pub struct Db {
    pool: PgPool,
    map: Arc<SchemaMap>,
}

impl Db {
    fn pool_options() -> PgPoolOptions {
        // asyncpg parity: it runs RESET ALL when a connection returns to
        // the pool; sqlx does not, so a session GUC set on one borrower
        // (e.g. hnsw.ef_search) would LEAK to every later one and make
        // results connection-history-dependent.
        PgPoolOptions::new()
            .max_connections(10)
            .after_release(|conn, _meta| {
                Box::pin(async move {
                    sqlx::query("RESET ALL").execute(conn).await?;
                    Ok(true)
                })
            })
    }

    /// Connect with default table mapping.
    pub async fn connect(dsn: &str) -> Result<Self, DbError> {
        let pool = Self::pool_options().connect(dsn).await?;
        Ok(Self {
            pool,
            map: Arc::new(SchemaMap::new()),
        })
    }

    /// Connect with an explicit map (tests, remapped deployments).
    pub async fn connect_with_map(dsn: &str, map: SchemaMap) -> Result<Self, DbError> {
        let pool = Self::pool_options().connect(dsn).await?;
        Ok(Self {
            pool,
            map: Arc::new(map),
        })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub fn map(&self) -> &SchemaMap {
        &self.map
    }

    /// Resolve a template's `{t_*}` placeholders (the `db.py` seam —
    /// every query text passes through here exactly once).
    pub fn sql(&self, template: &str) -> Result<String, DbError> {
        Ok(self.map.resolve(template)?)
    }

    /// SET a session GUC and fetch on ONE acquired connection
    /// (hnsw.ef_search only applies to the connection it ran on).
    /// `guc` must be a literal SET statement (no user input).
    pub async fn set_and_fetch(
        &self,
        guc: &str,
        sql: &str,
        binds: &[(&'static str, Bind)],
    ) -> Result<Vec<sqlx::postgres::PgRow>, DbError> {
        let mut conn = self.pool.acquire().await?;
        // a failed SET must propagate (Python: db.execute raises) — a
        // silently-missed GUC is a silently-wrong query plan
        sqlx::query(guc).execute(&mut *conn).await?;
        let mut q = sqlx::query(sql);
        for (_, b) in binds {
            q = b.apply(q);
        }
        Ok(q.fetch_all(&mut *conn).await?)
    }

    /// fetch (many rows) — the resolved-SQL `db.fetch`.
    pub async fn fetch(
        &self,
        template: &str,
        binds: &[(&'static str, Bind)],
    ) -> Result<Vec<sqlx::postgres::PgRow>, DbError> {
        let resolved = self.sql(template)?;
        let mut q = sqlx::query(&resolved);
        for (_, b) in binds {
            q = b.apply(q);
        }
        Ok(q.fetch_all(&self.pool).await?)
    }
}

/// A typed bind parameter (keeps call sites readable without sqlx
/// type inference gymnastics over enums).
pub enum Bind {
    I64(i64),
    F64(f64),
    Text(String),
    TextArray(Vec<String>),
    I64Array(Vec<i64>),
    OptI64(Option<i64>),
    OptText(Option<String>),
    /// pgvector vector bind (the user-vector ANN channels bind the
    /// query vector as a parameter; KNN seeds stay in-statement).
    Vector(Vec<f32>),
}

impl Bind {
    fn apply<'q>(
        &'q self,
        query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    ) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
        match self {
            Bind::I64(v) => query.bind(v),
            Bind::F64(v) => query.bind(v),
            Bind::Text(v) => query.bind(v),
            Bind::TextArray(v) => query.bind(v),
            Bind::I64Array(v) => query.bind(v),
            Bind::OptI64(v) => query.bind(v),
            Bind::OptText(v) => query.bind(v),
            Bind::Vector(v) => query.bind(pgvector::Vector::from(v.clone())),
        }
    }
}

/// Row-extraction helpers matching the asyncpg record access the
/// Python rows use (`dict.get` semantics):
/// * NULL or column ABSENT → `Ok(None)` — Python's `r.get(k)` returns
///   None for a missing key (channels select different column sets)
/// * column PRESENT but the wrong TYPE (e.g. an INT4 read as text) is
///   an ERROR — swallowing it hid real bugs (an INT4 read as i64 once
///   silently became None everywhere)
///
/// INT columns decode as `i32` first (sqlx is strict about width) and
/// widen to `i64`.
pub mod row {
    use sqlx::Row;

    pub type RowResult<T> = Result<T, sqlx::Error>;

    fn missing_is_none<T>(r: RowResult<Option<T>>) -> RowResult<Option<T>> {
        match r {
            Err(sqlx::Error::ColumnNotFound(_)) => Ok(None),
            other => other,
        }
    }

    pub fn opt_i64(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<i64>> {
        if let Ok(v) = row.try_get::<Option<i32>, _>(col) {
            return Ok(v.map(i64::from));
        }
        missing_is_none(row.try_get::<Option<i64>, _>(col))
    }

    pub fn opt_f64(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<f64>> {
        if let Ok(v) = row.try_get::<Option<f32>, _>(col) {
            return Ok(v.map(f64::from));
        }
        missing_is_none(row.try_get::<Option<f64>, _>(col))
    }

    pub fn opt_bool(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<bool>> {
        missing_is_none(row.try_get::<Option<bool>, _>(col))
    }

    pub fn opt_text(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<String>> {
        missing_is_none(row.try_get::<Option<String>, _>(col))
    }

    pub fn opt_text_array(
        row: &sqlx::postgres::PgRow,
        col: &str,
    ) -> RowResult<Option<Vec<String>>> {
        missing_is_none(row.try_get::<Option<Vec<String>>, _>(col))
    }

    pub fn opt_i64_array(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<Vec<i64>>> {
        if let Ok(v) = row.try_get::<Option<Vec<i32>>, _>(col) {
            return Ok(v.map(|a| a.into_iter().map(i64::from).collect()));
        }
        missing_is_none(row.try_get::<Option<Vec<i64>>, _>(col))
    }

    pub fn opt_vector(row: &sqlx::postgres::PgRow, col: &str) -> RowResult<Option<Vec<f32>>> {
        missing_is_none(row.try_get::<Option<pgvector::Vector>, _>(col))
            .map(|v| v.map(|vec| vec.to_vec()))
    }
}
