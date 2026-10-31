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
    /// Connect with default table mapping.
    pub async fn connect(dsn: &str) -> Result<Self, DbError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(dsn)
            .await?;
        Ok(Self {
            pool,
            map: Arc::new(SchemaMap::new()),
        })
    }

    /// Connect with an explicit map (tests, remapped deployments).
    pub async fn connect_with_map(dsn: &str, map: SchemaMap) -> Result<Self, DbError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(dsn)
            .await?;
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
/// Python rows use (missing/NULL → None).
pub mod row {
    use sqlx::Row;

    pub fn opt_i64(row: &sqlx::postgres::PgRow, col: &str) -> Option<i64> {
        row.try_get(col).ok()
    }

    pub fn opt_f64(row: &sqlx::postgres::PgRow, col: &str) -> Option<f64> {
        row.try_get(col).ok()
    }

    pub fn opt_text(row: &sqlx::postgres::PgRow, col: &str) -> Option<String> {
        row.try_get(col).ok()
    }

    pub fn opt_text_array(row: &sqlx::postgres::PgRow, col: &str) -> Option<Vec<String>> {
        row.try_get(col).ok()
    }

    pub fn opt_i64_array(row: &sqlx::postgres::PgRow, col: &str) -> Option<Vec<i64>> {
        row.try_get(col).ok()
    }

    pub fn opt_vector(row: &sqlx::postgres::PgRow, col: &str) -> Option<Vec<f32>> {
        row.try_get::<pgvector::Vector, _>(col)
            .ok()
            .map(|v| v.to_vec())
    }
}
