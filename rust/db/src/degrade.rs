//! Degrade classification — which Postgres errors mean "optional
//! channel missing", never "request failed".
//!
//! The Python side catches asyncpg's `UndefinedColumnError` /
//! `UndefinedObjectError` (and, for channels reading optional tables,
//! `UndefinedTableError`) and switches the channel off in place. The
//! SQLSTATE equivalents:
//!
//! * `42703` undefined_column (embedding column absent)
//! * `42704` undefined_object (vector type/extension absent)
//! * `42883` undefined_function (`<=>` without the pgvector extension)
//! * `42P01` undefined_table (optional enrichment table absent)

/// The SQLSTATE of a database error when it is one of the four
/// "optional dependency missing" codes, else `None`.
pub fn pg_code(err: &sqlx::Error) -> Option<&'static str> {
    let code = match err {
        sqlx::Error::Database(db) => db.code(),
        _ => None,
    }?;
    match code.as_ref() {
        "42703" => Some("42703"),
        "42704" => Some("42704"),
        "42883" => Some("42883"),
        "42P01" => Some("42P01"),
        _ => None,
    }
}

/// Does this code mean an optional dependency is missing?
pub fn is_missing_dependency(code: &str) -> bool {
    matches!(code, "42703" | "42704" | "42883" | "42P01")
}

/// Human-readable degrade reason for logging.
pub fn reason(code: &str, err: &sqlx::Error) -> String {
    let what = match code {
        "42703" => "column missing",
        "42704" => "object/type missing (extension not installed?)",
        "42883" => "function missing (pgvector operator unavailable?)",
        "42P01" => "table missing",
        _ => "unknown dependency missing",
    };
    format!("{what}: {err}")
}
