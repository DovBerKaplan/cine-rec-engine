//! cine-rec-db — the Rust port of the engine's DB layer (rust-port
//! phase 3).
//!
//! Three pieces:
//!
//! * [`schema_map`] — the logical table-name registry (`tables.py`):
//!   defaults, JSON/env remapping, loud validation, `{t_*}` SQL
//!   placeholder resolution.
//! * [`executor`] — a pool wrapper where EVERY query text passes the
//!   resolver before execution (the `db.py` seam), with typed binds
//!   and row helpers.
//!
//! * [`degrade`] — classifying Postgres errors into "optional channel
//!   missing" signals. DEGRADE, NEVER FAIL: an absent embedding
//!   column / extension / optional table turns its channel off
//!   ([`DbError::ChannelOff`]); it never fails the request.
//!
//! Queries are DYNAMIC (sqlx without compile-time verification) by
//! design: physical table names are runtime data through the registry.
//!
//! Integration tests run against a live Postgres when
//! `CINE_REC_INTEGRATION_DSN` is set (same harness as the Python
//! integration suite); otherwise they skip — the offline story stays
//! the default dev loop.

pub mod degrade;
pub mod executor;
pub mod info;
pub mod personalization;
pub mod recall;
pub mod recall2;
pub mod schema_map;
pub mod user_vector;
pub mod watched;
pub mod weights;

pub use executor::{Bind, Db, DbError};
pub use recall::CandidateRow;
pub use schema_map::{SchemaMap, SchemaMapError, LOGICAL_TABLES};
