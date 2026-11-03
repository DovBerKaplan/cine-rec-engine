//! PyO3 bridge: exposes the service compute path as the optional
//! extension module `cine_rec_engine._service_rs`.
//!
//! Boundary law: ONE crossing per request-stage with COARSE
//! batches — the whole compute spec enters as a single JSON payload,
//! the whole result leaves as a single JSON payload; no per-item
//! calls. A hosted tokio runtime drives the async work; the GIL is
//! released for its ENTIRE duration (`allow_threads` around the
//! `block_on`), so the embedding event loop can run other tasks.
//!
//! The caller (Python) wraps this sync entry in `asyncio.to_thread` —
//! the adapter `_service_accel.py` owns that. DSN-keyed Db handles are
//! cached across calls (the Python asyncpg pool has a Rust twin; one
//! crossing per stage keeps the boundary cheap).



use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::recommend::{recommend_for_user, RecommendSpec};
use crate::similar::{compute_find_similar, ComputeSpec};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("hosted tokio runtime")
    })
}

fn dbs_map() -> &'static Mutex<HashMap<String, Arc<cine_rec_db::Db>>> {
    static DBS: OnceLock<Mutex<HashMap<String, Arc<cine_rec_db::Db>>>> = OnceLock::new();
    DBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn db_for(dsn: &str) -> Result<Arc<cine_rec_db::Db>, String> {
    let dbs = dbs_map();
    let mut guard = dbs.lock().expect("db cache");
    if let Some(db) = guard.get(dsn) {
        return Ok(db.clone());
    }
    let db = runtime()
        .block_on(cine_rec_db::Db::connect(dsn))
        .map_err(|e| format!("connect {dsn}: {e}"))?;
    let db = Arc::new(db);
    guard.insert(dsn.to_string(), db.clone());
    Ok(db)
}

/// The one boundary crossing: `compute(dsn, spec_json) -> result_json`.
/// Spec/result shapes are `similar::ComputeSpec` / `ComputeResult`
/// (serde; float_roundtrip on the serde_json build — JSON floats
/// crossing the boundary need the same fidelity as the golden files).
#[pyfunction]
fn compute(py: Python, dsn: String, spec_json: String) -> PyResult<String> {
    let spec: ComputeSpec = serde_json::from_str(&spec_json)
        .map_err(|e| PyRuntimeError::new_err(format!("bad spec json: {e}")))?;
    let db = db_for(&dsn).map_err(PyRuntimeError::new_err)?;
    // GIL released for the whole stage — DB I/O and scoring run free
    let result = py.allow_threads(|| {
        runtime().block_on(async move { compute_find_similar(&db, &spec).await })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("compute failed: {e}")))?;
    serde_json::to_string(&result)
        .map_err(|e| PyRuntimeError::new_err(format!("result serialize: {e}")))
}

/// recommend_for_user, whole request in ONE crossing (spec → result).
#[pyfunction]
fn recommend(py: Python, dsn: String, spec_json: String) -> PyResult<String> {
    let spec: RecommendSpec = serde_json::from_str(&spec_json)
        .map_err(|e| PyRuntimeError::new_err(format!("bad recommend spec json: {e}")))?;
    let db = db_for(&dsn).map_err(PyRuntimeError::new_err)?;
    let result = py.allow_threads(|| {
        runtime().block_on(async move { recommend_for_user(&db, &spec).await })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("recommend failed: {e}")))?;
    serde_json::to_string(&result)
        .map_err(|e| PyRuntimeError::new_err(format!("result serialize: {e}")))
}

#[pyfunction]
fn cached_dsn_count(py: Python) -> PyResult<usize> {
    let n = py.allow_threads(|| dbs_map().lock().map(|g| g.len()).unwrap_or(0));
    Ok(n)
}

#[pymodule]
fn _service_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute, m)?)?;
    m.add_function(wrap_pyfunction!(recommend, m)?)?;
    m.add_function(wrap_pyfunction!(cached_dsn_count, m)?)?;
    m.add("VERSION", VERSION)?;
    Ok(())
}
