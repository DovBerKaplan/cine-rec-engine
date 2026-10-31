//! PyO3 bridge: exposes the ported scoring core as the optional
//! extension module `cine_rec_engine._scoring_rs`.
//!
//! Design:
//! * [`PyProfile`] wraps the pure-Rust profile in an `Arc` — methods
//!   clone the `Arc`s and release the GIL (`allow_threads`) for the
//!   actual math, so the warm-path thread pool parallelizes for real.
//! * The pure crate (and its parity harness) has NO pyo3 dependency —
//!   this module only compiles under the `pyo3` feature, and it is a
//!   thin conversion layer over `crate::real`.
//! * Entity dicts convert leniently (missing/None → field defaults),
//!   mirroring Python `dict.get` — except an EMPTY `entity_data` dict,
//!   which the Python source treats inconsistently (falsy for
//!   narrative fields, not-None for the enriched source) and is
//!   therefore rejected loudly here instead of silently diverging.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString};

use crate::real;
use crate::types::{CastKey, Entity, Profile};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ── conversion helpers ─────────────────────────────────────────────

fn dict_get<'py>(obj: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err(format!("expected a dict, got {obj:?}")))?;
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v)),
        None => Ok(None),
    }
}

fn opt_string(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<String>> {
    match dict_get(obj, key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(
            v.downcast::<PyString>()
                .map_err(|_| PyTypeError::new_err(format!("{key} must be a string")))?
                .to_str()?
                .to_string(),
        )),
    }
}

fn opt_i64(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<i64>> {
    match dict_get(obj, key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => v
            .extract::<i64>()
            .map(Some)
            .map_err(|_| PyTypeError::new_err(format!("{key} must be an int"))),
    }
}

fn opt_f64(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<f64>> {
    match dict_get(obj, key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        // Python rows carry int or float ratings — float(x) accepts both
        Some(v) => v
            .extract::<f64>()
            .map(Some)
            .map_err(|_| PyTypeError::new_err(format!("{key} must be a number"))),
    }
}

fn str_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<String>> {
    match dict_get(obj, key)? {
        None => Ok(Vec::new()),
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => v
            .extract::<Vec<String>>()
            .map_err(|_| PyTypeError::new_err(format!("{key} must be a list of strings"))),
    }
}

fn i64_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<i64>> {
    match dict_get(obj, key)? {
        None => Ok(Vec::new()),
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => v
            .extract::<Vec<i64>>()
            .map_err(|_| PyTypeError::new_err(format!("{key} must be a list of ints"))),
    }
}

/// cast_list members are actor NAMES in the recall rows and person ids
/// in some older shapes — accept both, per item.
fn cast_key_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<CastKey>> {
    match dict_get(obj, key)? {
        None => Ok(Vec::new()),
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => {
            let items = v
                .extract::<Vec<Bound<'_, PyAny>>>()
                .map_err(|_| PyTypeError::new_err(format!("{key} must be a list")))?;
            items
                .iter()
                .map(|item| {
                    if let Ok(id) = item.extract::<i64>() {
                        Ok(CastKey::Id(id))
                    } else if let Ok(name) = item.extract::<String>() {
                        Ok(CastKey::Name(name))
                    } else {
                        Err(PyTypeError::new_err(format!(
                            "{key} items must be ints or strings"
                        )))
                    }
                })
                .collect()
        }
    }
}

/// Python dict → [`Entity`]. Missing/None keys become field defaults
/// (the same values `_entity_profile`'s `dict.get(...)` fallbacks see).
pub fn py_to_entity(obj: &Bound<'_, PyAny>) -> PyResult<Entity> {
    Ok(Entity {
        id: opt_i64(obj, "id")?.unwrap_or(0),
        media_type: opt_string(obj, "media_type")?.unwrap_or_default(),
        title: opt_string(obj, "title")?,
        title_en: opt_string(obj, "title_en")?,
        genres: str_list(obj, "genres")?,
        keywords: str_list(obj, "keywords")?,
        cast_ids: i64_list(obj, "cast_ids")?,
        cast_list: cast_key_list(obj, "cast_list")?,
        director_ids: i64_list(obj, "director_ids")?,
        writer_ids: i64_list(obj, "writer_ids")?,
        composer_ids: i64_list(obj, "composer_ids")?,
        dp_ids: i64_list(obj, "dp_ids")?,
        companies: str_list(obj, "companies")?,
        networks: str_list(obj, "networks")?,
        original_language: opt_string(obj, "original_language")?,
        overview: opt_string(obj, "overview")?,
        overview_en: opt_string(obj, "overview_en")?,
        release_year: opt_i64(obj, "release_year")?,
        vote_average: opt_f64(obj, "vote_average")?,
        vote_count: opt_i64(obj, "vote_count")?,
        collection_id: opt_i64(obj, "collection_id")?,
        adult: dict_get(obj, "adult")?
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        via: opt_string(obj, "via")?,
        tmdb_rank: opt_i64(obj, "tmdb_rank")?,
        narrative: opt_string(obj, "narrative")?,
        pacing: opt_string(obj, "pacing")?,
        emotional_arc: match dict_get(obj, "emotional_arc")? {
            None => None,
            Some(v) if v.is_none() => None,
            Some(v) => Some(
                v.extract::<Vec<String>>()
                    .map_err(|_| PyTypeError::new_err("emotional_arc must be a list of strings"))?,
            ),
        },
        director: opt_string(obj, "director")?,
    })
}

// ── opaque profile handle ──────────────────────────────────────────

/// Derive-once profile, built by [`entity_profile`] and consumed by
/// [`PairScorer`] — opaque from Python on purpose (the Python code
/// only ever passes it back to this module).
#[pyclass]
pub struct PyProfile {
    inner: Arc<Profile>,
}

// ── module functions ───────────────────────────────────────────────

/// `_entity_profile(entity, entity_data=None, genres_override=None)` —
/// identical semantics to the Python function; returns an opaque
/// profile handle.
#[pyfunction]
#[pyo3(signature = (entity, entity_data = None, genres_override = None))]
fn entity_profile(
    entity: &Bound<'_, PyAny>,
    entity_data: Option<&Bound<'_, PyAny>>,
    genres_override: Option<Vec<String>>,
) -> PyResult<PyProfile> {
    let ent = py_to_entity(entity)?;
    let data: Option<Entity> = match entity_data {
        None => None,
        Some(d) if d.is_none() => None,
        Some(d) => {
            let as_dict = d
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("entity_data must be a dict or None"))?;
            // an EMPTY dict is the one input Python treats
            // inconsistently — reject loudly rather than diverge
            if as_dict.len() == 0 {
                return Err(PyTypeError::new_err(
                    "entity_data must be a non-empty dict or None",
                ));
            }
            Some(py_to_entity(d)?)
        }
    };
    let profile = real::entity_profile(&ent, data.as_ref(), genres_override.as_deref());
    Ok(PyProfile {
        inner: Arc::new(profile),
    })
}

// ── the pair scorer ────────────────────────────────────────────────

/// Holds the two weight lists (the per-request `w_list` and the
/// module-global `HEURISTIC_WEIGHTS_LIST` the residual term uses) so
/// the per-pair call converts no weight data.
#[pyclass]
pub struct PairScorer {
    w: Vec<f64>,
    heuristic_w27: Vec<f64>,
}

#[pymethods]
impl PairScorer {
    #[new]
    #[pyo3(signature = (w_list, heuristic_w27))]
    fn new(w_list: Vec<f64>, heuristic_w27: Vec<f64>) -> PyResult<Self> {
        if w_list.len() != crate::VEC_LEN {
            return Err(PyTypeError::new_err(format!(
                "w_list must have {} entries, got {}",
                crate::VEC_LEN,
                w_list.len()
            )));
        }
        if heuristic_w27.len() != crate::VEC_LEN - 1 {
            return Err(PyTypeError::new_err(format!(
                "heuristic_w27 must have {} entries, got {}",
                crate::VEC_LEN - 1,
                heuristic_w27.len()
            )));
        }
        Ok(Self {
            w: w_list,
            heuristic_w27,
        })
    }

    /// `_score_pair_fast(sp, cp, cand, cand_data, cand_genres_set,
    /// overview_sim, w_list)` — the w_list is bound at construction;
    /// `cand`/`cand_data` dicts are read only for the director-name
    /// fallback, exactly the fields the Python body touches.
    #[pyo3(signature = (sp, cp, cand_data, cand_genres_set, overview_sim = 0.0))]
    fn score_pair_fast(
        &self,
        py: Python<'_>,
        sp: &PyProfile,
        cp: &PyProfile,
        cand_data: &Bound<'_, PyAny>,
        cand_genres_set: Option<&Bound<'_, PySet>>,
        overview_sim: f64,
    ) -> PyResult<f64> {
        let director = opt_string(cand_data, "director")?;
        let cgs: Option<HashSet<String>> = match cand_genres_set {
            // no identity cache here: the hot loop is multi-threaded
            // (warm path) and a shared cache both races and mis-hits on
            // recycled addresses — measured gain was ~5%, not worth it
            Some(s) => Some(
                s.iter()
                    .map(|item| {
                        item.extract::<String>().map_err(|_| {
                            PyTypeError::new_err("cand_genres_set items must be strings")
                        })
                    })
                    .collect::<PyResult<HashSet<String>>>()?,
            ),
            None => None,
        };
        let sp = sp.inner.clone();
        let cp = cp.inner.clone();
        let w = self.w.clone();
        let hw = self.heuristic_w27.clone();
        Ok(py.allow_threads(move || {
            real::score_pair_fast(
                &sp,
                &cp,
                director.as_deref(),
                cgs.as_ref(),
                overview_sim,
                &w,
                &hw,
            )
        }))
    }

    /// `feature_vector(cand, cand_data, cand_genres, seed,
    /// overview_sim=0.0)` — the raw 28-entry vector.
    #[pyo3(signature = (cand, cand_data, cand_genres, seed, overview_sim = 0.0))]
    fn feature_vector(
        &self,
        py: Python<'_>,
        cand: &Bound<'_, PyAny>,
        cand_data: &Bound<'_, PyAny>,
        cand_genres: Vec<String>,
        seed: &Bound<'_, PyAny>,
        overview_sim: f64,
    ) -> PyResult<Vec<f64>> {
        let cand_e = py_to_entity(cand)?;
        let data_e = py_to_entity(cand_data)?;
        let seed_e = py_to_entity(seed)?;
        let hw = self.heuristic_w27.clone();
        Ok(py.allow_threads(move || {
            let sp = real::entity_profile(&seed_e, None, None);
            let cp = real::entity_profile(&cand_e, Some(&data_e), Some(&cand_genres));
            real::vec_from_profiles(&sp, &cp, &cand_e, &data_e, overview_sim, &hw)
        }))
    }
}

// ── the extension module ───────────────────────────────────────────

#[pymodule]
fn _scoring_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add("VEC_LEN", crate::VEC_LEN)?;
    m.add("MAX_SCORE_ULP", crate::MAX_SCORE_ULP)?;
    m.add_function(wrap_pyfunction!(entity_profile, m)?)?;
    m.add_class::<PairScorer>()?;
    let _ = py; // interpreter handle reserved for future additions
    Ok(())
}
