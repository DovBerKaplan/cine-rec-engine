//! PyO3 bridge: exposes the ported scoring core as the optional
//! extension module `cine_rec_engine._scoring_rs`.
//!
//! Design:
//! * [`PyProfile`] wraps the pure-Rust profile in an `Arc` — methods
//!   clone the `Arc`s and release the GIL (`allow_threads`) for the
//!   actual math, so EMBEDDERS running multi-threaded parallelize for
//!   real (the package itself is single-threaded today).
//! * The pure crate (and its parity harness) has NO pyo3 dependency —
//!   this module only compiles under the `pyo3` feature, and it is a
//!   thin conversion layer over `crate::real`.
//! * Entity dicts convert with missing/None → field defaults,
//!   mirroring Python `dict.get`. An EMPTY `entity_data` dict — a
//!   regular production input (`enriched.get(cid, {})`) — carries
//!   Python's exact distinction: enriched-source fields read the empty
//!   dict, narrative-side fields fall back to the entity (see
//!   `real::entity_profile_ex`). Items that cannot convert RAISE
//!   (TypeError); the per-call fallback in _scoring_accel recomputes
//!   that call in pure Python — silent skipping is not an option,
//!   Python's set() compares values and sizes enter the similarity
//!   denominators.

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
        // ints are the norm. An INTEGRAL float (15000.0) converts
        // exactly; a NON-integral float raises instead of truncating —
        // Python keeps the raw value (`entity.get(...) or 0`) and
        // compares it directly, so silent truncation would diverge
        // (e.g. vote_count=15000.5 vs the >15000 threshold). The
        // per-call fallback in _scoring_accel covers such rows.
        Some(v) => {
            if let Ok(i) = v.extract::<i64>() {
                Ok(Some(i))
            } else if let Ok(f) = v.extract::<f64>() {
                if f.fract() == 0.0 && f.is_finite() {
                    Ok(Some(f as i64))
                } else {
                    Err(PyTypeError::new_err(format!(
                        "{key} is a non-integral float; refusing to truncate"
                    )))
                }
            } else {
                Err(PyTypeError::new_err(format!("{key} must be a number")))
            }
        }
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

/// STRICT per-item conversion: any item that cannot convert raises
/// TypeError, which the per-call fallback in _scoring_accel turns into
/// a pure-Python computation for that call. Skipping items is NOT an
/// option — Python's set() compares VALUES (None == None, equal
/// strings match), the set SIZE enters the Jaccard/Dice denominators,
/// and cast/crew lists are sliced [:5] BEFORE any filtering could run,
/// so silent skipping produced wrong-but-plausible scores (found by
/// the phase-2 review, repro cases pinned in the bridge tests).
fn str_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<String>> {
    items_of(obj, key, "strings", |item| {
        item.extract::<String>()
            .map_err(|_| PyTypeError::new_err(format!("{key} items must be strings")))
    })
}

fn i64_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<i64>> {
    // ints convert; an INTEGRAL float (2.0, 15000.0) equals its int in
    // Python's set semantics and converts too; anything else —
    // non-integral floats included — raises
    items_of(obj, key, "ints", |item| {
        if let Ok(i) = item.extract::<i64>() {
            return Ok(i);
        }
        if let Ok(f) = item.extract::<f64>() {
            if f.is_finite() && f.fract() == 0.0 {
                return Ok(f as i64);
            }
        }
        Err(PyTypeError::new_err(format!(
            "{key} items must be ints (or integral floats)"
        )))
    })
}

/// cast_list members are actor NAMES in the recall rows and person ids
/// in some older shapes — both accepted, anything else raises.
fn cast_key_list(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<CastKey>> {
    items_of(obj, key, "ints or strings", |item| {
        if let Ok(id) = item.extract::<i64>() {
            return Ok(CastKey::Id(id));
        }
        if let Ok(name) = item.extract::<String>() {
            return Ok(CastKey::Name(name));
        }
        Err(PyTypeError::new_err(format!(
            "{key} items must be ints or strings"
        )))
    })
}

/// Iterate a list-ish field strictly: every item must convert.
fn items_of<T>(
    obj: &Bound<'_, PyAny>,
    key: &str,
    what: &str,
    convert: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    match dict_get(obj, key)? {
        None => Ok(Vec::new()),
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => v
            .try_iter()
            .map_err(|_| PyTypeError::new_err(format!("{key} must be a list of {what}")))?
            .map(|item| convert(&item?))
            .collect(),
    }
}

/// A non-string TRUTHY `via` (e.g. True, 1): gate-exempt like any
/// truthy value in Python, but never equal to a channel name — a
/// sentinel string nobody can spell in a real row models exactly that.
const NON_STRING_VIA: &str = "\u{0}non-string-via\u{0}";

fn opt_via(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<String>> {
    match dict_get(obj, key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => {
            if let Ok(s) = v.extract::<String>() {
                Ok(Some(s))
            } else if v.is_truthy()? {
                Ok(Some(NON_STRING_VIA.to_string()))
            } else {
                Ok(None)
            }
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
        // bool(entity.get("adult")) — Python truthiness of ANY value
        adult: match dict_get(obj, "adult")? {
            None => false,
            Some(v) if v.is_none() => false,
            Some(v) => v.is_truthy()?,
        },
        via: opt_via(obj, "via")?,
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
///
/// The ORIGINAL inputs are kept alongside (the `origin` getter) so the
/// pure-Python path can rebuild this side exactly when a call ends up
/// mixed (one profile converted, the other not) or when a Rust call
/// degrades — a mixed pair must never crash a request. The origins
/// are REFERENCES to the caller's dicts, not copies: the service never
/// mutates entity rows between profile construction and scoring
/// (documented in _scoring_accel). Python objects are touched ONLY
/// under the GIL; the hot scoring closure sees just the `Arc<Profile>`.
#[pyclass]
pub struct PyProfile {
    inner: Arc<Profile>,
    origin_entity: PyObject,
    origin_data: Option<PyObject>,
    origin_genres: Option<PyObject>,
}

#[pymethods]
impl PyProfile {
    /// (entity, entity_data, genres_override) as originally passed —
    /// for the pure-Python rebuild path in _scoring_accel.
    #[getter]
    fn origin(
        &self,
        py: Python<'_>,
    ) -> (PyObject, Option<PyObject>, Option<PyObject>) {
        (
            self.origin_entity.clone_ref(py),
            self.origin_data.as_ref().map(|o| o.clone_ref(py)),
            self.origin_genres.as_ref().map(|o| o.clone_ref(py)),
        )
    }

    /// GC support: the held origin references could theoretically join
    /// a reference cycle (an entity dict that somewhere stores this
    /// profile); without traversal such a cycle would never collect.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.origin_entity)?;
        if let Some(d) = &self.origin_data {
            visit.call(d)?;
        }
        if let Some(g) = &self.origin_genres {
            visit.call(g)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        // release the origin references; the Rust profile stays usable
        self.origin_data = None;
        self.origin_genres = None;
        // origin_entity is not Option; clearing it wholesale would need
        // a None sentinel — the cycle is already broken by the two above
        // plus Python's own tp_clear on the holder.
    }
}

// ── module functions ───────────────────────────────────────────────

/// `_entity_profile(entity, entity_data=None, genres_override=None)` —
/// identical semantics to the Python function; returns an opaque
/// profile handle.
#[pyfunction]
#[pyo3(signature = (entity, entity_data = None, genres_override = None))]
fn entity_profile(
    py: Python<'_>,
    entity: &Bound<'_, PyAny>,
    entity_data: Option<&Bound<'_, PyAny>>,
    genres_override: Option<Vec<String>>,
) -> PyResult<PyProfile> {
    let ent = py_to_entity(entity)?;
    let mut data_truthy = false;
    let data: Option<Entity> = match entity_data {
        None => None,
        Some(d) if d.is_none() => None,
        Some(d) => {
            let as_dict = d
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("entity_data must be a dict or None"))?;
            // EMPTY dict is a regular production input (service.py:
            // `enriched.get(cid, {})` for every non-enriched candidate).
            // Python semantics: enriched-source fields read the empty
            // dict; narrative/pacing/emotional-arc fields fall back to
            // the ENTITY (an empty dict is falsy). entity_profile_ex
            // carries exactly that distinction.
            data_truthy = as_dict.len() > 0;
            Some(py_to_entity(d)?)
        }
    };
    let profile =
        real::entity_profile_ex(&ent, data.as_ref(), data_truthy, genres_override.as_deref());
    Ok(PyProfile {
        inner: Arc::new(profile),
        origin_entity: entity.clone().unbind(),
        origin_data: match entity_data {
            Some(d) if !d.is_none() => Some(d.clone().unbind()),
            _ => None,
        },
        origin_genres: match genres_override {
            Some(g) => Some(g.into_pyobject(py)?.unbind()),
            None => None,
        },
    })
}

// ── the pair scorer ────────────────────────────────────────────────

/// Holds the two weight lists (the per-request `w_list` and the
/// module-global `HEURISTIC_WEIGHTS_LIST` the residual term uses) so
/// the per-pair call converts no weight data.
#[pyclass]
pub struct PairScorer {
    w: std::sync::Arc<[f64]>,
    heuristic_w27: std::sync::Arc<[f64]>,
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
            w: w_list.into(),
            heuristic_w27: heuristic_w27.into(),
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
        // a non-string director can never equal the (string) seed
        // director in Python — mapping it to None is identical
        let director = match dict_get(cand_data, "director")? {
            Some(v) if !v.is_none() => v.extract::<String>().ok(),
            _ => None,
        };
        // STRICT: a non-string genre item raises (Python's set compares
        // values and the gate reads set semantics — skipping would
        // silently change the genre gate). No identity cache: callers
        // may run multi-threaded, and a shared cache both races and
        // mis-hits on recycled addresses (~5% measured, not worth it).
        let cgs: Option<HashSet<String>> = match cand_genres_set {
            Some(s) => Some(
                s.iter()
                    .map(|item| {
                        item.extract::<String>().map_err(|_| {
                            PyTypeError::new_err(
                                "cand_genres_set items must be strings",
                            )
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
        // the canonical call shape passes cand as its own data —
        // convert once when they are the same object
        let cand_e = py_to_entity(cand)?;
        let data_e = if cand.is(cand_data) {
            cand_e.clone()
        } else {
            py_to_entity(cand_data)?
        };
        // an EMPTY data dict is falsy for the narrative-side fields;
        // cand-as-its-own-data is always a real (truthy) row
        let data_truthy = if cand.is(cand_data) {
            true
        } else {
            cand_data
                .downcast::<PyDict>()
                .map(|d| d.len() > 0)
                .unwrap_or(true)
        };
        let seed_e = py_to_entity(seed)?;
        let hw = self.heuristic_w27.clone();
        Ok(py.allow_threads(move || {
            let sp = real::entity_profile(&seed_e, None, None);
            let cp =
                real::entity_profile_ex(&cand_e, Some(&data_e), data_truthy, Some(&cand_genres));
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
