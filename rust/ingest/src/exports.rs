//! TMDB daily ID exports — the pure core of `ingest/exports.py`.
//!
//! TMDB publishes the full id lists daily (~07:00 UTC) as gzipped
//! JSON, one object per line:
//! `{"adult": bool, "id": int, "original_title"/"original_name": str,
//! "popularity": float, ...}`. Only `id`/`adult`/`popularity` are
//! consumed; every conversion follows PYTHON semantics (see
//! [`crate::pyval`]) because the field expressions are
//! `int(obj["id"])`, `bool(obj.get("adult", False))` and
//! `float(obj.get("popularity") or 0.0)` — the `or` makes truthiness
//! part of the contract (popularity `""`/`0`/`null` all read as 0.0).
//!
//! Two parse paths, faithfully different (ingest/exports.py:49-109):
//!
//! * [`parse_export`] — strict: strict utf-8, and ANY failure raises
//!   (`KeyError` for a missing id, `ValueError` for a bad literal or
//!   bad utf-8 or bad JSON, `TypeError` for a non-object line).
//! * [`parse_line`] — lenient (`_parse_line`): utf-8 with
//!   replacement chars, and `KeyError`/`ValueError` are swallowed
//!   into `Ok(None)` (skip the line) — but `TypeError` propagates,
//!   exactly like the Python `except (KeyError, ValueError)`.
//!
//! Parity is pinned by fixtures generated from the REAL Python
//! module (`tests/fixtures/gen_exports_fixtures.py`): entry values
//! (popularity bits included) must be byte-compatible, and error
//! outcomes must match class-for-class.
//!
//! Wave-D TODO (needs db traits / a runtime edge, NOT ported here):
//! `iter_export_ids` — streaming `zlib.decompressobj` over HTTP
//! chunks feeding [`LineBuffer`], plus `latest_export_url` reading
//! the real clock. The pure pieces it is made of — the line splitter
//! and the starter filters — ARE here and pinned.

use std::fmt;
use std::io::{BufRead, BufReader};

use serde_json::Value;

use crate::date::CivilDate;
use crate::pyval::{is_truthy, py_float, py_int, py_repr, py_strip, py_type_name, PyConvError};

pub const EXPORT_BASE: &str = "http://files.tmdb.org/p/exports";

/// One consumed line of a daily export file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExportEntry {
    pub id: i64,
    pub adult: bool,
    pub popularity: f64,
}

/// Errors of the strict/lenient parse paths, carrying the Python
/// exception family so callers can reproduce swallow decisions.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    /// A conversion failure — the payload says which Python family:
    /// `Key`/`Value`/`Type` (see [`PyConvError`]).
    Conv(PyConvError),
    /// gzip/IO failure (Python: `OSError`/`BadGzipFile`/`EOFError`
    /// family from the gzip module; message texts are
    /// implementation-specific, the failure itself is not).
    Io(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::Conv(e) => write!(f, "{e}"),
            ExportError::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ExportError {}

/// `ingest.exports.EXPORT_NAMES` — the two valid media keys.
pub fn export_name(medium: &str) -> Option<&'static str> {
    match medium {
        "movie" => Some("movie_ids"),
        "tv" => Some("tv_series_ids"),
        _ => None,
    }
}

/// Errors of the URL builders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportUrlError {
    /// `ValueError` parity — `medium must be 'movie' or 'tv', got …`
    /// (message exact).
    InvalidMedium(String),
    /// `ValueError` parity from `date` construction (message exact).
    InvalidDate(String),
    /// `OverflowError` parity — `0001-01-01` minus a day.
    DateOutOfRange,
}

impl fmt::Display for ExportUrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportUrlError::InvalidMedium(m) | ExportUrlError::InvalidDate(m) => f.write_str(m),
            ExportUrlError::DateOutOfRange => f.write_str("date value out of range"),
        }
    }
}

impl std::error::Error for ExportUrlError {}

/// `export_url_for(day, medium)` — byte-compatible URL construction
/// (`f"{EXPORT_BASE}/{EXPORT_NAMES[medium]}_{day:%m_%d_%Y}.json.gz"`).
pub fn export_url_for(day: CivilDate, medium: &str) -> Result<String, ExportUrlError> {
    let name = export_name(medium).ok_or_else(|| {
        ExportUrlError::InvalidMedium(format!(
            "medium must be 'movie' or 'tv', got {}",
            py_repr(medium)
        ))
    })?;
    Ok(format!(
        "{EXPORT_BASE}/{name}_{}.json.gz",
        day.url_fragment()
    ))
}

/// `latest_export_url(medium, today=…)` — yesterday's file (the
/// export job runs ~07:00 UTC; before that, today's file is absent).
pub fn latest_export_url_at(medium: &str, today: CivilDate) -> Result<String, ExportUrlError> {
    let day = today.previous_day().ok_or(ExportUrlError::DateOutOfRange)?;
    export_url_for(day, medium)
}

/// The TypeError `obj["id"]` raises for a non-object line, with
/// Python's per-type message shapes (pinned exactly by fixtures):
/// scalars are "not subscriptable", but str/list have their own
/// subscript messages.
pub(crate) fn subscript_error(obj: &Value) -> PyConvError {
    match obj {
        Value::String(_) => PyConvError::Type("string indices must be integers, not 'str'".into()),
        Value::Array(_) => {
            // CPython 3.12 dropped the quotes on THIS one while
            // keeping them on the string case — pinned by fixtures.
            PyConvError::Type("list indices must be integers or slices, not str".into())
        }
        other => PyConvError::Type(format!(
            "'{}' object is not subscriptable",
            py_type_name(other)
        )),
    }
}

/// Build an entry from a parsed line using the exact Python field
/// expressions.
pub(crate) fn entry_from_value(obj: &Value) -> Result<ExportEntry, PyConvError> {
    // A non-object line is a TypeError in Python (obj["id"] on an
    // int/float/str/None), and TypeError is never swallowed.
    let map = obj.as_object().ok_or_else(|| subscript_error(obj))?;
    // int(obj["id"]) — missing key is a KeyError.
    let id = py_int(
        map.get("id")
            .ok_or_else(|| PyConvError::Key(py_repr("id")))?,
    )?;
    // bool(obj.get("adult", False)) — any truthy JSON value counts.
    let adult = match map.get("adult") {
        Some(v) => is_truthy(v),
        None => false,
    };
    // float(obj.get("popularity") or 0.0) — falsy reads as 0.0.
    let popularity = match map.get("popularity") {
        Some(v) if is_truthy(v) => py_float(v)?,
        _ => 0.0,
    };
    Ok(ExportEntry {
        id,
        adult,
        popularity,
    })
}

/// `parse_export(gz_bytes)` — strict whole-file parse. Reads ALL
/// concatenated gzip members (Python's `gzip.GzipFile` does; a
/// single-member decoder would diverge on multi-member files).
/// Entries stream out lazily in Python; here the whole vector is
/// returned — the hundreds-of-MB streaming concern belongs to the
/// wave-D `iter_export_ids` port, not to this whole-buffer helper.
pub fn parse_export(gz_bytes: &[u8]) -> Result<Vec<ExportEntry>, ExportError> {
    // Python's GzipFile over EMPTY input iterates to EOF without an
    // error (0 entries); flate2's decoder reports a truncated header.
    // Pinned by the "empty" fixture case.
    if gz_bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    let mut reader = BufReader::new(flate2::read::MultiGzDecoder::new(gz_bytes));
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        // `for raw in f` — lines split on b"\n", delimiter included;
        // a trailing unterminated line IS yielded (unlike the
        // streaming path — see LineBuffer).
        let n = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| ExportError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        let text = std::str::from_utf8(&line).map_err(|e| {
            // UnicodeDecodeError is a ValueError subclass.
            ExportError::Conv(PyConvError::Value(format!(
                "'utf-8' codec can't decode: {e}"
            )))
        })?;
        let text = py_strip(text);
        if text.is_empty() {
            continue;
        }
        // json.loads — syntax errors are ValueError-class; the
        // message text is implementation-specific (kind matches).
        let obj: Value = serde_json::from_str(text)
            .map_err(|e| ExportError::Conv(PyConvError::Value(e.to_string())))?;
        entries.push(entry_from_value(&obj).map_err(ExportError::Conv)?);
    }
    Ok(entries)
}

/// `_parse_line(line)` — the lenient streaming-path line parser.
/// `KeyError`/`ValueError` → `Ok(None)` (skip); `TypeError` → loud
/// error, exactly like `except (KeyError, ValueError): return None`.
pub fn parse_line(line: &[u8]) -> Result<Option<ExportEntry>, ExportError> {
    // errors="replace" — from_utf8_lossy uses the same maximal-subpart
    // replacement algorithm as CPython's utf-8 error handler.
    let text = String::from_utf8_lossy(line);
    let text = py_strip(&text);
    if text.is_empty() {
        return Ok(None);
    }
    let obj: Value = match serde_json::from_str(text) {
        Ok(obj) => obj,
        Err(_) => return Ok(None), // JSONDecodeError ⊂ ValueError
    };
    match entry_from_value(&obj) {
        Ok(entry) => Ok(Some(entry)),
        Err(PyConvError::Key(_) | PyConvError::Value(_)) => Ok(None),
        Err(type_err @ PyConvError::Type(_)) => Err(ExportError::Conv(type_err)),
    }
}

/// The `pending` line splitter of `iter_export_ids`
/// (ingest/exports.py:83-94): decompressed chunks go in, complete
/// lines come out.
///
/// FAITHFUL QUIRK: a final line WITHOUT a trailing newline is never
/// parsed — the Python loop only splits on `b"\n"` and there is no
/// post-loop flush, so the trailing bytes are dropped silently.
/// Wave D must preserve this; [`LineBuffer::pending`] exposes the
/// dropped bytes so tests can pin the behavior.
#[derive(Debug, Default)]
pub struct LineBuffer {
    pending: Vec<u8>,
}

impl LineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `pending += decompressor.decompress(chunk)`.
    pub fn extend(&mut self, chunk: &[u8]) {
        self.pending.extend_from_slice(chunk);
    }

    /// `while b"\n" in pending: line, pending = pending.split(b"\n", 1)`
    /// — one line at a time, delimiter stripped.
    pub fn pop_line(&mut self) -> Option<Vec<u8>> {
        let pos = self.pending.iter().position(|&b| b == b'\n')?;
        let line: Vec<u8> = self.pending.drain(..pos).collect();
        self.pending.remove(0); // the b'\n'
        Some(line)
    }

    /// The still-unterminated tail — never parsed by the Python
    /// streaming loop (see the type docs). Read-only on purpose.
    pub fn pending(&self) -> &[u8] {
        &self.pending
    }
}

/// The `iter_export_ids` starter filters (ingest/exports.py:90-93):
/// `adult` entries are dropped unless `include_adult`, and entries
/// below the popularity floor are dropped.
///
/// NaN popularity never compares below the floor (`nan < x` is false
/// in Python), so NaN entries PASS — parity pinned by fixtures.
pub fn entry_passes(entry: &ExportEntry, min_popularity: f64, include_adult: bool) -> bool {
    // `!(pop < min)` via partial_cmp: Equal, Greater and the NaN case
    // (None) all pass — exactly Python's `popularity < min_popularity`
    // being false for NaN.
    (!entry.adult || include_adult)
        && !matches!(
            entry.popularity.partial_cmp(&min_popularity),
            Some(std::cmp::Ordering::Less)
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gz(lines: &[&str]) -> Vec<u8> {
        let mut w = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        use std::io::Write;
        for l in lines {
            writeln!(w, "{l}").unwrap();
        }
        w.finish().unwrap()
    }

    #[test]
    fn parse_export_happy_path() {
        let bytes = gz(&[
            r#"{"id": 1, "adult": false, "popularity": 1.25}"#,
            r#"{"id": 2, "original_title": "אור: סרט", "popularity": 0}"#, // unicode + falsy pop
            "",                                                            // blank line skipped
            r#"{"adult": true, "id": 3}"#, // popularity missing → 0.0
        ]);
        let entries = parse_export(&bytes).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0],
            ExportEntry {
                id: 1,
                adult: false,
                popularity: 1.25
            }
        );
        assert_eq!(
            entries[1],
            ExportEntry {
                id: 2,
                adult: false,
                popularity: 0.0
            }
        );
        assert_eq!(
            entries[2],
            ExportEntry {
                id: 3,
                adult: true,
                popularity: 0.0
            }
        );
    }

    #[test]
    fn parse_export_error_families() {
        // Missing id → KeyError class, strict path raises.
        let bytes = gz(&[r#"{"adult": false}"#]);
        assert!(matches!(
            parse_export(&bytes),
            Err(ExportError::Conv(PyConvError::Key(_)))
        ));
        // Non-object line → TypeError class (never swallowed anywhere).
        let bytes = gz(&["42"]);
        assert!(matches!(
            parse_export(&bytes),
            Err(ExportError::Conv(PyConvError::Type(_)))
        ));
        // Mid-payload corruption that still inflates to garbage lines:
        // the JSON parse fails BEFORE the reader ever reaches the gzip
        // trailer — so both Python (JSONDecodeError ⊂ ValueError) and
        // this port report the ValueError family, not an IO error.
        let mut bad = gz(&[r#"{"id": 1}"#]);
        bad[15] ^= 0xff;
        assert!(matches!(
            parse_export(&bad),
            Err(ExportError::Conv(PyConvError::Value(_)))
        ));
        // Strict decode failure: hand-build a gz whose payload has an
        // invalid utf-8 byte inside a line.
        use std::io::Write;
        let payload = b"{\"id\": 1, \"x\": \"\xff\"}\n";
        let mut w = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        w.write_all(payload).unwrap();
        let bytes = w.finish().unwrap();
        assert!(matches!(
            parse_export(&bytes),
            Err(ExportError::Conv(PyConvError::Value(_)))
        ));
    }

    #[test]
    fn parse_line_swallow_vs_propagate() {
        // ValueError-class → skipped.
        assert_eq!(parse_line(b"not json"), Ok(None));
        assert_eq!(parse_line(b"   "), Ok(None));
        assert_eq!(parse_line(b""), Ok(None));
        assert_eq!(parse_line(br#"{"adult": false}"#), Ok(None)); // KeyError → skip
                                                                  // ...but works in the lenient replace-decode path:
        assert_eq!(
            parse_line(b"{\"id\": 7, \"x\": \"\xff\"}"),
            Ok(Some(ExportEntry {
                id: 7,
                adult: false,
                popularity: 0.0
            }))
        );
        // TypeError-class → loud, on every path.
        assert!(matches!(
            parse_line(b"[1, 2]"),
            Err(ExportError::Conv(PyConvError::Type(_)))
        ));
        assert!(matches!(
            parse_line(b"null"),
            Err(ExportError::Conv(PyConvError::Type(_)))
        ));
    }

    #[test]
    fn line_buffer_splits_and_drops_the_tail() {
        let mut buf = LineBuffer::new();
        buf.extend(b"{\"id\": 1}\n{\"id");
        assert_eq!(buf.pop_line(), Some(b"{\"id\": 1}".to_vec()));
        assert_eq!(buf.pop_line(), None);
        buf.extend(b"\": 2}\n{\"id\": 3}\n");
        assert_eq!(buf.pop_line(), Some(b"{\"id\": 2}".to_vec()));
        assert_eq!(buf.pop_line(), Some(b"{\"id\": 3}".to_vec()));
        let mut buf = LineBuffer::new();
        buf.extend(b"tail without newline");
        assert_eq!(buf.pop_line(), None);
        assert_eq!(buf.pending(), b"tail without newline");
    }

    #[test]
    fn filters_match_python_comparisons() {
        let e = |id, adult, pop| ExportEntry {
            id,
            adult,
            popularity: pop,
        };
        assert!(entry_passes(&e(1, false, 0.0), 0.0, false));
        assert!(!entry_passes(&e(1, true, 0.0), 0.0, false)); // adult dropped
        assert!(entry_passes(&e(1, true, 0.0), 0.0, true));
        assert!(!entry_passes(&e(1, false, 0.4), 0.5, false)); // below floor
        assert!(entry_passes(&e(1, false, 0.5), 0.5, false)); // 0.5 < 0.5 is false → passes
        assert!(entry_passes(&e(1, false, f64::NAN), 0.5, false)); // NaN never below
    }

    #[test]
    fn urls_are_byte_compatible() {
        let d = CivilDate::new(2026, 10, 31).unwrap();
        assert_eq!(
            export_url_for(d, "movie").unwrap(),
            "http://files.tmdb.org/p/exports/movie_ids_10_31_2026.json.gz"
        );
        assert_eq!(
            export_url_for(d, "tv").unwrap(),
            "http://files.tmdb.org/p/exports/tv_series_ids_10_31_2026.json.gz"
        );
        assert_eq!(
            export_url_for(d, "film").unwrap_err().to_string(),
            "medium must be 'movie' or 'tv', got 'film'"
        );
        let today = CivilDate::new(2024, 3, 1).unwrap();
        assert_eq!(
            latest_export_url_at("movie", today).unwrap(),
            "http://files.tmdb.org/p/exports/movie_ids_02_29_2024.json.gz"
        );
    }
}
