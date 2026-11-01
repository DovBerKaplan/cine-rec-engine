//! Python value-conversion semantics over JSON values.
//!
//! Phase-2 lesson (`rust/scoring/README.md`, "Portability
//! discoveries"): **Python truthiness is part of the contract**. The
//! export parser feeds its JSON objects through Python expressions
//! whose accepted shapes are wider than JSON's numeric types —
//! `float("1.5")` on a string, `int(true)` → 1, `"" or 0.0` → 0.0 —
//! and whose failures come in three exception classes the Python code
//! treats differently: `_parse_line` swallows `KeyError`/`ValueError`
//! but lets `TypeError` propagate (ingest/exports.py:97-109).
//!
//! [`PyConvError`] mirrors those classes so callers can reproduce the
//! same swallow/propagate decisions, with Python's message texts where
//! they are deterministic.
//!
//! Known divergences (documented; all pathological for TMDB input):
//! * Python builds arbitrary-precision ints; values outside `i64` are
//!   a loud [`PyConvError::Value`] here instead of a bignum.
//! * Python `int()`/`float()` accept non-ASCII decimal digits
//!   (`int("٤٢")` is 42); only ASCII digits parse here.
//! * Python's `json` accepts the non-standard `NaN`/`Infinity`
//!   literals and produces non-finite floats; serde_json is strict
//!   JSON, so such a LINE fails at parse — a ValueError-class
//!   divergence (skip in the lenient path, raise in the strict one)
//!   instead of a NaN-valued entry.
//! * JSON *syntax*-error message texts are implementation-specific.
//!   Python's `json.JSONDecodeError` is a `ValueError` subclass, so
//!   the class (kind) matches; the message does not.

use std::fmt;

use serde_json::Value;

/// A Python exception family raised while converting a JSON value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyConvError {
    /// `KeyError` — a missing dict key. `str()` of a KeyError is the
    /// repr'd key (`'id'`), with no class prefix.
    Key(String),
    /// `ValueError` — right type, unconvertible value (bad literal,
    /// NaN→int, out-of-range for this port).
    Value(String),
    /// `AttributeError` — a `.get` on a non-dict (truthy containers
    /// where Python would raise before reaching the data).
    Attr(String),
    /// `TypeError` — the value's type is not convertible at all
    /// (null/list/dict where a number was needed, non-subscriptable
    /// JSON). The ingest code NEVER swallows this class.
    Type(String),
}

impl fmt::Display for PyConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyConvError::Key(m)
            | PyConvError::Value(m)
            | PyConvError::Type(m)
            | PyConvError::Attr(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for PyConvError {}

/// Python `bool(x)` over a JSON value: `null`/`false`/`0`/`±0.0`/
/// `""`/`[]`/`{}` are falsy; everything else — **including NaN** —
/// is truthy (Python: `bool(float("nan"))` is `True`).
pub fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        // NaN != 0.0 → true; -0.0 == 0.0 → false. Every JSON number
        // converts to f64 (u64/i64/f64), and any non-zero integer
        // rounds to a non-zero f64, so the zero test is exact.
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python type name of a JSON value, for TypeError message parity.
pub fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python `str.strip()` / `str.isspace()`: Unicode whitespace plus the
/// C0 control chars `\x1c`–`\x1f` (CPython's `Py_UNICODE_ISSPACE` adds
/// them; Rust's `char::is_whitespace` does not).
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()` — trim every [`py_isspace`] char from both ends.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// Python `repr()` for strings (quote preference and escapes). Exact
/// for the identifier-ish inputs this crate's error messages carry.
pub fn py_repr(s: &str) -> String {
    // Prefer single quotes; switch to double quotes only when the
    // string contains ' but not " (CPython's rule).
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `int(x)` over a JSON value (CPython 3.12 semantics for the
/// shapes JSON can carry).
///
/// * `bool` → 0/1; JSON int → itself; JSON float truncates toward zero
///   (Python `int(2.9)` is 2); JSON strings may carry sign, surrounding
///   whitespace and `_` between digits (Python 3.6+ rule).
/// * NaN/±inf → `ValueError` ("cannot convert float NaN/infinity to
///   integer"), exactly like Python.
/// * Out-of-`i64` values are a loud `ValueError` here — the documented
///   bignum divergence.
pub fn py_int(v: &Value) -> Result<i64, PyConvError> {
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                Err(PyConvError::Value(format!(
                    "int() argument {u} exceeds the i64 range this port \
                     supports (Python would build an arbitrary-precision int)"
                )))
            } else if let Some(f) = n.as_f64() {
                if f.is_nan() {
                    Err(PyConvError::Value(
                        "cannot convert float NaN to integer".into(),
                    ))
                } else if f.is_infinite() {
                    Err(PyConvError::Value(
                        "cannot convert float infinity to integer".into(),
                    ))
                } else {
                    float_to_i64(f)
                }
            } else {
                // serde_json numbers are u64 | i64 | f64 — unreachable.
                Err(PyConvError::Value(format!("unsupported number {n}")))
            }
        }
        Value::String(s) => parse_int_str(s),
        Value::Null | Value::Array(_) | Value::Object(_) => Err(PyConvError::Type(format!(
            "int() argument must be a string, a bytes-like object or a \
             real number, not '{}'",
            py_type_name(v)
        ))),
    }
}

/// `int()` on a float with Python's truncation and the exact i64
/// boundary: the half-open range −2⁶³..2⁶³ (−2⁶³ is `i64::MIN`;
/// 2⁶³ itself is out of range).
fn float_to_i64(f: f64) -> Result<i64, PyConvError> {
    let t = f.trunc();
    const I64_RANGE: std::ops::Range<f64> = std::ops::Range {
        start: -9_223_372_036_854_775_808.0,
        end: 9_223_372_036_854_775_808.0,
    };
    if !I64_RANGE.contains(&t) {
        Err(PyConvError::Value(format!(
            "int() argument {t} exceeds the i64 range this port \
             supports (Python would build an arbitrary-precision int)"
        )))
    } else {
        Ok(t as i64)
    }
}

/// Python `int(str)` — optional sign, ASCII digits, underscores only
/// between digits, surrounding whitespace stripped.
fn parse_int_str(orig: &str) -> Result<i64, PyConvError> {
    let invalid = || {
        PyConvError::Value(format!(
            "invalid literal for int() with base 10: {}",
            py_repr(orig)
        ))
    };
    let bignum = |acc: i128| {
        PyConvError::Value(format!(
            "int() argument {acc} exceeds the i64 range this port \
             supports (Python would build an arbitrary-precision int)"
        ))
    };
    let s = py_strip(orig).as_bytes();
    let (neg, digits) = match s.first() {
        Some(b'+') => (false, &s[1..]),
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    // i128 accumulator: absorbs "-9223372036854775808" (magnitude 2⁶³)
    // without overflow, then range-checks once.
    let mut acc: i128 = 0;
    let mut prev_digit = false;
    let mut seen_digit = false;
    for (i, &c) in digits.iter().enumerate() {
        match c {
            b'0'..=b'9' => {
                acc = acc * 10 + i128::from(c - b'0');
                prev_digit = true;
                seen_digit = true;
            }
            b'_' => {
                // Underscores must sit between two digits: the
                // previous char is one (prev_digit) and the next byte
                // is an ASCII digit.
                let next_digit = digits.get(i + 1).is_some_and(|d| d.is_ascii_digit());
                if !prev_digit || !next_digit {
                    return Err(invalid());
                }
                prev_digit = false;
            }
            _ => return Err(invalid()),
        }
    }
    if !seen_digit || !prev_digit {
        // "", "+", "1_", "_1" — no digits, or a trailing underscore.
        return Err(invalid());
    }
    let signed = if neg { -acc } else { acc };
    i64::try_from(signed).map_err(|_| bignum(signed))
}

/// Python `float(x)` over a JSON value: numbers pass through (integers
/// round to the nearest f64 exactly as Python's conversion does),
/// `bool` → 1.0/0.0, and strings accept surrounding whitespace,
/// `inf`/`infinity`/`nan` (any case, optional sign) and underscores
/// between digits.
pub fn py_float(v: &Value) -> Result<f64, PyConvError> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => Ok(n.as_f64().expect("json numbers convert to f64")),
        Value::String(s) => parse_float_str(s),
        Value::Null | Value::Array(_) | Value::Object(_) => Err(PyConvError::Type(format!(
            "float() argument must be a string or a real number, not '{}'",
            py_type_name(v)
        ))),
    }
}

/// Python `float(str)`.
fn parse_float_str(orig: &str) -> Result<f64, PyConvError> {
    let could_not = || {
        PyConvError::Value(format!(
            "could not convert string to float: {}",
            py_repr(orig)
        ))
    };
    let s = py_strip(orig);
    let b = s.as_bytes();
    // Validate underscore placement first (between ASCII digits only);
    // afterwards every underscore is safe to drop in one pass.
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' {
            let prev = i > 0 && b[i - 1].is_ascii_digit();
            let next = b.get(i + 1).is_some_and(|d| d.is_ascii_digit());
            if !prev || !next {
                return Err(could_not());
            }
        }
    }
    let cleaned: String = s.chars().filter(|&c| c != '_').collect();
    if cleaned.is_empty() {
        return Err(could_not());
    }
    // Rust's f64 parser accepts inf/infinity/nan (any case, optional
    // sign) and the decimal/exponent forms Python does — both sides
    // are correctly rounded, so the bits agree.
    cleaned.parse::<f64>().map_err(|_| could_not())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truthiness_matches_python() {
        // ±0.0 falsy — the case that bites ports. NaN/Infinity cannot
        // be reached through serde_json values (see the module docs:
        // strict JSON rejects those literals at parse); the
        // `f != 0.0` form keeps NaN truthy anyway, matching Python.
        assert!(!is_truthy(&json!(0)));
        assert!(!is_truthy(&json!(0.0)));
        assert!(!is_truthy(&json!(-0.0)));
        assert_eq!(json!(f64::NAN), json!(null)); // serde_json reality
        assert!(!is_truthy(&json!("")));
        assert!(!is_truthy(&json!([])));
        assert!(!is_truthy(&json!({})));
        assert!(!is_truthy(&Value::Null));
        assert!(is_truthy(&json!("0"))); // non-empty string
        assert!(is_truthy(&json!([0])));
    }

    #[test]
    fn py_int_truncates_and_validates() {
        assert_eq!(py_int(&json!(42)), Ok(42));
        assert_eq!(py_int(&json!(42.0)), Ok(42));
        assert_eq!(py_int(&json!(42.9)), Ok(42));
        assert_eq!(py_int(&json!(-42.9)), Ok(-42));
        assert_eq!(py_int(&json!(true)), Ok(1));
        assert_eq!(py_int(&json!(" 42 ")), Ok(42));
        assert_eq!(py_int(&json!("1_000")), Ok(1000));
        assert_eq!(py_int(&json!(1e18)), Ok(1_000_000_000_000_000_000)); // float truncation
        assert!(py_int(&json!(1e300)).is_err()); // bignum divergence, loud
        assert_eq!(
            py_int(&json!("abc")).unwrap_err(),
            PyConvError::Value("invalid literal for int() with base 10: 'abc'".into())
        );
        assert_eq!(
            py_int(&json!("1__0")).unwrap_err(),
            PyConvError::Value("invalid literal for int() with base 10: '1__0'".into())
        );
        assert_eq!(
            py_int(&Value::Null).unwrap_err(),
            PyConvError::Type(
                "int() argument must be a string, a bytes-like object or \
                 a real number, not 'NoneType'"
                    .into()
            )
        );
        assert_eq!(
            py_int(&json!([1])).unwrap_err(),
            PyConvError::Type(
                "int() argument must be a string, a bytes-like object or \
                 a real number, not 'list'"
                    .into()
            )
        );
    }

    #[test]
    fn py_float_accepts_python_string_shapes() {
        assert_eq!(py_float(&json!("1.5")), Ok(1.5));
        assert_eq!(py_float(&json!(" 1.5 ")), Ok(1.5));
        assert_eq!(py_float(&json!("1e3")), Ok(1000.0));
        assert_eq!(py_float(&json!("-Infinity")), Ok(f64::NEG_INFINITY));
        assert!(py_float(&json!("nan")).unwrap().is_nan());
        assert_eq!(py_float(&json!("1_0.5")), Ok(10.5));
        assert_eq!(py_float(&json!(true)), Ok(1.0));
        assert_eq!(
            py_float(&json!("abc")).unwrap_err(),
            PyConvError::Value("could not convert string to float: 'abc'".into())
        );
        assert_eq!(
            py_float(&json!([1])).unwrap_err(),
            PyConvError::Type(
                "float() argument must be a string or a real number, \
                 not 'list'"
                    .into()
            )
        );
    }

    #[test]
    fn py_repr_quotes_like_python() {
        assert_eq!(py_repr("plain"), "'plain'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a\"b"), "'a\"b'");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("line\n"), "'line\\n'");
    }

    #[test]
    fn py_strip_includes_c0_controls() {
        assert_eq!(py_strip("  x  "), "x");
        assert_eq!(py_strip("\u{1c}x\u{1f}"), "x");
        // NBSP is whitespace for both Python isspace() and Rust.
        assert_eq!(py_strip("\u{a0}x\u{a0}"), "x");
    }
}
