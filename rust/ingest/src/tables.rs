//! Logical table-name registry — the pure core of
//! `cine_rec_engine/tables.py`.
//!
//! The engine and the ingest loader reference tables by LOGICAL name
//! (the identifiers in docs/schema.sql and docs/user_data.sql); the
//! physical name defaults to the logical one. Integrators remap
//! without touching code via a JSON file (`CINE_REC_SCHEMA_MAP`) or
//! per-table env (`CINE_REC_TABLE_<NAME>`, which wins over the file).
//!
//! This module ports the VALIDATION, MERGE and RESOLUTION logic as
//! pure functions — no IO, no globals, no sqlx. Reading the env and
//! the map file is a runtime edge (wave D/E wiring on top of
//! [`TableMap::from_file_and_env`]); the stateful lazy-loading cache
//! (`_ensure_loaded`/`set_table_map`/`reset`) is likewise an edge —
//! the semantics of every decision are here and pinned by fixtures
//! generated from the REAL `tables.py` (`tests/fixtures/gen_tables_cases.py`),
//! including the exact error texts.
//!
//! A remapped name is explicit intent: invalid maps fail LOUDLY
//! (there is no silent degradation for a typo'd map — that is the
//! design; a typo'd map must not silently empty a channel).
//!
//! Valid physical names are lower-case identifiers, optionally
//! schema-qualified (`schema.table`, one dot) — names are interpolated
//! into SQL unquoted, so mixed case is out of scope by construction.
//!
//! Known divergence (documented): with MULTIPLE invalid entries,
//! Python reports the first offender in file/env iteration order;
//! `serde_json`'s default map is sorted, so the file side iterates
//! alphabetically here. Single-offender maps — the realistic case —
//! are message-identical.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::Value;

/// Every table/view the engine or the ingest loader can touch, in the
/// same grouping order as `tables.py::LOGICAL_TABLES` (names must
/// match docs/schema.sql and docs/user_data.sql).
pub const LOGICAL_TABLES: &[&str] = &[
    // catalog compatibility views (docs/schema.sql §6) — the read path
    "tmdb_media",
    "tmdb_genres",
    "tmdb_media_genres",
    "tmdb_media_keywords",
    "tmdb_media_companies",
    "tmdb_media_networks",
    "tmdb_cast",
    "tmdb_crew",
    // catalog facts (§1) — KNN/embedding paths read these directly
    "tmdb_movies",
    "tmdb_tv",
    // dimensions (§2) and genre lists (§3)
    "tmdb_people",
    "tmdb_keywords",
    "tmdb_production_companies",
    "tmdb_networks",
    "tmdb_movie_genres",
    "tmdb_tv_genres",
    // per-medium bridges (§4)
    "tmdb_movie_genres_map",
    "tmdb_tv_genres_map",
    "tmdb_movie_keywords",
    "tmdb_tv_keywords",
    "tmdb_movie_companies",
    "tmdb_tv_companies",
    "tmdb_movie_cast",
    "tmdb_tv_cast",
    "tmdb_movie_crew",
    "tmdb_tv_crew",
    "tmdb_tv_networks_map",
    // behavioral graph (§5) — tmdb_recommendations is physical (UPSERTed)
    "tmdb_recommendations",
    "tmdb_movie_recommendations",
    "tmdb_tv_recommendations",
    // optional enrichment — not in the shipped schema, degrade if absent
    "tmdb_cinematic",
    // user layer (docs/user_data.sql); user_watches is a derived view
    "user_watch_events",
    "user_feedback",
    "user_searches",
    "title_ratings",
    "user_title_stats",
    "user_stats",
    "user_genre_stats",
    "user_vectors",
    "user_watches",
    // integrator-supplied episode detail (progress SQL); not shipped
    "media_episodes",
];

/// `TableMapError` — invalid table map or unknown logical name, always
/// a loud failure. The message mirrors the Python text exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableMapError(pub String);

impl fmt::Display for TableMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TableMapError {}

/// The `CINE_REC_TABLE_` env prefix (stripped and lowercased to form
/// the logical key).
const ENV_PREFIX: &str = "CINE_REC_TABLE_";

fn sorted_logical_csv() -> String {
    // ', '.join(sorted(LOGICAL_TABLES)) — Python's error lists.
    let mut names: Vec<&str> = LOGICAL_TABLES.to_vec();
    names.sort_unstable();
    names.join(", ")
}

/// `^[a-z_][a-z0-9_]*$` — ASCII scan equivalent of the Python regex.
fn is_ident(s: &str) -> bool {
    let mut bytes = s.bytes();
    match bytes.next() {
        Some(b'a'..=b'z') | Some(b'_') => {}
        _ => return false,
    }
    bytes.all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

/// `_validate_physical` — lower-case identifier, optionally
/// schema-qualified with ONE dot. Empty parts fail the identifier
/// check (`"".split(".")` yields `[""]` in Python, which does not
/// match either).
pub fn validate_physical_name(name: &str) -> Result<(), TableMapError> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() > 2 || !parts.iter().all(|p| is_ident(p)) {
        return Err(TableMapError(format!(
            "invalid table name {}: expected a lower-case identifier, \
             optionally schema-qualified (schema.table)",
            crate::pyval::py_repr(name)
        )));
    }
    Ok(())
}

fn lookup_logical(logical: &str) -> Option<&'static str> {
    LOGICAL_TABLES.iter().copied().find(|t| *t == logical)
}

fn check_value(logical: &str, physical: &str) -> Result<String, TableMapError> {
    // `not isinstance(physical, str) or not physical.strip()` — the
    // non-string case cannot arise for (&str, &str) pairs; empty /
    // whitespace-only fails here with Python's message.
    if py_trim(physical).is_empty() {
        return Err(TableMapError(format!(
            "table name for {} must be a non-empty string",
            crate::pyval::py_repr(logical)
        )));
    }
    validate_physical_name(py_trim(physical))?;
    Ok(py_trim(physical).to_string())
}

/// Python `str.strip()` for map values (see `pyval::py_strip`).
fn py_trim(s: &str) -> &str {
    crate::pyval::py_strip(s)
}

/// A validated logical→physical mapping. Only non-default mappings
/// are stored; unmapped names resolve to themselves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableMap {
    map: BTreeMap<&'static str, String>,
}

impl TableMap {
    /// `set_table_map(mapping)` semantics — programmatic install. An
    /// empty iterator gives the identity map (all defaults).
    pub fn from_pairs<'a, I>(pairs: I) -> Result<Self, TableMapError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut map = BTreeMap::new();
        for (logical, physical) in pairs {
            let logical = lookup_logical(logical).ok_or_else(|| {
                TableMapError(format!(
                    "unknown table key {}; valid keys: {}",
                    crate::pyval::py_repr(logical),
                    sorted_logical_csv()
                ))
            })?;
            let value = check_value(logical, physical)?;
            map.insert(logical, value);
        }
        Ok(Self { map })
    }

    /// `load_table_map()` semantics, with the IO already done: the
    /// parsed JSON map (`None` when no path was configured — missing
    /// file / unreadable path cases belong to the runtime edge that
    /// reads it) plus the raw environment pairs. Env wins over file.
    ///
    /// `path` is only used for message parity on `must be a JSON
    /// object` (Python embeds the configured path).
    pub fn from_file_and_env(
        path: Option<&str>,
        file_map: Option<&Value>,
        env: &[(String, String)],
    ) -> Result<Self, TableMapError> {
        let mut entries: Vec<(String, String)> = Vec::new();
        match (path, file_map) {
            (None, _) | (_, None) => {}
            (Some(_path), Some(Value::Object(obj))) => {
                for (k, v) in obj {
                    let physical = match v {
                        Value::String(s) => s.clone(),
                        // Non-string values fail with the shared
                        // "non-empty string" message, like Python's
                        // isinstance check.
                        _ => {
                            return Err(TableMapError(format!(
                                "table name for {} must be a non-empty string",
                                crate::pyval::py_repr(k)
                            )));
                        }
                    };
                    entries.push((k.clone(), physical));
                }
            }
            (Some(path), Some(_)) => {
                return Err(TableMapError(format!(
                    "schema map {} must be a JSON object",
                    crate::pyval::py_repr(path)
                )));
            }
        }
        // `_env_overrides`: only CINE_REC_TABLE_* vars with non-blank
        // values count; the suffix is lowercased into the key.
        let mut merged: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in entries {
            merged.insert(k, v);
        }
        for (var, val) in env {
            if let Some(suffix) = var.strip_prefix(ENV_PREFIX) {
                if !val.trim().is_empty() {
                    merged.insert(suffix.to_lowercase(), val.clone());
                }
            }
        }
        // Validate + install, mirroring the Python loop (unknown key
        // first, then value checks, values stored stripped).
        let mut map = BTreeMap::new();
        for (logical, physical) in &merged {
            let logical_ref = lookup_logical(logical).ok_or_else(|| {
                TableMapError(format!(
                    "unknown table key {} in schema map; valid keys: {}",
                    crate::pyval::py_repr(logical),
                    sorted_logical_csv()
                ))
            })?;
            let value = check_value(logical_ref, physical)?;
            map.insert(logical_ref, value);
        }
        Ok(Self { map })
    }

    /// `tables.name(logical)` — physical name for a logical table,
    /// identity when nothing is mapped. Unknown names fail loudly.
    pub fn name<'a>(&'a self, logical: &'a str) -> Result<&'a str, TableMapError> {
        if !LOGICAL_TABLES.contains(&logical) {
            return Err(TableMapError(format!(
                "unknown logical table {}; valid names are defined in \
                 cine_rec_engine.tables.LOGICAL_TABLES",
                crate::pyval::py_repr(logical)
            )));
        }
        Ok(self.map.get(logical).map(String::as_str).unwrap_or(logical))
    }

    /// `tables.active_map()` — only the non-default mappings (for
    /// startup verification/logging).
    pub fn active_map(&self) -> &BTreeMap<&'static str, String> {
        &self.map
    }

    /// `tables.resolve(template)` — substitute `{t_<logical>}`
    /// placeholders with physical names; an unknown logical inside a
    /// placeholder fails loudly mid-substitution, like the Python
    /// `re.sub` calling `name()`.
    ///
    /// The placeholder grammar is `\{t_([a-z_]+)\}` exactly: digits
    /// and upper-case never match, so `{t_ab2c}` / `{T_x}` / `{t_}`
    /// stay literal.
    pub fn resolve(&self, template: &str) -> Result<String, TableMapError> {
        let bytes = template.as_bytes();
        let mut out = String::with_capacity(template.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i..].starts_with(b"{t_") {
                let start = i + 3;
                let mut end = start;
                while end < bytes.len() && (bytes[end].is_ascii_lowercase() || bytes[end] == b'_') {
                    end += 1;
                }
                if end > start && bytes.get(end) == Some(&b'}') {
                    // A placeholder: [a-z_]+ between "{t_" and "}".
                    let logical = &template[start..end];
                    out.push_str(self.name(logical)?);
                    i = end + 1;
                    continue;
                }
                // Not a placeholder — advance one byte and keep
                // scanning (the regex engine tries later positions).
            }
            // Multi-byte safety: only push complete chars. Placeholder
            // bytes are ASCII, so push a char boundary at a time.
            let ch_len = utf8_char_len(bytes[i]);
            out.push_str(&template[i..i + ch_len]);
            i += ch_len;
        }
        Ok(out)
    }
}

/// Length in bytes of the UTF-8 char starting at `b`.
fn utf8_char_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_table_list_matches_python_shape() {
        // 41 tables; unique; every name is itself a valid identifier.
        let mut seen = std::collections::HashSet::new();
        for t in LOGICAL_TABLES {
            assert!(seen.insert(*t), "duplicate logical name {t}");
            assert!(is_ident(t), "{t} must be a valid identifier");
        }
        assert_eq!(LOGICAL_TABLES.len(), 41);
    }

    #[test]
    fn physical_name_validation() {
        for ok in [
            "tmdb_media",
            "app_media",
            "app.user_watches",
            "a",
            "_x",
            "x1",
        ] {
            assert!(validate_physical_name(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            ".",
            "..",
            "a.b.c",
            "AppMedia",
            "1x",
            "-x",
            "app. user",
            "app user",
            "app.",
            ".app",
            "has-dash",
        ] {
            assert!(
                validate_physical_name(bad).is_err(),
                "{bad} must be rejected"
            );
        }
        assert_eq!(
            validate_physical_name("AppMedia").unwrap_err().to_string(),
            "invalid table name 'AppMedia': expected a lower-case identifier, \
             optionally schema-qualified (schema.table)"
        );
    }

    #[test]
    fn from_pairs_installs_and_reports() {
        let m = TableMap::from_pairs([("tmdb_media", "app_media"), ("user_watches", "app.uw")])
            .unwrap();
        assert_eq!(m.name("tmdb_media"), Ok("app_media"));
        assert_eq!(m.name("user_watches"), Ok("app.uw"));
        assert_eq!(m.name("tmdb_movies"), Ok("tmdb_movies")); // identity
        assert_eq!(m.active_map().len(), 2);
        assert_eq!(
            m.name("nope").unwrap_err().to_string(),
            "unknown logical table 'nope'; valid names are defined in \
             cine_rec_engine.tables.LOGICAL_TABLES"
        );
        assert_eq!(
            TableMap::from_pairs([("nope", "x")])
                .unwrap_err()
                .to_string(),
            format!(
                "unknown table key 'nope'; valid keys: {}",
                sorted_logical_csv()
            )
        );
        // Whitespace-only / empty values rejected with Python's text.
        assert_eq!(
            TableMap::from_pairs([("tmdb_media", "  ")])
                .unwrap_err()
                .to_string(),
            "table name for 'tmdb_media' must be a non-empty string"
        );
        // ...but surrounding whitespace is stripped off valid names.
        let m = TableMap::from_pairs([("tmdb_media", "  app_media  ")]).unwrap();
        assert_eq!(m.name("tmdb_media"), Ok("app_media"));
    }

    #[test]
    fn env_merges_over_file_and_ignores_blank() {
        let file: Value =
            serde_json::from_str(r#"{"tmdb_media": "file_media", "tmdb_tv": "file_tv"}"#).unwrap();
        let env = vec![
            (
                "CINE_REC_TABLE_TMDB_MEDIA".to_string(),
                "env_media".to_string(),
            ), // wins
            ("CINE_REC_TABLE_TMDB_MOVIES".to_string(), "   ".to_string()), // blank → ignored
            ("CINE_REC_TABLE_tmdb_tv".to_string(), "env_tv".to_string()),  // suffix lowercased
            ("OTHER_VAR".to_string(), "x".to_string()),                    // not a table var
        ];
        let m = TableMap::from_file_and_env(Some("map.json"), Some(&file), &env).unwrap();
        assert_eq!(m.name("tmdb_media"), Ok("env_media"));
        assert_eq!(m.name("tmdb_tv"), Ok("env_tv"));
        assert_eq!(m.name("tmdb_movies"), Ok("tmdb_movies")); // blank env ignored
                                                              // Unknown env key fails with the load-path message.
        let env = vec![("CINE_REC_TABLE_BOGUS".to_string(), "x".to_string())];
        assert_eq!(
            TableMap::from_file_and_env(None, None, &env)
                .unwrap_err()
                .to_string(),
            format!(
                "unknown table key 'bogus' in schema map; valid keys: {}",
                sorted_logical_csv()
            )
        );
        // Non-object file map fails with Python's message (path repr'd).
        let arr = serde_json::json!([1, 2]);
        assert_eq!(
            TableMap::from_file_and_env(Some("/tmp/m.json"), Some(&arr), &[])
                .unwrap_err()
                .to_string(),
            "schema map '/tmp/m.json' must be a JSON object"
        );
        // Non-string file value fails with the shared message.
        let bad = serde_json::json!({"tmdb_media": 5});
        assert_eq!(
            TableMap::from_file_and_env(Some("/tmp/m.json"), Some(&bad), &[])
                .unwrap_err()
                .to_string(),
            "table name for 'tmdb_media' must be a non-empty string"
        );
    }

    #[test]
    fn resolve_replaces_only_lowercase_underscore_placeholders() {
        let m = TableMap::from_pairs([("tmdb_media", "app_media")]).unwrap();
        assert_eq!(
            m.resolve("SELECT * FROM {t_tmdb_media} JOIN {t_user_watches}")
                .unwrap(),
            "SELECT * FROM app_media JOIN user_watches"
        );
        // Non-placeholders stay literal — the exact regex grammar.
        assert_eq!(
            m.resolve("{T_tmdb_media} {t_} {t_ab2c} {t_} {} t_x {t_tmdb_media")
                .unwrap(),
            "{T_tmdb_media} {t_} {t_ab2c} {t_} {} t_x {t_tmdb_media"
        );
        // Unknown logical inside a placeholder fails loudly.
        assert_eq!(
            m.resolve("SELECT * FROM {t_bogus}")
                .unwrap_err()
                .to_string(),
            "unknown logical table 'bogus'; valid names are defined in \
             cine_rec_engine.tables.LOGICAL_TABLES"
        );
        // Unicode passes through untouched.
        assert_eq!(
            m.resolve("SELECT 'אור' FROM {t_tmdb_media}").unwrap(),
            "SELECT 'אור' FROM app_media"
        );
        // Every logical name round-trips through resolve unharmed.
        let identity = TableMap::default();
        for t in LOGICAL_TABLES {
            let sql = format!("{{t_{t}}}");
            assert_eq!(identity.resolve(&sql).unwrap(), *t);
        }
    }
}
