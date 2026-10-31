//! Logical table-name registry — the Rust side of `tables.py`.
//!
//! The engine and the ingest loader reference tables by LOGICAL name;
//! the physical name defaults to the logical one and integrators remap
//! without touching code via `CINE_REC_SCHEMA_MAP` (JSON file) and
//! per-table `CINE_REC_TABLE_<LOGICAL>` env vars (env wins).
//!
//! A remapped name is explicit intent: invalid keys or names fail
//! LOUDLY ([`SchemaMapError`]) — never a silent channel degradation.
//!
//! Design note: Python keeps this in module globals with
//! `set_table_map()` for tests; here the map is a VALUE, cloned into
//! whoever needs it ([`std::sync::Arc`] in the executor), so tests
//! construct maps explicitly and nothing global mutates.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

/// Every table/view the engine or the ingest loader can touch — the
/// exact set from `tables.py::LOGICAL_TABLES` (keep in sync; the
/// parity fixture below fails if the semantics drift).
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

/// Invalid table map or unknown logical name — always a loud failure
/// (the port of `tables.TableMapError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaMapError(pub String);

impl fmt::Display for SchemaMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SchemaMapError {}

fn logical_set() -> HashSet<&'static str> {
    LOGICAL_TABLES.iter().copied().collect()
}

/// `^[a-z_][a-z0-9_]*$` per dot-part, at most one dot — physical names
/// are interpolated UNQUOTED into SQL, so this is a safety boundary.
fn validate_physical(name: &str) -> Result<(), SchemaMapError> {
    let parts: Vec<&str> = name.split('.').collect();
    let ident_ok = |p: &str| {
        let mut chars = p.chars();
        match chars.next() {
            Some(c) if c.is_ascii_lowercase() || c == '_' => {}
            _ => return false,
        }
        chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    };
    if parts.len() > 2 || !parts.iter().all(|p| ident_ok(p)) {
        return Err(SchemaMapError(format!(
            "invalid table name {name:?}: expected a lower-case identifier, \
             optionally schema-qualified (schema.table)"
        )));
    }
    Ok(())
}

/// The active table map (only the non-default mappings).
#[derive(Debug, Clone, Default)]
pub struct SchemaMap {
    tables: BTreeMap<String, String>,
}

impl SchemaMap {
    /// All defaults (identity mapping).
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a programmatic map (tests, tooling) — the port of
    /// `set_table_map`: an empty map restores defaults; keys/values are
    /// validated exactly like a file map.
    pub fn from_mapping<S: Into<String>>(
        mapping: impl IntoIterator<Item = (S, S)>,
    ) -> Result<Self, SchemaMapError> {
        let logicals = logical_set();
        let mut tables = BTreeMap::new();
        for (logical, physical) in mapping {
            let (logical, physical): (String, String) = (logical.into(), physical.into());
            if !logicals.contains(logical.as_str()) {
                return Err(SchemaMapError(format!(
                    "unknown table key {logical:?} in schema map"
                )));
            }
            let physical = physical.trim().to_string();
            if physical.is_empty() {
                return Err(SchemaMapError(format!(
                    "table name for {logical:?} must be a non-empty string"
                )));
            }
            validate_physical(&physical)?;
            tables.insert(logical, physical);
        }
        Ok(Self { tables })
    }

    /// Merge the JSON file (`CINE_REC_SCHEMA_MAP`) with per-table env
    /// overrides (`CINE_REC_TABLE_*`, env wins) — the port of
    /// `load_table_map`. The env is passed as an explicit pair list so
    /// tests inject it without process-global state.
    pub fn from_env_pairs(
        env_pairs: &[(String, String)],
        read_file: &dyn Fn(&str) -> std::io::Result<String>,
    ) -> Result<Self, SchemaMapError> {
        let mut file_map: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::new();
        let path = env_pairs
            .iter()
            .find(|(k, _)| k == "CINE_REC_SCHEMA_MAP")
            .map(|(_, v)| v.trim().to_string())
            .unwrap_or_default();
        if !path.is_empty() {
            let text = read_file(&path).map_err(|e| {
                SchemaMapError(format!("cannot load schema map {path:?}: {e}"))
            })?;
            let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                SchemaMapError(format!("cannot load schema map {path:?}: {e}"))
            })?;
            file_map = value
                .as_object()
                .ok_or_else(|| {
                    SchemaMapError(format!("schema map {path:?} must be a JSON object"))
                })?
                .clone();
        }

        let mut merged: Vec<(String, String)> = Vec::new();
        for (k, v) in &file_map {
            let physical = v.as_str().ok_or_else(|| {
                SchemaMapError(format!("table name for {k:?} must be a non-empty string"))
            })?;
            merged.push((k.clone(), physical.to_string()));
        }
        // env overrides win; CINE_REC_TABLE_<LOGICAL> with a blank value
        // is ignored (Python checks val.strip() truthiness)
        for (var, val) in env_pairs {
            if let Some(key) = var.strip_prefix("CINE_REC_TABLE_") {
                let val = val.trim();
                if !val.is_empty() {
                    let key = key.to_lowercase();
                    merged.retain(|(k, _)| k != &key);
                    merged.push((key, val.to_string()));
                }
            }
        }
        Self::from_mapping(merged)
    }

    /// `from_env_pairs` over the process environment.
    pub fn from_env() -> Result<Self, SchemaMapError> {
        let pairs: Vec<(String, String)> = std::env::vars().collect();
        Self::from_env_pairs(&pairs, &|p| std::fs::read_to_string(p))
    }

    /// Physical name for a logical table; identity when unmapped.
    /// Unknown logical names fail loudly.
    pub fn name(&self, logical: &str) -> Result<String, SchemaMapError> {
        if !logical_set().contains(logical) {
            return Err(SchemaMapError(format!(
                "unknown logical table {logical:?}; valid names are defined in \
                 cine_rec_db::schema_map::LOGICAL_TABLES"
            )));
        }
        match self.tables.get(logical) {
            Some(physical) => Ok(physical.clone()),
            None => Ok(logical.to_string()),
        }
    }

    /// Only the non-default mappings (startup verification/logging).
    pub fn active_map(&self) -> BTreeMap<String, String> {
        self.tables.clone()
    }

    /// Substitute `{t_<logical>}` placeholders with physical names —
    /// the port of `tables.resolve`. The placeholder grammar is exactly
    /// Python's `\{t_([a-z_]+)\}`: lower-case letters/underscores only,
    /// digits or dots inside the braces mean NO match and the text
    /// passes through untouched (the dotted-receiver escape lives on
    /// this exact behavior).
    pub fn resolve(&self, template: &str) -> Result<String, SchemaMapError> {
        let bytes = template.as_bytes();
        let mut out = String::with_capacity(template.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'{' && template[i..].starts_with("{t_") {
                // scan [a-z_]+ greedily after the prefix
                let mut j = i + 3;
                while j < bytes.len()
                    && (bytes[j] == b'_'
                        || bytes[j].is_ascii_lowercase())
                {
                    j += 1;
                }
                if j > i + 3 && j < bytes.len() && bytes[j] == b'}' {
                    let logical = &template[i + 3..j];
                    out.push_str(&self.name(logical)?);
                    i = j + 1;
                    continue;
                }
            }
            // push one char (templates are ASCII-ish SQL; char-boundary
            // safe because '{'/'_'/'a'..'z'/'}' are single-byte and any
            // non-match falls through byte-by-byte until the next '{')
            let ch = template[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
        Ok(out)
    }
}
