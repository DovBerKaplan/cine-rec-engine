//! SchemaMap unit tests — replay of the Python-generated parity
//! fixture (exact resolve/name semantics, placeholder grammar edges,
//! the dotted-receiver escape) plus ported cases from
//! tests/test_tables.py.

use std::collections::BTreeMap;
use std::path::Path;

use cine_rec_db::{SchemaMap, SchemaMapError, LOGICAL_TABLES};

fn fixture() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("schema_map_cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture json")
}

fn map_from_json(v: &serde_json::Value) -> SchemaMap {
    let pairs: Vec<(String, String)> = v
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, val)| (k.clone(), val.as_str().unwrap().to_string()))
        .collect();
    SchemaMap::from_mapping(pairs).expect("fixture maps are valid")
}

#[test]
fn logical_tables_match_python_exactly() {
    let fx = fixture();
    let py: Vec<&str> = fx["logical_tables_sorted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let mut rs: Vec<&str> = LOGICAL_TABLES.to_vec();
    rs.sort_unstable();
    assert_eq!(rs, py, "LOGICAL_TABLES drifted from tables.py");
}

#[test]
fn resolve_replays_every_python_case() {
    let fx = fixture();
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() > 80);
    let mut checked = 0;
    for case in cases {
        let map = map_from_json(&case["map"]);
        let template = case["template"].as_str().unwrap();
        let outcome = map.resolve(template);
        if case["ok"].as_bool().unwrap() {
            assert_eq!(
                outcome.as_deref(),
                Ok(case["resolved"].as_str().unwrap()),
                "template {template:?} under map {:?}",
                case["map"]
            );
        } else {
            assert!(outcome.is_err(), "template {template:?} must fail");
        }
        checked += 1;
    }
    assert!(checked > 80); // a parity test that replayed nothing is a bug
}

#[test]
fn name_replays_python_cases() {
    let fx = fixture();
    let default = SchemaMap::new();
    for case in fx["names"].as_array().unwrap() {
        let logical = case["logical"].as_str().unwrap();
        let outcome = default.name(logical);
        if case["ok"].as_bool().unwrap() {
            assert_eq!(outcome.as_deref(), Ok(case["name"].as_str().unwrap()));
        } else {
            assert!(outcome.is_err());
        }
    }
}

#[test]
fn every_logical_name_resolves_by_default() {
    let m = SchemaMap::new();
    for logical in LOGICAL_TABLES {
        assert_eq!(m.name(logical).as_deref(), Ok(*logical));
    }
}

// ── ported from tests/test_tables.py ──────────────────────────────

#[test]
fn invalid_names_are_loud() {
    for bad in [
        "AppMedia",
        "app media",
        "app; DROP TABLE x",
        "a.b.c",
        "1app",
        "-x",
        "app.x.y",
        "has-dash",
    ] {
        assert!(
            SchemaMap::from_mapping([("tmdb_media", bad)]).is_err(),
            "{bad:?} must be rejected"
        );
    }
}

#[test]
fn schema_qualified_names_allowed() {
    let m = SchemaMap::from_mapping([("tmdb_media", "app.media")]).unwrap();
    assert_eq!(m.name("tmdb_media").as_deref(), Ok("app.media"));
}

#[test]
fn unknown_key_is_loud() {
    let err = SchemaMap::from_mapping([("nope", "x")]).unwrap_err();
    assert!(err.0.contains("unknown table key"));
}

#[test]
fn blank_values_rejected() {
    assert!(SchemaMap::from_mapping([("tmdb_media", "   ")]).is_err());
}

#[test]
fn active_map_lists_only_remapped() {
    let m = SchemaMap::from_mapping([("tmdb_media", "app_media"), ("tmdb_tv", "app_tv")]).unwrap();
    let active: BTreeMap<String, String> = m.active_map();
    assert_eq!(active.len(), 2);
    assert_eq!(active["tmdb_media"], "app_media");
}

#[test]
fn env_pairs_file_and_override_semantics() {
    let dir = std::env::temp_dir().join(format!("smap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let map_file = dir.join("map.json");
    std::fs::write(
        &map_file,
        r#"{"tmdb_media": "file_media", "tmdb_tv": "file_tv"}"#,
    )
    .unwrap();

    let read_ok = |p: &str| std::fs::read_to_string(p);

    // file only
    let env: Vec<(String, String)> = vec![(
        "CINE_REC_SCHEMA_MAP".into(),
        map_file.to_str().unwrap().into(),
    )];
    let m = SchemaMap::from_env_pairs(&env, &read_ok).unwrap();
    assert_eq!(m.name("tmdb_media").as_deref(), Ok("file_media"));

    // env wins over file; blank env value ignored (file value stands)
    let env2 = vec![
        (
            "CINE_REC_SCHEMA_MAP".to_string(),
            map_file.to_str().unwrap().into(),
        ),
        ("CINE_REC_TABLE_TMDB_MEDIA".to_string(), "env_media".into()),
        ("CINE_REC_TABLE_TMDB_TV".to_string(), "   ".into()),
    ];
    let m2 = SchemaMap::from_env_pairs(&env2, &read_ok).unwrap();
    assert_eq!(m2.name("tmdb_media").as_deref(), Ok("env_media"));
    assert_eq!(m2.name("tmdb_tv").as_deref(), Ok("file_tv"));

    // missing file is LOUD
    let env3 = vec![(
        "CINE_REC_SCHEMA_MAP".to_string(),
        "/nonexistent/x.json".into(),
    )];
    assert!(matches!(
        SchemaMap::from_env_pairs(&env3, &read_ok),
        Err(SchemaMapError(_))
    ));

    // broken JSON is LOUD
    let bad = dir.join("bad.json");
    std::fs::write(&bad, "{not json").unwrap();
    let env4 = vec![(
        "CINE_REC_SCHEMA_MAP".to_string(),
        bad.to_str().unwrap().into(),
    )];
    assert!(SchemaMap::from_env_pairs(&env4, &read_ok).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn placeholder_edges_match_the_regex_grammar() {
    let m = SchemaMap::new();
    // digits/uppercase/empty group never match; unclosed braces pass through
    assert_eq!(m.resolve("{t_}").unwrap(), "{t_}");
    assert_eq!(m.resolve("{t_1abc}").unwrap(), "{t_1abc}");
    assert_eq!(m.resolve("{t_Tmdb_Media}").unwrap(), "{t_Tmdb_Media}");
    assert_eq!(m.resolve("{t_tmdb_media").unwrap(), "{t_tmdb_media");
    assert_eq!(
        m.resolve("'{t_tmdb_media}'").unwrap(),
        "'tmdb_media'" // quotes do NOT protect the placeholder
    );
}
