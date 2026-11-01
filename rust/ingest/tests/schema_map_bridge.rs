//! Cross-check of the SINGLE registry (`cine_rec_db::SchemaMap`,
//! session-1's crate) against fixtures generated from the REAL
//! `cine_rec_engine/tables.py` (tests/fixtures/tables_cases.json —
//! produced in wave C by fixtures/gen_tables_cases.py).
//!
//! What must agree: every accept/reject DECISION, the active maps,
//! and the rendered `{t_*}` SQL. Error MESSAGE TEXTS are intentionally
//! NOT compared — SchemaMap's messages differ from tables.py's
//! (Rust-Debug repr instead of Python repr, no valid-keys list, one
//! unknown-key message instead of Python's two per-path variants).
//! That divergence is recorded in the queue for session 1 (it is
//! user-visible through `cine-rec check`); this test pins everything
//! BEHAVIORAL so the two registries can never drift silently.

use cine_rec_db::SchemaMap;
use serde_json::Value;

#[test]
fn schema_map_decisions_match_python_tables_py() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/tables_cases.json")).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 25, "only {} cases", cases.len());

    let mut decided = 0;
    for case in cases {
        if case["io_edge"].as_bool().unwrap_or(false) {
            continue; // file-IO edge: wave E wiring, not registry logic
        }
        decided += 1;
        let name = case["name"].as_str().unwrap();
        let expect = &case["expect"];

        let map = match case["install"].as_str().unwrap() {
            "set" | "identity" => {
                let pairs: Vec<(String, String)> = case["pairs"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|p| {
                                (
                                    p[0].as_str().unwrap().to_string(),
                                    p[1].as_str().unwrap_or("").to_string(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                SchemaMap::from_mapping(pairs)
            }
            "load" => {
                let mut env: Vec<(String, String)> = case["env"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        (
                            p[0].as_str().unwrap().to_string(),
                            p[1].as_str().unwrap().to_string(),
                        )
                    })
                    .collect();
                let file = case.get("file").filter(|f| !f.is_null()).cloned();
                match file {
                    // The generator wrote the file and pointed
                    // CINE_REC_SCHEMA_MAP at it — model exactly that.
                    Some(body) => {
                        let path = case
                            .get("file_path")
                            .and_then(|p| p.as_str())
                            .unwrap_or("/tmp/cine_rec_tables_fixture_map.json")
                            .to_string();
                        env.push(("CINE_REC_SCHEMA_MAP".to_string(), path.clone()));
                        let text = serde_json::to_string(&body).unwrap();
                        let reader = move |p: &str| -> std::io::Result<String> {
                            if p == path {
                                Ok(text.clone())
                            } else {
                                Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    "no such file",
                                ))
                            }
                        };
                        SchemaMap::from_env_pairs(&env, &reader)
                    }
                    None => SchemaMap::from_env_pairs(&env, &|_p| {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "no such file",
                        ))
                    }),
                }
            }
            other => panic!("unknown install kind {other}"),
        };

        match (map, expect.get("error")) {
            (Err(_), Some(_)) => { /* decision agrees: rejected */ }
            (Err(e), None) => panic!("{name}: rejected ({e}), Python accepted"),
            (Ok(_), Some(_)) => panic!("{name}: accepted, Python rejected"),
            (Ok(m), None) => {
                let want = expect["active"].as_object().unwrap();
                let got = m.active_map();
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{name}: active map differs: {got:?} vs {want:?}"
                );
                for (k, v) in want {
                    assert_eq!(
                        got.get(k).map(String::as_str),
                        Some(v.as_str().unwrap()),
                        "{name}: active[{k}]"
                    );
                }
                for probe in expect["names"].as_array().unwrap() {
                    let logical = probe[0].as_str().unwrap();
                    let want = &probe[1];
                    match m.name(logical) {
                        Ok(got) => {
                            assert_eq!(got, want.as_str().unwrap(), "{name}: name({logical})")
                        }
                        Err(_) => assert!(
                            want.get("error").is_some(),
                            "{name}: name({logical}) errored, Python succeeded"
                        ),
                    }
                }
                for probe in expect["resolves"].as_array().unwrap() {
                    let template = probe[0].as_str().unwrap();
                    let want = &probe[1];
                    match m.resolve(template) {
                        Ok(got) => {
                            assert_eq!(got, want.as_str().unwrap(), "{name}: resolve({template:?})")
                        }
                        Err(_) => assert!(
                            want.get("error").is_some(),
                            "{name}: resolve({template:?}) errored, Python succeeded"
                        ),
                    }
                }
            }
        }
    }
    assert!(decided >= 25, "only {decided} decisions exercised");
}
