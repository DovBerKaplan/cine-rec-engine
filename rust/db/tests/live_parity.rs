//! FIELD-LEVEL parity against the Python queries, replayed from a
//! fixture the real `queries.py` generated against the seeded demo DB
//! (`tests/fixtures/gen_live_parity.py`). This is the test class the
//! phase-3 review demanded: ids, ORDER, and every field must agree.

use std::path::Path;

use cine_rec_db::recall::{enrich_candidates_batch, generate_candidates, generate_knn_candidates};
use cine_rec_db::recall2::{
    fetch_embeddings_batch, generate_director_candidates, generate_director_candidates_batch,
    generate_discovery_candidates, generate_popular_candidates, generate_tmdb_rec_candidates,
    generate_tmdb_rec_candidates_batch, generate_user_vector_candidates,
};
use cine_rec_db::CandidateRow;
use serde_json::Value;

fn fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("live_parity_cases.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn dsn() -> Option<String> {
    std::env::var("CINE_REC_INTEGRATION_DSN")
        .ok()
        .filter(|s| !s.is_empty())
}

fn genres(v: &[Value]) -> Vec<String> {
    v.iter().map(|g| g.as_str().unwrap().to_string()).collect()
}

/// Field-by-field comparison of one Rust row against the Python dict.
fn assert_row_matches(py: &Value, rs: &CandidateRow, ctx: &str) {
    let get = |k: &str| py.get(k);
    assert_eq!(
        rs.id,
        get("id").and_then(Value::as_i64).unwrap_or_default(),
        "{ctx}: id"
    );
    assert_eq!(
        rs.title.as_deref(),
        get("title").and_then(Value::as_str),
        "{ctx}: title"
    );
    assert_eq!(
        rs.title_en.as_deref(),
        get("title_en").and_then(Value::as_str),
        "{ctx}: title_en"
    );
    assert_eq!(
        rs.media_type.as_deref(),
        get("media_type").and_then(Value::as_str),
        "{ctx}: media_type"
    );
    let f_eq = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(
        f_eq(
            rs.vote_average.unwrap_or(0.0),
            get("vote_average").and_then(Value::as_f64).unwrap_or(0.0)
        ),
        "{ctx}: vote_average {:?} vs {:?}",
        rs.vote_average,
        get("vote_average")
    );
    assert_eq!(
        rs.vote_count,
        get("vote_count").and_then(Value::as_i64),
        "{ctx}: vote_count — INT4 decode regression?"
    );
    assert!(
        f_eq(
            rs.popularity.unwrap_or(0.0),
            get("popularity").and_then(Value::as_f64).unwrap_or(0.0)
        ),
        "{ctx}: popularity"
    );
    assert_eq!(
        rs.collection_id,
        get("collection_id").and_then(Value::as_i64),
        "{ctx}: collection_id"
    );
    assert_eq!(
        rs.release_year,
        get("release_year").and_then(Value::as_i64),
        "{ctx}: release_year — INT4 decode regression?"
    );
    assert_eq!(
        rs.genres,
        get("genres")
            .and_then(Value::as_array)
            .map(|a| a
                .iter()
                .filter_map(|g| g.as_str().map(String::from))
                .collect::<Vec<String>>())
            .unwrap_or_default(),
        "{ctx}: genres"
    );
    assert_eq!(
        rs.via.as_deref(),
        get("via").and_then(Value::as_str),
        "{ctx}: via"
    );
    assert_eq!(
        rs.tmdb_rank,
        get("tmdb_rank").and_then(Value::as_i64),
        "{ctx}: tmdb_rank"
    );
    if let Some(py_sim) = get("knn_similarity").and_then(Value::as_f64) {
        let rs_sim = rs.knn_similarity.unwrap_or(0.0);
        assert!(
            (rs_sim - py_sim).abs() < 1e-5,
            "{ctx}: knn_similarity {rs_sim} vs {py_sim}"
        );
    }
}

fn assert_list(py: &Value, rs: &[CandidateRow], label: &str) {
    let py_rows = py.as_array().unwrap();
    assert_eq!(
        rs.len(),
        py_rows.len(),
        "{label}: row count {} vs {}",
        rs.len(),
        py_rows.len()
    );
    for (i, (p, r)) in py_rows.iter().zip(rs).enumerate() {
        assert_row_matches(p, r, &format!("{label}[{i}]"));
    }
}

#[tokio::test]
async fn live_parity_replays_python_fields() {
    let Some(dsn) = dsn() else {
        eprintln!("skipping (no CINE_REC_INTEGRATION_DSN)");
        return;
    };
    let fx = fixture();
    let seeds = &fx["seeds"];
    let emb = fx["emb_column"].as_str().unwrap();
    let db = cine_rec_db::Db::connect(&dsn).await.unwrap();
    let cases = &fx["cases"];

    let movie_id = seeds["movie_id"].as_i64().unwrap();
    let tv_id = seeds["tv_id"].as_i64().unwrap();
    let mg = genres(seeds["movie_genres"].as_array().unwrap());
    let _tg = genres(seeds["tv_genres"].as_array().unwrap());
    let dir_seed = seeds["dir_seed"].as_i64().unwrap();
    let rec_seed = seeds["rec_seed"].as_array().unwrap();
    let rec_id = rec_seed[0].as_i64().unwrap();
    let rec_mt = rec_seed[1].as_str().unwrap();
    let uvec: Vec<f32> = seeds["user_vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    // genre channel (popularity + quality slices, both arms)
    let same = generate_candidates(&db, movie_id, &mg, "movie", 30, false)
        .await
        .unwrap();
    assert_list(&cases["genre_same"], &same, "genre_same");

    let cross = generate_candidates(&db, movie_id, &mg, "movie", 30, true)
        .await
        .unwrap();
    assert_list(&cases["genre_cross"], &cross, "genre_cross");

    // knn (same/cross/tv — the tv floor regression H-4 shows here)
    let knn_m = generate_knn_candidates(&db, movie_id, "movie", 20, false, emb, None)
        .await
        .unwrap();
    assert_list(&cases["knn_movie"], &knn_m, "knn_movie");
    let knn_t = generate_knn_candidates(&db, tv_id, "tv", 20, false, emb, None)
        .await
        .unwrap();
    assert_list(&cases["knn_tv"], &knn_t, "knn_tv");
    let knn_x = generate_knn_candidates(&db, movie_id, "movie", 20, true, emb, None)
        .await
        .unwrap();
    assert_list(&cases["knn_cross"], &knn_x, "knn_cross");

    // tmdb recs (single + batch equivalence vs PYTHON, not just self)
    let tm = generate_tmdb_rec_candidates(&db, rec_id, rec_mt, 20)
        .await
        .unwrap();
    assert_list(&cases["tmdb_single"], &tm, "tmdb_single");
    let tb = generate_tmdb_rec_candidates_batch(&db, &[(rec_id, rec_mt.to_string())], 20)
        .await
        .unwrap();
    assert_list(&cases["tmdb_batch"], &tb[0], "tmdb_batch");

    // auteur (via=director regression H-2 shows here)
    let dr = generate_director_candidates(&db, dir_seed, "movie", 15)
        .await
        .unwrap();
    assert_list(&cases["director_single"], &dr, "director_single");
    let db_ = generate_director_candidates_batch(&db, &[(dir_seed, "movie".to_string())], 15)
        .await
        .unwrap();
    assert_list(&cases["director_batch"], &db_[0], "director_batch");

    // vector channels
    let uv = generate_user_vector_candidates(&db, &uvec, 10, emb)
        .await
        .unwrap();
    assert_list(&cases["uvec"], &uv, "uvec");
    let disc = generate_discovery_candidates(&db, &uvec, &[1, 2], 10, emb)
        .await
        .unwrap();
    assert_list(&cases["discovery"], &disc, "discovery");

    // popular page rows (keys + fields)
    let pop = generate_popular_candidates(&db, 8, None, None)
        .await
        .unwrap();
    let py_pop = cases["popular"].as_array().unwrap();
    assert_eq!(pop.len(), py_pop.len(), "popular: row count");
    for (i, (p, r)) in py_pop.iter().zip(&pop).enumerate() {
        assert_eq!(
            r.tmdb_id,
            p["tmdb_id"].as_i64().unwrap(),
            "popular[{i}].tmdb_id"
        );
        assert_eq!(
            r.title.as_deref(),
            p["title"].as_str(),
            "popular[{i}].title"
        );
        assert_eq!(
            r.title_en.as_deref(),
            p["title_en"].as_str(),
            "popular[{i}].title_en — PageRow key parity"
        );
        assert_eq!(
            r.vote_count,
            p["vote_count"].as_i64(),
            "popular[{i}].vote_count"
        );
        assert_eq!(
            r.release_year,
            p["release_year"].as_i64(),
            "popular[{i}].release_year"
        );
    }

    // embeddings batch (keys only — float32 exactness already proven)
    let emb_pairs: Vec<(i64, String)> = knn_m
        .iter()
        .take(6)
        .map(|c| (c.id, c.media_type.clone().unwrap_or_default()))
        .collect();
    let embs = fetch_embeddings_batch(&db, &emb_pairs, emb).await.unwrap();
    let got: Vec<String> = {
        let mut v: Vec<String> = embs.keys().map(|(id, mt)| format!("{id}|{mt}")).collect();
        v.sort();
        v
    };
    let want: Vec<String> = cases["embeddings_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    assert_eq!(got, want, "embeddings keys");

    // enrichment (spot fields per candidate)
    let ids: Vec<i64> = same.iter().take(25).map(|c| c.id).collect();
    let mts: Vec<String> = same
        .iter()
        .take(25)
        .map(|c| c.media_type.clone().unwrap_or_default())
        .collect();
    let enr = enrich_candidates_batch(&db, &ids, Some(&mts))
        .await
        .unwrap();
    let py_enr = cases["enrich"].as_object().unwrap();
    assert_eq!(enr.len(), py_enr.len(), "enrich: bundle count");
    for (k, want) in py_enr {
        let id: i64 = k.parse().unwrap();
        let got = enr
            .get(&id)
            .unwrap_or_else(|| panic!("enrich missing {id}"));
        assert_eq!(
            got.director.as_deref(),
            want["director"].as_str(),
            "enrich[{id}].director"
        );
        // director_ids comes from unordered ARRAY_AGG — compared as a
        // SET below together with the other id fields
        // the fixture stores the first 3 names (spot-check budget)
        let want_cast: Vec<String> = want["cast_list"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        assert!(
            got.cast_list.starts_with(&want_cast),
            "enrich[{id}].cast_list {:?} vs {:?}",
            got.cast_list,
            want_cast
        );
        let want_ids: Vec<i64> = want["cast_ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        assert!(got.cast_ids.starts_with(&want_ids), "enrich[{id}].cast_ids");
        // unordered fields compare as SETS — the batch SQL carries no
        // ORDER BY on keywords/companies/networks, so the order is
        // instance-dependent (the review's L-6)
        let s: Vec<&str> = Vec::new();
        let _ = s;
        let str_set = |v: &Value| -> std::collections::BTreeSet<String> {
            v.as_array()
                .unwrap()
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        };
        let i_set = |v: &Value| -> std::collections::BTreeSet<i64> {
            v.as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_i64)
                .collect()
        };
        assert_eq!(
            got.keywords
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            str_set(&want["keywords"]),
            "enrich[{id}].keywords (set)"
        );
        assert_eq!(
            got.companies
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            str_set(&want["companies"]),
            "enrich[{id}].companies (set)"
        );
        assert_eq!(
            got.networks
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            str_set(&want["networks"]),
            "enrich[{id}].networks (set)"
        );
        assert_eq!(
            got.director_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            i_set(&want["director_ids"]),
            "enrich[{id}].director_ids (set)"
        );
        assert_eq!(
            got.writer_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            i_set(&want["writer_ids"]),
            "enrich[{id}].writer_ids (set)"
        );
        assert_eq!(
            got.composer_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            i_set(&want["composer_ids"]),
            "enrich[{id}].composer_ids (set)"
        );
        assert_eq!(
            got.dp_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            i_set(&want["dp_ids"]),
            "enrich[{id}].dp_ids (set)"
        );
        assert_eq!(
            got.collection_id,
            want["collection_id"].as_i64(),
            "enrich[{id}].collection_id"
        );
        assert_eq!(
            got.original_language.as_deref(),
            want["original_language"].as_str(),
            "enrich[{id}].original_language"
        );
    }
}
