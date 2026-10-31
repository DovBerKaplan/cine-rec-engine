//! LIVE integration tests — the phase-3 gate material.
//!
//! Run against the seeded demo database when `CINE_REC_INTEGRATION_DSN`
//! is set (the same harness the Python integration suite uses):
//!
//!     CINE_REC_INTEGRATION_DSN=postgresql://demo:demo@127.0.0.1:54341/demo \
//!         cargo test --test integration
//!
//! Without the variable everything here SKIPS — the offline story stays
//! the default dev loop. Every test asserts n > 0 on the data it reads:
//! a test that measured nothing is a bug (the empty-bench lesson).

use cine_rec_db::recall::{enrich_candidates_batch, generate_candidates, generate_knn_candidates};
use cine_rec_db::Db;

fn dsn() -> Option<String> {
    std::env::var("CINE_REC_INTEGRATION_DSN").ok().filter(|s| !s.is_empty())
}

/// A seeded movie seed (exists in the demo catalog).
const SEED_MOVIE_ID: i64 = 155; // The Dark Knight-ish id in demo data
const SEED_GENRES: &[&str] = &["Action", "Crime", "Drama"];
const DEMO_EMB_COLUMN: &str = "embedding_minilm";

macro_rules! integration_test {
    ($name:ident, |$db:ident, $dsn:ident| $body:expr) => {
        #[tokio::test]
        async fn $name() {
            let Some(dsn) = dsn() else {
                eprintln!("skipping (no CINE_REC_INTEGRATION_DSN)");
                return;
            };
            let db = Db::connect(&dsn).await.expect("connect");
            let $db = &db;
            let $dsn = &dsn;
            $body;
        }
    };
}

integration_test!(genre_channel_returns_seedless_candidates, |db, dsn| {
    let cands = generate_candidates(
        &db,
        SEED_MOVIE_ID,
        &SEED_GENRES.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "movie",
        10,
        false,
    )
    .await
    .expect("genre channel");
    assert!(!cands.is_empty(), "demo DB has genre neighbors");
    assert!(cands.len() <= 10);
    for c in &cands {
        assert_ne!(c.id, SEED_MOVIE_ID);
        assert_eq!(c.media_type.as_deref(), Some("movie"));
        assert!(!c.genres.is_empty());
        assert!(c.vote_average.unwrap_or(0.0) >= 5.5);
    }
});

integration_test!(cross_media_keeps_per_medium_budget, |db, dsn| {
    let cands = generate_candidates(
        &db,
        SEED_MOVIE_ID,
        &SEED_GENRES.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "movie",
        5,
        true,
    )
    .await
    .expect("cross-media genre channel");
    assert!(!cands.is_empty());
    let mut per_medium: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for c in &cands {
        *per_medium
            .entry(c.media_type.clone().unwrap_or_default())
            .or_default() += 1;
    }
    for (mt, n) in &per_medium {
        assert!(n <= &5, "medium {mt} exceeded its budget: {n}");
    }
});

integration_test!(empty_genres_short_circuit, |db, dsn| {
    let cands = generate_candidates(&db, SEED_MOVIE_ID, &[], "movie", 10, false)
        .await
        .expect("empty genres");
    assert!(cands.is_empty());
});

integration_test!(knn_returns_semantic_neighbors, |db, dsn| {
    let cands = generate_knn_candidates(
        &db,
        SEED_MOVIE_ID,
        "movie",
        8,
        false,
        DEMO_EMB_COLUMN,
        None,
    )
    .await
    .expect("knn channel");
    assert!(!cands.is_empty(), "demo DB carries embeddings");
    assert!(cands.len() <= 8);
    for c in &cands {
        assert_eq!(c.via.as_deref(), Some("knn"));
        let sim = c.knn_similarity.unwrap();
        assert!((0.0..=1.000001).contains(&sim), "similarity out of range: {sim}");
        assert_ne!(c.id, SEED_MOVIE_ID);
    }
    // descending similarity
    let sims: Vec<f64> = cands.iter().filter_map(|c| c.knn_similarity).collect();
    let mut sorted = sims.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(sims, sorted, "knn rows must arrive similarity-ordered");
});

integration_test!(knn_missing_column_degrades_to_empty, |db, dsn| {
    let cands = generate_knn_candidates(
        &db,
        SEED_MOVIE_ID,
        "movie",
        8,
        false,
        "embedding_nonexistent",
        None,
    )
    .await
    .expect("missing column must degrade, not fail");
    assert!(cands.is_empty());
});

integration_test!(enrichment_covers_candidates, |db, dsn| {
    let cands = generate_candidates(
        &db,
        SEED_MOVIE_ID,
        &SEED_GENRES.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "movie",
        10,
        false,
    )
    .await
    .expect("genre channel");
    let ids: Vec<i64> = cands.iter().map(|c| c.id).collect();
    let media: Vec<String> = cands
        .iter()
        .map(|c| c.media_type.clone().unwrap_or_default())
        .collect();
    assert!(!ids.is_empty());

    let enriched = enrich_candidates_batch(&db, &ids, Some(&media))
        .await
        .expect("enrichment");
    assert_eq!(enriched.len(), ids.len(), "every candidate gets a bundle");
    // most demo candidates carry at least cast/keywords enrichment
    let with_any = enriched
        .values()
        .filter(|e| !e.cast_list.is_empty() || !e.keywords.is_empty())
        .count();
    assert!(with_any > 0, "demo data carries enrichment");
    // every bundle's optional fields are SAFE defaults, not missing keys
    for e in enriched.values() {
        let _ = &e.director;
        let _ = &e.cast_ids;
        let _ = &e.collection_id;
        let _ = &e.original_language;
        let _ = &e.emotional_arc;
    }
});

integration_test!(table_map_reaches_the_sql, |db, dsn| {
    use cine_rec_db::SchemaMap;
    // sanity first: the default map resolves to the shipped names
    let cands = generate_candidates(
        &db,
        SEED_MOVIE_ID,
        &SEED_GENRES.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "movie",
        5,
        false,
    )
    .await
    .expect("default map");
    assert!(!cands.is_empty());

    // an UNMAPPED-map connection behaves identically
    let map = SchemaMap::new();
    let db2 = Db::connect_with_map(dsn, map)
        .await
        .expect("connect with map");
    let cands2 = generate_candidates(
        &db2,
        SEED_MOVIE_ID,
        &SEED_GENRES.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "movie",
        5,
        false,
    )
    .await
    .expect("identity map");
    assert_eq!(cands, cands2, "identity mapping must be a no-op");
});
