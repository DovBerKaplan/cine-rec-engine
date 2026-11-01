//! Live smoke for the phase-3 tail (info.rs): the four functions the
//! find_similar compute path calls. Catches SQL-shape and decode
//! errors early; full field parity belongs to the phase-5 harness.

use cine_rec_db::info::{
    apply_recall_filters, fetch_cinematic_tags, get_movie_info_batch,
    semantic_overview_similarity_batch, RecallFilters,
};
use cine_rec_db::Db;

async fn db() -> Db {
    let dsn = std::env::var("CINE_REC_INTEGRATION_DSN").unwrap_or_else(|_| {
        eprintln!("skipping (no CINE_REC_INTEGRATION_DSN)");
        std::process::exit(0);
    });
    Db::connect(&dsn).await.unwrap()
}

#[tokio::test]
async fn seed_info_and_cinematic_smoke() {
    let db = db().await;
    // two known demo titles from the personalization fixture: 11 (movie)
    // and 43 (tv)
    let infos = get_movie_info_batch(&db, &[11, 43], None).await.unwrap();
    assert!(infos.contains_key(&11) && infos.contains_key(&43), "{:?}", infos.keys());
    let m = &infos[&11];
    assert_eq!(m.media_type, "movie");
    assert!(!m.title.is_empty());
    assert!(!m.genres.is_empty(), "movie 11 has genres");
    // media_type-filtered variant: asking for tv only drops the movie
    let tv_only = get_movie_info_batch(&db, &[11, 43], Some("tv")).await.unwrap();
    assert!(tv_only.contains_key(&43) && !tv_only.contains_key(&11));

    let tags = fetch_cinematic_tags(&db, &[11, 43]).await.unwrap();
    // demo data may or may not carry cinematic rows — only shape-check
    for (_narr, _pacing, _arc) in tags.values() {}
}

#[tokio::test]
async fn recall_filters_smoke() {
    let db = db().await;
    let cands: Vec<(i64, String)> = vec![(11, "movie".into()), (43, "tv".into())];
    // empty filters → everything eligible, no query needed
    let out = apply_recall_filters(&db, &cands, &RecallFilters::default())
        .await
        .unwrap();
    assert_eq!(out.eligible.len(), 2);
    assert!(!out.degraded);

    // a genre filter: find one real genre id first (any genre of title 11)
    let rows = db
        .fetch(
            &db.sql("SELECT g.id AS gid FROM {t_tmdb_media_genres} mg \
                    JOIN {t_tmdb_genres} g ON g.id = mg.genre_id \
                    WHERE mg.media_id = 11 LIMIT 1")
                .unwrap(),
            &[],
        )
        .await
        .unwrap();
    let gid = cine_rec_db::executor::row::opt_i64(&rows[0], "gid").unwrap().unwrap();
    let f = RecallFilters {
        genre_ids: Some(vec![gid]),
        ..Default::default()
    };
    let out = apply_recall_filters(&db, &cands, &f).await.unwrap();
    assert!(!out.degraded, "demo schema has the genre map tables");
    assert!(out.eligible.contains(&(11, "movie".to_string())));
}

#[tokio::test]
async fn semantic_batch_smoke() {
    let db = db().await;
    // the demo default column: check it exists on tmdb_movies first
    let rows = db
        .fetch(
            "SELECT column_name FROM information_schema.columns \
             WHERE table_name = 'tmdb_movies' AND column_name LIKE 'embedding%' \
             ORDER BY column_name LIMIT 5",
            &[],
        )
        .await
        .unwrap();
    let cols: Vec<String> = rows
        .iter()
        .filter_map(|r| {
            cine_rec_db::executor::row::opt_text(r, "column_name")
                .ok()
                .flatten()
        })
        .collect();
    let emb = cols
        .iter()
        .find(|c| c.as_str() == "embedding")
        .or_else(|| cols.first())
        .expect("some embedding column on tmdb_movies");
    eprintln!("semantic smoke column: {emb}");

    let sims = semantic_overview_similarity_batch(
        &db,
        &[11],
        &[43],
        None,
        emb,
        None,
    )
    .await
    .unwrap();
    // 11 and 43 both carry embeddings in the demo data (the phase-4
    // vector fixture built from them); shape-check only here
    for cmap in sims.values() {
        assert!(!cmap.is_empty());
        for v in cmap.values() {
            assert!((0.0..=1.0).contains(v), "cosine in range: {v}");
        }
    }
}
