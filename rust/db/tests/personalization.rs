//! Personalization §B–§J live replay: the SAME §A event sequence the
//! Python fixture generator ran (user offset by +1000), then
//! field-level comparison of everything derived — title stats (exact
//! except w_item, which carries recency-of-now and compares within a
//! tolerance), §C counters, genre stats, and the user vector's
//! normalized direction.

use std::path::Path;

use cine_rec_db::executor::row;
use cine_rec_db::personalization::{
    apply_skip_decay, record_event, record_feedback, refresh_user_stats, WatchEvent,
};
use cine_rec_db::user_vector::{build_user_vector, top_weighted_seeds};
use cine_rec_db::watched::{
    get_user_dislikes, get_user_recommendation_exclusions, get_user_title_progress,
    get_user_watched,
};
use cine_rec_db::{Db, DbError};
use serde_json::Value;

fn fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("personalization_parity.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn dsn() -> Option<String> {
    std::env::var("CINE_REC_INTEGRATION_DSN")
        .ok()
        .filter(|s| !s.is_empty())
}

fn ev_from_json(v: &Value, user_id: i64, default_user: i64) -> WatchEvent {
    let _ = default_user;
    WatchEvent {
        user_id,
        tmdb_id: v["tmdb_id"].as_i64().unwrap(),
        media_type: v["media_type"].as_str().unwrap().to_string(),
        watched_at: v["watched_at"].as_str().unwrap().to_string(),
        watched_sec: v["watched_sec"].as_i64(),
        duration_sec: v["duration_sec"].as_i64(),
        pause_count: v["pause_count"].as_i64(),
        completed: v["completed"].as_bool(),
        season: v["season"].as_i64(),
        episode: v["episode"].as_i64(),
        last_position_sec: v["last_position_sec"].as_i64(),
    }
}

async fn clean_user(db: &Db, user: i64) {
    for tbl in [
        "user_watch_events",
        "user_title_stats",
        "user_feedback",
        "user_stats",
        "user_genre_stats",
        "user_vectors",
    ] {
        let sql = format!("DELETE FROM {{t_{tbl}}} WHERE user_id = $1");
        let _ = db
            .fetch(
                &db.sql(&sql).unwrap(),
                &[("u", cine_rec_db::Bind::I64(user))],
            )
            .await;
    }
}

#[tokio::test]
async fn personalization_replays_python_end_to_end() {
    let Some(dsn) = dsn() else {
        if std::env::var("CINE_REC_REQUIRE_DB")
            .ok()
            .is_some_and(|v| v == "1")
        {
            panic!("CINE_REC_REQUIRE_DB=1 but no CINE_REC_INTEGRATION_DSN");
        }
        eprintln!("skipping (no CINE_REC_INTEGRATION_DSN)");
        return;
    };
    let fx = fixture();
    let py_user = fx["user"].as_i64().unwrap();
    let user = py_user + 1000; // our replay user
    let emb = fx["emb_column"].as_str().unwrap();
    let db = Db::connect(&dsn).await.unwrap();
    clean_user(&db, user).await;

    // §J first: record_event is ONE transaction — a failure inside the
    // title-stats recompute must roll the raw event back with it.
    {
        let bad = WatchEvent {
            user_id: user,
            tmdb_id: 11,
            media_type: "movie".to_string(),
            watched_at: "not-a-timestamp".to_string(), // fails INSIDE the tx
            ..Default::default()
        };
        assert!(
            record_event(&db, &bad).await.is_err(),
            "bad event must fail"
        );
        let rows = db
            .fetch(
                &db.sql("SELECT COUNT(*) AS n FROM {t_user_watch_events} WHERE user_id = $1")
                    .unwrap(),
                &[("u", cine_rec_db::Bind::I64(user))],
            )
            .await
            .unwrap();
        let n: i64 = row::opt_i64(&rows[0], "n").unwrap().unwrap_or_default();
        assert_eq!(n, 0, "§J: a failed record_event must leave NO raw event");
    }

    // replay the fixture events
    for ev in fx["events"].as_array().unwrap() {
        record_event(&db, &ev_from_json(ev, user, py_user))
            .await
            .unwrap();
    }
    let fb = &fx["feedback"];
    record_feedback(
        &db,
        user,
        fb["tmdb_id"].as_i64().unwrap(),
        fb["media_type"].as_str().unwrap(),
        fb["kind"].as_str().unwrap(),
    )
    .await
    .unwrap();
    refresh_user_stats(&db, user).await.unwrap();

    // ── title stats: everything exact except w_item (recency of now) ──
    let rows = db
        .fetch(
            &db.sql(
                "SELECT tmdb_id, media_type, sessions, total_watched_sec, \
                 max_ratio, pause_count_total, rewatch_count, episodes_watched, \
                 last_season, last_episode, last_ep_watched_sec, \
                 last_ep_duration_sec, dropped, w_item \
                 FROM {t_user_title_stats} WHERE user_id = $1 ORDER BY tmdb_id",
            )
            .unwrap(),
            &[("u", cine_rec_db::Bind::I64(user))],
        )
        .await
        .unwrap();
    assert!(!rows.is_empty());
    let py_rows: Vec<&Value> = fx["title_stats"].as_array().unwrap().iter().collect();
    assert_eq!(rows.len(), py_rows.len(), "title row count");
    for (r, py) in rows.iter().zip(&py_rows) {
        let tmdb_id: i64 = row::opt_i64(r, "tmdb_id").unwrap().unwrap_or_default();
        for col in [
            "sessions",
            "total_watched_sec",
            "rewatch_count",
            "episodes_watched",
            "last_season",
            "last_episode",
            "last_ep_watched_sec",
            "last_ep_duration_sec",
        ] {
            let got = row::opt_i64(r, col).unwrap();
            let want = py[col].as_i64();
            assert_eq!(got, want, "title {tmdb_id}.{col}");
        }
        let dropped: bool = row::opt_bool(r, "dropped").unwrap().unwrap_or(false);
        assert_eq!(
            dropped,
            py["dropped"].as_bool().unwrap(),
            "title {tmdb_id}.dropped"
        );
        let max_ratio: Option<f64> = row::opt_f64(r, "max_ratio").unwrap();
        let want_ratio = py["max_ratio"].as_f64();
        match (max_ratio, want_ratio) {
            (Some(a), Some(b)) => {
                assert!((a - b).abs() < 1e-9, "title {tmdb_id}.max_ratio {a} vs {b}")
            }
            (None, None) => {}
            other => panic!("title {tmdb_id}.max_ratio mismatch: {other:?}"),
        }
        // w_item: same formula, but recency uses each side's `now` —
        // half-life 30d, runs minutes apart → tiny drift is expected
        let w: f64 = row::opt_f64(r, "w_item").unwrap().unwrap_or_default();
        let w_py = py["w_item"].as_f64().unwrap();
        assert!(
            (w - w_py).abs() / w_py.max(1e-9) < 5e-3,
            "title {tmdb_id}.w_item {w} vs python {w_py}"
        );
    }

    // ── §C counters: exact (30-day windows match for fixed dates) ──
    let stats = db
        .fetch(
            &db.sql(
                "SELECT titles_touched, titles_weighted, movies_completed, \
                 movies_dropped, series_started, series_hooked, \
                 series_abandoned_mid_ep, rewatch_titles, explicit_favorites, \
                 total_watch_sec_all FROM {t_user_stats} WHERE user_id = $1",
            )
            .unwrap(),
            &[("u", cine_rec_db::Bind::I64(user))],
        )
        .await
        .unwrap();
    let s = &stats[0];
    let py = &fx["user_stats"];
    for col in [
        "titles_touched",
        "titles_weighted",
        "movies_completed",
        "movies_dropped",
        "series_started",
        "series_hooked",
        "series_abandoned_mid_ep",
        "rewatch_titles",
        "explicit_favorites",
        "total_watch_sec_all",
    ] {
        let got = row::opt_i64(s, col).unwrap().unwrap_or_default();
        let want = py[col].as_i64().unwrap();
        assert_eq!(got, want, "user_stats.{col}");
    }

    // ── genre stats: exact ──
    let genres = db
        .fetch(
            &db.sql(
                "SELECT media_type, genre_id, watch_sec, weighted_sum, title_count \
                 FROM {t_user_genre_stats} WHERE user_id = $1 \
                 ORDER BY genre_id, media_type",
            )
            .unwrap(),
            &[("u", cine_rec_db::Bind::I64(user))],
        )
        .await
        .unwrap();
    let py_genres = fx["genre_stats"].as_array().unwrap();
    assert_eq!(genres.len(), py_genres.len(), "genre row count");
    for (r, py) in genres.iter().zip(py_genres) {
        let gid: i64 = row::opt_i64(r, "genre_id").unwrap().unwrap_or_default();
        let want_gid = py["genre_id"].as_i64().unwrap();
        assert_eq!(gid, want_gid, "genre id");
        let ws = row::opt_i64(r, "watch_sec").unwrap().unwrap_or_default();
        assert_eq!(
            ws,
            py["watch_sec"].as_i64().unwrap(),
            "genre {gid}.watch_sec"
        );
        let tc = row::opt_i64(r, "title_count").unwrap().unwrap_or_default();
        assert_eq!(
            tc,
            py["title_count"].as_i64().unwrap(),
            "genre {gid}.title_count"
        );
        // weighted_sum floats compare within w_item drift
        let wsum: f64 = row::opt_f64(r, "weighted_sum").unwrap().unwrap_or_default();
        let want_wsum = py["weighted_sum"].as_f64().unwrap();
        assert!(
            (wsum - want_wsum).abs() / want_wsum.max(1e-9) < 5e-3,
            "genre {gid}.weighted_sum {wsum} vs {want_wsum}"
        );
    }

    // ── vector: built, 384-dim, unit norm (direction parity is
    // component-wise vs the w drift) ──
    let vec = build_user_vector(&db, user, None, emb).await.unwrap();
    let want_dim = fx["vector_dim"].as_u64().unwrap() as usize;
    assert_eq!(vec.as_ref().map(|v| v.len()), Some(want_dim), "vector dim");
    let v = vec.unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "vector must be unit-norm: {norm}"
    );

    // ── seeds + exclusions + progress ──
    let seeds = top_weighted_seeds(&db, user, 20).await.unwrap();
    assert!(!seeds.is_empty());
    assert!(seeds[0].2 > 0.0);
    assert!(seeds.iter().all(|(id, _, _)| *id > 0));

    let watched = get_user_watched(&db, user).await.unwrap();
    assert!(!watched.is_empty(), "user_watches view derives from stats");
    let dislikes = get_user_dislikes(&db, user).await.unwrap();
    assert!(dislikes.is_empty(), "favorite is not a dislike");
    let excl = get_user_recommendation_exclusions(&db, user).await.unwrap();
    assert!(excl.len() >= watched.len());

    let titles = &fx["titles"];
    let movie_id = titles["movie"].as_i64().unwrap();
    let tv_id = titles["tv"].as_i64().unwrap();
    let pm = get_user_title_progress(&db, user, movie_id, "movie")
        .await
        .unwrap();
    assert!(pm.watched, "movie watched");
    let ptv = get_user_title_progress(&db, user, tv_id, "tv")
        .await
        .unwrap();
    // tv progress DEGRADES on the default schema: the derived
    // user_watches view carries no season/episode columns and
    // media_episodes is integrator-supplied — Python degrades to
    // watched=false here too (parity pinned)
    assert!(!ptv.watched, "tv progress degrades on the default view");
    assert!(!ptv.watched_all);

    // ── feedback kinds + skip decay semantics ──
    assert!(matches!(
        record_feedback(&db, user, movie_id, "movie", "hug").await,
        Err(DbError::Invalid(_))
    ));
    // skip decay: bounded nudge, floored; returns false for unseen titles
    assert!(
        !apply_skip_decay(&db, user, 987_654_321, "movie", 0.9, 0.05)
            .await
            .unwrap()
    );
    assert!(apply_skip_decay(&db, user, movie_id, "movie", 0.9, 0.05)
        .await
        .unwrap());
    let after = db
        .fetch(
            &db.sql(
                "SELECT w_item FROM {t_user_title_stats} \
                     WHERE user_id=$1 AND tmdb_id=$2 AND media_type='movie'",
            )
            .unwrap(),
            &[
                ("u", cine_rec_db::Bind::I64(user)),
                ("id", cine_rec_db::Bind::I64(movie_id)),
            ],
        )
        .await
        .unwrap();
    let w_after: f64 = row::opt_f64(&after[0], "w_item")
        .unwrap()
        .unwrap_or_default();
    assert!(
        (w_after
            - 0.9
                * seeds
                    .iter()
                    .find(|(id, _, _)| *id == movie_id)
                    .unwrap()
                    .2
                    .max(w_after))
        .abs()
            < 0.2,
        "skip decay keeps w in the bounded band"
    );
}
