//! Personalization §B–§J live replay: the SAME §A event sequence the
//! Python fixture generator ran (user offset by +1000), then
//! field-level comparison of everything derived — title stats (exact
//! fields exact; w recomputed through the RUST §D pipeline at the
//! fixture's frozen instant and compared at 1e-9), §C counters, genre
//! stats (clock-free fields; weighted_sum is written at event-time
//! `now()` and cannot cross instants), and the user vector BIT-EXACT
//! against Python's build on the same weights.

use std::path::Path;

use cine_rec_db::executor::row;
use cine_rec_db::personalization::{
    apply_skip_decay, catalog_episode_count, record_event, record_feedback, refresh_user_stats,
    title_w, WatchEvent,
};
use cine_rec_db::schema_map::SchemaMap;
use cine_rec_db::user_vector::{aggregate_f64, build_user_vector, top_weighted_seeds};
use cine_rec_db::watched::{
    get_title_rating, get_user_dislikes, get_user_recommendation_exclusions,
    get_user_title_progress, get_user_watched,
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

/// Parse a Python float.hex() string ("[-]0x1.xxxp±n") into f64 —
/// STRICTLY: malformed input panics instead of silently decoding to a
/// wrong number (a misparsed component could squeak past the cosine
/// check; L-a). Exact by construction: the mantissa's hex digits are
/// read as one integer (< 2^53) scaled by a power of two.
fn parse_py_hex(s: &str) -> f64 {
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let body = body
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("parse_py_hex: {s:?}: missing 0x prefix"));
    let (mantissa, exp_s) = body
        .split_once('p')
        .unwrap_or_else(|| panic!("parse_py_hex: {s:?}: missing p exponent"));
    let exp: i32 = exp_s
        .parse()
        .unwrap_or_else(|e| panic!("parse_py_hex: {s:?}: bad exponent {exp_s:?}: {e}"));
    let (int_part, frac) = mantissa
        .split_once('.')
        .unwrap_or_else(|| panic!("parse_py_hex: {s:?}: missing radix point"));
    assert!(
        int_part == "0" || int_part == "1",
        "parse_py_hex: {s:?}: mantissa int part {int_part:?} not 0/1"
    );
    assert!(
        !frac.is_empty() && frac.chars().all(|c| c.is_ascii_hexdigit()),
        "parse_py_hex: {s:?}: bad fraction {frac:?}"
    );
    // int_digits||frac_digits read as a single hex integer, then
    // scaled: mantissa = int.frac = joined / 16^frac_len. The scale is
    // split in two: a single powi underflows to 0 below ~2^-1023, and
    // subnormal hex inputs (Python emits down to 2^-1074) would
    // silently read as 0 — two halves stay representable (L-δ)
    let joined = format!("{int_part}{frac}");
    let mant_int: u64 =
        u64::from_str_radix(&joined, 16).unwrap_or_else(|e| panic!("parse_py_hex: {s:?}: {e}"));
    let shift = exp - 4 * frac.len() as i32;
    let scale = 2.0f64.powi(shift / 2) * 2.0f64.powi(shift - shift / 2);
    let v = (mant_int as f64) * scale;
    assert!(v.is_finite(), "parse_py_hex: {s:?}: non-finite result");
    if neg {
        -v
    } else {
        v
    }
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
    let feedback: Vec<&Value> = fx["feedback"].as_array().unwrap().iter().collect();
    for f in &feedback {
        record_feedback(
            &db,
            user,
            f["tmdb_id"].as_i64().unwrap(),
            f["media_type"].as_str().unwrap(),
            f["kind"].as_str().unwrap(),
        )
        .await
        .unwrap();
    }
    refresh_user_stats(&db, user).await.unwrap();

    // ── title stats: everything exact except w_item (recency of now) ──
    // the SELECT carries the PRODUCTION column shape (favorite/disliked
    // subselects + last_epoch) so `title_w` reads what production reads
    let rows = db
        .fetch(
            &db.sql(
                "SELECT uts.tmdb_id, uts.media_type, uts.sessions, \
                 uts.total_watched_sec, uts.max_ratio, uts.pause_count_total, \
                 uts.rewatch_count, uts.episodes_watched, uts.last_season, \
                 uts.last_episode, uts.last_ep_watched_sec, \
                 uts.last_ep_duration_sec, uts.dropped, uts.w_item, \
                 COALESCE((SELECT TRUE FROM {t_user_feedback} f \
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id \
                     AND f.media_type = uts.media_type AND f.kind = 'favorite'), \
                   FALSE) AS favorite, \
                 EXISTS (SELECT 1 FROM {t_user_feedback} f \
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id \
                     AND f.media_type = uts.media_type AND f.kind = 'dislike') \
                   AS disliked, \
                 EXTRACT(EPOCH FROM uts.last_watched_at)::float8 AS last_epoch \
                 FROM {t_user_title_stats} uts WHERE uts.user_id = $1 \
                 ORDER BY uts.tmdb_id",
            )
            .unwrap(),
            &[("u", cine_rec_db::Bind::I64(user))],
        )
        .await
        .unwrap();
    assert!(!rows.is_empty());
    let py_rows: Vec<&Value> = fx["title_stats"].as_array().unwrap().iter().collect();
    assert_eq!(rows.len(), py_rows.len(), "title row count");
    // M-α meta-guard: the corpus itself must carry each engagement path
    // on its OWN title, or the pins below silently evaporate
    let favorites: std::collections::HashSet<i64> = feedback
        .iter()
        .filter(|f| f["kind"].as_str() == Some("favorite"))
        .filter_map(|f| f["tmdb_id"].as_i64())
        .collect();
    let disliked_fb: std::collections::HashSet<i64> = feedback
        .iter()
        .filter(|f| f["kind"].as_str() == Some("dislike"))
        .filter_map(|f| f["tmdb_id"].as_i64())
        .collect();
    assert!(
        py_rows.iter().any(|p| {
            p["rewatch_count"].as_i64().unwrap_or_default() >= 1
                && !favorites.contains(&p["tmdb_id"].as_i64().unwrap_or_default())
        }),
        "corpus must include a rewatch-ONLY title (M-α)"
    );
    assert!(
        py_rows.iter().any(|p| {
            p["rewatch_count"].as_i64().unwrap_or_default() == 0
                && favorites.contains(&p["tmdb_id"].as_i64().unwrap_or_default())
        }),
        "corpus must include a favorite-ONLY title (M-α)"
    );
    assert!(
        !disliked_fb.is_empty(),
        "corpus must include a disliked title (M-α)"
    );
    // NOTE: injecting ONE engagement constant alone is structurally
    // invisible — the formula is max(REWATCH, FAVORITE) on BOTH sides
    // (Python included), so the other 1.5 always masks the drift. The
    // corpus above pins the FLAG logic (each path on its own title);
    // a BOTH-constants drift is caught by the 1e-9 w gate below.
    //
    // L-ε: pin the PRODUCTION episode-count fetch itself — the exact
    // gate below feeds `title_w` the fixture number, so a broken fetch
    // (wrong table/decode) would otherwise ride only the coarse band
    {
        let tv_id = fx["titles"]["tv"].as_i64().unwrap();
        let want = fx["tv_total_episodes"].as_i64().unwrap_or_default();
        let mut conn = db.pool().acquire().await.unwrap();
        let fetched = catalog_episode_count(&mut conn, &db, tv_id).await.unwrap();
        assert_eq!(fetched, want, "catalog_episode_count vs fixture");
    }
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
        // H-2″: push the row RUST replayed through the PRODUCTION
        // assembly — `personalization::title_w`, the same helper both
        // recompute paths use, so the SeriesSignals construction is
        // exercised, not a test-side copy (design note) — at the
        // fixture's frozen instant, and match Python's w_at_py_now at
        // 1e-9. Corpus coverage (M-α): a rewatch-ONLY title, a
        // favorite-ONLY title, and a disliked title each pin one
        // engagement constant alone, so neither 1.5 can hide behind
        // the max() of the other; disliked pins the w=0 override.
        let s_side = &fx["s_side"][&tmdb_id.to_string()];
        let py_now = fx["python_now_epoch"].as_f64().unwrap();
        let w_at_py_now = s_side["w_at_py_now"].as_f64().unwrap();
        let n_eps = fx["tv_total_episodes"].as_i64().unwrap_or_default();
        let is_tv = row::opt_text(r, "media_type").unwrap().as_deref() == Some("tv");
        let (w_rs, dropped_rs) = title_w(r, if is_tv { n_eps } else { 0 }, py_now).unwrap();
        assert!(
            (w_at_py_now - w_rs).abs() < 1e-9,
            "title {tmdb_id}: python w_at_py_now {w_at_py_now} != rust-pipeline w {w_rs}"
        );
        assert_eq!(
            dropped_rs, dropped,
            "title {tmdb_id}: production is_dropped"
        );
        // the STORED Rust w used its own now — sanity band only
        // (decay ∈ (0,1] → w ≤ base); the exact check above is the gate
        let w: f64 = row::opt_f64(r, "w_item").unwrap().unwrap_or_default();
        let base = s_side["s"].as_f64().unwrap() * s_side["engagement"].as_f64().unwrap();
        if base > 0.0 && w < 1.5 {
            let recency_rs = w / base;
            assert!(
                (0.0..=1.0 + 1e-9).contains(&recency_rs),
                "title {tmdb_id}: rust stored recency {recency_rs} out of band"
            );
        }
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

    // ── genre stats: clock-free fields only ──
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
        // weighted_sum is NOT gated: refresh_user_stats rebuilds it as
        // SUM(w_item) at ITS wall-clock now(), so it decays ~2.3%/day
        // after every regen — a tolerance here just re-arms the 5.2h
        // time bomb this check was (H-2′). The w it is made of is
        // pinned exactly, per title and clock-free, by the H-2″ check
        // above against w_at_py_now.
    }

    // ── vector: stored-build sanity, then BIT-EXACT parity (M-2/L-b) ──
    let vec = build_user_vector(&db, user, None, emb).await.unwrap();
    let want_dim = fx["vector_dim"].as_u64().unwrap() as usize;
    assert_eq!(vec.as_ref().map(|v| v.len()), Some(want_dim), "vector dim");
    let v = vec.unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "vector must be unit-norm: {norm}"
    );
    let py_bits = fx["vector_bits"].as_array().unwrap();
    assert_eq!(v.len(), py_bits.len(), "vector component count");
    let py_vec: Vec<f32> = py_bits
        .iter()
        .map(|h| parse_py_hex(h.as_str().unwrap()) as f32)
        .collect();
    // the stored build used the weights stored at RUST's instant —
    // direction sanity against the fixture vector only (bit equality
    // across different instants is unattainable, recency drifts)
    let dot: f32 = v.iter().zip(&py_vec).map(|(a, b)| a * b).sum();
    assert!(dot > 1.0 - 1e-5, "vector direction diverged: cosine={dot}");

    // BIT-EXACT gate (L-b): re-aggregate the live embeddings with the
    // FROZEN w_at_py_now and match, bit for bit, the fixture's
    // `vector_bits_frozen` — Python's OWN user_vector.aggregate over
    // the same embeddings and weights, summed in the recorded order
    // (`frozen_order`: frozen-w DESC, tmdb_id ASC — the summation
    // order is part of the contract). Attainable and required: same
    // f64 accumulation, and both norms are compensated sums (py_sum ↔
    // CPython ≥3.12 sum()). The stored `vector_bits` above CANNOT be
    // reproduced bit-exactly — it was built from the stored w at a
    // different instant — which is why the fixture carries this
    // frozen twin.
    let emb_sql = db
        .sql(&format!(
            "SELECT uts.tmdb_id, m.emb FROM {{t_user_title_stats}} uts \
             JOIN (SELECT id, 'movie'::text AS media_type, {emb} AS emb \
                   FROM {{t_tmdb_movies}} WHERE {emb} IS NOT NULL \
                   UNION ALL SELECT id, 'tv'::text, {emb} \
                   FROM {{t_tmdb_tv}} WHERE {emb} IS NOT NULL) m \
             ON m.id = uts.tmdb_id AND m.media_type = uts.media_type \
             WHERE uts.user_id = $1 AND uts.w_item > 0",
        ))
        .unwrap();
    let emb_rows = db
        .fetch(&emb_sql, &[("u", cine_rec_db::Bind::I64(user))])
        .await
        .unwrap();
    assert!(!emb_rows.is_empty(), "vector source rows");
    let w_of = |id: i64| {
        fx["s_side"][&id.to_string()]["w_at_py_now"]
            .as_f64()
            .unwrap()
    };
    let mut pairs: Vec<(i64, Vec<f32>)> = emb_rows
        .iter()
        .filter_map(|er| {
            let id = row::opt_i64(er, "tmdb_id").unwrap().unwrap_or_default();
            let emb_v = row::opt_vector(er, "emb").unwrap()?;
            Some((id, emb_v))
        })
        .collect();
    // the recorded summation order: frozen-w DESC, tmdb_id ASC — NOT
    // the (drifted) stored-w order the SQL happened to return
    pairs.sort_by(|a, b| w_of(b.0).total_cmp(&w_of(a.0)).then_with(|| a.0.cmp(&b.0)));
    let got_order: Vec<i64> = pairs.iter().map(|(id, _)| *id).collect();
    let want_order: Vec<i64> = fx["frozen_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(got_order, want_order, "frozen summation order");
    let vectors: Vec<Vec<f32>> = pairs.iter().map(|(_, e)| e.clone()).collect();
    let frozen_ws: Vec<f64> = pairs.iter().map(|(id, _)| w_of(*id)).collect();
    // M-β: compare in f64 — the f32 cast rounds away exactly the
    // sub-f32 differences that separate a compensated norm from a
    // naive fold (measured on this fixture: 381/384 components differ
    // in f64, 0 after rounding to f32), so an f32 comparison could
    // not lock py_sum. Python's aggregate returns f64 components and
    // the fixture stores their hex.
    let agg = aggregate_f64(&vectors, &frozen_ws).expect("aggregate over fixture weights");
    let py_frozen: Vec<f64> = fx["vector_bits_frozen"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| parse_py_hex(h.as_str().unwrap()))
        .collect();
    assert_eq!(agg.len(), py_frozen.len(), "frozen vector component count");
    for (i, x) in agg.iter().enumerate() {
        assert_eq!(
            x.to_bits(),
            py_frozen[i].to_bits(),
            "vector[{i}]: rust-aggregated {x} != python-frozen {}",
            py_frozen[i]
        );
    }

    // ── seeds + exclusions + progress ──
    let seeds = top_weighted_seeds(&db, user, 20).await.unwrap();
    assert!(!seeds.is_empty());
    assert!(seeds[0].2 > 0.0);
    assert!(seeds.iter().all(|(id, _, _)| *id > 0));

    let watched = get_user_watched(&db, user).await.unwrap();
    assert!(!watched.is_empty(), "user_watches view derives from stats");
    let dislikes = get_user_dislikes(&db, user).await.unwrap();
    // the corpus dislikes exactly the disliked titles (M-α) — and a
    // favorite is never a dislike
    let dis_ids: std::collections::HashSet<i64> = dislikes.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        dis_ids, disliked_fb,
        "dislikes = exactly the disliked titles"
    );
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
    // H-1 regression: get_title_rating on an UNRATED title returns
    // None — the common case that used to panic on rows[0]
    assert_eq!(
        get_title_rating(&db, user, 987_654_321, "movie")
            .await
            .unwrap(),
        None,
        "unrated title must be None, never a panic"
    );
    // M-3 regression: factor=0 is rejected (it would bury to the
    // floor — the opposite of "a bounded nudge"), and so is NaN
    assert!(matches!(
        apply_skip_decay(&db, user, movie_id, "movie", 0.0, 0.05).await,
        Err(DbError::Invalid(_))
    ));
    assert!(matches!(
        apply_skip_decay(&db, user, movie_id, "movie", f64::NAN, 0.05).await,
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

/// M-1′: an episodes table that EXISTS but has no row for this title
/// (empty, or just missing it) must not fabricate `reached_last` —
/// Rust's Option ordering makes `Some(x) >= None` true, so the naive
/// port claimed completion. Runs on a dedicated temp schema:
/// user_watches WITH season/episode columns + a controllable episodes
/// table, wired through an explicit SchemaMap (the registry seam, no
/// process-global state).
#[tokio::test]
async fn tv_progress_reached_last_needs_a_known_catalog_end() {
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
    let db = Db::connect(&dsn).await.unwrap();
    let schema = "p4c_prog";
    let (user, tv_id): (i64, i64) = (931_001, 42_424_242);
    for stmt in [
        format!("DROP SCHEMA IF EXISTS {schema} CASCADE"),
        format!("CREATE SCHEMA {schema}"),
        format!(
            "CREATE TABLE {schema}.watches (user_id bigint, tmdb_id bigint, \
             media_type text, season int, episode int)"
        ),
        format!("CREATE TABLE {schema}.episodes (tmdb_id bigint, season int, episode int)"),
    ] {
        db.fetch(&stmt, &[]).await.unwrap();
    }
    // the user watched S1E1..S1E5
    for ep in 1..=5 {
        db.fetch(
            &format!("INSERT INTO {schema}.watches VALUES ($1, $2, 'tv', 1, $3)"),
            &[
                ("u", cine_rec_db::Bind::I64(user)),
                ("t", cine_rec_db::Bind::I64(tv_id)),
                ("e", cine_rec_db::Bind::I64(ep)),
            ],
        )
        .await
        .unwrap();
    }
    let map = SchemaMap::from_mapping([
        ("user_watches".to_string(), format!("{schema}.watches")),
        ("media_episodes".to_string(), format!("{schema}.episodes")),
    ])
    .unwrap();
    let db2 = Db::connect_with_map(&dsn, map).await.unwrap();

    // episodes table exists but is EMPTY: progress is real, completion
    // is NOT claimed
    let p = get_user_title_progress(&db2, user, tv_id, "tv")
        .await
        .unwrap();
    assert!(p.watched, "five seen episodes");
    assert_eq!(p.max_season, Some(1));
    assert_eq!(p.max_episode, Some(5));
    assert!(!p.watched_all, "empty catalog: cat_n = 0");
    assert!(
        !p.reached_last,
        "no known catalog end — reached_last must NOT be fabricated"
    );

    // catalog shorter than progress → both completion claims are real
    for ep in 1..=3 {
        db.fetch(
            &format!("INSERT INTO {schema}.episodes VALUES ($1, 1, $2)"),
            &[
                ("t", cine_rec_db::Bind::I64(tv_id)),
                ("e", cine_rec_db::Bind::I64(ep)),
            ],
        )
        .await
        .unwrap();
    }
    let p = get_user_title_progress(&db2, user, tv_id, "tv")
        .await
        .unwrap();
    assert!(p.watched_all, "5 seen >= 3 catalog episodes");
    assert!(p.reached_last, "seen (1,5) >= catalog end (1,3)");

    // catalog longer than progress → both completion flags fall
    for ep in 4..=8 {
        db.fetch(
            &format!("INSERT INTO {schema}.episodes VALUES ($1, 1, $2)"),
            &[
                ("t", cine_rec_db::Bind::I64(tv_id)),
                ("e", cine_rec_db::Bind::I64(ep)),
            ],
        )
        .await
        .unwrap();
    }
    let p = get_user_title_progress(&db2, user, tv_id, "tv")
        .await
        .unwrap();
    assert!(!p.watched_all && !p.reached_last, "8 catalog eps, 5 seen");

    db.fetch(&format!("DROP SCHEMA {schema} CASCADE"), &[])
        .await
        .unwrap();
}
