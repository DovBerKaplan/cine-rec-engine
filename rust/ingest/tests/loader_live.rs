//! LIVE loader parity: the Rust loader vs the REAL ingest/loader.py
//! on the SAME database, in two schemas — the same pattern as
//! rust/db's live_parity (field-level, not just counts).
//!
//! Gated by `CINE_REC_S3_DSN` (plus optional `CINE_REC_S3_PY`, the
//! venv python; defaults to <repo>/.venv/bin/python). The test:
//!
//! 1. runs fixtures/gen_loader_live.py — the real TmdbIngest with a
//!    per-case patched _fetch_title — into schema `pyload`;
//! 2. applies docs/schema.sql into schema `rsload` and runs the RUST
//!    loader over the SAME plan (payload + gate parameters), with all
//!    20 loader tables remapped through SchemaMap (the registry
//!    working as an integrator would use it);
//! 3. compares every loader-touched table with symmetric EXCEPT
//!    (updated_at/synced_at excluded — now() differs by construction)
//!    and the stats counters field-for-field.
//!
//! The plan includes failure paths (bind failures, gate rejections, a
//! credited person without an id) — both sides must reach the SAME
//! partial state and the SAME counters, never just "no error".

use std::path::{Path, PathBuf};
use std::process::Command;

use cine_rec_db::{Db, SchemaMap};
use cine_rec_ingest::loader::{TitleFetcher, TmdbIngest};
use serde_json::Value;
use sqlx::Row;

/// Every table the loader touches, with its comparison columns
/// (now()-columns excluded).
const COMPARE: &[(&str, &str)] = &[
    ("tmdb_movies", "id,title,original_title,original_language,overview,release_date,runtime,status,adult,vote_average,vote_count,popularity,poster_path,backdrop_path,collection_id,imdb_id,budget,revenue"),
    ("tmdb_tv", "id,name,original_name,original_language,overview,first_air_date,last_air_date,status,in_production,number_of_seasons,number_of_episodes,adult,vote_average,vote_count,popularity,poster_path,backdrop_path,imdb_id"),
    ("tmdb_people", "id,name"),
    ("tmdb_keywords", "id,name"),
    ("tmdb_production_companies", "id,name"),
    ("tmdb_networks", "id,name"),
    ("tmdb_movie_genres", "id,name"),
    ("tmdb_tv_genres", "id,name"),
    ("tmdb_movie_genres_map", "movie_id,genre_id"),
    ("tmdb_tv_genres_map", "tv_id,genre_id"),
    ("tmdb_movie_keywords", "movie_id,keyword_id"),
    ("tmdb_tv_keywords", "tv_id,keyword_id"),
    ("tmdb_movie_companies", "movie_id,company_id"),
    ("tmdb_tv_companies", "tv_id,company_id"),
    ("tmdb_tv_networks_map", "tv_id,network_id"),
    ("tmdb_movie_cast", "movie_id,person_id,character,cast_order"),
    ("tmdb_tv_cast", "tv_id,person_id,character,cast_order"),
    ("tmdb_movie_crew", "movie_id,person_id,job,department"),
    ("tmdb_tv_crew", "tv_id,person_id,job,department"),
    ("tmdb_recommendations", "media_id,media_type,rec_media_id,rec_media_type,rank,popularity"),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

struct OnePayload(Option<Value>);

impl TitleFetcher for OnePayload {
    fn fetch_title(
        &mut self,
        _media_id: i64,
        _medium: &str,
    ) -> impl std::future::Future<Output = Option<Value>> + Send {
        let payload = self.0.clone();
        async move { payload }
    }
}

fn fixtures() -> Value {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut payloads: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("loader_payloads.json")).unwrap())
            .unwrap();
    let stats_path = std::env::temp_dir().join("cine_s3_loader_live_stats.json");
    payloads.as_object_mut().unwrap().insert(
        "__stats_path".into(),
        Value::String(stats_path.to_string_lossy().into()),
    );
    payloads
}

#[tokio::test]
async fn loader_table_state_matches_python_live() {
    let Some(dsn) = std::env::var("CINE_REC_S3_DSN")
        .ok()
        .filter(|s| !s.is_empty())
    else {
        eprintln!("skipping: CINE_REC_S3_DSN not set");
        return;
    };
    let fixtures = fixtures();
    let stats_path = PathBuf::from(fixtures["__stats_path"].as_str().unwrap());

    // 1. python side: the REAL loader into `pyload`.
    let python = std::env::var("CINE_REC_S3_PY").unwrap_or_else(|_| {
        repo_root()
            .join(".venv/bin/python")
            .to_string_lossy()
            .into()
    });
    let gen = repo_root().join("rust/ingest/tests/fixtures/gen_loader_live.py");
    let out = Command::new(&python)
        .arg(&gen)
        .arg("--dsn")
        .arg(&dsn)
        .arg("--out")
        .arg(&stats_path)
        .output()
        .expect("failed to run the python generator");
    assert!(
        out.status.success(),
        "python generator failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let live: Value = serde_json::from_str(&std::fs::read_to_string(&stats_path).unwrap()).unwrap();
    let plan = live["plan"].as_array().unwrap();
    assert!(plan.len() >= 15, "suspiciously small plan");

    // 2. rust side: fresh `rsload` schema + the schema file, applied
    //    in one simple-protocol batch so view bodies bind to rsload.
    let setup = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&dsn)
        .await
        .unwrap();
    let schema_sql = std::fs::read_to_string(repo_root().join("docs/schema.sql")).unwrap();
    let mut conn = setup.acquire().await.unwrap();
    sqlx::query("DROP SCHEMA IF EXISTS rsload CASCADE")
        .execute(&mut *conn)
        .await
        .unwrap();
    sqlx::query("CREATE SCHEMA rsload")
        .execute(&mut *conn)
        .await
        .unwrap();
    sqlx::raw_sql(&format!("SET search_path TO rsload;\n{schema_sql}"))
        .execute(&mut *conn)
        .await
        .unwrap();
    drop(conn);
    setup.close().await;

    // 3. the Rust loader with every table remapped to rsload.* — the
    //    registry path an integrator takes.
    let map = SchemaMap::from_mapping(
        COMPARE
            .iter()
            .map(|(table, _)| (table.to_string(), format!("rsload.{table}"))),
    )
    .unwrap();
    let db = Db::connect_with_map(&dsn, map).await.unwrap();
    let ingest = TmdbIngest::new(db, Some("fixture-key"), 40).unwrap();

    for step in plan {
        let name = step["name"].as_str().unwrap();
        let medium = step["medium"].as_str().unwrap();
        let min_pop = step["min_popularity"].as_f64().unwrap();
        let include_adult = step["include_adult"].as_bool().unwrap();
        let fetch_none = step["fetch_none"].as_bool().unwrap_or(false);
        let payload = fixtures["payloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            // fetch_none steps are synthetic — no payload needed.
            .map(|c| c["payload"].clone())
            .unwrap_or_else(|| {
                assert!(fetch_none, "payload {name} missing and not fetch_none");
                Value::Null
            });
        let media_id = payload
            .get("id")
            .and_then(|v| cine_rec_ingest::pyval::py_int(v).ok())
            .unwrap_or(0);
        let mut fetcher = OnePayload(if fetch_none { None } else { Some(payload) });
        ingest
            .ingest_one(&mut fetcher, media_id, medium, min_pop, include_adult)
            .await;
    }

    // 4. stats — field for field.
    let want = live["stats"].as_object().unwrap();
    let got = ingest.stats();
    for (k, v) in want {
        let actual = match k.as_str() {
            "attempted" => got.attempted,
            "ingested" => got.ingested,
            "filtered_adult" => got.filtered_adult,
            "filtered_popularity" => got.filtered_popularity,
            "failed" => got.failed,
            "complete" => got.complete,
            other => panic!("unknown stat {other}"),
        };
        assert_eq!(
            actual,
            v.as_i64().unwrap(),
            "stats.{k} differs (rust {actual} vs python {})",
            v.as_i64().unwrap()
        );
    }
    assert!(got.ingested > 0, "nothing was ingested — empty measurement");

    // 5. table states — symmetric EXCEPT per table, plus non-empty
    //    checks where the plan guarantees rows.
    let cmp_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(2)
        .connect(&dsn)
        .await
        .unwrap();

    // H-1 INVARIANT pin (the EXCEPT alone cannot catch a bug BOTH
    // sides share): the plan ingests movie 155 AFTER tv 155 — the
    // series' networks must STILL be there afterwards. Pre-fix, both
    // sides deleted them (identical, silently wrong).
    for schema in ["pyload", "rsload"] {
        let n: i64 = sqlx::query(&format!(
            "SELECT count(*)::bigint FROM {schema}.tmdb_tv_networks_map WHERE tv_id = 155"
        ))
        .fetch_one(&cmp_pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert_eq!(
            n, 2,
            "{schema}: tv 155 networks must survive movie 155 (H-1)"
        );
    }

    let mut non_empty_seen = 0;
    for (table, cols) in COMPARE {
        let sql = format!(
            "SELECT count(*) AS n FROM (\n\
             (SELECT {cols} FROM pyload.{table} EXCEPT SELECT {cols} FROM rsload.{table})\n\
             UNION ALL\n\
             (SELECT {cols} FROM rsload.{table} EXCEPT SELECT {cols} FROM pyload.{table})) d"
        );
        let row = sqlx::query(&sql).fetch_one(&cmp_pool).await.unwrap();
        let diff: i64 = row.try_get("n").unwrap();
        assert_eq!(diff, 0, "table {table} diverged ({diff} differing rows)");

        let counts = sqlx::query(&format!(
            "SELECT (SELECT count(*) FROM pyload.{table}) AS py, \
             (SELECT count(*) FROM rsload.{table}) AS rs"
        ))
        .fetch_one(&cmp_pool)
        .await
        .unwrap();
        let py: i64 = counts.try_get("py").unwrap();
        let rs: i64 = counts.try_get("rs").unwrap();
        assert_eq!(py, rs, "table {table} count differs");
        if py > 0 {
            non_empty_seen += 1;
        }
    }
    // The plan must have produced a real catalog, not an empty one.
    // The specific list matters more than a raw count: movie-155-again
    // legitimately EMPTIES movie-side bridges by mirroring (its
    // minimal payload deletes 155's cast/crew/keywords — correct on
    // both sides), so the count alone would be brittle.
    assert!(
        non_empty_seen >= 15,
        "only {non_empty_seen} non-empty tables — empty measurement"
    );
    for must in [
        "tmdb_movies",
        "tmdb_tv",
        "tmdb_people",
        "tmdb_keywords",
        "tmdb_production_companies",
        "tmdb_networks",
        "tmdb_recommendations",
        "tmdb_tv_cast",
        "tmdb_tv_crew",
        "tmdb_tv_genres",
        "tmdb_tv_genres_map",
        "tmdb_tv_networks_map",
    ] {
        let n: i64 = sqlx::query(&format!(
            "SELECT (SELECT count(*) FROM pyload.{must}) + \
             (SELECT count(*) FROM rsload.{must})"
        ))
        .fetch_one(&cmp_pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert!(n > 0, "{must} is empty on both sides — empty measurement");
    }
    cmp_pool.close().await;
}
