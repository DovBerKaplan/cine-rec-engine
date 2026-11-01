//! Per-user watch history for recommendation filtering — port of
//! `watched.py` (minus the Redis cache layer: the Rust side always
//! reads fresh; caching is a service-phase concern).
//!
//! The engine never recommends a title the requesting user already
//! watched. Exclusion keys are (tmdb_id, media_type) pairs — numeric
//! ids collide between movies and series in TMDB.

use std::collections::HashSet;

use crate::executor::{row, Bind, Db, DbError};

/// (tmdb_id, media_type) — the exclusion key.
pub type Key = (i64, String);

async fn fetch_keys(db: &Db, sql_template: String, user_id: i64) -> Result<HashSet<Key>, DbError> {
    // degrade: a history lookup must never break a recommendation —
    // Python catches Exception and returns an empty set
    match db
        .fetch(&sql_template, &[("user", Bind::I64(user_id))])
        .await
    {
        Ok(rows) => Ok(rows
            .iter()
            .filter_map(|r| {
                let id = row::opt_i64(r, "tmdb_id").ok().flatten()?;
                let mt = row::opt_text(r, "media_type").ok().flatten()?;
                Some((id, mt))
            })
            .collect()),
        Err(_) => Ok(HashSet::new()),
    }
}

/// `get_user_watched` — (tmdb_id, media_type) pairs the user already
/// watched. Empty on failure (never breaks a recommendation).
pub async fn get_user_watched(db: &Db, user_id: i64) -> Result<HashSet<Key>, DbError> {
    const SQL: &str = "SELECT DISTINCT tmdb_id::bigint, media_type \
         FROM {t_user_watches} WHERE user_id = $1";
    fetch_keys(db, db.sql(SQL)?, user_id).await
}

/// `get_user_rated` — pairs the user rated (any rating).
pub async fn get_user_rated(db: &Db, user_id: i64) -> Result<HashSet<Key>, DbError> {
    const SQL: &str = "SELECT tmdb_id::bigint AS tmdb_id, media_type \
         FROM {t_title_ratings} WHERE user_id = $1";
    fetch_keys(db, db.sql(SQL)?, user_id).await
}

/// `get_user_dislikes` — explicit dislikes, hard-filtered from every
/// recommendation (§F.2).
pub async fn get_user_dislikes(db: &Db, user_id: i64) -> Result<HashSet<Key>, DbError> {
    const SQL: &str = "SELECT tmdb_id::bigint, media_type FROM {t_user_feedback} \
         WHERE user_id = $1 AND kind = 'dislike'";
    fetch_keys(db, db.sql(SQL)?, user_id).await
}

/// `get_user_recommendation_exclusions` — watched OR rated: a rated
/// title never returns, no matter which button was pressed.
pub async fn get_user_recommendation_exclusions(
    db: &Db,
    user_id: i64,
) -> Result<HashSet<Key>, DbError> {
    let mut all = get_user_watched(db, user_id).await?;
    all.extend(get_user_rated(db, user_id).await?);
    Ok(all)
}

/// `get_title_rating` — the user's current rating ('love'/'like'/
/// 'dislike') or None.
pub async fn get_title_rating(
    db: &Db,
    user_id: i64,
    tmdb_id: i64,
    media_type: &str,
) -> Result<Option<String>, DbError> {
    let sql = db.sql(
        "SELECT rating FROM {t_title_ratings} \
         WHERE user_id = $1 AND tmdb_id = $2 AND media_type = $3",
    )?;
    let rows = db
        .fetch(
            &sql,
            &[
                ("user", Bind::I64(user_id)),
                ("id", Bind::I64(tmdb_id)),
                ("mt", Bind::Text(media_type.to_string())),
            ],
        )
        .await;
    let Ok(rows) = rows else {
        return Ok(None); // read failure degrades, never breaks a request
    };
    // no row = the user never rated this title (the COMMON case) —
    // Python returns None; a panic here would break every card render
    let Some(r) = rows.first() else {
        return Ok(None);
    };
    Ok(row::opt_text(r, "rating").ok().flatten())
}

/// `get_user_title_progress` — the user's watch state for ONE title.
/// movie → watched; tv → watched_all / reached_last / max_season /
/// max_episode. Degrades to `watched: false` on any read failure.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TitleProgress {
    pub watched: bool,
    pub watched_all: bool,
    pub reached_last: bool,
    pub max_season: Option<i64>,
    pub max_episode: Option<i64>,
}

pub async fn get_user_title_progress(
    db: &Db,
    user_id: i64,
    tmdb_id: i64,
    media_type: &str,
) -> Result<TitleProgress, DbError> {
    let out = TitleProgress::default();
    if media_type == "tv" {
        let sql = db.sql(
            r#"
                WITH seen AS (
                    SELECT uw.season, uw.episode
                    FROM {t_user_watches} uw
                    WHERE uw.user_id = $1
                      AND uw.tmdb_id = $2::bigint
                      AND uw.media_type = 'tv'
                      AND uw.season >= 1 AND uw.episode >= 1
                    GROUP BY uw.season, uw.episode
                ),
                catalog AS (
                    SELECT MAX(e.season) AS cat_s,
                           MAX(e.episode) AS cat_e,
                           COUNT(DISTINCT (e.season, e.episode)) AS cat_n
                    FROM {t_media_episodes} e
                    WHERE e.tmdb_id = $2::bigint
                      AND e.season >= 1 AND e.episode >= 1
                )
                SELECT
                    (SELECT season FROM seen ORDER BY season DESC, episode DESC LIMIT 1) AS max_s,
                    (SELECT episode FROM seen ORDER BY season DESC, episode DESC LIMIT 1) AS max_e,
                    (SELECT COUNT(*) FROM seen) AS seen_n,
                    catalog.cat_s, catalog.cat_e, catalog.cat_n
                FROM catalog
                "#,
        )?;
        let Ok(rows) = db
            .fetch(
                &sql,
                &[("user", Bind::I64(user_id)), ("id", Bind::I64(tmdb_id))],
            )
            .await
        else {
            return Ok(out); // integrator episodes table missing — degrade
        };
        let Some(r) = rows.first() else {
            return Ok(out);
        };
        // Python wraps the WHOLE row decode in `except Exception →
        // return out` — any decode failure degrades to the default
        // progress and never breaks a card render (L-2). A missing
        // column reads as None (dict.get semantics) which lands in the
        // same place; a wrong TYPE degrades instead of erroring.
        let parsed: Result<Option<TitleProgress>, sqlx::Error> = (|| {
            let seen_n = row::opt_i64(r, "seen_n")?.unwrap_or_default();
            if seen_n == 0 {
                return Ok(None);
            }
            let max_s = row::opt_i64(r, "max_s")?;
            let max_e = row::opt_i64(r, "max_e")?;
            let cat_s = row::opt_i64(r, "cat_s")?;
            let cat_e = row::opt_i64(r, "cat_e")?;
            let cat_n = row::opt_i64(r, "cat_n")?.unwrap_or_default();
            Ok(Some(TitleProgress {
                watched: true,
                watched_all: seen_n >= cat_n && cat_n > 0,
                // reached_last claims the catalog's LAST episode was
                // seen — that needs a KNOWN catalog end. Rust's Option
                // ordering makes `Some(x) >= None` TRUE, so the naive
                // port fabricated completion whenever the episodes
                // table exists but is empty (or lacks this title) —
                // same explicit call the Python mirror made (M-1).
                reached_last: matches!((cat_s, cat_e), (Some(_), Some(_)))
                    && (max_s, max_e) >= (cat_s, cat_e),
                max_season: max_s,
                max_episode: max_e,
            }))
        })();
        return match parsed {
            Ok(Some(p)) => Ok(p),
            // nothing seen OR a decode failure — default progress,
            // never an error
            _ => Ok(out),
        };
    }
    let sql = db.sql(
        "SELECT EXISTS (SELECT 1 FROM {t_user_watches} \
         WHERE user_id = $1 AND tmdb_id = $2::bigint AND media_type = 'movie') AS watched",
    )?;
    let Ok(rows) = db
        .fetch(
            &sql,
            &[("user", Bind::I64(user_id)), ("id", Bind::I64(tmdb_id))],
        )
        .await
    else {
        return Ok(out);
    };
    Ok(TitleProgress {
        watched: rows
            .first()
            .and_then(|r| row::opt_bool(r, "watched").ok().flatten())
            .unwrap_or(false),
        ..out
    })
}
