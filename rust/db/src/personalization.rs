//! User statistics layer (spec v0.2 §B.3, §C, §G) — the DB plumbing
//! port of `user_stats.py`. Owns every write to user_title_stats /
//! user_stats / user_genre_stats:
//!
//! * [`record_event`] — ONE transaction: raw event + title-stats
//!   recompute + w_i + `user_stats.last_event_at` (acceptance §J).
//! * [`refresh_user_stats`] — full §C recompute (profile + genres).
//! * [`recompute_weights`] / [`nightly_recompute`] — §G.2 recency.
//! * [`record_feedback`] / [`apply_skip_decay`] — §B.2 explicit signals.
//!
//! The raw events are the source of truth; every derived table here can
//! be rebuilt from them at any time.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::executor::{row, Bind, Db, DbError};
use crate::weights::{self, SeriesSignals};

/// An episode counts as watched at ≥ 50% (`EPISODE_THRESHOLD_RATIO`).
pub const EPISODE_THRESHOLD_RATIO: f64 = 0.50;

/// The §A event contract. `watched_at` is an ISO-8601 timestamptz
/// string, bound through an explicit `::timestamptz` cast.
#[derive(Debug, Clone, Default)]
pub struct WatchEvent {
    pub user_id: i64,
    pub tmdb_id: i64,
    pub media_type: String,
    pub watched_at: String,
    pub watched_sec: Option<i64>,
    pub duration_sec: Option<i64>,
    pub pause_count: Option<i64>,
    pub completed: Option<bool>,
    pub season: Option<i64>,
    pub episode: Option<i64>,
    pub last_position_sec: Option<i64>,
}

fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

const UPSERT_TITLE_STATS: &str = r#"
    INSERT INTO {t_user_title_stats} (
        user_id, tmdb_id, media_type, sessions, total_watched_sec, max_ratio,
        pause_count_total, rewatch_count, last_watched_at, episodes_watched,
        last_season, last_episode, last_ep_watched_sec, last_ep_duration_sec,
        dropped, w_item, updated_at
    )
    SELECT
        $1, $2, $3,
        COUNT(*),
        COALESCE(SUM(e.watched_sec), 0),
        MAX(CASE WHEN COALESCE(e.duration_sec, 0) > 0
                 THEN e.watched_sec::float / e.duration_sec END),
        COALESCE(SUM(e.pause_count), 0),
        -- rewatch = a session of a unit already completed earlier
        -- (same season/episode for tv; the movie itself for movies) —
        -- continuing to the NEXT episode is progress, never a rewatch.
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM {t_user_watch_events} p
            WHERE p.user_id = e.user_id AND p.tmdb_id = e.tmdb_id
              AND p.media_type = e.media_type AND p.completed
              AND p.watched_at < e.watched_at
              AND p.season IS NOT DISTINCT FROM e.season
              AND p.episode IS NOT DISTINCT FROM e.episode)),
        MAX(e.watched_at),
        COUNT(DISTINCT CASE WHEN e.completed
              OR (COALESCE(e.duration_sec,0) > 0
                  AND e.watched_sec::float / e.duration_sec >= $4)
              THEN (e.season, e.episode) END),
        last_ev.season,
        last_ev.episode,
        last_ev.watched_sec,
        last_ev.duration_sec,
        FALSE, 0, now()
    FROM {t_user_watch_events} e
    LEFT JOIN LATERAL (
        SELECT h.season, h.episode, h.watched_sec, h.duration_sec
        FROM {t_user_watch_events} h
        WHERE h.user_id = e.user_id AND h.tmdb_id = e.tmdb_id
          AND h.media_type = e.media_type
        ORDER BY h.watched_at DESC LIMIT 1
    ) last_ev ON true
    WHERE e.user_id = $1 AND e.tmdb_id = $2 AND e.media_type = $3
    GROUP BY last_ev.season, last_ev.episode,
             last_ev.watched_sec, last_ev.duration_sec
    ON CONFLICT (user_id, tmdb_id, media_type) DO UPDATE SET
        sessions = EXCLUDED.sessions,
        total_watched_sec = EXCLUDED.total_watched_sec,
        max_ratio = EXCLUDED.max_ratio,
        pause_count_total = EXCLUDED.pause_count_total,
        rewatch_count = EXCLUDED.rewatch_count,
        last_watched_at = EXCLUDED.last_watched_at,
        episodes_watched = EXCLUDED.episodes_watched,
        last_season = EXCLUDED.last_season,
        last_episode = EXCLUDED.last_episode,
        last_ep_watched_sec = EXCLUDED.last_ep_watched_sec,
        last_ep_duration_sec = EXCLUDED.last_ep_duration_sec,
        dropped = EXCLUDED.dropped,
        w_item = EXCLUDED.w_item,
        updated_at = now()
"#;

/// The w_i + dropped pair a recompute produces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecomputeOutcome {
    pub w_item: f64,
    pub dropped: bool,
}

/// `record_event` — append one watch event and update everything it
/// touches (§G.1), all in ONE transaction: raw event + title-stats
/// recompute + w_i + `user_stats.last_event_at` (§J). Vector rebuild is
/// the caller's business — never on the player's hot path.
pub async fn record_event(db: &Db, event: &WatchEvent) -> Result<(), DbError> {
    // §A required fields — loud ValueError in Python, loud error here
    if event.user_id == 0
        || event.tmdb_id == 0
        || event.media_type.is_empty()
        || event.watched_at.is_empty()
    {
        return Err(DbError::Invalid(format!(
            "watch event missing fields: user_id={}, tmdb_id={}, media_type={:?}, watched_at={:?}",
            event.user_id, event.tmdb_id, event.media_type, event.watched_at
        )));
    }

    let insert_sql = db.sql(
        r#"
                INSERT INTO {t_user_watch_events} (
                    user_id, tmdb_id, media_type, watched_at, watched_sec,
                    duration_sec, pause_count, completed, season, episode,
                    last_position_sec
                ) VALUES ($1,$2,$3,$4::timestamptz,$5,$6,$7,$8,$9,$10,$11)
                "#,
    )?;
    let bump_sql = db.sql(
        r#"
                INSERT INTO {t_user_stats} (user_id, last_event_at, updated_at)
                VALUES ($1, $2::timestamptz, now())
                ON CONFLICT (user_id) DO UPDATE SET
                    last_event_at = EXCLUDED.last_event_at, updated_at = now()
                "#,
    )?;

    let mut tx = db.pool().begin().await?;
    {
        let mut q = sqlx::query(&insert_sql);
        q = q
            .bind(event.user_id)
            .bind(event.tmdb_id)
            .bind(event.media_type.clone())
            .bind(event.watched_at.clone())
            .bind(event.watched_sec.unwrap_or(0))
            .bind(event.duration_sec)
            .bind(event.pause_count.unwrap_or(0))
            .bind(event.completed.unwrap_or(false))
            .bind(event.season)
            .bind(event.episode)
            .bind(event.last_position_sec);
        q.execute(&mut *tx).await?;

        recompute_on_conn(&mut tx, db, event.user_id, event.tmdb_id, &event.media_type).await?;

        sqlx::query(&bump_sql)
            .bind(event.user_id)
            .bind(event.watched_at.clone())
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// `_recompute_title_stats` — rebuild one user_title_stats row from the
/// raw events + feedback, then store w_i/dropped. Runs INSIDE the
/// caller's transaction when one is open.
async fn recompute_on_conn(
    conn: &mut sqlx::postgres::PgConnection,
    db: &Db,
    user_id: i64,
    tmdb_id: i64,
    media_type: &str,
) -> Result<Option<RecomputeOutcome>, DbError> {
    let stats_sql = db.sql(UPSERT_TITLE_STATS)?;
    sqlx::query(&stats_sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .bind(EPISODE_THRESHOLD_RATIO)
        .execute(&mut *conn)
        .await?;

    let fetch_sql = db.sql(
        r#"SELECT uts.*, COALESCE(
               (SELECT TRUE FROM {t_user_feedback} f
                WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                  AND f.media_type = uts.media_type AND f.kind = 'favorite'), FALSE)
           AS favorite,
           EXISTS (SELECT 1 FROM {t_user_feedback} f
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                     AND f.media_type = uts.media_type AND f.kind = 'dislike')
           AS disliked,
           EXTRACT(EPOCH FROM uts.last_watched_at)::float8 AS last_epoch
           FROM {t_user_title_stats} uts
           WHERE uts.user_id = $1 AND uts.tmdb_id = $2 AND uts.media_type = $3"#,
    )?;
    let row = match sqlx::query(&fetch_sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .fetch_optional(&mut *conn)
        .await?
    {
        Some(r) => r,
        None => return Ok(None),
    };

    let total_eps = if media_type == "tv" {
        catalog_episode_count(conn, db, tmdb_id).await?
    } else {
        0
    };

    let episodes_watched = row::opt_i64(&row, "episodes_watched")?.unwrap_or_default();
    let last_ep_watched = row::opt_i64(&row, "last_ep_watched_sec")?.unwrap_or_default();
    let last_ep_duration = row::opt_i64(&row, "last_ep_duration_sec")?.unwrap_or_default();
    let series = SeriesSignals {
        episodes_watched,
        total_episodes: total_eps,
        last_ep_ratio: if last_ep_watched > 0 && last_ep_duration > 0 {
            Some(last_ep_watched as f64 / last_ep_duration as f64)
        } else {
            None
        },
        last_ep_watched_sec: last_ep_watched,
        total_watched_sec: row::opt_i64(&row, "total_watched_sec")?.unwrap_or_default(),
    };
    let w = weights::w_item(
        media_type,
        row::opt_f64(&row, "max_ratio")?,
        row::opt_i64(&row, "pause_count_total")?.unwrap_or_default(),
        Some(&series),
        row::opt_f64(&row, "last_epoch")?,
        now_epoch(),
        row::opt_i64(&row, "rewatch_count")?.unwrap_or_default(),
        row::opt_bool(&row, "favorite")?.unwrap_or(false),
        false,
        row::opt_bool(&row, "disliked")?.unwrap_or(false),
    );
    // §D.1 only — an explicit dislike zeroes w_item but is NOT a
    // drop-off (movies_dropped counts genuine quit-early, §C.1)
    let dropped = weights::is_dropped(
        media_type,
        row::opt_f64(&row, "max_ratio")?,
        series.episodes_watched,
        series.last_ep_ratio,
    );

    let update_sql = db.sql(
        r#"UPDATE {t_user_title_stats}
           SET w_item = $4, dropped = $5, updated_at = now()
           WHERE user_id = $1 AND tmdb_id = $2 AND media_type = $3"#,
    )?;
    sqlx::query(&update_sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .bind(w)
        .bind(dropped)
        .execute(&mut *conn)
        .await?;
    Ok(Some(RecomputeOutcome { w_item: w, dropped }))
}

/// `_catalog_episode_count` — total episodes for a series from the
/// catalog (0 when unknown).
async fn catalog_episode_count(
    conn: &mut sqlx::postgres::PgConnection,
    db: &Db,
    tmdb_id: i64,
) -> Result<i64, DbError> {
    let sql = db.sql("SELECT number_of_episodes FROM {t_tmdb_tv} WHERE id = $1")?;
    match sqlx::query(&sql).bind(tmdb_id).fetch_one(conn).await {
        Ok(r) => Ok(row::opt_i64(&r, "number_of_episodes")?.unwrap_or_default()),
        Err(_) => Ok(0), // unknown series — Python catches Exception too
    }
}

/// `refresh_user_stats` — §C full recompute of user_stats +
/// user_genre_stats for one user (one transaction).
pub async fn refresh_user_stats(db: &Db, user_id: i64) -> Result<(), DbError> {
    let profile_sql = db.sql(
        r#"
                INSERT INTO {t_user_stats} (
                    user_id, titles_touched, titles_weighted,
                    movies_completed, movies_dropped,
                    series_started, series_hooked, series_abandoned_mid_ep,
                    rewatch_titles, explicit_dislikes, explicit_favorites,
                    watchlist_open,
                    total_watch_sec_30d, total_watch_sec_all,
                    movie_share_30d, tv_share_30d,
                    median_session_sec, avg_pauses_per_unfinished_movie,
                    updated_at
                )
                SELECT
                    $1,
                    COUNT(*),
                    COUNT(*) FILTER (WHERE w_item > 0),
                    COUNT(*) FILTER (WHERE media_type='movie'
                                     AND max_ratio >= 0.75),
                    COUNT(*) FILTER (WHERE media_type='movie' AND dropped),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND episodes_watched >= 1),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND episodes_watched >= 3),
                    COUNT(*) FILTER (WHERE media_type='tv'
                                     AND last_ep_watched_sec > 0
                                     AND COALESCE(
                                         last_ep_watched_sec::float
                                         / NULLIF(last_ep_duration_sec,0), 1)
                                         < 0.20),
                    COUNT(*) FILTER (WHERE rewatch_count >= 1),
                    (SELECT COUNT(*) FROM {t_user_feedback} f
                     WHERE f.user_id = $1 AND f.kind = 'dislike'),
                    (SELECT COUNT(*) FROM {t_user_feedback} f
                     WHERE f.user_id = $1 AND f.kind = 'favorite'),
                    (SELECT COUNT(*) FROM {t_user_feedback} f
                     WHERE f.user_id = $1 AND f.kind = 'watchlist'
                       AND NOT EXISTS (
                           SELECT 1 FROM {t_user_title_stats} uts
                           WHERE uts.user_id = $1 AND uts.tmdb_id = f.tmdb_id
                             AND uts.media_type = f.media_type)),
                    COALESCE(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0),
                    COALESCE(SUM(total_watched_sec), 0),
                    COALESCE((
                        SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'
                            AND media_type='movie')
                        ::float / NULLIF(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0)), 0),
                    COALESCE((
                        SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'
                            AND media_type='tv')
                        ::float / NULLIF(SUM(total_watched_sec) FILTER (WHERE last_watched_at > now() - interval '30 days'), 0)), 0),
                    (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY watched_sec)
                     FROM {t_user_watch_events} WHERE user_id = $1)::int,
                    COALESCE((
                        SUM(uts.pause_count_total) FILTER (WHERE uts.media_type='movie' AND uts.max_ratio < 0.75)
                        ::float / NULLIF(COUNT(*) FILTER (WHERE uts.media_type='movie' AND uts.max_ratio < 0.75), 0)), 0),
                    now()
                FROM {t_user_title_stats} uts
                WHERE uts.user_id = $1
                ON CONFLICT (user_id) DO UPDATE SET
                    titles_touched = EXCLUDED.titles_touched,
                    titles_weighted = EXCLUDED.titles_weighted,
                    movies_completed = EXCLUDED.movies_completed,
                    movies_dropped = EXCLUDED.movies_dropped,
                    series_started = EXCLUDED.series_started,
                    series_hooked = EXCLUDED.series_hooked,
                    series_abandoned_mid_ep = EXCLUDED.series_abandoned_mid_ep,
                    rewatch_titles = EXCLUDED.rewatch_titles,
                    explicit_dislikes = EXCLUDED.explicit_dislikes,
                    explicit_favorites = EXCLUDED.explicit_favorites,
                    watchlist_open = EXCLUDED.watchlist_open,
                    total_watch_sec_30d = EXCLUDED.total_watch_sec_30d,
                    total_watch_sec_all = EXCLUDED.total_watch_sec_all,
                    movie_share_30d = EXCLUDED.movie_share_30d,
                    tv_share_30d = EXCLUDED.tv_share_30d,
                    median_session_sec = EXCLUDED.median_session_sec,
                    avg_pauses_per_unfinished_movie =
                        EXCLUDED.avg_pauses_per_unfinished_movie,
                    updated_at = now()
                "#,
    )?;
    let delete_genres_sql = db.sql("DELETE FROM {t_user_genre_stats} WHERE user_id = $1")?;
    let genres_sql = db.sql(
        r#"
                INSERT INTO {t_user_genre_stats} (
                    user_id, media_type, genre_id, watch_sec, weighted_sum,
                    title_count, last_watched_at
                )
                SELECT $1, gm.media_type, gm.genre_id,
                       SUM(uts.total_watched_sec),
                       SUM(uts.w_item),
                       COUNT(*),
                       MAX(uts.last_watched_at)
                FROM {t_user_title_stats} uts
                JOIN {t_tmdb_media_genres} gm
                  ON gm.media_id = uts.tmdb_id AND gm.media_type = uts.media_type
                WHERE uts.user_id = $1
                GROUP BY gm.media_type, gm.genre_id
                "#,
    )?;

    let mut tx = db.pool().begin().await?;
    sqlx::query(&profile_sql)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(&delete_genres_sql)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(&genres_sql)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

/// `recompute_weights` — recency moves → recompute w_i for every title
/// of one user (§G.2).
pub async fn recompute_weights(db: &Db, user_id: i64) -> Result<u64, DbError> {
    let fetch_sql = db.sql(
        r#"SELECT uts.*, COALESCE((SELECT TRUE FROM {t_user_feedback} f
               WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                 AND f.media_type = uts.media_type AND f.kind='favorite'),
               FALSE) AS favorite,
           EXISTS (SELECT 1 FROM {t_user_feedback} f
                   WHERE f.user_id = uts.user_id AND f.tmdb_id = uts.tmdb_id
                     AND f.media_type = uts.media_type AND f.kind='dislike')
           AS disliked,
           EXTRACT(EPOCH FROM uts.last_watched_at)::float8 AS last_epoch
           FROM {t_user_title_stats} uts WHERE uts.user_id = $1"#,
    )?;
    let rows = db
        .fetch(&fetch_sql, &[("user", Bind::I64(user_id))])
        .await?;

    let tv_ids: Vec<i64> = rows
        .iter()
        .filter(|r| {
            row::opt_text(r, "media_type")
                .unwrap_or_default()
                .as_deref()
                == Some("tv")
        })
        .filter_map(|r| row::opt_i64(r, "tmdb_id").ok().flatten())
        .collect();
    let mut ep_counts: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    if !tv_ids.is_empty() {
        let cat_sql =
            db.sql("SELECT id, number_of_episodes FROM {t_tmdb_tv} WHERE id = ANY($1::bigint[])")?;
        for r in db
            .fetch(&cat_sql, &[("ids", Bind::I64Array(tv_ids))])
            .await?
        {
            ep_counts.insert(
                row::opt_i64(&r, "id")?.unwrap_or_default(),
                row::opt_i64(&r, "number_of_episodes")?.unwrap_or_default(),
            );
        }
    }

    let update_sql = db.sql(
        r#"UPDATE {t_user_title_stats} SET w_item = $4, updated_at = now()
           WHERE user_id=$1 AND tmdb_id=$2 AND media_type=$3"#,
    )?;
    let mut tx = db.pool().begin().await?;
    let mut updated = 0u64;
    for r in &rows {
        let media_type = row::opt_text(r, "media_type")?.unwrap_or_default();
        let tmdb_id = row::opt_i64(r, "tmdb_id")?.unwrap_or_default();
        let last_ep_watched = row::opt_i64(r, "last_ep_watched_sec")?.unwrap_or_default();
        let last_ep_duration = row::opt_i64(r, "last_ep_duration_sec")?.unwrap_or_default();
        let series = SeriesSignals {
            episodes_watched: row::opt_i64(r, "episodes_watched")?.unwrap_or_default(),
            total_episodes: *ep_counts.get(&tmdb_id).unwrap_or(&0),
            last_ep_ratio: if last_ep_watched > 0 && last_ep_duration > 0 {
                Some(last_ep_watched as f64 / last_ep_duration as f64)
            } else {
                None
            },
            last_ep_watched_sec: last_ep_watched,
            total_watched_sec: row::opt_i64(r, "total_watched_sec")?.unwrap_or_default(),
        };
        let w = weights::w_item(
            &media_type,
            row::opt_f64(r, "max_ratio")?,
            row::opt_i64(r, "pause_count_total")?.unwrap_or_default(),
            Some(&series),
            row::opt_f64(r, "last_epoch")?,
            now_epoch(),
            row::opt_i64(r, "rewatch_count")?.unwrap_or_default(),
            row::opt_bool(r, "favorite")?.unwrap_or(false),
            false,
            row::opt_bool(r, "disliked")?.unwrap_or(false),
        );
        let res = sqlx::query(&update_sql)
            .bind(user_id)
            .bind(tmdb_id)
            .bind(&media_type)
            .bind(w)
            .execute(&mut *tx)
            .await?;
        updated += res.rows_affected();
    }
    tx.commit().await?;
    Ok(updated)
}

/// `record_feedback` — §B.2 + §G.1: upsert an explicit signal and
/// refresh what it touches. dislike/favorite change w_i; click/skip are
/// recorded as-is (a click is attribution; a skip's bounded nudge is
/// [`apply_skip_decay`]).
pub async fn record_feedback(
    db: &Db,
    user_id: i64,
    tmdb_id: i64,
    media_type: &str,
    kind: &str,
) -> Result<(), DbError> {
    if !matches!(
        kind,
        "dislike" | "favorite" | "watchlist" | "click" | "skip"
    ) {
        return Err(DbError::Invalid(format!(
            "kind must be dislike|favorite|watchlist|click|skip, got {kind:?}"
        )));
    }
    let upsert_sql = db.sql(
        r#"INSERT INTO {t_user_feedback} (user_id, tmdb_id, media_type, kind)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (user_id, tmdb_id, media_type, kind) DO NOTHING"#,
    )?;
    let mut tx = db.pool().begin().await?;
    sqlx::query(&upsert_sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .bind(kind)
        .execute(&mut *tx)
        .await?;

    let has_stats_sql = db.sql(
        "SELECT 1 FROM {t_user_title_stats} \
         WHERE user_id=$1 AND tmdb_id=$2 AND media_type=$3",
    )?;
    let has_stats = sqlx::query(&has_stats_sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .fetch_optional(&mut *tx)
        .await?
        .is_some();
    if has_stats && matches!(kind, "dislike" | "favorite") {
        recompute_on_conn(&mut tx, db, user_id, tmdb_id, media_type).await?;
    }
    tx.commit().await?;
    Ok(())
}

/// `apply_skip_decay` — one skip = a bounded nudge, never a bury: w_item
/// *= factor, floored at `skip_decay_floor`. Returns whether a row was
/// touched (skipping an unseen title is recorded, not weighted).
pub async fn apply_skip_decay(
    db: &Db,
    user_id: i64,
    tmdb_id: i64,
    media_type: &str,
    factor: f64,
    skip_decay_floor: f64,
) -> Result<bool, DbError> {
    // strictly (0, 1) like Python's `0.0 < factor < 1.0` — 0 would
    // bury to the floor (the opposite of "a bounded nudge"), NaN fails
    // both bounds
    if !(factor > 0.0 && factor < 1.0) {
        return Err(DbError::Invalid(format!(
            "skip decay factor must be in (0, 1), got {factor}"
        )));
    }
    let sql = db.sql(
        r#"UPDATE {t_user_title_stats}
           SET w_item = GREATEST(w_item * $4, $5)
           WHERE user_id=$1 AND tmdb_id=$2 AND media_type=$3"#,
    )?;
    let res = sqlx::query(&sql)
        .bind(user_id)
        .bind(tmdb_id)
        .bind(media_type)
        .bind(factor)
        .bind(skip_decay_floor)
        .execute(db.pool())
        .await?;
    Ok(res.rows_affected() > 0)
}

/// `nightly_recompute` result counts.
pub struct NightlyOutcome {
    pub users: usize,
    pub vectors_rebuilt: usize,
}

/// `nightly_recompute` — §G.2 optional fleet batch: weights for active
/// users, stats rebuild, stale-vector rebuild (> max_age). `vector_builder`
/// mirrors the Python injectable (default: user_vector::build_user_vector).
pub async fn nightly_recompute(
    db: &Db,
    active_days: i64,
    vector_max_age_hours: i64,
    emb_column: &str,
) -> Result<NightlyOutcome, DbError> {
    let users_sql = db.sql(
        "SELECT user_id FROM {t_user_stats} \
         WHERE last_event_at > now() - ($1 || ' days')::interval",
    )?;
    let users = db
        .fetch(&users_sql, &[("days", Bind::Text(active_days.to_string()))])
        .await?;
    let mut rebuilt = 0usize;
    for u in &users {
        let user_id = row::opt_i64(u, "user_id")?.unwrap_or_default();
        recompute_weights(db, user_id).await?;
        refresh_user_stats(db, user_id).await?;
        let stale_sql = db.sql(
            r#"SELECT vector_updated_at IS NULL
                    OR vector_updated_at < now() - ($2 || ' hours')::interval
               FROM {t_user_stats} WHERE user_id = $1"#,
        )?;
        let stale = db
            .fetch(
                &stale_sql,
                &[
                    ("user", Bind::I64(user_id)),
                    ("hours", Bind::Text(vector_max_age_hours.to_string())),
                ],
            )
            .await?;
        let is_stale = stale
            .first()
            .and_then(|r| row::opt_bool(r, "?column?").ok().flatten())
            .unwrap_or(false);
        if is_stale {
            crate::user_vector::build_user_vector(db, user_id, None, emb_column).await?;
            rebuilt += 1;
        }
    }
    Ok(NightlyOutcome {
        users: users.len(),
        vectors_rebuilt: rebuilt,
    })
}
