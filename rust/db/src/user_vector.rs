//! User vector construction (spec v0.2 §E) — port of `user_vector.py`.
//!
//! `u_raw = Σ w_i · v_i` over items with w_i > 0 AND an embedding,
//! then L2-normalized. No weighted items → NO vector at all (never a
//! fake zero vector). Multi-persona (§E.2) is deferred exactly like the
//! Python side; the storage is ready.

use crate::executor::{row, Bind, Db, DbError};

/// Weighted sum + L2 normalize. `None` when nothing weighs in.
///
/// Raises on dimension mismatch — a mixed-dimension sum is a
/// configuration error (two encoders in one space), not silence.
pub fn aggregate(vectors: &[Vec<f32>], weights: &[f64]) -> Option<Vec<f32>> {
    if vectors.is_empty() || weights.is_empty() {
        return None;
    }
    let dim = vectors[0].len();
    let mut acc = vec![0.0f64; dim];
    let mut total_w = 0.0f64;
    for (vec, w) in vectors.iter().zip(weights) {
        if *w <= 0.0 {
            continue;
        }
        if vec.len() != dim {
            // parity: Python raises ValueError on dimension mismatch
            return None;
        }
        for (i, x) in vec.iter().enumerate() {
            acc[i] += w * *x as f64;
        }
        total_w += w;
    }
    if total_w <= 0.0 {
        return None;
    }
    let norm = acc.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    Some(acc.iter().map(|x| (*x / norm) as f32).collect())
}

/// `build_user_vector` — rebuild and store the user's vector (persona
/// 0). `None` if not buildable (missing column/extension, or nothing
/// weighs in). Updates `user_stats.vector_updated_at` + persona_count.
///
/// Degrades to `None` on ANY error — never raises on the request path
/// (the Python body catches broadly around the build too).
pub async fn build_user_vector(
    db: &Db,
    user_id: i64,
    space: Option<&str>,
    emb_column: &str,
) -> Result<Option<Vec<f32>>, DbError> {
    // dynamic column + registry names, mirroring the Python f-string
    let sql = format!(
        r#"
                SELECT uts.w_item, m.emb
                FROM {uts_table} uts
                JOIN (
                    SELECT id, 'movie'::text AS media_type, {emb} AS emb
                    FROM {movies} WHERE {emb} IS NOT NULL
                    UNION ALL
                    SELECT id, 'tv'::text, {emb}
                    FROM {tv} WHERE {emb} IS NOT NULL
                ) m ON m.id = uts.tmdb_id AND m.media_type = uts.media_type
                WHERE uts.user_id = $1 AND uts.w_item > 0
                ORDER BY uts.w_item DESC
                "#,
        uts_table = db.map().name("user_title_stats")?,
        emb = emb_column,
        movies = db.map().name("tmdb_movies")?,
        tv = db.map().name("tmdb_tv")?,
    );
    let rows = match db.fetch(&sql, &[("user", Bind::I64(user_id))]).await {
        Ok(r) => r,
        // Python: except UndefinedColumnError -> None; the vector path
        // degrades on any missing piece
        Err(_) => return Ok(None),
    };

    let vectors: Vec<Vec<f32>> = rows
        .iter()
        .filter_map(|r| row::opt_vector(r, "emb").ok().flatten())
        .collect();
    let weights: Vec<f64> = rows
        .iter()
        .filter_map(|r| row::opt_f64(r, "w_item").ok().flatten())
        .collect();
    let vec = aggregate(&vectors, &weights);

    let delete_sql =
        db.sql("DELETE FROM {t_user_vectors} WHERE user_id = $1 AND persona_id = 0")?;
    let reset_sql = db.sql(
        "UPDATE {t_user_stats} SET vector_updated_at = now(), persona_count = 0 \
         WHERE user_id = $1",
    )?;
    let insert_sql = db.sql(
        r#"
            INSERT INTO {t_user_vectors} (user_id, persona_id, space, embedding,
                                      weight, updated_at)
            VALUES ($1, 0, $2, $3, 1.0, now())
            ON CONFLICT (user_id, persona_id, space) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                weight = 1.0,
                updated_at = now()
            "#,
    )?;
    let mark_sql = db.sql(
        "UPDATE {t_user_stats} SET vector_updated_at = now(), persona_count = 1 \
         WHERE user_id = $1",
    )?;

    let Some(vec) = vec else {
        let mut tx = db.pool().begin().await?;
        sqlx::query(&delete_sql)
            .bind(user_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query(&reset_sql)
            .bind(user_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        return Ok(None);
    };

    let mut tx = db.pool().begin().await?;
    sqlx::query(&insert_sql)
        .bind(user_id)
        .bind(space.unwrap_or("default"))
        .bind(pgvector::Vector::from(vec.clone()))
        .execute(&mut *tx)
        .await?;
    sqlx::query(&mark_sql)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(Some(vec))
}

/// `ensure_user_vector` — the stored vector if fresh; rebuild first when
/// absent or stale (the plug-and-play path — the ANN channel works
/// without the nightly cron). Degrades to `None` on any error.
pub async fn ensure_user_vector(
    db: &Db,
    user_id: i64,
    space: Option<&str>,
    emb_column: &str,
    max_age_hours: f64,
) -> Result<Option<Vec<f32>>, DbError> {
    let probe_sql = db.sql(
        r#"SELECT embedding FROM {t_user_vectors}
           WHERE user_id = $1 AND persona_id = 0
             AND space = COALESCE($2, space)
             AND updated_at > now() - ($3 || ' hours')::interval"#,
    )?;
    let rows = match db
        .fetch(
            &probe_sql,
            &[
                ("user", Bind::I64(user_id)),
                ("space", Bind::OptText(space.map(String::from))),
                ("hours", Bind::Text(max_age_hours.to_string())),
            ],
        )
        .await
    {
        Ok(r) => r,
        // A failed probe means the vector layer is unusable — a rebuild
        // would fail the same way (Python catches Exception here)
        Err(_) => return Ok(None),
    };
    if let Some(r) = rows.first() {
        if let Some(v) = row::opt_vector(r, "embedding").ok().flatten() {
            return Ok(Some(v));
        }
    }
    build_user_vector(db, user_id, space, emb_column).await
}

/// `load_user_vector` — the stored vector for retrieval (None absent).
pub async fn load_user_vector(
    db: &Db,
    user_id: i64,
    space: Option<&str>,
) -> Result<Option<Vec<f32>>, DbError> {
    let sql = db.sql(
        r#"SELECT embedding FROM {t_user_vectors}
           WHERE user_id = $1 AND persona_id = 0
             AND space = COALESCE($2, space)"#,
    )?;
    let rows = db
        .fetch(
            &sql,
            &[
                ("user", Bind::I64(user_id)),
                ("space", Bind::OptText(space.map(String::from))),
            ],
        )
        .await?;
    Ok(rows
        .first()
        .and_then(|r| row::opt_vector(r, "embedding").ok().flatten()))
}

/// `top_weighted_seeds` — highest-w_i (id, media_type, w) tuples, the
/// LTR seed set (§F.1).
pub async fn top_weighted_seeds(
    db: &Db,
    user_id: i64,
    limit: i64,
) -> Result<Vec<(i64, String, f64)>, DbError> {
    let sql = db.sql(
        r#"SELECT tmdb_id, media_type, w_item FROM {t_user_title_stats}
           WHERE user_id = $1 AND w_item > 0
           ORDER BY w_item DESC, last_watched_at DESC LIMIT $2"#,
    )?;
    let rows = db
        .fetch(
            &sql,
            &[("user", Bind::I64(user_id)), ("limit", Bind::I64(limit))],
        )
        .await?;
    rows.iter()
        .map(|r| {
            Ok((
                row::opt_i64(r, "tmdb_id")?.unwrap_or_default(),
                row::opt_text(r, "media_type")?.unwrap_or_default(),
                row::opt_f64(r, "w_item")?.unwrap_or_default(),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_basics() {
        assert!(aggregate(&[], &[]).is_none());
        // zero weights → None
        assert!(aggregate(&[vec![1.0, 0.0]], &[0.0]).is_none());
        // single unit vector → itself
        let v = aggregate(&[vec![3.0, 4.0]], &[1.0]).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        // weighting scales before normalizing: 2:1 mix of orthogonal axes
        let v = aggregate(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[2.0, 1.0]).unwrap();
        assert!((v[0] as f64 - 2.0 / 5f64.sqrt()).abs() < 1e-6);
        // dimension mismatch → None (config error semantics)
        assert!(aggregate(&[vec![1.0, 0.0], vec![1.0, 0.0, 0.0]], &[1.0, 1.0]).is_none());
    }
}
