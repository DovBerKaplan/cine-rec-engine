//! User vector construction (spec v0.2 §E) — port of `user_vector.py`.
//!
//! `u_raw = Σ w_i · v_i` over items with w_i > 0 AND an embedding,
//! then L2-normalized. No weighted items → NO vector at all (never a
//! fake zero vector). Multi-persona (§E.2) is deferred exactly like the
//! Python side; the storage is ready.

use crate::degrade;
use crate::executor::{row, Bind, Db, DbError};

/// CPython ≥3.12 `sum()` is Neumaier-compensated — every ported float
/// sum must fold the same way (the phase-2 lesson; the norm below is
/// Python's `sum(x * x for x in acc)`).
fn py_sum(items: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for x in items {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    if c != 0.0 {
        sum += c;
    }
    sum
}

/// Weighted sum + L2 normalize, in f64 — Python's `aggregate` returns
/// f64 components; the f32 [`aggregate`] is the stored-vector wrapper.
/// Bit-parity checks must use THIS one (comparing after the f32 cast
/// rounds away exactly the sub-f32 differences that distinguish a
/// compensated norm from a naive fold — review M-β).
pub fn aggregate_f64(vectors: &[Vec<f32>], weights: &[f64]) -> Option<Vec<f64>> {
    if dims_mismatch(vectors) {
        // the caller turns this into an ERR before any write — a mixed
        // dimension sum is a configuration error, not silence (Python
        // raises ValueError before the write block too)
        return None;
    }
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
        for (i, x) in vec.iter().enumerate() {
            acc[i] += w * *x as f64;
        }
        total_w += w;
    }
    if total_w <= 0.0 {
        return None;
    }
    let norm = py_sum(acc.iter().map(|x| x * x)).sqrt();
    if norm == 0.0 {
        return None;
    }
    Some(acc.iter().map(|x| x / norm).collect())
}

/// The stored-vector shape: the f64 aggregation rounded to f32 (the
/// embedding columns store f32; the f64 pass stays the parity target).
pub fn aggregate(vectors: &[Vec<f32>], weights: &[f64]) -> Option<Vec<f32>> {
    aggregate_f64(vectors, weights).map(|v| v.into_iter().map(|x| x as f32).collect())
}

/// True when the vectors carry mixed dimensions (a configuration
/// error — two encoders in one space).
pub fn dims_mismatch(vectors: &[Vec<f32>]) -> bool {
    match vectors.first() {
        Some(first) => vectors.iter().any(|v| v.len() != first.len()),
        None => false,
    }
}

/// `build_user_vector` — rebuild and store the user's vector (persona
/// 0). `None` if not buildable (missing column/extension, or nothing
/// weighs in). Updates `user_stats.vector_updated_at` + persona_count.
///
/// Degrades to `None` ONLY when the embedding column is missing
/// (SQLSTATE 42703 — Python's `UndefinedColumnError`) or nothing weighs
/// in; every other failure propagates as `Err` (the NIGHTLY path in
/// Python lets those rise too — only `ensure_user_vector` wraps them
/// away). A dimension mismatch is `Err` BEFORE any write (the stored
/// vector survives), mirroring Python's ValueError.
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
        // Python catches ONLY UndefinedColumnError here — the embedding
        // column is the optional piece. Any OTHER failure (connection,
        // syntax, permissions) propagates: the nightly cron wants to
        // hear about it; only ensure_user_vector wraps errors away.
        Err(e) => {
            if let DbError::Sqlx(se) = &e {
                if degrade::pg_code(se) == Some("42703") {
                    return Ok(None);
                }
            }
            return Err(e);
        }
    };

    // L-7: paired extraction — one pass over the rows so a decode
    // failure drops BOTH fields of that row (two separate filter_maps
    // could silently misalign weights to the wrong vectors)
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
    let mut weights: Vec<f64> = Vec::with_capacity(rows.len());
    for r in &rows {
        if let (Ok(Some(v)), Ok(Some(w))) = (row::opt_vector(r, "emb"), row::opt_f64(r, "w_item")) {
            vectors.push(v);
            weights.push(w);
        }
    }
    if dims_mismatch(&vectors) {
        // Python raises ValueError BEFORE the write block: the stored
        // vector survives, no delete/marker happens
        return Err(DbError::Invalid(
            "user vector: embedding dimension mismatch".to_string(),
        ));
    }
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
    // never raises on the request path — the rebuild degrades to None
    // on ANY error, exactly like the Python except Exception
    Ok(build_user_vector(db, user_id, space, emb_column)
        .await
        .ok()
        .flatten())
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

    #[test]
    fn py_sum_compensates() {
        // CPython ≥3.12 sum() is Neumaier-compensated — the norm in
        // aggregate ports that sum, so it must fold the same way or
        // vectors drift 1 f64 ULP (the phase-2 lesson, applied here)
        let xs = [1.0f64, 1e100, 1.0, -1e100];
        assert_eq!(py_sum(xs.into_iter()), 2.0);
        // a naive sequential fold loses both 1.0s inside the ±1e100
        assert_ne!(xs.iter().copied().fold(0.0f64, |a, b| a + b), 2.0);
    }

    #[test]
    fn aggregate_f64_norm_is_compensated() {
        // Python-anchored golden: one unit component + 40 × 2^-28. A
        // naive norm fold loses every 2^-56 square (below ulp(1)) and
        // normalizes to exactly 1.0 / 2^-28; the compensated fold
        // (CPython ≥3.12 sum) yields the bits Python's aggregate
        // produced. The live fixture's data rounds IDENTICALLY either
        // way (measured), so only this golden catches swapping py_sum
        // for a plain fold inside aggregate (review M-β).
        let v: Vec<f32> = std::iter::once(1.0f32)
            .chain(std::iter::repeat_n(2.0f32.powi(-28), 40))
            .collect();
        let out = aggregate_f64(&[v], &[1.0]).unwrap();
        // from Python: user_vector.aggregate([[1.0] + [2**-28]*40], [1.0])
        // — 0x1.ffffffffffffep-1 / 0x1.ffffffffffffep-29 as f64 bits
        // (hex float literals are not stable Rust)
        assert_eq!(out[0], f64::from_bits(0x3feffffffffffffe));
        assert_eq!(out[1], f64::from_bits(0x3e2ffffffffffffe));
        assert_eq!(out[40], f64::from_bits(0x3e2ffffffffffffe));
    }
}
