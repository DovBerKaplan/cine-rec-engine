//! The real port: a [`ScoringBackend`] translating the Python
//! scoring core (`service.py::_entity_profile` / `_vec_from_profiles`
//! / `_score_pair_fast` + the pure helpers of `scoring.py`) with the
//! EXACT operation order of the Python code.
//!
//! ULP distances depend on that order — every sum here folds left the
//! way the Python expression tree evaluates, and nothing reorders
//! float operations. The parity harness (`tests/real_backend_parity.rs`)
//! proves bit-exact vectors and ≤1-ULP fast scores against BOTH golden
//! sets.
//!
//! Two faithfulness rules worth calling out:
//! * Python truthiness (`x or default`) is ported as "non-zero /
//!   non-empty", including `if year:` treating 0 as missing.
//! * The vec path reads the candidate/entity dicts directly (its own
//!   `via` / `tmdb_rank` / `collection_id or fallback` semantics),
//!   while the fast path reads the same values from the prebuilt
//!   candidate profile — the two Python paths differ subtly and both
//!   behaviors are preserved here.

use std::collections::HashSet;

use crate::backend::ScoringBackend;
use crate::constants;
use crate::types::{CastKey, Entity, Profile};

// ── pure helpers (scoring.py) ──────────────────────────────────────

/// Character bigrams of the lowercased string (`scoring.bigram_set`).
pub fn bigram_set(text: &str) -> HashSet<String> {
    let low = text.to_lowercase();
    let chars: Vec<char> = low.chars().collect();
    let mut out = HashSet::with_capacity(chars.len().saturating_sub(1));
    if chars.len() < 2 {
        return out;
    }
    for w in chars.windows(2) {
        out.insert(w.iter().collect());
    }
    out
}

/// `scoring.canonical_genres` — Hebrew labels map to canonical English
/// names; everything else passes through stripped.
pub fn canonical_genres(genres: &[String]) -> HashSet<String> {
    // genres are canonical English (ingest syncs with en-US); strip + dedup
    genres
        .iter()
        .map(|g| g.trim().to_string())
        .filter(|g| !g.is_empty())
        .collect()
}

/// `scoring.detect_style` — style labels whose keyword sets overlap.
pub fn detect_style(keywords: &[String]) -> HashSet<String> {
    let mut out = HashSet::new();
    if keywords.is_empty() {
        return out;
    }
    for (label, keys) in constants::KEYWORDS_STYLE {
        if keys.iter().any(|k| keywords.iter().any(|kw| kw == k)) {
            out.insert(label.to_string());
        }
    }
    out
}

/// `scoring.keyword_set` — meta-stopword-filtered keyword set.
pub fn keyword_set(keywords: &[String]) -> HashSet<String> {
    keywords
        .iter()
        .filter(|k| !constants::META_KEYWORD_STOPWORDS.contains(&k.as_str()))
        .cloned()
        .collect()
}

/// `service._modality` — anime / animation / live from canonical
/// genres + original language.
fn modality(param_genres: &[String], original_language: Option<&str>) -> String {
    if canonical_genres(param_genres).contains("Animation") {
        if original_language
            .unwrap_or("")
            .to_lowercase()
            .starts_with("ja")
        {
            "anime".to_string()
        } else {
            "animation".to_string()
        }
    } else {
        "live".to_string()
    }
}

/// `scoring.audience_score` (canonical sets precomputed by callers).
pub fn audience_score(
    seed_canonical: &HashSet<String>,
    cand_canonical: &HashSet<String>,
    _seed_adult: bool, // mirrors the Python signature; the youth gate
    //                     reads only the candidate's adult signal today
    cand_adult: bool,
) -> f64 {
    let youth = |g: &HashSet<String>| constants::YOUTH_GENRES.iter().any(|y| g.contains(*y));
    let seed_y = youth(seed_canonical);
    let cand_y = youth(cand_canonical);
    let cand_adult_signal = cand_adult
        || (!cand_y
            && constants::ADULT_LEANING_GENRES
                .iter()
                .any(|g| cand_canonical.contains(*g)));

    if seed_y && cand_adult_signal {
        return 0.0;
    }
    if seed_y == cand_y {
        return 1.0;
    }
    0.5
}

/// `scoring.tone_score` (canonical sets + arc tags precomputed).
pub fn tone_score(
    seed_canonical: &HashSet<String>,
    cand_canonical: &HashSet<String>,
    seed_arc_lower: &HashSet<String>,
    cand_arc_lower: &HashSet<String>,
) -> f64 {
    let count_in =
        |g: &HashSet<String>, set: &[&str]| set.iter().filter(|s| g.contains(**s)).count();
    let seed_light_count = count_in(seed_canonical, constants::LIGHT_GENRES);
    let cand_light_count = count_in(cand_canonical, constants::LIGHT_GENRES);
    let seed_dark_count = count_in(seed_canonical, constants::DARK_GENRES);
    let cand_dark_count = count_in(cand_canonical, constants::DARK_GENRES);

    let mut seed_light = seed_light_count > 0;
    let mut seed_dark = seed_dark_count > 0;
    let mut cand_light = cand_light_count > 0;
    let mut cand_dark = cand_dark_count > 0;

    // Emotional arc tags refine when genres alone are ambiguous
    let arc_hit = |tags: &HashSet<String>, set: &[&str]| set.iter().any(|t| tags.contains(*t));
    if arc_hit(seed_arc_lower, constants::LIGHT_ARC_TAGS) {
        seed_light = true;
    }
    if arc_hit(seed_arc_lower, constants::DARK_ARC_TAGS) {
        seed_dark = true;
    }
    if arc_hit(cand_arc_lower, constants::LIGHT_ARC_TAGS) {
        cand_light = true;
    }
    if arc_hit(cand_arc_lower, constants::DARK_ARC_TAGS) {
        cand_dark = true;
    }

    if (seed_light && cand_light) || (seed_dark && cand_dark) {
        return 1.0;
    }
    if (seed_light && cand_dark) || (seed_dark && cand_light) {
        return 0.0;
    }
    // Strongly one-sided: seed CLEARLY light (2+ genres) but candidate
    // has ZERO light signal → penalize (0.2, not 0.5).
    if seed_light_count >= 2 && cand_light_count == 0 {
        return 0.2;
    }
    if seed_dark_count >= 2 && cand_dark_count == 0 {
        return 0.2;
    }
    0.5
}

/// `source.get("original_language") or entity.get("original_language")`
/// — first NON-EMPTY string wins (Python `or` skips falsy "").
fn lang_or<'a>(source: &'a Entity, entity: &'a Entity) -> Option<&'a str> {
    source
        .original_language
        .as_deref()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            entity
                .original_language
                .as_deref()
                .filter(|s| !s.is_empty())
        })
}

// ── entity profile (service._entity_profile) ───────────────────────

/// `service._entity_profile`. `entity_data` follows Python truthiness
/// where the source code does (`data if data else entity` — an empty
/// dict reads the entity; `data if entity_data is not None` — an
/// empty dict still counts as data for the enriched-source fields).
/// The harness only ever passes a real row or None, but both nuances
/// are preserved for the bridge.
pub fn entity_profile(
    entity: &Entity,
    entity_data: Option<&Entity>,
    genres_override: Option<&[String]>,
) -> Profile {
    // canonical calls: real row data (truthy) or None
    entity_profile_ex(entity, entity_data, entity_data.is_some(), genres_override)
}

/// The full-truthiness form used by the pyo3 bridge, where
/// `entity_data` can be an EMPTY dict — a regular production input
/// (`service.py`: `enriched.get(cid, {})`). Python's semantics there:
/// the ENRICHED-source fields read the empty dict (everything falls to
/// defaults), while the narrative/pacing/emotional-arc fields fall
/// back to the ENTITY (`data.get(...) if data else entity.get(...)`
/// — an empty dict is falsy). `data_truthy` carries exactly that.
pub fn entity_profile_ex(
    entity: &Entity,
    entity_data: Option<&Entity>,
    data_truthy: bool,
    genres_override: Option<&[String]>,
) -> Profile {
    // `source = data if entity_data is not None else entity` — the
    // enriched source follows is-not-None (an empty dict still counts)
    let source: &Entity = entity_data.unwrap_or(entity);
    // genres = entity.get("genres") or data.get("genres") or []
    let genres: Vec<String> = if !entity.genres.is_empty() {
        entity.genres.clone()
    } else if let Some(d) = entity_data {
        d.genres.clone()
    } else {
        Vec::new()
    };
    let param_genres: Vec<String> = match genres_override {
        Some(g) => g.to_vec(),
        None => genres.clone(),
    };
    // overview = entity.get("overview_en", "") or entity.get("overview", "") or ""
    let overview: String = entity
        .overview_en
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| entity.overview.clone().filter(|s| !s.is_empty()))
        .unwrap_or_default();

    let can_genres = canonical_genres(&param_genres);
    // `data.get(...) if data else entity.get(...)` — dict TRUTHINESS:
    // an empty data dict falls these back to the entity
    let arc_source: Option<&Vec<String>> = if data_truthy {
        entity_data.and_then(|d| d.emotional_arc.as_ref())
    } else {
        entity.emotional_arc.as_ref()
    };
    let arc_set: HashSet<String> = arc_source
        .map(|arc| {
            arc.iter()
                .map(|t| t.to_lowercase().trim().to_string())
                .collect()
        })
        .unwrap_or_default();
    let arc_lower_raw: HashSet<String> = arc_source
        .map(|arc| arc.iter().map(|t| t.to_lowercase()).collect())
        .unwrap_or_default();

    Profile {
        dir_ids: source.director_ids.iter().copied().collect(),
        writer_ids: source.writer_ids.iter().copied().collect(),
        composer_ids: source.composer_ids.iter().copied().collect(),
        dp_ids: source.dp_ids.iter().copied().collect(),
        cast5: if !source.cast_ids.is_empty() {
            source
                .cast_ids
                .iter()
                .take(5)
                .map(|id| CastKey::Id(*id))
                .collect()
        } else {
            source.cast_list.iter().take(5).cloned().collect()
        },
        keywords: keyword_set(&source.keywords),
        companies: source.companies.iter().cloned().collect(),
        networks: source.networks.iter().cloned().collect(),
        can_genres_raw: param_genres.iter().cloned().collect(),
        param_genres: param_genres.clone(),
        style: detect_style(&source.keywords),
        genres: genres.clone(),
        modality: modality(&param_genres, lang_or(source, entity)),
        director: source.director.clone(),
        coll: source.collection_id,
        overview_bigrams: if overview.is_empty() {
            None
        } else {
            Some(bigram_set(&overview))
        },
        overview,
        year: entity.release_year,
        votes: entity.vote_count.unwrap_or(0),
        rating: entity.vote_average.unwrap_or(0.0),
        narrative: if data_truthy {
            entity_data.and_then(|d| d.narrative.clone())
        } else {
            entity.narrative.clone()
        },
        pacing: if data_truthy {
            entity_data.and_then(|d| d.pacing.clone())
        } else {
            entity.pacing.clone()
        },
        emotional_arc: arc_source.cloned(),
        arc_set,
        arc_lower_raw,
        adult: entity.adult,
        via: entity.via.clone(),
        cand_coll: source.collection_id,
        tmdb_rec_decay: if entity.via.as_deref() == Some("tmdb") {
            let rank = entity.tmdb_rank.filter(|r| *r != 0).unwrap_or(1);
            (1.0 - ((rank - 1) as f64) * 0.15).max(0.25)
        } else {
            0.0
        },
        can_genres,
    }
}

// ── the 28-entry vector (service._vec_from_profiles) ───────────────

/// `service._vec_from_profiles` — reads `cand`/`cand_data` directly
/// (the vec path's own via/collection semantics, distinct from the
/// fast path's profile reads).
pub fn vec_from_profiles(
    sp: &Profile,
    cp: &Profile,
    cand: &Entity,
    cand_data: &Entity,
    overview_sim: f64,
    heuristic_w27: &[f64], // HEURISTIC_WEIGHTS_LIST (27 entries)
) -> Vec<f64> {
    // director_match (name fallback mirrors feature_vector)
    let director_name_truthy = matches!(sp.director.as_deref(), Some(d) if !d.is_empty());
    let director_match =
        (!sp.dir_ids.is_empty() && !cp.dir_ids.is_empty() && !sp.dir_ids.is_disjoint(&cp.dir_ids))
            || (director_name_truthy && sp.director == cand_data.director);

    let cand_coll = truthy_i64(cand_data.collection_id).or(truthy_i64(cand.collection_id));
    let shared_collection =
        truthy_i64(sp.coll).is_some() && cand_coll.is_some() && truthy_i64(sp.coll) == cand_coll;

    let mut director_channel = 0.0f64;
    if cand.via.as_deref() == Some("director") {
        if cp.rating >= 7.0 {
            director_channel = 1.0;
        } else {
            director_channel = ((cp.rating - 5.5) / 1.5).max(0.3);
        }
        if shared_collection {
            director_channel *= 0.5;
        }
    }

    let writer_match = f64::from(
        (!sp.writer_ids.is_empty()
            && !cp.writer_ids.is_empty()
            && !sp.writer_ids.is_disjoint(&cp.writer_ids))
            || cand.via.as_deref() == Some("writer"),
    );

    let medium_mismatch = f64::from(sp.modality != cp.modality);
    let composer_match = f64::from(!sp.composer_ids.is_disjoint(&cp.composer_ids));
    let dp_match = f64::from(!sp.dp_ids.is_disjoint(&cp.dp_ids));

    let mut tmdb_rec_decay = 0.0f64;
    if cand.via.as_deref() == Some("tmdb") {
        let tmdb_rank = cand.tmdb_rank.filter(|r| *r != 0).unwrap_or(1);
        tmdb_rec_decay = (1.0 - ((tmdb_rank - 1) as f64) * 0.15).max(0.25);
    }

    let cosine_sim = if overview_sim > 0.0 {
        overview_sim
    } else if let (Some(a), Some(b)) = (&sp.overview_bigrams, &cp.overview_bigrams) {
        let inter = a.intersection(b).count();
        let union = a.union(b).count();
        if union > 0 {
            inter as f64 / union as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    let keyword_sim = jaccard(&sp.keywords, &cp.keywords);
    let cast_sim = dice_cast5(&sp.cast5, &cp.cast5);

    // sum(GENRE_PRIORITY.get(g, 1.0) for g in sorted(intersection)) —
    // left fold in sorted order, starting from +0.0
    let genre_priority_sum: f64 = {
        let mut shared: Vec<&String> = sp.can_genres.intersection(&cp.can_genres).collect();
        shared.sort();
        py_sum(shared.iter().map(|g| constants::genre_priority(g)))
    };

    let style_match = f64::from(!sp.style.is_disjoint(&cp.style));
    let company_sim = jaccard(&sp.companies, &cp.companies);
    let shared_network = f64::from(
        !sp.networks.is_empty()
            && !cp.networks.is_empty()
            && !sp.networks.is_disjoint(&cp.networks),
    );

    let (year_le5, year_le10, year_gt25) = year_flags(sp.year, cp.year);

    let votes_gt15k = f64::from(cp.votes > 15000);
    let low_votes_high_rating = f64::from(cp.votes < 2000 && cp.rating > 7.0);
    let rating_bonus = ((cp.rating - 6.0).clamp(0.0, 2.0)) * 0.5;
    let pop_action_leak =
        f64::from(cp.votes > 10000 && cosine_sim < constants::POP_ACTION_SIM_FLOOR);

    let narrative_match = match (sp.narrative.as_deref(), cp.narrative.as_deref()) {
        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
            if a == b {
                1.0
            } else if (a == "linear" && b == "anthology") || (a == "anthology" && b == "linear") {
                -1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    let tone_compat = tone_score(
        &sp.can_genres,
        &cp.can_genres,
        &sp.arc_lower_raw,
        &cp.arc_lower_raw,
    );
    let arc_match = if !sp.arc_set.is_empty() && !cp.arc_set.is_empty() {
        let inter = sp.arc_set.intersection(&cp.arc_set).count();
        let union = sp.arc_set.union(&cp.arc_set).count();
        if union > 0 {
            inter as f64 / union as f64
        } else {
            0.0
        }
    } else {
        0.0
    };
    let pacing_truthy = matches!(sp.pacing.as_deref(), Some(p) if !p.is_empty());
    let pacing_match = if pacing_truthy && sp.pacing == cp.pacing {
        1.0
    } else {
        0.0
    };

    let audience_compat = audience_score(&sp.can_genres, &cp.can_genres, sp.adult, cp.adult);

    let vec = [
        cosine_sim,
        keyword_sim,
        cast_sim,
        genre_priority_sum,
        f64::from(director_match),
        director_channel,
        writer_match,
        tmdb_rec_decay,
        f64::from(shared_collection),
        shared_network,
        style_match,
        company_sim,
        year_le5,
        year_le10,
        year_gt25,
        votes_gt15k,
        low_votes_high_rating,
        rating_bonus,
        pop_action_leak,
        composer_match,
        dp_match,
        medium_mismatch,
        narrative_match,
        tone_compat,
        arc_match,
        pacing_match,
        audience_compat,
    ];

    // heuristic_score = sum(w * v for w, v in zip(w27, vec)) — left fold
    let heuristic_score = py_sum(heuristic_w27.iter().zip(vec.iter()).map(|(w, v)| w * v));

    let mut out = Vec::with_capacity(crate::VEC_LEN);
    out.extend_from_slice(&vec);
    out.push(heuristic_score);
    out
}

/// Python truthiness for ints: 0 is falsy (`sp["coll"] and ...`,
/// `cand_data.get("collection_id") or cand.get(...)`).
fn truthy_i64(v: Option<i64>) -> Option<i64> {
    v.filter(|x| *x != 0)
}

/// CPython 3.12's `sum()` over floats — Neumaier compensated summation,
/// same left-to-right order, the compensation folded in at the end.
/// The goldens were captured under 3.12 (where sum() stopped being a
/// naive fold); a plain fold lands 1 ULP off on some inputs. EVERY
/// Python `sum()` over floats ported here must go through this.
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

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let union = a.union(b).count();
    if union > 0 {
        a.intersection(b).count() as f64 / union as f64
    } else {
        0.0
    }
}

fn dice_cast5(a: &HashSet<CastKey>, b: &HashSet<CastKey>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let total = a.len() + b.len();
    if total > 0 {
        (2.0 * a.intersection(b).count() as f64) / total as f64
    } else {
        0.0
    }
}

/// `if sp["year"] and cp["year"]` — Python truthiness: year 0 counts
/// as missing on either side.
fn year_flags(sy: Option<i64>, cy: Option<i64>) -> (f64, f64, f64) {
    // Python truthiness: year 0 (or None) counts as missing on either
    // side — then all three flags stay 0.0
    let (a, b) = match (sy.filter(|v| *v != 0), cy.filter(|v| *v != 0)) {
        (Some(a), Some(b)) => (a, b),
        _ => return (0.0, 0.0, 0.0),
    };
    let diff = (a - b).abs();
    if diff <= 5 {
        (1.0, 0.0, 0.0)
    } else if diff <= 10 {
        (0.0, 1.0, 0.0)
    } else if diff > 25 {
        (0.0, 0.0, 1.0)
    } else {
        (0.0, 0.0, 0.0)
    }
}

// ── fast score (service._score_pair_fast) ──────────────────────────

/// `service._score_pair_fast` — one accumulation pass over the same
/// feature order, reading candidate-side values from `cp` exactly the
/// way the Python hot loop does. `heuristic_w27` is the module-global
/// `HEURISTIC_WEIGHTS_LIST` used by the residual term.
pub fn score_pair_fast(
    sp: &Profile,
    cp: &Profile,
    cand_data_director: Option<&str>,
    cand_genres_set: Option<&HashSet<String>>,
    overview_sim: f64,
    w: &[f64],
    heuristic_w27: &[f64],
) -> f64 {
    // genre gate: classic-recall candidates must share a genre; via
    // channels are exempt (their proximity signal is the qualification).
    // Python truthiness twice: an EMPTY genre set counts as missing,
    // and an EMPTY via string counts as no via ("" is falsy).
    let via_truthy = matches!(cp.via.as_deref(), Some(v) if !v.is_empty());
    if !sp.genres.is_empty() && !via_truthy {
        if let Some(cgs) = cand_genres_set.filter(|s| !s.is_empty()) {
            if sp.can_genres_raw.is_disjoint(cgs) {
                return 0.0;
            }
        }
    }

    let director_name_truthy = matches!(sp.director.as_deref(), Some(d) if !d.is_empty());
    let director_match =
        (!sp.dir_ids.is_empty() && !cp.dir_ids.is_empty() && !sp.dir_ids.is_disjoint(&cp.dir_ids))
            || (director_name_truthy && sp.director.as_deref() == cand_data_director);
    let shared_collection = truthy_i64(sp.coll).is_some()
        && truthy_i64(cp.cand_coll).is_some()
        && truthy_i64(sp.coll) == truthy_i64(cp.cand_coll);

    let mut director_channel = 0.0f64;
    if cp.via.as_deref() == Some("director") {
        if cp.rating >= 7.0 {
            director_channel = 1.0;
        } else {
            director_channel = ((cp.rating - 5.5) / 1.5).max(0.3);
        }
        if shared_collection {
            director_channel *= 0.5;
        }
    }

    let writer_match = f64::from(
        (!sp.writer_ids.is_empty()
            && !cp.writer_ids.is_empty()
            && !sp.writer_ids.is_disjoint(&cp.writer_ids))
            || cp.via.as_deref() == Some("writer"),
    );

    let medium_mismatch = f64::from(sp.modality != cp.modality);
    let composer_match = f64::from(!sp.composer_ids.is_disjoint(&cp.composer_ids));
    let dp_match = f64::from(!sp.dp_ids.is_disjoint(&cp.dp_ids));

    let cosine_sim = if overview_sim > 0.0 {
        overview_sim
    } else if let (Some(a), Some(b)) = (&sp.overview_bigrams, &cp.overview_bigrams) {
        let inter = a.intersection(b).count();
        let union = a.union(b).count();
        if union > 0 {
            inter as f64 / union as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    let keyword_sim = jaccard(&sp.keywords, &cp.keywords);
    let cast_sim = dice_cast5(&sp.cast5, &cp.cast5);

    let genre_priority_sum: f64 = {
        let mut shared: Vec<&String> = sp.can_genres.intersection(&cp.can_genres).collect();
        shared.sort();
        py_sum(shared.iter().map(|g| constants::genre_priority(g)))
    };

    let style_match = f64::from(!sp.style.is_disjoint(&cp.style));
    let company_sim = jaccard(&sp.companies, &cp.companies);
    let shared_network = f64::from(
        !sp.networks.is_empty()
            && !cp.networks.is_empty()
            && !sp.networks.is_disjoint(&cp.networks),
    );

    let (year_le5, year_le10, year_gt25) = year_flags(sp.year, cp.year);

    let votes_gt15k = f64::from(cp.votes > 15000);
    let low_votes_high_rating = f64::from(cp.votes < 2000 && cp.rating > 7.0);
    let rating_bonus = ((cp.rating - 6.0).clamp(0.0, 2.0)) * 0.5;
    let pop_action_leak =
        f64::from(cp.votes > 10000 && cosine_sim < constants::POP_ACTION_SIM_FLOOR);

    let narrative_match = match (sp.narrative.as_deref(), cp.narrative.as_deref()) {
        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
            if a == b {
                1.0
            } else if (a == "linear" && b == "anthology") || (a == "anthology" && b == "linear") {
                -1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    let tone_compat = tone_score(
        &sp.can_genres,
        &cp.can_genres,
        &sp.arc_lower_raw,
        &cp.arc_lower_raw,
    );
    let arc_match = if !sp.arc_set.is_empty() && !cp.arc_set.is_empty() {
        let inter = sp.arc_set.intersection(&cp.arc_set).count();
        let union = sp.arc_set.union(&cp.arc_set).count();
        if union > 0 {
            inter as f64 / union as f64
        } else {
            0.0
        }
    } else {
        0.0
    };
    let pacing_truthy = matches!(sp.pacing.as_deref(), Some(p) if !p.is_empty());
    let pacing_match = if pacing_truthy && sp.pacing == cp.pacing {
        1.0
    } else {
        0.0
    };

    let audience_compat = audience_score(&sp.can_genres, &cp.can_genres, sp.adult, cp.adult);

    // The single accumulation pass — the SAME left-to-right chain as
    // the Python expression, feature order locked by tests.
    let mut score = w[0] * cosine_sim;
    score += w[1] * keyword_sim;
    score += w[2] * cast_sim;
    score += w[3] * genre_priority_sum;
    score += w[4] * f64::from(director_match);
    score += w[5] * director_channel;
    score += w[6] * writer_match;
    score += w[7] * cp.tmdb_rec_decay;
    score += w[8] * f64::from(shared_collection);
    score += w[9] * shared_network;
    score += w[10] * style_match;
    score += w[11] * company_sim;
    score += w[12] * year_le5;
    score += w[13] * year_le10;
    score += w[14] * year_gt25;
    score += w[15] * votes_gt15k;
    score += w[16] * low_votes_high_rating;
    score += w[17] * rating_bonus;
    score += w[18] * pop_action_leak;
    score += w[19] * composer_match;
    score += w[20] * dp_match;
    score += w[21] * medium_mismatch;
    score += w[22] * narrative_match;
    score += w[23] * tone_compat;
    score += w[24] * arc_match;
    score += w[25] * pacing_match;
    score += w[26] * audience_compat;
    if w[27] != 0.0 {
        // residual heuristic score (LTR input; weight 0.0 in baseline)
        let mut residual = heuristic_w27[0] * cosine_sim;
        residual += heuristic_w27[1] * keyword_sim;
        residual += heuristic_w27[2] * cast_sim;
        residual += heuristic_w27[3] * genre_priority_sum;
        residual += heuristic_w27[4] * f64::from(director_match);
        residual += heuristic_w27[5] * director_channel;
        residual += heuristic_w27[6] * writer_match;
        residual += heuristic_w27[7] * cp.tmdb_rec_decay;
        residual += heuristic_w27[8] * f64::from(shared_collection);
        residual += heuristic_w27[9] * shared_network;
        residual += heuristic_w27[10] * style_match;
        residual += heuristic_w27[11] * company_sim;
        residual += heuristic_w27[12] * year_le5;
        residual += heuristic_w27[13] * year_le10;
        residual += heuristic_w27[14] * year_gt25;
        residual += heuristic_w27[15] * votes_gt15k;
        residual += heuristic_w27[16] * low_votes_high_rating;
        residual += heuristic_w27[17] * rating_bonus;
        residual += heuristic_w27[18] * pop_action_leak;
        residual += heuristic_w27[19] * composer_match;
        residual += heuristic_w27[20] * dp_match;
        residual += heuristic_w27[21] * medium_mismatch;
        residual += heuristic_w27[22] * narrative_match;
        residual += heuristic_w27[23] * tone_compat;
        residual += heuristic_w27[24] * arc_match;
        residual += heuristic_w27[25] * pacing_match;
        residual += heuristic_w27[26] * audience_compat;
        score += w[27] * residual;
    }
    score
}

// ── the real backend (plugs into the parity harness) ───────────────

/// Real port for the parity harness. `heuristic_w27` is Python's
/// module-global `HEURISTIC_WEIGHTS_LIST` (values[:27]) used by the
/// residual term and the trailing heuristic_score slot.
pub struct RealBackend {
    pub heuristic_w27: Vec<f64>,
}

impl RealBackend {
    pub fn new(heuristic_w27: Vec<f64>) -> Self {
        Self { heuristic_w27 }
    }
}

impl Default for RealBackend {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl ScoringBackend for RealBackend {
    fn entity_profile(
        &self,
        entity: &Entity,
        entity_data: Option<&Entity>,
        genres_override: Option<&[String]>,
    ) -> Profile {
        entity_profile(entity, entity_data, genres_override)
    }

    fn feature_vector(
        &self,
        cand: &Entity,
        cand_data: &Entity,
        cand_genres: &[String],
        seed: &Entity,
        overview_sim: f64,
    ) -> Vec<f64> {
        let sp = entity_profile(seed, None, None);
        let cp = entity_profile(cand, Some(cand_data), Some(cand_genres));
        vec_from_profiles(&sp, &cp, cand, cand_data, overview_sim, &self.heuristic_w27)
    }

    fn score_pair_fast(
        &self,
        sp: &Profile,
        cp: &Profile,
        _cand: &Entity,
        cand_data: &Entity,
        cand_genres_set: Option<&HashSet<String>>,
        overview_sim: f64,
        w_list: &[f64],
    ) -> f64 {
        score_pair_fast(
            sp,
            cp,
            cand_data.director.as_deref(),
            cand_genres_set,
            overview_sim,
            w_list,
            &self.heuristic_w27,
        )
    }
}
