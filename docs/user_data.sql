-- cine-rec-engine — user data layer (spec v0.2 §B, §C, §E.3)
-- ---------------------------------------------------------------------------
-- Three write-disjoint layers: catalog (schema.sql §1-§6) NEVER touches these;
-- the app writes raw events + explicit signals here; everything derived
-- (user_title_stats / user_stats / user_genre_stats / user_vectors) is
-- recomputed from the raw by user_stats.py / user_vector.py.
--
-- Raw truth:   user_watch_events, user_feedback, user_searches, title_ratings
-- Derived:     user_title_stats, user_stats, user_genre_stats, user_vectors
-- Compatibility: the legacy `user_watches` contract (engine/watched.py)
--   becomes a VIEW over user_title_stats — no engine code changes.

CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- §B.1 RAW EVENTS — one row per viewing session/unit
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_watch_events (
    event_id          bigserial PRIMARY KEY,
    user_id           bigint NOT NULL,
    tmdb_id           bigint NOT NULL,
    media_type        text   NOT NULL CHECK (media_type IN ('movie', 'tv')),
    watched_at        timestamptz NOT NULL,
    watched_sec       integer NOT NULL DEFAULT 0,
    duration_sec      integer,                 -- movie or current episode length
    pause_count       integer NOT NULL DEFAULT 0,
    completed         boolean NOT NULL DEFAULT false,
    season            integer,                 -- NULL for movies
    episode           integer,
    last_position_sec integer                  -- resume point within the unit
);

CREATE INDEX IF NOT EXISTS user_watch_events_user_time_idx
    ON user_watch_events (user_id, watched_at DESC);
CREATE INDEX IF NOT EXISTS user_watch_events_user_title_idx
    ON user_watch_events (user_id, media_type, tmdb_id);

-- ===========================================================================
-- §B.2 EXPLICIT SIGNALS
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_feedback (
    user_id    bigint NOT NULL,
    tmdb_id    bigint NOT NULL,
    media_type text   NOT NULL CHECK (media_type IN ('movie', 'tv')),
    kind       text   NOT NULL CHECK (kind IN ('dislike', 'favorite', 'watchlist', 'click', 'skip')),  -- click/skip: RFC §4 feedback loop
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, tmdb_id, media_type, kind)
);

CREATE TABLE IF NOT EXISTS user_searches (      -- optional, short-term intent
    user_id     bigint NOT NULL,
    query_text  text   NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS user_searches_user_idx ON user_searches (user_id, created_at DESC);

-- title_ratings stays the numeric-rating contract (schema.sql §7 legacy).

-- ===========================================================================
-- §B.3 TITLE-LEVEL AGGREGATE — direct input to w_i and the vector
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_title_stats (
    user_id              bigint NOT NULL,
    tmdb_id              bigint NOT NULL,
    media_type           text   NOT NULL CHECK (media_type IN ('movie', 'tv')),
    sessions             integer NOT NULL DEFAULT 0,
    total_watched_sec    bigint  NOT NULL DEFAULT 0,
    max_ratio            double precision,      -- max watched/duration per unit
    pause_count_total    integer NOT NULL DEFAULT 0,
    rewatch_count        integer NOT NULL DEFAULT 0,
    last_watched_at      timestamptz,
    episodes_watched     integer NOT NULL DEFAULT 0,  -- completed / above-threshold
    last_season          integer,
    last_episode         integer,
    last_ep_watched_sec  integer,
    last_ep_duration_sec integer,
    dropped              boolean NOT NULL DEFAULT false,
    w_item               double precision NOT NULL DEFAULT 0,
    updated_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, tmdb_id, media_type)
);

-- ===========================================================================
-- §C.1 USER PROFILE NUMBERS
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_stats (
    user_id                         bigint PRIMARY KEY,
    titles_touched                  integer NOT NULL DEFAULT 0,
    titles_weighted                 integer NOT NULL DEFAULT 0,
    movies_completed                integer NOT NULL DEFAULT 0,
    movies_dropped                  integer NOT NULL DEFAULT 0,
    series_started                  integer NOT NULL DEFAULT 0,
    series_hooked                   integer NOT NULL DEFAULT 0,
    series_abandoned_mid_ep         integer NOT NULL DEFAULT 0,
    rewatch_titles                  integer NOT NULL DEFAULT 0,
    explicit_dislikes               integer NOT NULL DEFAULT 0,
    explicit_favorites              integer NOT NULL DEFAULT 0,
    watchlist_open                  integer NOT NULL DEFAULT 0,
    total_watch_sec_30d             bigint  NOT NULL DEFAULT 0,
    total_watch_sec_all             bigint  NOT NULL DEFAULT 0,
    movie_share_30d                 double precision,
    tv_share_30d                    double precision,
    median_session_sec              integer,
    avg_pauses_per_unfinished_movie double precision,
    last_event_at                   timestamptz,
    vector_updated_at               timestamptz,
    persona_count                   smallint NOT NULL DEFAULT 0,
    updated_at                      timestamptz NOT NULL DEFAULT now()
);

-- ===========================================================================
-- §C.2 GENRE DISTRIBUTION — UI explanations + smearing detection
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_genre_stats (
    user_id        bigint NOT NULL,
    media_type     text   NOT NULL,
    genre_id       integer NOT NULL,
    watch_sec      bigint NOT NULL DEFAULT 0,
    weighted_sum   double precision NOT NULL DEFAULT 0,
    title_count    integer NOT NULL DEFAULT 0,
    last_watched_at timestamptz,
    PRIMARY KEY (user_id, media_type, genre_id)
);

-- ===========================================================================
-- §E.3 USER VECTORS — up to 3 personas (persona_id 0 = single vector)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS user_vectors (
    user_id    bigint NOT NULL,
    persona_id smallint NOT NULL DEFAULT 0,
    space      text    NOT NULL DEFAULT 'e5e',  -- encoder key (models/README.md)
    embedding  vector,                          -- unbounded dim: match your space
    weight     double precision NOT NULL DEFAULT 1.0,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, persona_id, space)
);

-- ===========================================================================
-- COMPATIBILITY — the legacy user_watches contract, now DERIVED (spec §H)
-- ===========================================================================
-- engine/watched.py reads (tmdb_id, media_type) pairs "already seen":
-- anything with positive weight, or a movie completed (ratio ≥ 0.75).

CREATE OR REPLACE VIEW user_watches AS
SELECT user_id, tmdb_id, media_type
FROM user_title_stats
WHERE w_item > 0
   OR (media_type = 'movie' AND max_ratio >= 0.75);
