-- cine-rec-engine — canonical TMDB mirror schema
-- ---------------------------------------------------------------------------
-- Source of truth: TMDB API only (language=en-US). No embeddings, no
-- narrative tags, no user views — those live elsewhere (or not at all).
-- Movies and TV are SPLIT like TMDB itself: two resources, two id spaces,
-- two fact tables. TMDB ids are unique WITHIN a medium, not across them:
-- movie 155 and tv 155 are both real, different titles.
--
-- Layout:
--   §1  Facts        — tmdb_movies / tmdb_tv
--   §2  Dimensions   — shared people / keywords / companies / networks
--   §3  Genre lists  — separate per medium (TMDB's lists differ!)
--   §4  Bridges      — N:N per medium, never text[] on the title row
--   §5  Behavior     — TMDB /recommendations graph, per medium
--   §6  Compatibility views — project the split onto the engine's shape
--   §7  User data    — your events (watches/ratings)
--
-- Only what the API returns is stored. No name normalization, no
-- translation, no guessing. §6 views are the default adapter until the
-- engine reads the two fact tables natively.

-- ===========================================================================
-- §1 FACTS — titles
-- ===========================================================================

CREATE TABLE IF NOT EXISTS tmdb_movies (
    id                bigint PRIMARY KEY,
    title             text NOT NULL,            -- en-US
    original_title    text,
    original_language text,
    overview          text,                     -- en-US
    release_date      date,
    runtime           integer,
    status            text,
    adult             boolean NOT NULL DEFAULT false,
    vote_average      numeric,
    vote_count        integer,
    popularity        numeric,
    poster_path       text,
    backdrop_path     text,
    collection_id     bigint,                   -- belongs_to_collection.id (nullable)
    imdb_id           text,
    budget            bigint,                   -- optional, never used for ranking
    revenue           bigint,                   -- optional, never used for ranking
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tmdb_tv (
    id                 bigint PRIMARY KEY,
    name               text NOT NULL,           -- en-US; TMDB field name kept
    original_name      text,
    original_language  text,
    overview           text,
    first_air_date     date,
    last_air_date      date,
    status             text,
    in_production      boolean,
    number_of_seasons  integer,
    number_of_episodes integer,
    adult              boolean NOT NULL DEFAULT false,
    vote_average       numeric,
    vote_count         integer,
    popularity         numeric,
    poster_path        text,
    backdrop_path      text,
    imdb_id            text,                    -- from external_ids if pulled
    updated_at         timestamptz NOT NULL DEFAULT now()
    -- no collection_id on series
);

-- ===========================================================================
-- §2 DIMENSIONS — shared
-- ===========================================================================

CREATE TABLE IF NOT EXISTS tmdb_people (
    id   bigint PRIMARY KEY,
    name text NOT NULL                          -- as returned in credits (en-US)
);

CREATE TABLE IF NOT EXISTS tmdb_keywords (
    id   bigint PRIMARY KEY,
    name text NOT NULL
);

CREATE TABLE IF NOT EXISTS tmdb_production_companies (
    id   bigint PRIMARY KEY,
    name text NOT NULL
);

CREATE TABLE IF NOT EXISTS tmdb_networks (      -- TV only
    id   bigint PRIMARY KEY,
    name text NOT NULL
);

-- ===========================================================================
-- §3 GENRE LISTS — separate per medium
-- ===========================================================================
-- TMDB's movie and tv genre lists are different (e.g. tv 10759
-- "Action & Adventure" has no movie counterpart). Overlapping ids carry
-- the same name in both lists (verified: 16/35/80/99/18/10751/37).

CREATE TABLE IF NOT EXISTS tmdb_movie_genres (
    id   integer PRIMARY KEY,
    name text NOT NULL                          -- /genre/movie/list
);

CREATE TABLE IF NOT EXISTS tmdb_tv_genres (
    id   integer PRIMARY KEY,
    name text NOT NULL                          -- /genre/tv/list
);

-- ===========================================================================
-- §4 BRIDGES — N:N per medium
-- ===========================================================================
-- Cast: top-5 by `order`. Same person can play two roles → cast_order in PK.
-- Crew: only Director / Creator(tv) / Writer / Original Music Composer /
--       Director of Photography. Everything else is dropped at ingest.

-- --- movies ---
CREATE TABLE IF NOT EXISTS tmdb_movie_genres_map (
    movie_id bigint NOT NULL REFERENCES tmdb_movies(id),
    genre_id integer NOT NULL REFERENCES tmdb_movie_genres(id),
    PRIMARY KEY (movie_id, genre_id)
);
CREATE TABLE IF NOT EXISTS tmdb_movie_keywords (
    movie_id   bigint NOT NULL REFERENCES tmdb_movies(id),
    keyword_id bigint NOT NULL REFERENCES tmdb_keywords(id),
    PRIMARY KEY (movie_id, keyword_id)
);
CREATE TABLE IF NOT EXISTS tmdb_movie_companies (
    movie_id   bigint NOT NULL REFERENCES tmdb_movies(id),
    company_id bigint NOT NULL REFERENCES tmdb_production_companies(id),
    PRIMARY KEY (movie_id, company_id)
);
CREATE TABLE IF NOT EXISTS tmdb_movie_cast (
    movie_id   bigint NOT NULL REFERENCES tmdb_movies(id),
    person_id  bigint NOT NULL REFERENCES tmdb_people(id),
    character  text,
    cast_order integer NOT NULL,
    PRIMARY KEY (movie_id, person_id, cast_order)
);
CREATE TABLE IF NOT EXISTS tmdb_movie_crew (
    movie_id   bigint NOT NULL REFERENCES tmdb_movies(id),
    person_id  bigint NOT NULL REFERENCES tmdb_people(id),
    job        text NOT NULL,
    department text,
    PRIMARY KEY (movie_id, person_id, job)
);

-- --- tv ---
CREATE TABLE IF NOT EXISTS tmdb_tv_genres_map (
    tv_id    bigint NOT NULL REFERENCES tmdb_tv(id),
    genre_id integer NOT NULL REFERENCES tmdb_tv_genres(id),
    PRIMARY KEY (tv_id, genre_id)
);
CREATE TABLE IF NOT EXISTS tmdb_tv_keywords (
    tv_id      bigint NOT NULL REFERENCES tmdb_tv(id),
    keyword_id bigint NOT NULL REFERENCES tmdb_keywords(id),
    PRIMARY KEY (tv_id, keyword_id)
);
CREATE TABLE IF NOT EXISTS tmdb_tv_companies (
    tv_id      bigint NOT NULL REFERENCES tmdb_tv(id),
    company_id bigint NOT NULL REFERENCES tmdb_production_companies(id),
    PRIMARY KEY (tv_id, company_id)
);
CREATE TABLE IF NOT EXISTS tmdb_tv_networks_map (
    tv_id      bigint NOT NULL REFERENCES tmdb_tv(id),
    network_id bigint NOT NULL REFERENCES tmdb_networks(id),
    PRIMARY KEY (tv_id, network_id)
);
CREATE TABLE IF NOT EXISTS tmdb_tv_cast (
    tv_id      bigint NOT NULL REFERENCES tmdb_tv(id),
    person_id  bigint NOT NULL REFERENCES tmdb_people(id),
    character  text,
    cast_order integer NOT NULL,
    PRIMARY KEY (tv_id, person_id, cast_order)
);
CREATE TABLE IF NOT EXISTS tmdb_tv_crew (
    tv_id     bigint NOT NULL REFERENCES tmdb_tv(id),
    person_id bigint NOT NULL REFERENCES tmdb_people(id),
    job       text NOT NULL,
    department text,
    PRIMARY KEY (tv_id, person_id, job)
);

-- ===========================================================================
-- §5 BEHAVIOR — TMDB /recommendations graph
-- ===========================================================================
-- TMDB recommendations stay within their medium (movie→movie, tv→tv),
-- exactly like the API.

CREATE TABLE IF NOT EXISTS tmdb_movie_recommendations (
    movie_id     bigint NOT NULL REFERENCES tmdb_movies(id),
    rec_movie_id bigint NOT NULL,
    rank         integer NOT NULL,
    popularity   double precision,
    synced_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (movie_id, rec_movie_id)
);

CREATE TABLE IF NOT EXISTS tmdb_tv_recommendations (
    tv_id     bigint NOT NULL REFERENCES tmdb_tv(id),
    rec_tv_id bigint NOT NULL,
    rank      integer NOT NULL,
    popularity double precision,
    synced_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tv_id, rec_tv_id)
);

-- Engine write-back cache (see §6 note): the engine UPSERTs the behavioral
-- rows it fetches on demand (ON CONFLICT DO UPDATE), which a view cannot
-- receive. This one shared table is therefore PHYSICAL; ingest mirrors
-- its per-medium rows into the two tables above when both exist.
CREATE TABLE IF NOT EXISTS tmdb_recommendations (
    media_id       bigint NOT NULL,
    media_type     text   NOT NULL CHECK (media_type IN ('movie', 'tv')),
    rec_media_id   bigint NOT NULL,
    rec_media_type text   NOT NULL CHECK (rec_media_type IN ('movie', 'tv')),
    rank           integer NOT NULL,
    popularity     double precision,
    synced_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (media_id, media_type, rec_media_id, rec_media_type)
);

-- ===========================================================================
-- §6 COMPATIBILITY VIEWS — the split, projected onto the engine's shape
-- ===========================================================================
-- The engine (v0.1) reads the unified names below. These views make the
-- split schema serve it unchanged. Reads only, except tmdb_recommendations
-- (physical, §5). en-US only: title_en IS title, overview_en IS overview.

CREATE OR REPLACE VIEW tmdb_media AS
SELECT id, 'movie'::text AS media_type,
       title, title AS title_en,
       overview, overview AS overview_en,
       original_language, vote_average::float, vote_count, popularity::float,
       poster_path, collection_id, adult, release_date, NULL::date AS first_air_date
FROM tmdb_movies
UNION ALL
SELECT id, 'tv'::text,
       name, name,
       overview, overview,
       original_language, vote_average::float, vote_count, popularity::float,
       poster_path, NULL::bigint, adult, NULL::date, first_air_date
FROM tmdb_tv;

CREATE OR REPLACE VIEW tmdb_genres AS
SELECT id, name FROM tmdb_movie_genres
UNION
SELECT id, name FROM tmdb_tv_genres;

CREATE OR REPLACE VIEW tmdb_media_genres AS
SELECT movie_id AS media_id, genre_id, 'movie'::text AS media_type
FROM tmdb_movie_genres_map
UNION ALL
SELECT tv_id, genre_id, 'tv'::text
FROM tmdb_tv_genres_map;

CREATE OR REPLACE VIEW tmdb_media_keywords AS
SELECT movie_id AS media_id, keyword_id, 'movie'::text AS media_type
FROM tmdb_movie_keywords
UNION ALL
SELECT tv_id, keyword_id, 'tv'::text
FROM tmdb_tv_keywords;

CREATE OR REPLACE VIEW tmdb_media_companies AS
SELECT movie_id AS media_id, company_id, 'movie'::text AS media_type
FROM tmdb_movie_companies
UNION ALL
SELECT tv_id, company_id, 'tv'::text
FROM tmdb_tv_companies;

CREATE OR REPLACE VIEW tmdb_media_networks AS
SELECT tv_id AS media_id, network_id, 'tv'::text AS media_type
FROM tmdb_tv_networks_map;

CREATE OR REPLACE VIEW tmdb_cast AS
SELECT movie_id AS media_id, person_id,
       character AS character_name, cast_order, 'movie'::text AS media_type
FROM tmdb_movie_cast
UNION ALL
SELECT tv_id, person_id, character, cast_order, 'tv'::text
FROM tmdb_tv_cast;

CREATE OR REPLACE VIEW tmdb_crew AS
SELECT movie_id AS media_id, person_id, job, department, 'movie'::text AS media_type
FROM tmdb_movie_crew
UNION ALL
SELECT tv_id, person_id, job, department, 'tv'::text
FROM tmdb_tv_crew;

-- ===========================================================================
-- §7 USER DATA — your events
-- ===========================================================================
-- Swap for your own tables: the engine only needs (tmdb_id, media_type)
-- per user — see engine/watched.py USER_WATCHES_SQL.

-- user_watches is now a DERIVED view over user_title_stats
-- (docs/user_data.sql §compatibility) — the raw events live there too.

CREATE TABLE IF NOT EXISTS title_ratings (
    user_id    bigint NOT NULL,
    tmdb_id    bigint NOT NULL,
    media_type text NOT NULL CHECK (media_type IN ('movie', 'tv')),
    rating     integer,
    rated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, tmdb_id, media_type)
);

-- ===========================================================================
-- INDEXES — minimal set
-- ===========================================================================

CREATE INDEX IF NOT EXISTS tmdb_movies_popularity_idx   ON tmdb_movies (popularity DESC);
CREATE INDEX IF NOT EXISTS tmdb_movies_votes_idx        ON tmdb_movies (vote_average, vote_count);
CREATE INDEX IF NOT EXISTS tmdb_tv_popularity_idx       ON tmdb_tv (popularity DESC);
CREATE INDEX IF NOT EXISTS tmdb_tv_votes_idx            ON tmdb_tv (vote_average, vote_count);

CREATE INDEX IF NOT EXISTS tmdb_movie_genres_map_movie_idx ON tmdb_movie_genres_map (movie_id);
CREATE INDEX IF NOT EXISTS tmdb_movie_keywords_movie_idx   ON tmdb_movie_keywords (movie_id);
CREATE INDEX IF NOT EXISTS tmdb_movie_companies_movie_idx  ON tmdb_movie_companies (movie_id);
CREATE INDEX IF NOT EXISTS tmdb_movie_cast_movie_idx       ON tmdb_movie_cast (movie_id);
CREATE INDEX IF NOT EXISTS tmdb_movie_crew_movie_idx       ON tmdb_movie_crew (movie_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_genres_map_tv_idx       ON tmdb_tv_genres_map (tv_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_keywords_tv_idx         ON tmdb_tv_keywords (tv_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_companies_tv_idx        ON tmdb_tv_companies (tv_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_networks_map_tv_idx     ON tmdb_tv_networks_map (tv_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_cast_tv_idx             ON tmdb_tv_cast (tv_id);
CREATE INDEX IF NOT EXISTS tmdb_tv_crew_tv_idx             ON tmdb_tv_crew (tv_id);

CREATE INDEX IF NOT EXISTS tmdb_movie_crew_person_job_idx ON tmdb_movie_crew (person_id, job);
CREATE INDEX IF NOT EXISTS tmdb_tv_crew_person_job_idx    ON tmdb_tv_crew (person_id, job);

CREATE INDEX IF NOT EXISTS tmdb_movie_recommendations_movie_idx ON tmdb_movie_recommendations (movie_id, rank);
CREATE INDEX IF NOT EXISTS tmdb_tv_recommendations_tv_idx       ON tmdb_tv_recommendations (tv_id, rank);

CREATE INDEX IF NOT EXISTS title_ratings_user_idx  ON title_ratings (user_id);
