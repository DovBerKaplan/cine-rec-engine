"""Scoring configuration for the recommendation engine.

Tuned feature weights for the similarity scorer (see weights.json provenance).
"""

# Scoring weights for recommendation components
COMPANY_SIMILARITY_WEIGHT: float = 0.75  # Production company overlap — a weak taste signal
# (Warner Bros produced half the action catalog; at 3.0 it drowned semantics)
COLLECTION_MATCH_BONUS: float = 2.0  # Same franchise/collection (flat +2)
NETWORK_MATCH_BONUS: float = 2.0  # Shared TV network (flat +2, TV only)
# Director DNA — person-id match from tmdb_crew (name-string match kept as
# fallback for rows the crew backfill hasn't reached yet). The channel bonus
# rides on top for candidates the director channel recalled: by construction
# they share the seed's director person_id, the strongest auteur signal the
# overview vectors cannot see (Coen brothers never match by name string).
DIRECTOR_MATCH_BONUS: float = 3.0
# Modest on purpose: writer/cast/behavioral signals carry the rest of the
# auteur signal, so the channel bonus only nudges.
DIRECTOR_CHANNEL_BONUS: float = 3.0
# Writer DNA — a shared screenwriter links taste clusters the director
# signal can't see (Taylor Sheridan's Sicario / Hell or High Water / Wind
# River are different directors, one voice). Weaker than an auteur stamp,
# so it starts lower and applies once per pair (match or channel recall).
WRITER_BONUS: float = 3.5
# Behavioral signal — a title TMDB's own users ranked in the seed's top
# recommendations ("people also watched"). Today it only escapes the
# genre gate; this gives the rank-1-3 recalls real points.
TMDB_REC_BONUS: float = 3.5
# Auteur diversification — geometric decay per additional same-director
# title in the final list (first keeps full score). (1.0, 1.0, 1.0) = off;
# diversification is handled by MMR instead.
AUTEUR_DECAY_FACTORS: tuple = (1.0, 1.0, 1.0)
# Genre-conditional auteur decay (7c): the second-plus same-director title
# whose genres share NOTHING with the seed decays by this factor — Sicario
# (Crime) under Arrival (Sci-Fi/Drama) is director brand without a thematic
# bridge, while Dune (Sci-Fi) keeps its full auteur score.
GENRE_MISMATCH_AUTEUR_FACTOR: float = 0.5
# --- MMR selection (step 7d) ------------------------------------------
# Greedy Maximal Marginal Relevance over the ranked list before the final
# limit: score(c) = λ·relevance_norm − (1−λ)·max_sim(c, selected).
# Relevance is min-max normalized per list so both terms live in [0,1]
# and λ has true 70/30 meaning. Similarity is graded: same collection 1.0
# (sequel pile-up), shared director 0.85 (a second masterpiece can still
# surface), genre Jaccard capped at 0.5 (genre overlap must not erase
# good picks). Replaces the point-penalty decays as THE diversifier.
MMR_ENABLED: bool = True
MMR_LAMBDA: float = 0.8  # tuned for a 70/30 relevance/diversity balance
# Pop-action leak damper — mega-vote titles (>10k votes) whose semantic
# similarity to the seed is below the floor lose POP_ACTION_PENALTY:
# Transformers has genre/company connectivity with Arrival but zero
# thematic overlap, and the light −1.0 popularity damping never stopped it.
POP_ACTION_SIM_FLOOR: float = 0.30
# 0.0 — off; the similarity floor above does the gating.
POP_ACTION_PENALTY: float = 0.0

# Genre priority weights — uniform 1.0 for most genres, with selective
# boosts for strong secondary-genre signals.
# - Music 1.5: when someone searches for a musical (telenovela, teen pop),
#   other musicals should outrank non-musical matches in the same tone
#   (a teen musical should steer toward other teen musicals).
# - Soap 1.3: telenovela format is a strong affinity signal.
# The old values (Mystery=1.6 vs Family=0.7) actively suppressed light
# content: a user searching for a teen musical got dark fantasy because
# dark genres "scored more" per match. Genre matching should reflect
# similarity, not prestige. Only truly niche formats get < 1.0.
GENRE_PRIORITY: dict[str, float] = {
    "Mystery": 1.0,
    "Psychological": 1.0,
    "Science Fiction": 1.0,
    "Horror": 1.0,
    "Thriller": 1.0,
    "Crime": 1.0,
    "Fantasy": 1.0,
    "Action": 1.0,
    "War": 1.0,
    "History": 1.0,
    "Western": 1.0,
    "Adventure": 1.0,
    "Drama": 1.0,
    "Comedy": 1.0,
    "Romance": 1.0,
    "Animation": 1.0,
    "Family": 1.0,
    "Music": 1.5,
    "Soap": 1.3,
    "Documentary": 0.5,
}

# Keyword → style mapping for style matching (+3 points)
KEYWORDS_STYLE: dict[str, set[str]] = {
    "mind-bending": {
        "dream",
        "subconscious",
        "lucid dream",
        "virtual reality",
        "simulation",
        "manipulation",
        "memory loss",
        "amnesia",
        "alternate reality",
        "hallucination",
        "unreliable narrator",
        "mind game",
        "twist ending",
        "psychological thriller",
    },
    "cyberpunk/dystopian": {
        "cyberpunk",
        "dystopia",
        "artificial intelligence",
        "android",
        "transhumanism",
        "megacorporation",
        "post-apocalyptic",
    },
    "time-travel": {
        "time travel",
        "time loop",
        "temporal paradox",
        "parallel timeline",
        "multiverse",
    },
    "noir": {
        "neo-noir",
        "femme fatale",
        "cynical detective",
        "dark city",
        "gritty",
        "conspiracy",
    },
    "buddy cop": {"buddy cop", "bromance", "police partners"},
    "parody": {"parody", "satire", "spoof"},
    "slapstick": {"absurd", "slapstick", "over-the-top"},
}

# Semantic (pgvector KNN) candidate recall: top-N nearest overview
# embeddings fetched per seed IN ADDITION to the genre+popularity candidates.
# Surfaces titles that are close in spirit but would never pass the
# genre-overlap + popularity ordering of the classic recall path.
KNN_CANDIDATES_PER_SEED: int = 60

# Local catalog stores Hebrew genre labels ('מותחן', 'מסתורין') while the
# scoring tables (GENRE_PRIORITY, style matching) use canonical English.
# Map every label found in tmdb_genres so priority weighting actually fires.
GENRE_LABELS_EN: dict[str, str] = {
    "אימה": "Horror",
    "אנימציה": "Animation",
    "אקשן": "Action",
    "אקשן והרפתקאות": "Action",
    "דוקומנטרי": "Documentary",
    "דיבורים": "Talk Show",
    "דרמה": "Drama",
    "הסטוריה": "History",
    "הרפתקאות": "Adventure",
    "חדשות": "News",
    "ילדים": "Family",
    "מדע בדיוני": "Science Fiction",
    "מדע בדיוני ופנטזיה": "Science Fiction",
    "מוסיקה": "Music",
    "מותחן": "Thriller",
    "מלחמה": "War",
    "מלחמה ופוליטיקה": "War",
    "מסתורין": "Mystery",
    "מערבון": "Western",
    "משפחה": "Family",
    "סבון": "Soap",
    "סרט טלויזיה": "TV Movie",
    "פנטזיה": "Fantasy",
    "פשע": "Crime",
    "קומדיה": "Comedy",
    "רומנטי": "Romance",
    "ריאליטי": "Reality",
}

# Candidates below this many TMDB votes are dropped during pre-filtering —
# thin-voted oddities (old/niche titles with a lucky 6.5) should not crowd
# proven classics out of the shortlist.
MIN_VOTE_COUNT: int = 500

# Media-type-aware vote floors (channels AND the pre-filter). Measured on
# a lower TV floor (100/200) widens series recall
# but nets -0.2 (noise outweighs hits), so both stay 500 for now — the
# plumbing remains so the split can be revisited with better tv embeddings.
VOTE_FLOOR_MOVIE: int = 500
VOTE_FLOOR_TV: int = 500


# ----------------------------------------------------------------------------
# Embedding space — "original" (384-dim MiniLM, column `embedding`) or
# "v4b" (768-dim fine-tuned bge, column `embedding_v4`). All engine reads
# (both KNN channels + the cosine feature) follow this switch; embeddings
# are written by scripts/eval/embed_catalog_v4.py with the canonical text.
# ----------------------------------------------------------------------------
import os

REC_EMBEDDING_SPACE = os.getenv("CINE_REC_EMBEDDING", os.getenv("REC_EMBEDDING", "original"))
# Bake-off spaces map to their embedding_<name> columns; "v4b" keeps its
# historical name and "original" the legacy 384-dim column.
EMBEDDING_COLUMN = {
    "v4b": "embedding_v4",
    "original": "embedding",
}.get(REC_EMBEDDING_SPACE, f"embedding_{REC_EMBEDDING_SPACE}")


# Optional cosine blend across several embedding columns:
#   REC_COSINE_BLEND="embedding_mpnetae:32.0,embedding_e5e:8.0,embedding_mpnetan:4.0"
# The final-score cosine becomes sum(w_i/sum(w) * cos_i) * cosine_sim_weight.
# Retrieval (KNN) stays on EMBEDDING_COLUMN — the blend rescores candidates.
def _parse_cosine_blend(raw: str) -> dict:
    blend = {}
    for part in (raw or "").split(","):
        if ":" in part:
            col, w = part.rsplit(":", 1)
            try:
                blend[col.strip()] = float(w)
            except ValueError:
                continue
    return {c: w for c, w in blend.items() if w > 0}


COSINE_BLEND: dict = _parse_cosine_blend(os.getenv("CINE_REC_COSINE_BLEND", os.getenv("REC_COSINE_BLEND", "")))


# On-request user-vector freshness: a stored vector older than this is
# rebuilt inline (ensure_user_vector) before the ANN channel runs, so
# recommend_for_user works with no cron at all — the rebuild costs at
# most one probe query plus one rebuild per window per user. Operators
# running nightly_recompute can raise this to keep the request path
# read-only.
USER_VECTOR_MAX_AGE_HOURS = float(
    os.getenv("CINE_REC_USER_VECTOR_MAX_AGE_HOURS", "24") or 24
)


# Item-to-item user tilt: with a user context present, candidates whose
# scores sit within a hair of each other break ties toward the user's
# taste — a boost-only, multiplicative affinity term capped at this
# fraction of the ORIGINAL score (affinity = cosine to the user vector,
# clamped at 0). The seed's relevance stays primary: a 2× better
# candidate can never be overtaken, and low affinity never buries.
USER_TILT_ALPHA = float(os.getenv("CINE_REC_USER_TILT_ALPHA", "0.15") or 0.15)


# Exploration budget (RFC §2): the share of every personalized list
# reserved for adjacent-cluster discovery. Eligibility is deterministic
# (genre-disjoint from the user's top clusters AND affinity ≥ the main
# list's median) — this targets serendipity, never random noise. 0
# disables; per-request `explore=` overrides (capped at 0.5).
EXPLORE_SHARE = float(os.getenv("CINE_REC_EXPLORE_SHARE", "0.12") or 0.12)

# One skip = a bounded nudge, never a bury: the skipped title's w_item
# is multiplied by this factor (RFC §4). 0.9 ≈ ten skips to halve.
SKIP_DECAY = float(os.getenv("CINE_REC_SKIP_DECAY", "0.9") or 0.9)
