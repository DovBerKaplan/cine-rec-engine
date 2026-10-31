//! Public scoring constants, mirrored from `cine_rec_engine/config.py`
//! and `scoring.py`. These are TASTE/BEHAVIOR data (genre prices,
//! style keyword lists, canonical label mapping), not tuned
//! coefficients — the fitted weights stay in the artifact the Python
//! side passes in at runtime.
//!
//! Keep in sync with config.py: the parity harness catches drift on
//! the golden inputs the moment any value here disagrees.

/// `config.GENRE_PRIORITY` — relative taste weight per canonical
/// genre (Music/Soap priced above blockbusters, Documentary below).
pub const GENRE_PRIORITY: &[(&str, f64)] = &[
    ("Mystery", 1.0),
    ("Psychological", 1.0),
    ("Science Fiction", 1.0),
    ("Horror", 1.0),
    ("Thriller", 1.0),
    ("Crime", 1.0),
    ("Fantasy", 1.0),
    ("Action", 1.0),
    ("War", 1.0),
    ("History", 1.0),
    ("Western", 1.0),
    ("Adventure", 1.0),
    ("Drama", 1.0),
    ("Comedy", 1.0),
    ("Romance", 1.0),
    ("Animation", 1.0),
    ("Family", 1.0),
    ("Music", 1.5),
    ("Soap", 1.3),
    ("Documentary", 0.5),
];

/// `config.POP_ACTION_SIM_FLOOR` — cosine similarity below which a
/// high-vote candidate reads as popular-noise leakage.
pub const POP_ACTION_SIM_FLOOR: f64 = 0.30;

/// `config.KEYWORDS_STYLE` — style labels and the plot keywords that
/// trigger them (order mirrors the dict for identical style output).
pub const KEYWORDS_STYLE: &[(&str, &[&str])] = &[
    (
        "mind-bending",
        &[
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
        ],
    ),
    (
        "cyberpunk/dystopian",
        &[
            "cyberpunk",
            "dystopia",
            "artificial intelligence",
            "android",
            "transhumanism",
            "megacorporation",
            "post-apocalyptic",
        ],
    ),
    (
        "time-travel",
        &[
            "time travel",
            "time loop",
            "temporal paradox",
            "parallel timeline",
            "multiverse",
        ],
    ),
    (
        "noir",
        &[
            "neo-noir",
            "femme fatale",
            "cynical detective",
            "dark city",
            "gritty",
            "conspiracy",
        ],
    ),
    ("buddy cop", &["buddy cop", "bromance", "police partners"]),
    ("parody", &["parody", "satire", "spoof"]),
    ("slapstick", &["absurd", "slapstick", "over-the-top"]),
];

/// `config.GENRE_LABELS_EN` — local Hebrew genre labels to canonical
/// English names (without this mapping genre weighting is dead code
/// on real catalog data).
pub const GENRE_LABELS_EN: &[(&str, &str)] = &[
    ("אימה", "Horror"),
    ("אנימציה", "Animation"),
    ("אקשן", "Action"),
    ("אקשן והרפתקאות", "Action"),
    ("דוקומנטרי", "Documentary"),
    ("דיבורים", "Talk Show"),
    ("דרמה", "Drama"),
    ("הסטוריה", "History"),
    ("הרפתקאות", "Adventure"),
    ("חדשות", "News"),
    ("ילדים", "Family"),
    ("מדע בדיוני", "Science Fiction"),
    ("מדע בדיוני ופנטזיה", "Science Fiction"),
    ("מוסיקה", "Music"),
    ("מותחן", "Thriller"),
    ("מלחמה", "War"),
    ("מלחמה ופוליטיקה", "War"),
    ("מסתורין", "Mystery"),
    ("מערבון", "Western"),
    ("משפחה", "Family"),
    ("סבון", "Soap"),
    ("סרט טליבזיה", "TV Movie"),
    ("פנטזיה", "Fantasy"),
    ("פשע", "Crime"),
    ("קומדיה", "Comedy"),
    ("רומנטי", "Romance"),
    ("ריאליטי", "Reality"),
];

/// `scoring.META_KEYWORD_STOPWORDS` — production-fact keywords with
/// zero taste signal; filtered before any keyword similarity.
pub const META_KEYWORD_STOPWORDS: &[&str] = &[
    "based on novel or book",
    "based on book",
    "based on novel",
    "based on play",
    "based on true story",
    "based on comic",
    "based on manga",
    "based on video game",
    "based on tv series",
    "woman director",
    "sequel",
    "prequel",
    "remake",
    "spin off",
    "duringcreditsstinger",
    "aftercreditsstinger",
    "short film",
    "biography",
    "silent film",
    "3d",
    "imax",
    "anime",
    "adaptation",
    "reboot",
];

/// `scoring.LIGHT_GENRES` / `DARK_GENRES` — emotional-register sets.
pub const LIGHT_GENRES: &[&str] = &[
    "Comedy",
    "Family",
    "Music",
    "Romance",
    "Soap",
    "Kids",
    "Animation",
];
pub const DARK_GENRES: &[&str] = &["Horror", "Thriller", "Mystery", "Crime"];

/// `scoring.LIGHT_ARC_TAGS` / `DARK_ARC_TAGS` — lowercase arc tags.
pub const LIGHT_ARC_TAGS: &[&str] = &[
    "feel-good",
    "heartwarming",
    "uplifting",
    "hopeful",
    "cozy",
    "feel-good",
];
pub const DARK_ARC_TAGS: &[&str] = &[
    "dark",
    "gritty",
    "bleak",
    "disturbing",
    "tense",
    "suspenseful",
];

/// `scoring.YOUTH_GENRES` — youth-audience markers.
pub const YOUTH_GENRES: &[&str] = &["Kids", "Family", "Animation", "Soap"];

/// Adult-leaning canonical genres (audience_score's inner check).
pub const ADULT_LEANING_GENRES: &[&str] = &["Crime", "Horror", "Thriller"];

pub fn genre_priority(label: &str) -> f64 {
    // GENRE_PRIORITY.get(g, 1.0) — everything unlisted prices at 1.0
    GENRE_PRIORITY
        .iter()
        .find(|(k, _)| *k == label)
        .map(|(_, v)| *v)
        .unwrap_or(1.0)
}
