"""Rust-acceleration bridge: output-identity with the pure-Python
implementations, on the frozen golden inputs AND on edge shapes the
goldens don't carry.

The suite passes in BOTH deployments:
* with the compiled extension installed — the accelerated path is
  compared head-to-head against the pure-Python aliases;
* without it (pure wheel) — the availability tests skip, and the
  fallback flag / disable-switch tests still run.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import cine_rec_engine.service as S

RUST_AVAILABLE = S.SCORING_RS.startswith("rust")

FIXTURES = Path(__file__).parent.parent / "rust" / "scoring" / "tests" / "fixtures"


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


SIGN = 0x8000000000000000
MASK = 0xFFFFFFFFFFFFFFFF


def _key(x: float) -> int:
    b = _bits(x)
    return (b ^ SIGN) if b & SIGN else (~b & MASK)


def _ulp(a: float, b: float) -> int:
    if a == b:
        return 0
    return abs(_key(a) - _key(b))


def _pairs(fx):
    for c in fx["cands"]:
        for s in fx["seeds"]:
            yield c, s


# ── availability & flags ───────────────────────────────────────────


def test_scoring_rs_flag_is_consistent():
    if (FIXTURES.parent.parent.parent / "cine_rec_engine" / "_scoring_rs.so").exists():
        assert RUST_AVAILABLE, "extension present but not active"
    # either way the engine must have booted to SOME valid mode
    assert S.SCORING_RS.startswith(("rust", "python-fallback", "disabled"))


def test_disable_switch_forces_python():
    """CINE_REC_SCORING_RS=0 must boot the module on the pure path
    (checked in a subprocess — the flag is read at import time)."""
    code = (
        "import cine_rec_engine.service as s; "
        "assert s.SCORING_RS.startswith('disabled'), s.SCORING_RS; "
        "assert s.feature_vector is s._py_feature_vector; "
        "assert s._score_pair_fast is s._py_score_pair_fast"
    )
    env = dict(os.environ, CINE_REC_SCORING_RS="0")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


# ── head-to-head parity on the frozen inputs ───────────────────────


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
@pytest.mark.parametrize("fixture_name", ["entities_120_4.json", "entities_enriched.json"])
def test_bridge_matches_python_on_frozen_sets(fixture_name):
    fx = json.loads((FIXTURES / fixture_name).read_text())
    w = list(S.HEURISTIC_WEIGHTS.values())
    checked = 0
    for cand, seed in _pairs(fx):
        # feature_vector: bit-exact
        vec_rs = S.feature_vector(cand, cand, cand["genres"], seed)
        vec_py = S._py_feature_vector(cand, cand, cand["genres"], seed)
        assert len(vec_rs) == 28
        assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py)), (
            f"vec diverged for cand={cand['id']} seed={seed['id']}"
        )
        # profiles + fast score: <= 1 ULP
        sp = S._entity_profile(seed, None)
        cp = S._entity_profile(cand, cand, genres_override=cand["genres"])
        fast_rs = S._score_pair_fast(
            sp, cp, cand, cand, set(cand["genres"]) or None, 0.0, w
        )
        sp_py = S._py_entity_profile(seed, None)
        cp_py = S._py_entity_profile(cand, cand, genres_override=cand["genres"])
        fast_py = S._py_score_pair_fast(
            sp_py, cp_py, cand, cand, set(cand["genres"]) or None, 0.0, w
        )
        assert _ulp(fast_rs, fast_py) <= 1, (
            f"score diverged ({_ulp(fast_rs, fast_py)} ULP) for "
            f"cand={cand['id']} seed={seed['id']}: {fast_rs!r} vs {fast_py!r}"
        )
        checked += 1
    assert checked > 0  # a bench that measured nothing is a bug


# ── edge shapes the goldens don't carry ────────────────────────────


def _edge_pairs():
    base = {
        "id": 1,
        "media_type": "movie",
        "genres": ["Drama"],
        "keywords": [],
        "cast_ids": [],
        "director_ids": [],
        "writer_ids": [],
        "composer_ids": [],
        "dp_ids": [],
        "companies": [],
        "networks": [],
        "original_language": None,
        "overview": None,
        "overview_en": None,
        "release_year": None,
        "vote_average": None,
        "vote_count": None,
        "collection_id": None,
        "adult": False,
        "via": None,
    }
    seed = dict(base, id=100, genres=["Action"], overview="A detective story")
    out = []

    # cast_list by NAMES on both sides (recall SQL shape) — matches by
    # name, and name-vs-id must never match
    c = dict(base, id=2, cast_ids=[], cast_list=["Alpha Actor", "Beta Actor"])
    s = dict(seed, id=101, cast_ids=[], cast_list=["Beta Actor", "Gamma Actor"])
    out.append(("cast-by-name", c, s))

    # empty via string: NOT gate-exempt (clash -> gated to 0.0)
    c = dict(base, id=3, genres=["War"], via="")
    out.append(("empty-via-clash", c, seed))

    # year 0 on one side (falsy in Python -> flags stay 0)
    c = dict(base, id=4, release_year=0, vote_average=7.0, vote_count=100)
    s2 = dict(seed, id=102, release_year=2000)
    out.append(("year-zero", c, s2))

    # Hebrew genre labels canonicalize
    c = dict(base, id=5, genres=["Thriller"], keywords=["dream"])
    s3 = dict(seed, id=103, genres=["Thriller"], keywords=["dream", "gritty"])
    out.append(("hebrew-labels", c, s3))

    # empty-string fields everywhere (falsy strings)
    c = dict(
        base, id=6, original_language="", overview="", overview_en="",
        director="Someone", via=None,
    )
    s4 = dict(seed, id=104, director="Someone")
    out.append(("empty-strings", c, s4))

    # tmdb_rank=0 (falsy -> treated as rank 1) with via=tmdb
    c = dict(base, id=7, via="tmdb", tmdb_rank=0, genres=["Action"], vote_average=8.0)
    out.append(("tmdb-rank-zero", c, seed))

    # adult youth-vs-adult audience clash
    c = dict(base, id=8, genres=["Horror", "Crime"], adult=True)
    s5 = dict(seed, id=105, genres=["Animation", "Family"])
    out.append(("audience-clash", c, s5))

    # narrative/pacing/emotional_arc agreement + unicode overview
    c = dict(
        base, id=9, genres=["Drama"], narrative="linear", pacing="contemplative",
        emotional_arc=["Feel-Good  "], overview="סיפור על גשם",
    )
    s6 = dict(
        seed, id=106, narrative="linear", pacing="contemplative",
        emotional_arc=["feel-good"],
    )
    out.append(("narrative-agreement", c, s6))

    # EMPTY genres on the candidate (cand_genres_set None vs empty set)
    c = dict(base, id=10, genres=[])
    out.append(("empty-genres-none", c, seed))
    out.append(("empty-genres-empty-set", c, seed))
    return out


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
@pytest.mark.parametrize(
    "label,cand,seed", [(p[0], p[1], p[2]) for p in _edge_pairs()]
)
def test_bridge_matches_python_on_edges(label, cand, seed):
    w = list(S.HEURISTIC_WEIGHTS.values())

    vec_rs = S.feature_vector(cand, cand, cand.get("genres", []), seed)
    vec_py = S._py_feature_vector(cand, cand, cand.get("genres", []), seed)
    assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py)), label

    sp = S._entity_profile(seed, None)
    cp = S._entity_profile(cand, cand, genres_override=cand.get("genres"))
    sp_py = S._py_entity_profile(seed, None)
    cp_py = S._py_entity_profile(cand, cand, genres_override=cand.get("genres"))

    if label == "empty-genres-empty-set":
        cgs_rs, cgs_py = set(), set()  # empty set (not None)
    else:
        cgs_rs = cgs_py = set(cand.get("genres", [])) or None

    fast_rs = S._score_pair_fast(sp, cp, cand, cand, cgs_rs, 0.0, w)
    fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, cand, cgs_py, 0.0, w)
    assert _ulp(fast_rs, fast_py) <= 1, f"{label}: {fast_rs!r} vs {fast_py!r}"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_bridge_empty_entity_data_matches_python():
    """`enriched.get(cid, {})` — an EMPTY entity_data dict is the
    regular production input for every non-enriched candidate (the
    user-vector ANN extras attach after enrichment). Python semantics:
    enriched-source fields read the empty dict (features zero),
    narrative/pacing/emotional-arc fall back to the ENTITY. The bridge
    must reproduce that exactly — this was the phase-2 blocker."""
    cand = {
        "id": 2, "media_type": "movie", "genres": ["Drama", "Thriller"],
        "keywords": ["dream"], "cast_ids": [7],
        "director_ids": [3], "writer_ids": [], "composer_ids": [], "dp_ids": [],
        "companies": [], "networks": [],
        "original_language": "en", "overview": "A story", "overview_en": None,
        "release_year": 1999, "vote_average": 7.5, "vote_count": 900,
        "collection_id": 55, "adult": False, "via": None,
        "narrative": "linear", "pacing": "contemplative",
        "emotional_arc": ["tense"],
    }
    seed = {
        "id": 100, "media_type": "movie", "genres": ["Drama"],
        "keywords": ["dream"], "cast_ids": [7], "director_ids": [3],
        "writer_ids": [], "composer_ids": [], "dp_ids": [], "companies": [],
        "networks": [], "original_language": "en",
        "overview": "Another story", "overview_en": None,
        "release_year": 2001, "vote_average": 8.0, "vote_count": 800,
        "collection_id": 55, "adult": False, "via": None,
        "narrative": "linear", "pacing": "contemplative",
        "emotional_arc": ["tense"],
    }
    w = list(S.HEURISTIC_WEIGHTS.values())

    for data in ({}, cand):  # the {} production shape AND the identity shape
        vec_rs = S.feature_vector(cand, data, cand["genres"], seed)
        vec_py = S._py_feature_vector(cand, data, cand["genres"], seed)
        assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py)), data

        sp = S._entity_profile(seed, None)
        cp = S._entity_profile(cand, data, genres_override=cand["genres"])
        sp_py = S._py_entity_profile(seed, None)
        cp_py = S._py_entity_profile(cand, data, genres_override=cand["genres"])
        fast_rs = S._score_pair_fast(sp, cp, cand, data, set(cand["genres"]), 0.0, w)
        fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, data, set(cand["genres"]), 0.0, w)
        assert _ulp(fast_rs, fast_py) <= 1, (data, fast_rs, fast_py)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_bridge_cand_data_differs_from_cand():
    """The production shape where cand_data is a DIFFERENT (enriched)
    row: director-name fallback, collection fallback, via and
    tmdb_rank read from cand while enrichment comes from the data."""
    cand = {
        "id": 3, "media_type": "movie", "genres": ["Action"],
        "keywords": [], "cast_ids": [], "director_ids": [], "writer_ids": [],
        "composer_ids": [], "dp_ids": [], "companies": [], "networks": [],
        "original_language": None, "overview": None, "overview_en": None,
        "release_year": 2010, "vote_average": 6.0, "vote_count": 50,
        "collection_id": 9, "adult": False, "via": "tmdb", "tmdb_rank": 2,
    }
    data = {
        "id": 3, "keywords": ["gritty", "conspiracy"], "cast_ids": [1, 2, 3],
        "director_ids": [42], "writer_ids": [], "composer_ids": [9], "dp_ids": [],
        "companies": ["Co"], "networks": [], "original_language": "en",
        "overview": "enriched overview", "overview_en": "enriched overview",
        "collection_id": 0,  # falsy in Python -> falls back to cand's 9
        "narrative": None, "pacing": "", "emotional_arc": None,
    }
    seed = {
        "id": 200, "media_type": "tv", "genres": ["Action", "Crime"],
        "keywords": ["conspiracy"], "cast_ids": [3, 4], "director_ids": [42],
        "writer_ids": [], "composer_ids": [], "dp_ids": [], "companies": [],
        "networks": ["NT"], "original_language": "en",
        "overview": "seed overview", "overview_en": None,
        "release_year": 2015, "vote_average": 7.0, "vote_count": 3000,
        "collection_id": 9, "adult": False, "via": None,
        "director": "Some Name",
    }
    data["director"] = "Some Name"  # name fallback via cand_data
    w = list(S.HEURISTIC_WEIGHTS.values())

    vec_rs = S.feature_vector(cand, data, cand["genres"], seed)
    vec_py = S._py_feature_vector(cand, data, cand["genres"], seed)
    assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py))

    sp = S._entity_profile(seed, None)
    cp = S._entity_profile(cand, data, genres_override=cand["genres"])
    sp_py = S._py_entity_profile(seed, None)
    cp_py = S._py_entity_profile(cand, data, genres_override=cand["genres"])
    fast_rs = S._score_pair_fast(sp, cp, cand, data, set(cand["genres"]), 0.0, w)
    fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, data, set(cand["genres"]), 0.0, w)
    assert _ulp(fast_rs, fast_py) <= 1


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_bridge_pacing_and_collection_truthiness():
    """pacing='' is falsy (no pacing match); collection_id=0 is falsy
    (no shared collection, and data's 0 falls back to cand's)."""
    base = {
        "id": 1, "media_type": "movie", "genres": ["Drama"], "keywords": [],
        "cast_ids": [], "director_ids": [], "writer_ids": [], "composer_ids": [],
        "dp_ids": [], "companies": [], "networks": [],
        "original_language": None, "overview": None, "overview_en": None,
        "release_year": 2000, "vote_average": 7.0, "vote_count": 100,
        "collection_id": None, "adult": False, "via": None,
    }
    cand = dict(base, id=2, pacing="", collection_id=0)
    seed = dict(base, id=300, pacing="", collection_id=0)
    vec_rs = S.feature_vector(cand, cand, cand["genres"], seed)
    vec_py = S._py_feature_vector(cand, cand, cand["genres"], seed)
    assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py))
    # pacing match must NOT fire for empty strings (slot 25 == 0.0)
    name_i = S.FEATURE_NAMES.index("pacing_match")
    assert vec_rs[name_i] == 0.0
    name_i = S.FEATURE_NAMES.index("shared_collection")
    assert vec_rs[name_i] == 0.0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_bridge_gil_released_path_is_thread_safe():
    """Parallel threads through the GIL-releasing scorer stay exact."""
    import threading

    fx = json.loads((FIXTURES / "entities_120_4.json").read_text())
    w = list(S.HEURISTIC_WEIGHTS.values())
    cands, seeds = fx["cands"][:20], fx["seeds"]
    expected = {}
    for c in cands:
        for s in seeds:
            expected[(c["id"], s["id"])] = S._py_score_pair_fast(
                S._py_entity_profile(s, None),
                S._py_entity_profile(c, c, genres_override=c["genres"]),
                c, c, set(c["genres"]) or None, 0.0, w,
            )
    got: dict = {}
    lock = threading.Lock()

    def work(chunk):
        sps = {s["id"]: S._entity_profile(s, None) for s in seeds}
        local = {}
        for c in chunk:
            cp = S._entity_profile(c, c, genres_override=c["genres"])
            for s in seeds:
                local[(c["id"], s["id"])] = S._score_pair_fast(
                    sps[s["id"]], cp, c, c, set(c["genres"]) or None, 0.0, w
                )
        with lock:
            got.update(local)

    threads = [
        threading.Thread(target=work, args=(cands[i::4],)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == len(expected) == 80
    for key, py_score in expected.items():
        assert _ulp(got[key], py_score) <= 1, key


# ── pure-Python fallback sanity (runs even without the extension) ──


def test_python_fallback_aliases_are_the_real_implementations():
    fx = json.loads((FIXTURES / "entities_enriched.json").read_text())
    cand, seed = fx["cands"][0], fx["seeds"][0]
    vec = S._py_feature_vector(cand, cand, cand["genres"], seed)
    assert len(vec) == 28
    golden = json.loads((FIXTURES / "goldens_enriched.json").read_text())
    assert all(_bits(a) == _bits(b) for a, b in zip(vec, golden[0]["vec"]))


# ── M-7: differential fuzz over the production-shaped input space ──
# Recipe from the phase-2 review: cand ≠ cand_data, overview_sim ≠ 0,
# empty data dicts, empty strings, falsy zeros — everything the frozen
# goldens (which always pass cand_data == cand, overview_sim = 0) do
# not reach.

_VIA = [None, "", "knn", "tmdb", "director", "writer", "text"]
_GENRES = ["Drama", "Action", "Thriller", "Animation", "War", ""]
_ARCS = [None, [], ["feel-good"], ["Feel-Good  "], ["dark", "Tense"], ["x"]]


@pytest.fixture(scope="module")
def _fuzz_entities():
    import random

    rng = random.Random(20261031)  # deterministic corpus
    rows = []
    for i in range(60):
        rows.append(
            {
                "id": 1000 + i,
                "media_type": rng.choice(["movie", "tv"]),
                "genres": rng.sample(_GENRES, k=rng.randint(0, 3)),
                "keywords": rng.sample(
                    ["dream", "gritty", "conspiracy", "time loop", "parody"], k=rng.randint(0, 3)
                ),
                "cast_ids": rng.sample(range(50), k=rng.randint(0, 5)),
                "director_ids": rng.sample(range(20), k=rng.randint(0, 2)),
                "writer_ids": rng.sample(range(20), k=rng.randint(0, 2)),
                "composer_ids": rng.sample(range(20), k=rng.randint(0, 1)),
                "dp_ids": rng.sample(range(20), k=rng.randint(0, 1)),
                "companies": rng.sample(["Co", "Studio"], k=rng.randint(0, 2)),
                "networks": rng.sample(["NT", "NT2"], k=rng.randint(0, 2)),
                "original_language": rng.choice([None, "en", "ja", ""]),
                "overview": rng.choice([None, "", "A detective story", "סיפור על גשם"]),
                "overview_en": rng.choice([None, "", "English text"]),
                "release_year": rng.choice([None, 0, 1960, 1999, 2020]),
                "vote_average": rng.choice([None, 0.0, 5.5, 7.2, 9.0]),
                "vote_count": rng.choice([None, 0, 100, 20000]),
                "collection_id": rng.choice([None, 0, 7, 300]),
                "adult": rng.choice([False, True]),
                "via": rng.choice(_VIA),
                "tmdb_rank": rng.choice([None, 0, 1, 4]),
                "narrative": rng.choice([None, "", "linear", "anthology"]),
                "pacing": rng.choice([None, "", "contemplative", "propulsive"]),
                "emotional_arc": rng.choice(_ARCS),
                "director": rng.choice([None, "", "Some Name"]),
            }
        )
    return rows


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_differential_fuzz_rust_vs_python(_fuzz_entities):
    """Every (cand, data, seed) shape in the corpus, both feature_vector
    and the fast score: Rust must equal pure Python to the contract
    (bit-exact vec, <=1 ULP score)."""
    import random

    rng = random.Random(7)
    rows = _fuzz_entities
    w = list(S.HEURISTIC_WEIGHTS.values())
    checked = 0
    for cand in rows:
        for seed in rng.sample(rows, 6):
            for data in (cand, {}, dict(cand, id=cand["id"], cast_ids=[], keywords=[])):
                overview_sim = rng.choice([0.0, rng.random(), 1.0])
                vec_rs = S.feature_vector(cand, data, cand.get("genres", []), seed, overview_sim)
                vec_py = S._py_feature_vector(
                    cand, data, cand.get("genres", []), seed, overview_sim
                )
                assert len(vec_rs) == 28
                for i, (a, b) in enumerate(zip(vec_rs, vec_py)):
                    assert _bits(a) == _bits(b), (
                        f"vec slot {i} ({S.FEATURE_NAMES[i]}) cand={cand['id']} "
                        f"seed={seed['id']} data={'{}' if data == {} else 'row'} "
                        f"sim={overview_sim:.3f}: {a!r} vs {b!r}"
                    )
                sp = S._entity_profile(seed, None)
                cp = S._entity_profile(cand, data, genres_override=cand.get("genres"))
                sp_py = S._py_entity_profile(seed, None)
                cp_py = S._py_entity_profile(cand, data, genres_override=cand.get("genres"))
                cgs = set(cand.get("genres", [])) or None
                fast_rs = S._score_pair_fast(sp, cp, cand, data, cgs, 0.0, w)
                fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, data, cgs, 0.0, w)
                assert _ulp(fast_rs, fast_py) <= 1, (
                    f"score cand={cand['id']} seed={seed['id']} "
                    f"data={'{}' if data == {} else 'row'}: {fast_rs!r} vs {fast_py!r}"
                )
                checked += 1
    assert checked > 500  # a differential test that checked nothing is a bug


# ── M-8: tests for the mixed-pair root fix (phase-2 review round 3) ──
# Python's set() compares VALUES (None == None), sizes enter the
# Jaccard/Dice denominators, and [:5] slices BEFORE any filtering —
# so item-skipping conversions produced wrong-but-plausible scores.
# These pin the exact repro cases + the rebuild-don't-crash contract.


def _repro_base(**overrides):
    """Deterministic base row (bench-shaped)."""
    row = {
        "id": 1,
        "media_type": "movie",
        "genres": ["Action"],
        "keywords": ["dream"],
        "cast_ids": [11],
        "director_ids": [4],
        "writer_ids": [],
        "composer_ids": [],
        "dp_ids": [],
        "companies": ["co0"],
        "networks": [],
        "original_language": "en",
        "overview": "A story",
        "overview_en": None,
        "release_year": 2001,
        "vote_average": 7.0,
        "vote_count": 100,
        "collection_id": None,
        "adult": False,
        "via": None,
    }
    row.update(overrides)
    return row


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
@pytest.mark.parametrize(
    "cand_over,seed_over",
    [
        # cast: None is a VALUE — matches None on the other side
        ({"cast_ids": [1, None]}, {"cast_ids": [2, None]}),
        # cast by names: equal names match
        ({"cast_ids": ["a", "b"]}, {"cast_ids": ["a", "c"]}),
        # [:5] slices BEFORE anything: the trailing 9 never enters the set
        ({"cast_ids": [None, 1, 2, 3, 4, 9]}, {"cast_ids": [9]}),
        # director None == None
        ({"director_ids": [None]}, {"director_ids": [None]}),
        # companies with None members
        ({"companies": ["co1", None]}, {"companies": [None]}),
        # non-integral numeric rows must not crash the request
        ({"vote_count": 15000.5}, {}),
        ({"release_year": 2001.6}, {}),
        # surrogate strings in overview: cand side
        ({"overview": "x \ud800"}, {}),
        # surrogate strings in overview: seed side
        ({}, {"overview": "y \ud800"}),
    ],
)
def test_mixed_and_value_semantics_repros(cand_over, seed_over):
    w = list(S.HEURISTIC_WEIGHTS.values())
    cand = _repro_base(id=2, **cand_over)
    seed = _repro_base(id=100, genres=["Drama"], **seed_over)

    vec_rs = S.feature_vector(cand, cand, cand["genres"], seed)
    vec_py = S._py_feature_vector(cand, cand, cand["genres"], seed)
    assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py)), (
        cand_over,
        seed_over,
    )

    sp = S._entity_profile(seed, None)
    cp = S._entity_profile(cand, cand, genres_override=cand["genres"])
    sp_py = S._py_entity_profile(seed, None)
    cp_py = S._py_entity_profile(cand, cand, genres_override=cand["genres"])
    fast_rs = S._score_pair_fast(sp, cp, cand, cand, set(cand["genres"]), 0.0, w)
    fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, cand, set(cand["genres"]), 0.0, w)
    assert _ulp(fast_rs, fast_py) <= 1, (cand_over, seed_over, fast_rs, fast_py)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_mixed_profile_pair_rebuilds_pure():
    """A Rust seed profile + a pure-Python candidate profile (the exact
    crash shape: an exotic candidate row fell back at conversion) must
    score identically to the pure path — never raise mixed-kinds."""
    w = list(S.HEURISTIC_WEIGHTS.values())
    clean_seed = _repro_base(id=100, genres=["Drama"], cast_ids=[7, 8])
    exotic_cand = _repro_base(id=2, vote_count=15000.5, cast_ids=[7, 9])

    sp_rust = S._entity_profile(clean_seed, None)  # Rust handle
    assert type(sp_rust).__name__ == "PyProfile"
    # the exotic row's conversion fails -> wrapper returns a pure dict
    cp_pure = S._entity_profile(exotic_cand, exotic_cand, genres_override=exotic_cand["genres"])
    assert isinstance(cp_pure, dict)

    got = S._score_pair_fast(sp_rust, cp_pure, exotic_cand, exotic_cand,
                             set(exotic_cand["genres"]), 0.0, w)
    sp_py = S._py_entity_profile(clean_seed, None)
    cp_py = S._py_entity_profile(exotic_cand, exotic_cand, genres_override=exotic_cand["genres"])
    want = S._py_score_pair_fast(sp_py, cp_py, exotic_cand, exotic_cand,
                                 set(exotic_cand["genres"]), 0.0, w)
    assert _ulp(got, want) <= 1

    # and the mirror mix (pure seed + Rust candidate)
    got2 = S._score_pair_fast(sp_py, S._entity_profile(exotic_cand, exotic_cand,
                                                       genres_override=exotic_cand["genres"]),
                              exotic_cand, exotic_cand, set(exotic_cand["genres"]), 0.0, w)
    assert _ulp(got2, want) <= 1


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled _scoring_rs not installed")
def test_differential_fuzz_typed_exotic_rows():
    """The reviewer's typed/exotic profile: non-integral floats, Decimals,
    None/str items inside id lists, sets/tuples as lists, via=True,
    adult variants, 2**70 ids, surrogates — the service call must never
    raise where _py_* doesn't, and outputs must match to the contract."""
    import random
    from decimal import Decimal

    rng = random.Random(4242)
    w = list(S.HEURISTIC_WEIGHTS.values())

    def exotic_row(i):
        return {
            "id": rng.choice([i, 2**70, -1]),
            "media_type": "movie",
            "genres": rng.choice([["Action"], ["Thriller"], [], ["Action", "Drama"]]),
            "keywords": rng.choice([("dream",), ["dream", "gritty"], [], ["x" * 3]]),
            "cast_ids": rng.choice(
                [[1, None], ["a", "b"], [None, 1, 2, 3, 4, 9], [], [2.0], ["a"]]
            ),
            "director_ids": rng.choice([[None], [4], [], [3, None]]),
            "writer_ids": [],
            "composer_ids": [],
            "dp_ids": [],
            "companies": rng.choice([{"co1", None}, ["co1", None], [], ["x"]]),
            "networks": [],
            "original_language": "en",
            "overview": rng.choice(["A story", "x \ud800", "", None]),
            "overview_en": None,
            "release_year": rng.choice([2001, 2001.6, 0, None, Decimal("2001")]),
            "vote_average": 7.0,
            "vote_count": rng.choice([100, 15000.5, 20000, Decimal("900"), 0]),
            "collection_id": rng.choice([None, 0, 7]),
            "adult": rng.choice([False, True, 0, 1, "", "yes", None]),
            "via": rng.choice([None, "", "knn", True, "tmdb"]),
        }

    rows = [exotic_row(i) for i in range(25)]
    checked = 0
    for cand in rows:
        for seed in rows:
            if seed is cand:
                continue
            for data in (cand, {}):
                overview_sim = rng.choice([0.0, rng.random(), 1.0])
                # the pure path defines what may not raise
                vec_py = S._py_feature_vector(
                    cand, data, cand.get("genres", []), seed, overview_sim
                )
                vec_rs = S.feature_vector(cand, data, cand.get("genres", []), seed, overview_sim)
                assert all(_bits(a) == _bits(b) for a, b in zip(vec_rs, vec_py)), (
                    f"vec diverged cand={cand['id']} seed={seed['id']}"
                )
                sp = S._entity_profile(seed, None)
                cp = S._entity_profile(cand, data, genres_override=cand.get("genres"))
                sp_py = S._py_entity_profile(seed, None)
                cp_py = S._py_entity_profile(cand, data, genres_override=cand.get("genres"))
                cgs = set(cand.get("genres", [])) or None
                fast_rs = S._score_pair_fast(sp, cp, cand, data, cgs, 0.0, w)
                fast_py = S._py_score_pair_fast(sp_py, cp_py, cand, data, cgs, 0.0, w)
                assert _ulp(fast_rs, fast_py) <= 1, (
                    f"score diverged cand={cand['id']} seed={seed['id']}"
                )
                checked += 1
    assert checked > 500


def test_extension_file_problem_flags_every_truncation():
    """The ELF structural gate, exercised on real bytes (the file is
    copied and truncated — NEVER imported)."""
    import tempfile

    from cine_rec_engine import _scoring_accel as A

    src = Path(__file__).parent.parent / "cine_rec_engine" / "_scoring_rs.so"
    if not src.exists():
        pytest.skip("compiled _scoring_rs not installed")
    full = src.read_bytes()
    assert len(full) > 400_000

    with tempfile.TemporaryDirectory() as d:
        full_copy = Path(d) / "full.so"
        full_copy.write_bytes(full)
        assert A._extension_file_problem(str(full_copy)) is None

        for cut in (0, 64, 4096, 400_000, len(full) // 2, len(full) - 1):
            p = Path(d) / f"cut_{cut}.so"
            p.write_bytes(full[:cut])
            problem = A._extension_file_problem(str(p))
            assert problem is not None, f"cut at {cut} must be flagged"
            assert "truncated" in problem
