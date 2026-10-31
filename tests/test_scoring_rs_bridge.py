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
    c = dict(base, id=5, genres=["מותחן"], keywords=["dream"])
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
def test_bridge_rejects_empty_entity_data_loudly():
    """Python treats an empty entity_data dict inconsistently (falsy
    for narrative fields, not-None for the enriched source); the
    bridge refuses it instead of silently diverging."""
    seed = {"id": 1, "media_type": "movie", "genres": ["Drama"]}
    with pytest.raises(TypeError):
        S._entity_profile(seed, {}, genres_override=seed["genres"])


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
