"""Optional Rust acceleration for the scoring hot path.

Rebinds the service module's `_entity_profile` / `_score_pair_fast` /
`feature_vector` dispatchers to the compiled `_scoring_rs` extension.
The pure-Python implementations (`_py_*` in the service module) remain
the source of truth; this module installs only when the extension
imports cleanly AND the interpreter matches its float-sum semantics —
degrade, never fail.

Guards, in order:
* Python >= 3.12 — CPython changed `sum()` over floats to compensated
  (Neumaier) summation in 3.12; the port replicates the 3.12 fold, so
  on 3.10/3.11 the extension would NOT be output-identical to the pure
  path and stays off.
* the extension file (when locatable) passes a minimal size sanity
  check — a TRUNCATED ``.so`` can kill the process with SIGBUS before
  any Python-level ``except`` can run, so obviously-broken files are
  skipped up front. This is a heuristic, not a hash: the full escape
  hatch is ``CINE_REC_SCORING_RS=0``.
* importing the module and building the scorers succeeds.

Per-call fallback: a conversion error inside one accelerated call
(exotic field types) falls back to the pure implementation FOR THAT
CALL. Profiles dispatch by type — a dict profile means the pure path,
an opaque extension profile means the Rust one; a MIXED pair cannot be
recovered (one side cannot be rebuilt) and raises loudly instead of
guessing.

Engine caching: `PairScorer` binds the weight lists at construction,
so converting them per pair would eat the win. Callers pass the same
list object per process (`HEURISTIC_ACTIVE_LIST`); the cache is a
single atomic tuple assignment — readers either see the old pair or
the new one, never a mixture. Do not mutate a weights list in place
after passing it here.

The per-pair call releases the GIL for the math, so EMBEDDERS running
the engine on multiple threads parallelize for real (the package
itself is single-threaded today).
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional

# CPython switched sum() to compensated summation in 3.12; the Rust
# port's py_sum replicates THAT fold. Below 3.12 the extension would
# differ from the pure path in the last ULP and must not install.
_MIN_PY = (3, 12)

# A healthy extension is ~850 KB; a truncated one (interrupted copy,
# full disk) can SIGBUS the process at import, below any except. This
# floor skips obviously-broken files early — not a substitute for the
# CINE_REC_SCORING_RS=0 escape hatch.
_MIN_SO_BYTES = 400_000


def _extension_path() -> Optional[str]:
    try:
        from importlib.util import find_spec

        spec = find_spec("cine_rec_engine._scoring_rs")
        # a plain .so has no loader-added origin sometimes; guard all
        return getattr(spec, "origin", None) if spec else None
    except Exception:
        return None


def install(ns: dict[str, Any]) -> str:
    """Bind accelerated implementations into the service globals.

    Returns a status string for logging. Raises on any problem — the
    caller turns that into the pure-Python fallback.
    """
    if sys.version_info < _MIN_PY:
        return (
            f"python-fallback (CPython {sys.version_info[:3]}'s sum() "
            f"semantics differ from the port's; need >= {_MIN_PY})"
        )

    origin = _extension_path()
    if origin:
        import os

        try:
            if os.path.getsize(origin) < _MIN_SO_BYTES:
                return (
                    "python-fallback (extension file looks truncated: "
                    f"{origin}); set CINE_REC_SCORING_RS=0 to silence"
                )
        except OSError:
            pass  # stat failed — let the import try (and fail) normally

    from . import _scoring_rs as rs

    hw = ns["HEURISTIC_WEIGHTS_LIST"]  # module global, values[:27]
    py_entity_profile = ns["_py_entity_profile"]
    py_score_pair_fast = ns["_py_score_pair_fast"]
    py_feature_vector = ns["_py_feature_vector"]

    # feature_vector's vec path uses only HEURISTIC_WEIGHTS_LIST; the
    # PairScorer API still wants a 28-entry list — pad with a residual
    # zero that the vec path never reads.
    vec_scorer = rs.PairScorer(list(hw) + [0.0], hw)

    cache: dict[str, Any] = {"pair": (None, None)}  # (w_list, scorer)

    def _scorer_for(w_list: list) -> Any:
        cached_list, cached_scorer = cache["pair"]
        if cached_list is not w_list:
            cached_scorer = rs.PairScorer(list(w_list), hw)
            # single atomic assignment — no torn cache under threads
            cache["pair"] = (w_list, cached_scorer)
        return cached_scorer

    def _is_rust_profile(p: Any) -> bool:
        return type(p).__name__ == "PyProfile"

    def entity_profile(
        entity: dict,
        entity_data: Optional[dict] = None,
        genres_override: Optional[list] = None,
    ):
        try:
            return rs.entity_profile(entity, entity_data, genres_override)
        except TypeError:
            # exotic field types — this call only, pure Python
            return py_entity_profile(entity, entity_data, genres_override)

    def score_pair_fast(
        sp, cp, cand: dict, cand_data: dict, cand_genres_set, overview_sim: float, w_list: list
    ) -> float:
        rust_sp = _is_rust_profile(sp)
        rust_cp = _is_rust_profile(cp)
        if rust_sp != rust_cp:
            raise TypeError(
                "mixed profile kinds (one Rust, one pure-Python) — cannot "
                "score; rebuild both profiles from the same path"
            )
        if not rust_sp:
            return py_score_pair_fast(
                sp, cp, cand, cand_data, cand_genres_set, overview_sim, w_list
            )
        try:
            return _scorer_for(w_list).score_pair_fast(
                sp, cp, cand_data, cand_genres_set, overview_sim
            )
        except TypeError as e:
            # the call reads only the director name and the genre set
            # from the dicts — both are lenient; a TypeError here means
            # something structural: loud is better than wrong
            raise TypeError(f"rust score_pair_fast failed: {e}") from e

    def feature_vector(
        cand: dict,
        cand_data: dict,
        cand_genres: list,
        seed: dict,
        overview_sim: float = 0.0,
    ) -> list:
        try:
            return vec_scorer.feature_vector(cand, cand_data, cand_genres, seed, overview_sim)
        except TypeError:
            return py_feature_vector(cand, cand_data, cand_genres, seed, overview_sim)

    ns["_entity_profile"] = entity_profile
    ns["_score_pair_fast"] = score_pair_fast
    ns["feature_vector"] = feature_vector
    return f"rust {getattr(rs, '__version__', '?')}"


def uninstall(ns: dict[str, Callable]) -> None:
    """Restore the pure-Python implementations (tests / debugging)."""
    ns["_entity_profile"] = ns["_py_entity_profile"]
    ns["_score_pair_fast"] = ns["_py_score_pair_fast"]
    ns["feature_vector"] = ns["_py_feature_vector"]
