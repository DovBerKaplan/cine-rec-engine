"""Optional Rust acceleration for the scoring hot path.

Rebinds the service module's `_entity_profile` / `_score_pair_fast` /
`feature_vector` to the compiled `_scoring_rs` extension. The
pure-Python implementations remain the source of truth (kept callable
under their `_py_*` aliases in the service module); this module only
installs when the extension imports cleanly — degrade, never fail.

Engine caching: `PairScorer` binds the weight lists at construction,
so converting them per pair would eat the win. Callers pass the same
list object per process (`HEURISTIC_ACTIVE_LIST`), so an identity
check is enough; a different list object simply builds a new scorer.
Do not mutate a weights list in place after passing it here.

The per-pair call releases the GIL for the math, so the warm-path
thread pool parallelizes for real.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def install(ns: dict[str, Any]) -> str:
    """Bind accelerated implementations into the service globals.

    Returns a status string for logging. Raises on any problem — the
    caller turns that into the pure-Python fallback.
    """
    from . import _scoring_rs as rs

    hw = ns["HEURISTIC_WEIGHTS_LIST"]  # module global, values[:27]
    # feature_vector's vec path uses only HEURISTIC_WEIGHTS_LIST; the
    # PairScorer API still wants a 28-entry list — pad with a residual
    # zero that the vec path never reads.
    vec_scorer = rs.PairScorer(list(hw) + [0.0], hw)

    cache: dict[str, Any] = {"list": None, "scorer": None}

    def _scorer_for(w_list: list) -> Any:
        if cache["list"] is not w_list:
            cache["list"] = w_list
            cache["scorer"] = rs.PairScorer(list(w_list), hw)
        return cache["scorer"]

    def entity_profile(
        entity: dict,
        entity_data: Optional[dict] = None,
        genres_override: Optional[list] = None,
    ):
        return rs.entity_profile(entity, entity_data, genres_override)

    def score_pair_fast(
        sp, cp, cand: dict, cand_data: dict, cand_genres_set, overview_sim: float, w_list: list
    ) -> float:
        return _scorer_for(w_list).score_pair_fast(
            sp, cp, cand_data, cand_genres_set, overview_sim
        )

    def feature_vector(
        cand: dict,
        cand_data: dict,
        cand_genres: list,
        seed: dict,
        overview_sim: float = 0.0,
    ) -> list:
        return vec_scorer.feature_vector(cand, cand_data, cand_genres, seed, overview_sim)

    ns["_entity_profile"] = entity_profile
    ns["_score_pair_fast"] = score_pair_fast
    ns["feature_vector"] = feature_vector
    return f"rust {_scoring_rs_version(rs)}"


def _scoring_rs_version(rs) -> str:
    return getattr(rs, "__version__", "?")


def uninstall(ns: dict[str, Callable]) -> None:
    """Restore the pure-Python implementations (tests / debugging)."""
    ns["_entity_profile"] = ns["_py_entity_profile"]
    ns["_score_pair_fast"] = ns["_py_score_pair_fast"]
    ns["feature_vector"] = ns["_py_feature_vector"]
