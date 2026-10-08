"""Embedding space registry — the seam between engine and models.

The engine's KNN recall channel and cosine-blend feature read OVERVIEW
EMBEDDINGS from columns on ``tmdb_media``. Which column(s) exist is a
property of YOUR database: you generate embeddings with whichever
sentence-encoder you like (see ``models/README.md``), store them as
``vector`` columns, and register them here.

Each entry names:
- ``column``: the tmdb_media column holding the vectors (None = the
  tuned multi-model ensemble driven by config.COSINE_BLEND).
- ``cosine_weight``: the cosine_sim feature weight to use when this
  space is selected solo (overrides the tuned default).

The published ``weights.json`` was tuned on the ensemble space; solo
weights below are safe defaults, not fitted constants.
"""

from __future__ import annotations

from typing import Optional

REC_MODELS: dict = {
    "ensemble": {
        "column": None,
        "cosine_weight": None,
    },
    "e5e": {
        "column": "embedding_e5e",
        "cosine_weight": 26.0,
    },
    "mpnetae": {
        "column": "embedding_mpnetae",
        "cosine_weight": 26.0,
    },
    "mpnetan": {
        "column": "embedding_mpnetan",
        "cosine_weight": 26.0,
    },
    "v4": {
        "column": "embedding_v4",
        "cosine_weight": 10.0,
    },
}


def normalize_model(name: Optional[str]) -> Optional[str]:
    """Case-insensitive match against REC_MODELS, or None."""
    if not name:
        return None
    low = name.strip().lower()
    for key in REC_MODELS:
        if key.lower() == low:
            return key
    return None


def column_for(space: Optional[str]) -> Optional[str]:
    """The embedding column a solo space builds and searches from.

    None = the default column (config.EMBEDDING_COLUMN) — used by the
    ensemble and unknown spaces. Single source for this mapping: the
    user-vector build, the ANN recall, and the request path must agree,
    or a vector built from one column gets searched against another.
    """
    if space:
        spec = REC_MODELS.get(space)
        if spec and spec["column"]:
            return spec["column"]
    return None


def is_solo(model_key: str) -> bool:
    """True when the key selects a single embedding column."""
    spec = REC_MODELS.get(model_key)
    return bool(spec and spec["column"])
