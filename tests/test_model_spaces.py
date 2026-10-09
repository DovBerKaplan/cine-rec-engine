"""Registry behavior: legacy-key back-compat and space/column mapping."""

from cine_rec_engine.model_spaces import (
    REC_MODELS,
    column_for,
    is_solo,
    normalize_model,
)


class TestLegacyKeys:
    """Stored model choices must survive an upgrade, not silently degrade."""

    def test_v4b_alias_resolves_to_v4(self):
        assert normalize_model("v4b") == "v4"

    def test_alias_is_case_insensitive_and_trimmed(self):
        assert normalize_model(" V4B ") == "v4"

    def test_original_is_a_real_entry(self):
        # the base MiniLM space — same key CINE_REC_EMBEDDING=original names
        assert normalize_model("original") == "original"
        assert column_for("original") == "embedding"
        assert is_solo("original")

    def test_alias_column_and_solo_follow_the_canonical_entry(self):
        assert column_for("v4b") == "embedding_v4"
        assert is_solo("v4b")

    def test_unknown_still_none(self):
        assert normalize_model("bogus") is None
        assert normalize_model("") is None
        assert normalize_model(None) is None

    def test_canonical_keys_unchanged(self):
        for key in ("ensemble", "e5e", "mpnetae", "mpnetan", "v4"):
            assert normalize_model(key) == key


class TestWeights:
    def test_v4_uses_the_strong_space_default(self):
        # embedding_v4 is a strong space like the other 768-dim columns;
        # a lightweight-space price here weakened solo v4 requests.
        assert REC_MODELS["v4"]["cosine_weight"] == 26.0

    def test_original_keeps_the_lightweight_default(self):
        assert REC_MODELS["original"]["cosine_weight"] == 10.0

    def test_ensemble_has_no_solo_column(self):
        assert column_for("ensemble") is None
        assert not is_solo("ensemble")


class TestServeAcceptsLegacyKeys:
    def test_validate_model_passes_alias_through_normalized(self):
        from cine_rec_engine.serve import _validate_model

        assert _validate_model("v4b") == "v4"
        assert _validate_model("original") == "original"
        assert _validate_model(None) is None
