"""Registry behavior: space/column mapping and name normalization."""

from cine_rec_engine.model_spaces import (
    REC_MODELS,
    column_for,
    is_solo,
    normalize_model,
)


class TestNameNormalization:
    def test_keys_resolve_case_insensitively_and_trimmed(self):
        assert normalize_model(" MiniLM ") == "minilm"
        assert normalize_model("original") == "original"

    def test_original_is_a_real_entry(self):
        # the base space — same key CINE_REC_EMBEDDING=original names
        assert column_for("original") == "embedding"
        assert is_solo("original")

    def test_minilm_is_the_public_reference_space(self):
        assert column_for("minilm") == "embedding_minilm"
        assert is_solo("minilm")

    def test_unknown_still_none(self):
        assert normalize_model("bogus") is None
        assert normalize_model("") is None
        assert normalize_model(None) is None


class TestWeights:
    def test_solo_spaces_keep_the_safe_default(self):
        # solo cosine weights are safe defaults, not fitted constants
        assert REC_MODELS["original"]["cosine_weight"] == 10.0
        assert REC_MODELS["minilm"]["cosine_weight"] == 10.0

    def test_ensemble_has_no_solo_column(self):
        assert column_for("ensemble") is None
        assert not is_solo("ensemble")


class TestServeAcceptsRegistryKeys:
    def test_validate_model_passes_through_normalized(self):
        from cine_rec_engine.serve import _validate_model

        assert _validate_model("minilm") == "minilm"
        assert _validate_model("original") == "original"
        assert _validate_model(None) is None
