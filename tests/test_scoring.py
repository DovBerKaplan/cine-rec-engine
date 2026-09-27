"""Tests for recommendation scoring functions.

Source: cine_rec_engine/scoring.py

All functions are pure (no I/O), so no mocking is needed.
"""

from cine_rec_engine.scoring import (
    company_similarity,
    detect_style,
    fast_cast_similarity,
    fast_keyword_similarity,
    optimize_candidate_selection,
    sentence_similarity,
)


class TestSentenceSimilarity:
    """Test character-bigram Jaccard similarity."""

    def test_identical_strings(self):
        """Identical non-empty strings should score 1.0."""
        assert sentence_similarity("hello", "hello") == 1.0

    def test_empty_first_argument(self):
        """Empty first argument returns 0.0."""
        assert sentence_similarity("", "hello") == 0.0

    def test_empty_second_argument(self):
        """Empty second argument returns 0.0."""
        assert sentence_similarity("hello", "") == 0.0

    def test_both_empty(self):
        """Both empty returns 0.0."""
        assert sentence_similarity("", "") == 0.0

    def test_case_insensitive(self):
        """Similarity is case-insensitive."""
        score = sentence_similarity("Hello World", "hello world")
        assert score == 1.0

    def test_partial_overlap(self):
        """Strings with partial overlap return value between 0 and 1."""
        score = sentence_similarity("action movie", "action film")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        """Completely different strings return 0.0."""
        # "ab" -> bigrams {"ab"}, "cd" -> bigrams {"cd"}, no intersection
        score = sentence_similarity("ab", "cd")
        assert score == 0.0

    def test_single_char_strings(self):
        """Single-character strings produce no bigrams => 0.0."""
        assert sentence_similarity("a", "a") == 0.0


class TestDetectStyle:
    """Test keyword-to-style mapping."""

    def test_empty_keywords(self):
        """Empty keyword list returns empty list."""
        assert detect_style([]) == []

    def test_parody_style(self):
        """Keywords matching 'parody' style are detected."""
        result = detect_style(["parody", "comedy", "slapstick"])
        assert "parody" in result
        assert "slapstick" in result

    def test_buddy_cop_style(self):
        """Keywords matching 'buddy cop' style are detected."""
        result = detect_style(["buddy cop", "action"])
        assert "buddy cop" in result

    def test_no_matching_style(self):
        """Unrelated keywords return empty list."""
        result = detect_style(["drama", "romance", "thriller"])
        assert result == []

    def test_multiple_styles(self):
        """Multiple matching styles are all returned."""
        result = detect_style(["parody", "bromance", "slapstick"])
        assert "parody" in result
        assert "buddy cop" in result
        assert "slapstick" in result


class TestFastKeywordSimilarity:
    """Test keyword set-overlap scoring."""

    def test_identical_keywords(self):
        """Identical keyword lists score 1.0."""
        kws = ["action", "thriller", "crime"]
        assert fast_keyword_similarity(kws, kws) == 1.0

    def test_no_overlap(self):
        """Disjoint keyword lists score 0.0."""
        score = fast_keyword_similarity(["a", "b"], ["c", "d"])
        assert score == 0.0

    def test_partial_overlap(self):
        """Partial overlap returns a fractional score."""
        score = fast_keyword_similarity(["a", "b", "c"], ["b", "c", "d"])
        # Jaccard: intersection 2 / union 4
        assert abs(score - 0.5) < 1e-9

    def test_empty_first_list(self):
        """Empty first list returns 0.0."""
        assert fast_keyword_similarity([], ["a"]) == 0.0

    def test_empty_second_list(self):
        """Empty second list returns 0.0."""
        assert fast_keyword_similarity(["a"], []) == 0.0

    def test_single_overlap(self):
        """Jaccard: intersection 1 / union 3."""
        score = fast_keyword_similarity(["a"], ["a", "b", "c"])
        assert abs(score - 1 / 3) < 1e-9


class TestFastCastSimilarity:
    """Test cast-member set-overlap scoring (top-5)."""

    def test_identical_cast(self):
        """Identical cast lists score 1.0."""
        cast = ["Actor A", "Actor B", "Actor C"]
        assert fast_cast_similarity(cast, cast) == 1.0

    def test_only_top_five_compared(self):
        """Only the first 5 members of each list are compared."""
        cast1 = ["A", "B", "C", "D", "E", "F", "G"]
        cast2 = ["A", "B", "C", "D", "E", "X", "Y"]
        # top 5 are identical => score 1.0
        assert fast_cast_similarity(cast1, cast2) == 1.0

    def test_empty_cast(self):
        """Empty cast lists return 0.0."""
        assert fast_cast_similarity([], ["A"]) == 0.0

    def test_no_overlap(self):
        """Completely different cast members score 0.0."""
        score = fast_cast_similarity(["A", "B"], ["C", "D"])
        assert score == 0.0


class TestCompanySimilarity:
    """Test production company set-overlap scoring."""

    def test_identical_companies(self):
        """Identical company lists score 1.0."""
        companies = ["Marvel Studios", "Disney"]
        assert company_similarity(companies, companies) == 1.0

    def test_empty_companies(self):
        """Empty company list returns 0.0."""
        assert company_similarity([], ["Studio"]) == 0.0

    def test_partial_overlap(self):
        """Partial overlap returns fractional score."""
        score = company_similarity(["A", "B", "C"], ["B", "C", "D"])
        # Jaccard: intersection 2 / union 4
        assert abs(score - 0.5) < 1e-9


class TestOptimizeCandidateSelection:
    """Test pre-filtering and sorting of candidates."""

    def test_empty_candidates(self):
        """Empty candidate list returns empty."""
        assert optimize_candidate_selection([], ["Action"]) == []

    def test_filters_low_rated(self):
        """Candidates below 5.5 rating are filtered out."""
        candidates = [
            {"id": 1, "genres": ["Action"], "vote_average": 4.0, "vote_count": 1000},
            {"id": 2, "genres": ["Action"], "vote_average": 7.0, "vote_count": 1000},
        ]
        result = optimize_candidate_selection(candidates, ["Action"])
        ids = [c["id"] for c in result]
        assert 1 not in ids
        assert 2 in ids

    def test_filters_no_genre_overlap(self):
        """Candidates with no genre overlap are filtered out."""
        candidates = [
            {"id": 1, "genres": ["Drama"], "vote_average": 8.0, "vote_count": 1000},
            {"id": 2, "genres": ["Action"], "vote_average": 8.0, "vote_count": 1000},
        ]
        result = optimize_candidate_selection(candidates, ["Comedy", "Horror"])
        ids = [c["id"] for c in result]
        # Neither "Drama" nor "Action" overlap with ["Comedy","Horror"]
        # within the first 3 genres of each side
        assert 1 not in ids
        assert 2 not in ids

    def test_sorts_by_composite_score(self):
        """Candidates are sorted by composite of rating, popularity, votes."""
        candidates = [
            {
                "id": 1,
                "genres": ["Action"],
                "vote_average": 6.0,
                "popularity": 10.0,
                "vote_count": 1000,
            },
            {
                "id": 2,
                "genres": ["Action"],
                "vote_average": 9.0,
                "popularity": 50.0,
                "vote_count": 500,
            },
        ]
        result = optimize_candidate_selection(candidates, ["Action"])
        # Higher-rated candidate should come first
        assert result[0]["id"] == 2

    def test_respects_target_count_times_two(self):
        """Result is capped at target_count * 2."""
        candidates = [
            {
                "id": i,
                "genres": ["Action"],
                "vote_average": 7.0,
                "popularity": 10.0,
                "vote_count": 1000,
            }
            for i in range(100)
        ]
        result = optimize_candidate_selection(candidates, ["Action"], target_count=10)
        assert len(result) == 20  # 10 * 2
