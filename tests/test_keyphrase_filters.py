"""Unit tests for candidate phrase filtering logic."""

import pytest

from jobdistill.normalize import is_valid_candidate, normalize_phrase


class TestCandidateFilters:
    """Verify that the filters correctly accept/reject typical candidate phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Python",
            "machine learning",
            "React Native",
            "CI/CD",
            "C++",
            ".NET",
            "natural language processing",
            "AWS",
            "TensorFlow",
        ],
    )
    def test_accept_valid_skills(self, phrase: str):
        assert is_valid_candidate(phrase), f"Should accept: {phrase}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "",
            "   ",
            "the",
            "and or but",
            "apply now",
            "click here",
            "12345",
            "$50,000",
            "a very long phrase that has more than five words total here",
        ],
    )
    def test_reject_invalid(self, phrase: str):
        assert not is_valid_candidate(phrase), f"Should reject: {phrase}"

    def test_min_max_tokens(self):
        assert is_valid_candidate("Python", min_tokens=1, max_tokens=1) is True
        assert is_valid_candidate("React Native", min_tokens=1, max_tokens=1) is False
        assert is_valid_candidate("React Native", min_tokens=2, max_tokens=3) is True

    def test_min_max_chars(self):
        assert is_valid_candidate("Go", min_chars=3) is False
        assert is_valid_candidate("Go", min_chars=1) is True


class TestNormalizePhraseEdgeCases:
    def test_trailing_period(self):
        assert normalize_phrase("Python.") == "Python"

    def test_surrounding_parens(self):
        assert normalize_phrase("(Django)") == "Django"

    def test_mixed_whitespace(self):
        assert normalize_phrase(" React\t Native ") == "React Native"

    def test_unicode_dash(self):
        assert normalize_phrase("CI\u2013CD") == "CI-CD"

    def test_ellipsis(self):
        assert normalize_phrase("Python…") == "Python"
