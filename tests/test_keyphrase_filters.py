"""Unit tests for candidate phrase filtering logic."""

import pytest

from jobdistill.normalize import (
    has_tech_indicator,
    is_valid_candidate,
    normalize_phrase,
    stopword_ratio,
)


class TestHasTechIndicator:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Python",
            "C++",
            ".NET",
            "CI/CD",
            "C#",
            "AI",
            "AWS",
            "SQL",
            "JavaScript",
            "TypeScript",
            "React",
            "TensorFlow",
            "REST",
            "Node.js",
            "5G",
            "Docker",
            "Git",
            "Kubernetes",
            "React Native",
        ],
    )
    def test_accepts_tech_phrases(self, phrase: str):
        assert has_tech_indicator(phrase), f"Should accept: {phrase}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "the and or",
            "in on at",
            "submit applications soon",
            "canada job posting",
            "students submit applications",
            "equal opportunity employer",
            "before the deadline",
        ],
    )
    def test_rejects_non_tech_phrases(self, phrase: str):
        assert not has_tech_indicator(phrase), f"Should reject: {phrase}"


class TestStopwordRatio:
    def test_all_stopwords(self):
        assert stopword_ratio("the and or") == 1.0

    def test_no_stopwords(self):
        assert stopword_ratio("Python React AWS") == 0.0

    def test_mixed(self):
        ratio = stopword_ratio("the Python and React")
        assert 0.4 < ratio < 0.6

    def test_empty(self):
        assert stopword_ratio("") == 0.0


class TestCandidateFilters:
    """Verify that the filters correctly accept/reject typical candidate phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Python",
            "React Native",
            "CI/CD",
            "C++",
            ".NET",
            "AWS",
            "TensorFlow",
            "SQL",
            "JavaScript",
            "Docker",
            "Git",
            "REST",
            "Node.js",
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
            "submit applications soon",
            "canada job posting",
            "students submit applications",
            "equal opportunity employer",
            "before the deadline",
            "a very long phrase that has more than five words total here",
        ],
    )
    def test_reject_invalid(self, phrase: str):
        assert not is_valid_candidate(phrase), f"Should reject: {phrase}"

    def test_min_max_tokens(self):
        assert is_valid_candidate("Python", min_tokens=1, max_tokens=1) is True
        assert is_valid_candidate("React Native", min_tokens=1, max_tokens=1) is False
        assert is_valid_candidate("React Native", min_tokens=2, max_tokens=3) is True


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
