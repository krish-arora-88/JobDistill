"""Unit tests for jobdistill.normalize."""

import pytest

from jobdistill.normalize import (
    clean_text,
    deduplicate_phrases,
    has_tech_indicator,
    is_junk_phrase,
    is_pure_numeric,
    is_stopword_only,
    is_valid_candidate,
    normalize_phrase,
    normalize_unicode,
    normalize_whitespace,
    stopword_ratio,
    token_count,
)


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_strips_edges(self):
        assert normalize_whitespace("  foo  ") == "foo"

    def test_collapses_newlines(self):
        assert normalize_whitespace("a\n\nb\tc") == "a b c"


class TestNormalizeUnicode:
    def test_smart_quotes(self):
        assert normalize_unicode("\u201chello\u201d") == '"hello"'

    def test_en_dash(self):
        assert normalize_unicode("A\u2013B") == "A-B"

    def test_curly_apostrophe(self):
        assert normalize_unicode("it\u2019s") == "it's"


class TestNormalizePhrase:
    def test_strip_punctuation(self):
        assert normalize_phrase("Python,") == "Python"
        assert normalize_phrase("(React)") == "React"
        assert normalize_phrase('"Docker"') == "Docker"

    def test_preserves_internal_special(self):
        assert normalize_phrase("C++") == "C++"
        assert normalize_phrase(".NET") == ".NET"
        assert normalize_phrase("CI/CD") == "CI/CD"

    def test_whitespace(self):
        assert normalize_phrase("  React  Native  ") == "React Native"


class TestStopwordOnly:
    def test_all_stopwords(self):
        assert is_stopword_only("the and or") is True

    def test_not_all(self):
        assert is_stopword_only("the Python") is False

    def test_single_stopword(self):
        assert is_stopword_only("a") is True


class TestJunkPhrase:
    def test_known_junk(self):
        assert is_junk_phrase("apply now") is True
        assert is_junk_phrase("Apply Now") is True

    def test_not_junk(self):
        assert is_junk_phrase("Python") is False


class TestPureNumeric:
    def test_numbers(self):
        assert is_pure_numeric("12345") is True
        assert is_pure_numeric("3.14") is True
        assert is_pure_numeric("$50,000") is True

    def test_not_numeric(self):
        assert is_pure_numeric("5G") is False
        assert is_pure_numeric("Python3") is False


class TestTokenCount:
    def test_single(self):
        assert token_count("Python") == 1

    def test_multi(self):
        assert token_count("React Native") == 2


class TestIsValidCandidate:
    def test_valid(self):
        assert is_valid_candidate("Python") is True
        assert is_valid_candidate("React Native") is True
        assert is_valid_candidate("CI/CD") is True

    def test_empty(self):
        assert is_valid_candidate("") is False

    def test_too_long(self):
        phrase = " ".join(["word"] * 6)
        assert is_valid_candidate(phrase) is False

    def test_stopword_only(self):
        assert is_valid_candidate("the and or") is False

    def test_junk(self):
        assert is_valid_candidate("apply now") is False

    def test_numeric(self):
        assert is_valid_candidate("12345") is False

    def test_single_char(self):
        assert is_valid_candidate("x", min_chars=2) is False


class TestDeduplicatePhrases:
    def test_dedup(self):
        result = deduplicate_phrases(["Python", "python", "PYTHON", "Java"])
        assert len(result) == 2
        assert result[0] == "Python"

    def test_preserves_order(self):
        result = deduplicate_phrases(["C++", "Java", "c++"])
        assert result == ["C++", "Java"]
