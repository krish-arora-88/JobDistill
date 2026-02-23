"""Text and phrase normalization utilities for skill extraction."""

from __future__ import annotations

import re
import unicodedata
from typing import List

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "about", "above", "after", "again", "also", "any", "as", "if",
    "into", "once", "then", "there", "up", "out", "over", "under",
})

_JUNK_PHRASES = frozenset({
    "apply now", "click here", "learn more", "read more", "see more",
    "job description", "job posting", "job title", "job type",
    "equal opportunity", "please apply", "we are looking",
    "good communication", "strong communication", "excellent communication",
    "team player", "self starter", "fast paced", "detail oriented",
    "years experience", "year experience", "work experience",
    "cover letter", "resume", "salary", "compensation", "benefits",
})

# Matches strings that are purely numeric or single punctuation
_PURE_NUMERIC_RE = re.compile(r"^[\d.,/%$€£¥+\-×÷=<>]+$")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """NFC-normalize and replace common unicode variants."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_text(text: str) -> str:
    """Full text cleaning: unicode normalize, collapse whitespace."""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def normalize_phrase(phrase: str) -> str:
    """Normalize a candidate skill phrase for deduplication.

    Lowercases, strips outer punctuation, collapses whitespace.
    Preserves internal special chars (e.g. C++, .NET, CI/CD).
    """
    phrase = normalize_unicode(phrase)
    phrase = normalize_whitespace(phrase)
    phrase = phrase.strip(",;:!?\"'()[]{}…–—")
    # Strip trailing dots but preserve leading dots (e.g. ".NET")
    phrase = phrase.rstrip(".")
    phrase = normalize_whitespace(phrase)
    return phrase


def is_stopword_only(phrase: str) -> bool:
    """True if every token in the phrase is a stopword."""
    tokens = phrase.lower().split()
    return all(t in _STOPWORDS for t in tokens)


def is_junk_phrase(phrase: str) -> bool:
    """True if the phrase is a known junk/boilerplate string."""
    return phrase.lower().strip() in _JUNK_PHRASES


def is_pure_numeric(phrase: str) -> bool:
    """True if the phrase contains only numbers / math symbols."""
    return bool(_PURE_NUMERIC_RE.match(phrase.strip()))


def token_count(phrase: str) -> int:
    return len(phrase.split())


def is_valid_candidate(
    phrase: str,
    min_tokens: int = 1,
    max_tokens: int = 5,
    min_chars: int = 1,
    max_chars: int = 60,
) -> bool:
    """Check whether a candidate phrase passes basic quality filters.

    Filters:
    - Token count within [min_tokens, max_tokens]
    - Character length within [min_chars, max_chars]
    - Not stopword-only
    - Not a junk phrase
    - Not purely numeric
    """
    phrase = phrase.strip()
    if not phrase:
        return False
    n_tokens = token_count(phrase)
    if n_tokens < min_tokens or n_tokens > max_tokens:
        return False
    if len(phrase) < min_chars or len(phrase) > max_chars:
        return False
    if is_stopword_only(phrase):
        return False
    if is_junk_phrase(phrase):
        return False
    if is_pure_numeric(phrase):
        return False
    return True


def deduplicate_phrases(phrases: List[str]) -> List[str]:
    """Deduplicate phrases by normalized form, keeping first occurrence."""
    seen: set[str] = set()
    result: list[str] = []
    for p in phrases:
        key = normalize_phrase(p).lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result
