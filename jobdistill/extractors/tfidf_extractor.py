"""TF-IDF based fallback candidate extractor (no heavy deps beyond sklearn)."""

from __future__ import annotations

import logging
from typing import List, Tuple

from jobdistill.normalize import is_valid_candidate, normalize_phrase

logger = logging.getLogger(__name__)


class TFIDFCandidateExtractor:
    """Extract candidate phrases using TF-IDF weights.

    Used as a fallback when KeyBERT is unavailable or returns too few
    candidates.  Fits on a single document and returns top ngrams by
    TF-IDF weight.  Only sklearn is required (already a project dep).
    """

    def __init__(self, top_k: int = 30, ngram_range: Tuple[int, int] = (1, 3)) -> None:
        self.top_k = top_k
        self.ngram_range = ngram_range
        self._available = True
        self._warned = False

    @property
    def available(self) -> bool:
        return self._available

    def extract_candidates(self, text: str) -> List[Tuple[str, float]]:
        """Return (normalized_phrase, tfidf_score) pairs from a single doc."""
        if not self._available or not text or not text.strip():
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            if not self._warned:
                logger.warning("scikit-learn not installed; TF-IDF fallback unavailable")
                self._warned = True
            self._available = False
            return []

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            stop_words="english",
            max_features=500,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([text])
        except ValueError:
            return []

        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        indexed = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )

        seen: set[str] = set()
        candidates: list[Tuple[str, float]] = []
        for idx, score in indexed:
            phrase = feature_names[idx]
            norm = normalize_phrase(phrase)
            key = norm.lower()
            if key in seen:
                continue
            if is_valid_candidate(norm):
                seen.add(key)
                candidates.append((key, float(score)))
            if len(candidates) >= self.top_k:
                break

        return candidates
