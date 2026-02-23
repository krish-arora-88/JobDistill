"""ML-based skill extractor: KeyBERT candidates -> classifier -> dedup."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from jobdistill.extractors.base import ExtractionResult, SkillExtractor
from jobdistill.extractors.classifier import SkillClassifier
from jobdistill.extractors.keybert_extractor import KeyBERTExtractor
from jobdistill.normalize import normalize_phrase

logger = logging.getLogger(__name__)


def _find_example_mentions(text: str, phrase: str, max_examples: int = 3) -> List[str]:
    """Find short context windows around phrase occurrences in text."""
    examples: list[str] = []
    try:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    except re.error:
        return examples
    for match in pattern.finditer(text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        snippet = text[start:end].strip().replace("\n", " ")
        examples.append(f"...{snippet}...")
        if len(examples) >= max_examples:
            break
    return examples


def _deduplicate_by_embedding(
    phrases_with_scores: List[Tuple[str, float]],
    embed_fn,
    similarity_threshold: float = 0.85,
) -> List[Tuple[str, float]]:
    """Cluster near-duplicate phrases by cosine similarity, keep highest-scored representative."""
    if len(phrases_with_scores) <= 1:
        return phrases_with_scores

    phrases = [p for p, _ in phrases_with_scores]
    scores = [s for _, s in phrases_with_scores]
    embeddings = embed_fn(phrases)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    kept: list[int] = []
    merged: set[int] = set()
    indices_by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    for idx in indices_by_score:
        if idx in merged:
            continue
        kept.append(idx)
        for other in range(len(phrases)):
            if other != idx and other not in merged and sim_matrix[idx, other] >= similarity_threshold:
                merged.add(other)

    return [(phrases[i], scores[i]) for i in kept]


class MLSkillExtractor(SkillExtractor):
    """Two-stage ML extractor: keyphrase candidates -> skill classifier.

    If no trained classifier is available, falls back to returning
    all valid KeyBERT candidates (useful before first training run).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_dir: Optional[str] = None,
        top_k: int = 30,
        min_confidence: float = 0.60,
        deduplicate: bool = True,
        dedup_threshold: float = 0.85,
    ) -> None:
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.deduplicate = deduplicate
        self.dedup_threshold = dedup_threshold
        self._embedding_model = embedding_model

        self._keyphrase = KeyBERTExtractor(
            embedding_model=embedding_model,
            top_k=top_k,
        )
        self._classifier = SkillClassifier(
            embedding_model=embedding_model,
            threshold=min_confidence,
        )
        self._classifier_loaded = False

        if model_dir:
            self._try_load_classifier(model_dir)

    def _try_load_classifier(self, model_dir: str) -> None:
        try:
            self._classifier.load(model_dir)
            self._classifier_loaded = True
            logger.info("Loaded skill classifier from %s", model_dir)
        except FileNotFoundError:
            logger.warning(
                "No classifier found at %s; will use unfiltered KeyBERT candidates",
                model_dir,
            )

    @property
    def name(self) -> str:
        return "ml"

    def extract(self, text: str) -> ExtractionResult:
        candidates = self._keyphrase.extract_candidates(text)
        if not candidates:
            return ExtractionResult(skills={}, candidates_considered=0)

        phrases = [p for p, _ in candidates]
        keyphrase_scores = {p: s for p, s in candidates}

        if self._classifier_loaded:
            probas = self._classifier.predict_proba(phrases)
            scored = [
                (phrase, prob)
                for phrase, prob in zip(phrases, probas)
                if prob >= self.min_confidence
            ]
        else:
            scored = [(p, s) for p, s in candidates]

        if self.deduplicate and len(scored) > 1:
            self._keyphrase._ensure_model()
            backend = self._keyphrase._kw_model.model  # type: ignore[union-attr]
            embed_fn = lambda docs: backend.embedding_model.encode(docs, show_progress_bar=False)
            scored = _deduplicate_by_embedding(scored, embed_fn, self.dedup_threshold)

        skills_dict = {normalize_phrase(p): float(conf) for p, conf in scored}

        example_mentions: Dict[str, List[str]] = {}
        for phrase in list(skills_dict.keys())[:10]:
            mentions = _find_example_mentions(text, phrase)
            if mentions:
                example_mentions[phrase] = mentions

        return ExtractionResult(
            skills=skills_dict,
            candidates_considered=len(candidates),
            example_mentions=example_mentions,
        )
