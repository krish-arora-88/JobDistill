"""ML-based skill extractor: multi-source candidates -> classifier -> dedup.

Candidate sources (unioned before classification):
  1. KeyBERT keyphrases (primary, unsupervised)
  2. TF-IDF ngrams (fallback when KeyBERT unavailable / returns < 5 results)
  3. Tech-token regex patterns (always, catches symbols/acronyms KeyBERT misses)

Classification uses trained LogisticRegression when available, otherwise
anchor-phrase cosine similarity.  Threshold auto-relaxes if too few skills
pass so that extraction never collapses to zero output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jobdistill.extractors.base import ExtractionResult, SkillExtractor
from jobdistill.extractors.classifier import SkillClassifier
from jobdistill.extractors.keybert_extractor import KeyBERTExtractor
from jobdistill.extractors.tfidf_extractor import TFIDFCandidateExtractor
from jobdistill.normalize import is_valid_candidate, normalize_phrase

logger = logging.getLogger(__name__)

MIN_KEYBERT_CANDIDATES = 5
MIN_SKILLS_PER_DOC = 5
CLASSIFIER_THRESHOLD_FLOOR = 0.3
CLASSIFIER_THRESHOLD_STEP = 0.1
TECH_TOKEN_PRIOR_SCORE = 0.8

# ---------------------------------------------------------------------------
# Tech-token regex extraction (pattern-based, NOT a skills list)
# ---------------------------------------------------------------------------

_TECH_TOKEN_PATTERNS = [
    re.compile(r"\.NET\b"),                              # .NET
    re.compile(r"(?<![A-Za-z])C\+\+(?![A-Za-z])"),      # C++
    re.compile(r"(?<![A-Za-z])[CF]#(?![A-Za-z])"),       # C#, F#
    re.compile(r"\b[A-Z]{2,6}/[A-Z]{2,6}\b"),           # CI/CD, TCP/IP
    re.compile(r"\b[A-Z]{2,6}\b"),                       # AWS, SQL, AI, ML, REST
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"),     # JavaScript, TypeScript
    re.compile(r"\b[A-Z][a-z0-9]{2,15}\b"),              # Python, Linux, Docker
]


def extract_tech_tokens(text: str) -> List[Tuple[str, float]]:
    """Pattern-based extraction of tech-shaped tokens from raw text.

    Returns (normalized_phrase, prior_score) pairs that pass is_valid_candidate.
    """
    seen: set[str] = set()
    results: list[Tuple[str, float]] = []
    for pattern in _TECH_TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group()
            norm = normalize_phrase(raw)
            key = norm.lower()
            if key in seen:
                continue
            if not is_valid_candidate(norm):
                continue
            seen.add(key)
            results.append((key, TECH_TOKEN_PRIOR_SCORE))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Cluster near-duplicate phrases by cosine similarity, keep highest-scored."""
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


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class MLSkillExtractor(SkillExtractor):
    """Multi-source ML extractor: KeyBERT + TF-IDF + tech-tokens -> classifier -> dedup."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_dir: Optional[str] = None,
        top_k: int = 30,
        min_confidence: float = 0.75,
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
        self._tfidf = TFIDFCandidateExtractor(top_k=top_k)
        self._classifier = SkillClassifier(
            embedding_model=embedding_model,
            threshold=min_confidence,
        )
        self._classifier_loaded = False

        self.classifier_floor_triggered_count = 0
        self._docs_processed = 0
        self._total_skills_extracted = 0

        if model_dir:
            self._try_load_classifier(model_dir)

    def _try_load_classifier(self, model_dir: str) -> None:
        try:
            self._classifier.load(model_dir)
            self._classifier_loaded = True
            logger.info("Loaded skill classifier from %s", model_dir)
            meta = self._classifier._meta
            if meta:
                metrics = meta.get("metrics", {})
                if metrics.get("train_samples", 0) < 100:
                    logger.warning(
                        "Classifier trained on only %d samples - results may be unreliable",
                        metrics.get("train_samples", 0),
                    )
                if metrics.get("eval_f1", 1.0) < 0.5:
                    logger.warning(
                        "Classifier eval F1=%.2f (poor) - consider retraining",
                        metrics.get("eval_f1", 0),
                    )
        except FileNotFoundError:
            logger.warning(
                "No classifier found at %s; using anchor-phrase fallback scorer",
                model_dir,
            )

    @property
    def name(self) -> str:
        return "ml"

    def extract(self, text: str) -> ExtractionResult:
        # --- Stage 1: Gather candidates from multiple sources ---
        keybert_candidates = self._keyphrase.extract_candidates(text)

        use_tfidf = (
            not self._keyphrase.available
            or len(keybert_candidates) < MIN_KEYBERT_CANDIDATES
        )
        tfidf_candidates = self._tfidf.extract_candidates(text) if use_tfidf else []

        tech_candidates = extract_tech_tokens(text)

        # --- Stage 2: Union candidates, keep best pre-classifier score ---
        best_scores: dict[str, float] = {}
        for phrase, score in keybert_candidates + tfidf_candidates + tech_candidates:
            key = phrase  # already lowercase-normalized from all sources
            if key not in best_scores or score > best_scores[key]:
                best_scores[key] = score

        all_candidates = list(best_scores.items())

        if not all_candidates:
            return ExtractionResult(skills={}, candidates_considered=0)

        phrases = [p for p, _ in all_candidates]

        # --- Stage 3: Classify ---
        probas = self._classifier.predict_proba(phrases)

        # --- Stage 4: Filter with auto-relaxation ---
        threshold = self.min_confidence
        scored = [
            (phrase, prob)
            for phrase, prob in zip(phrases, probas)
            if prob >= threshold
        ]

        threshold_relaxed = False

        while len(scored) < MIN_SKILLS_PER_DOC and threshold > CLASSIFIER_THRESHOLD_FLOOR:
            threshold = max(threshold - CLASSIFIER_THRESHOLD_STEP, CLASSIFIER_THRESHOLD_FLOOR)
            scored = [
                (phrase, prob)
                for phrase, prob in zip(phrases, probas)
                if prob >= threshold
            ]
            threshold_relaxed = True

        if len(scored) < MIN_SKILLS_PER_DOC and len(phrases) >= MIN_SKILLS_PER_DOC:
            sorted_by_prob = sorted(
                zip(phrases, probas), key=lambda x: x[1], reverse=True
            )
            scored = list(sorted_by_prob[:MIN_SKILLS_PER_DOC])
            threshold_relaxed = True

        if threshold_relaxed:
            self.classifier_floor_triggered_count += 1

        # --- Stage 5: Deduplicate by embedding similarity ---
        if self.deduplicate and len(scored) > 1:
            try:
                self._keyphrase._ensure_model()
                if self._keyphrase.available and self._keyphrase._kw_model is not None:
                    backend = self._keyphrase._kw_model.model  # type: ignore[union-attr]
                    embed_fn = lambda docs: backend.embedding_model.encode(
                        docs, show_progress_bar=False
                    )
                    scored = _deduplicate_by_embedding(scored, embed_fn, self.dedup_threshold)
            except Exception as e:
                logger.debug("Embedding dedup skipped: %s", e)

        skills_dict = {normalize_phrase(p): float(conf) for p, conf in scored}

        # --- Track stats for auto-warning ---
        self._docs_processed += 1
        self._total_skills_extracted += len(skills_dict)
        if self._docs_processed == 20:
            avg = self._total_skills_extracted / 20
            if avg < 3:
                logger.warning(
                    "Low extraction rate: avg %.1f skills/doc over first 20 docs "
                    "(expected >= 3). Check boilerplate removal and candidate generation.",
                    avg,
                )

        # --- Example mentions ---
        example_mentions: Dict[str, List[str]] = {}
        for phrase in list(skills_dict.keys())[:10]:
            mentions = _find_example_mentions(text, phrase)
            if mentions:
                example_mentions[phrase] = mentions

        # --- Build debug info ---
        debug_info: Dict[str, Any] = {
            "keybert_candidates": len(keybert_candidates),
            "tfidf_candidates": len(tfidf_candidates),
            "tech_token_candidates": len(tech_candidates),
            "total_union_candidates": len(all_candidates),
            "threshold_used": round(threshold, 2),
            "threshold_relaxed": threshold_relaxed,
            "final_skills": len(skills_dict),
            "top_candidates": [
                {
                    "phrase": p,
                    "classifier_prob": round(float(prob), 4),
                    "kept": prob >= threshold,
                }
                for p, prob in sorted(
                    zip(phrases, probas), key=lambda x: x[1], reverse=True
                )[:15]
            ],
        }

        return ExtractionResult(
            skills=skills_dict,
            candidates_considered=len(all_candidates),
            example_mentions=example_mentions,
            debug_info=debug_info,
        )
