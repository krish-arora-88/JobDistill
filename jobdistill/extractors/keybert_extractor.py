"""Candidate skill-phrase extraction using KeyBERT (unsupervised)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from jobdistill.normalize import is_valid_candidate, normalize_phrase

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 2000  # chars
DEFAULT_CHUNK_OVERLAP = 200


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class KeyBERTExtractor:
    """Extract candidate keyphrases from text using KeyBERT.

    This is the *unsupervised* first stage of the ML pipeline.  It returns
    ranked (phrase, score) pairs without judging whether they are "skills."
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 30,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        use_noun_chunks: bool = True,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_noun_chunks = use_noun_chunks

        self._kw_model: Optional[object] = None
        self._nlp: Optional[object] = None
        self._available = True
        self._warned = False

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_model(self) -> None:
        """Lazy-load KeyBERT (and optionally spaCy) on first use."""
        if self._kw_model is not None:
            return
        if not self._available:
            return

        try:
            from keybert import KeyBERT  # type: ignore[import-untyped]
        except ImportError:
            if not self._warned:
                logger.warning(
                    "keybert package not installed; KeyBERT extractor unavailable. "
                    "Install with: pip install keybert"
                )
                self._warned = True
            self._available = False
            return
        except Exception as exc:
            if not self._warned:
                logger.warning("Failed to import keybert: %s", exc)
                self._warned = True
            self._available = False
            return

        try:
            logger.info("Loading KeyBERT with model %s", self.embedding_model_name)
            self._kw_model = KeyBERT(model=self.embedding_model_name)
        except Exception as exc:
            if not self._warned:
                logger.warning("Failed to load KeyBERT model: %s", exc)
                self._warned = True
            self._available = False
            return

        if self.use_noun_chunks:
            try:
                import spacy  # type: ignore[import-untyped]

                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning(
                        "spaCy model 'en_core_web_sm' not found; "
                        "falling back to n-gram candidates. "
                        "Install with: python -m spacy download en_core_web_sm"
                    )
            except ImportError:
                logger.info("spaCy not installed; using n-gram candidates only")

    def extract_candidates(self, text: str) -> List[Tuple[str, float]]:
        """Return deduplicated (phrase, best_score) list from the document."""
        self._ensure_model()
        if not self._available or not text or not text.strip():
            return []

        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        phrase_scores: Dict[str, float] = {}

        seed_keywords = self._get_noun_chunks(text) if self._nlp else None

        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue
            try:
                keywords = self._kw_model.extract_keywords(  # type: ignore[union-attr]
                    chunk,
                    keyphrase_ngram_range=self.keyphrase_ngram_range,
                    stop_words="english",
                    top_n=self.top_k,
                    use_mmr=True,
                    diversity=0.5,
                    seed_keywords=seed_keywords,
                )
            except Exception as e:
                logger.warning("KeyBERT extraction failed on chunk: %s", e)
                continue

            for phrase, score in keywords:
                norm = normalize_phrase(phrase)
                if not is_valid_candidate(norm):
                    continue
                if norm.lower() not in phrase_scores or score > phrase_scores[norm.lower()]:
                    phrase_scores[norm.lower()] = score

        ranked = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_k]

    def _get_noun_chunks(self, text: str) -> Optional[List[str]]:
        """Extract noun chunks via spaCy for seed guidance (capped for speed)."""
        if self._nlp is None:
            return None
        try:
            max_chars = 50_000
            doc = self._nlp(text[:max_chars])  # type: ignore[misc]
            chunks = list({chunk.text.strip().lower() for chunk in doc.noun_chunks})
            return chunks[:200] if chunks else None
        except Exception:
            return None
