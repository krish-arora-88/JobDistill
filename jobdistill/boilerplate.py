"""Corpus-level boilerplate removal.

Removes lines that appear across many documents (high document frequency),
as well as lines that are too short, purely punctuation, or UI-like.
This runs BEFORE skill extraction to prevent platform text from being
treated as keyphrases.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_LETTER_OR_DIGIT_RE = re.compile(r"[a-zA-Z0-9]")
_MOSTLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_MAX_LINE_KEY_LEN = 200
_DIGIT_RUN_RE = re.compile(r"\d+")


def canonicalize_line(line: str) -> str:
    """Canonicalize a line for document-frequency comparison.

    Lowercase, strip, replace digit runs with <NUM>, collapse whitespace,
    truncate to 200 chars.  This ensures lines that differ only by
    IDs / dates / page numbers are treated as identical for boilerplate
    detection.
    """
    line = line.strip().lower()
    line = _DIGIT_RUN_RE.sub("<NUM>", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line[:_MAX_LINE_KEY_LEN]


def _is_ui_like(line: str) -> bool:
    """True if a line is too short, mostly punctuation, or has no alphanum."""
    stripped = line.strip()
    if len(stripped) < 6:
        return True
    if not _LETTER_OR_DIGIT_RE.search(stripped):
        return True
    alpha_count = sum(1 for c in stripped if c.isalpha() or c.isdigit())
    if alpha_count < len(stripped) * 0.3:
        return True
    return False


@dataclass
class BoilerplateStats:
    """Metrics from a boilerplate removal pass."""

    total_lines: int = 0
    removed_lines: int = 0
    top_removed: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def removed_ratio(self) -> float:
        return self.removed_lines / max(self.total_lines, 1)


def compute_line_df(doc_texts: List[str]) -> Dict[str, int]:
    """Compute document frequency for each normalized line across docs.

    Returns {normalized_line: count_of_docs_containing_it}.
    """
    line_df: Counter = Counter()

    for text in doc_texts:
        seen_in_doc: set = set()
        for raw_line in text.split("\n"):
            key = canonicalize_line(raw_line)
            if not key:
                continue
            if key not in seen_in_doc:
                seen_in_doc.add(key)
                line_df[key] += 1

    return dict(line_df)


def strip_boilerplate(
    text: str,
    line_df: Dict[str, int],
    num_docs: int,
    df_threshold: float = 0.05,
) -> Tuple[str, int]:
    """Remove boilerplate lines from a single document.

    Lines are removed if:
      - They appear in >= df_threshold fraction of documents, OR
      - They are "UI-like" (short, no alphanumeric, mostly punctuation).

    Returns (cleaned_text, lines_removed_count).
    """
    min_doc_count = max(2, int(num_docs * df_threshold))

    kept: list = []
    removed = 0

    for raw_line in text.split("\n"):
        key = canonicalize_line(raw_line)

        if not key:
            kept.append(raw_line)
            continue

        if _is_ui_like(raw_line.strip()):
            removed += 1
            continue

        if line_df.get(key, 0) >= min_doc_count:
            removed += 1
            continue

        kept.append(raw_line)

    return "\n".join(kept), removed


def strip_boilerplate_corpus(
    doc_texts: List[str],
    df_threshold: float = 0.05,
) -> Tuple[List[str], BoilerplateStats]:
    """Two-pass boilerplate removal across a corpus.

    Pass 1: Compute line document frequency.
    Pass 2: Strip high-DF and UI-like lines from each doc.

    Returns (cleaned_texts, stats).
    """
    num_docs = len(doc_texts)
    if num_docs == 0:
        return [], BoilerplateStats()

    line_df = compute_line_df(doc_texts)

    min_doc_count = max(2, int(num_docs * df_threshold))

    cleaned: list = []
    stats = BoilerplateStats()

    for text in doc_texts:
        clean_text, n_removed = strip_boilerplate(
            text, line_df, num_docs, df_threshold
        )
        n_total = len([ln for ln in text.split("\n") if ln.strip()])
        stats.total_lines += n_total
        stats.removed_lines += n_removed
        cleaned.append(clean_text)

    top_removed = sorted(
        [(line, df) for line, df in line_df.items() if df >= min_doc_count],
        key=lambda x: -x[1],
    )[:20]
    stats.top_removed = top_removed

    logger.info(
        "Boilerplate removal: %d/%d lines removed (%.1f%%), threshold=%d docs",
        stats.removed_lines,
        stats.total_lines,
        stats.removed_ratio * 100,
        min_doc_count,
    )

    return cleaned, stats
