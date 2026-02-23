"""PDF text extraction with caching support."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text

from jobdistill.normalize import clean_text

logger = logging.getLogger(__name__)


def _cache_key(pdf_path: str) -> str:
    """Deterministic cache key from absolute path + mtime."""
    abs_path = os.path.abspath(pdf_path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0
    raw = f"{abs_path}::{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()


def extract_pdf(pdf_path: str, cache_dir: Optional[str] = None) -> str:
    """Extract and clean text from a PDF, with optional disk cache.

    Returns cleaned text (may be empty string on failure).
    """
    if cache_dir:
        cache_path = Path(cache_dir) / "text" / f"{_cache_key(pdf_path)}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    text = _raw_extract(pdf_path)
    text = clean_text(text)

    if cache_dir:
        cache_path = Path(cache_dir) / "text" / f"{_cache_key(pdf_path)}.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")

    return text


def _raw_extract(pdf_path: str) -> str:
    """Call pdfminer; return empty string on error."""
    try:
        return extract_text(pdf_path) or ""
    except Exception as e:
        logger.error("Error extracting text from %s: %s", pdf_path, e)
        return ""
