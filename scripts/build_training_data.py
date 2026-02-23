#!/usr/bin/env python3
"""Generate a weakly-labeled training dataset for the skill classifier.

Reads PDFs (or cached text), extracts candidate phrases via KeyBERT,
then labels them using the legacy skills list as positive seeds.
Remaining candidates are sampled as negatives.

Usage:
    python scripts/build_training_data.py \
        --pdf_dirs Summer_2025_Co-op Winter_2026_Co-op \
        --out data/training_data.jsonl \
        --max_docs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobdistill.extractors.keybert_extractor import KeyBERTExtractor
from jobdistill.extractors.regex_extractor import get_possible_skills, get_skill_mapping
from jobdistill.normalize import normalize_phrase
from jobdistill.pdf_text import extract_pdf
from jobdistill.pipeline import collect_pdf_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_positive_set() -> set[str]:
    """All known skill strings (lowercased) from the legacy inventory + alias keys."""
    skills = set(s.lower() for s in get_possible_skills())
    mapping = get_skill_mapping()
    for alias, norm in mapping.items():
        skills.add(alias.lower())
        if isinstance(norm, list):
            for n in norm:
                skills.add(n.lower())
        else:
            skills.add(norm.lower())
    return skills


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak-label training data")
    parser.add_argument(
        "--pdf_dirs", nargs="+",
        default=["Summer_2025_Co-op", "Fall_2025_Co-op", "Winter_2026_Co-op", "Summer_2026_Co-op"],
    )
    parser.add_argument("--out", type=str, default="data/training_data.jsonl")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--cache_dir", type=str, default=".cache/jobdistill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_ratio", type=float, default=1.5,
                        help="Ratio of negatives to positives to sample")
    args = parser.parse_args()

    random.seed(args.seed)

    pdf_files = collect_pdf_files(args.pdf_dirs, max_docs=args.max_docs)
    if not pdf_files:
        logger.error("No PDF files found.")
        sys.exit(1)
    logger.info("Found %d PDFs", len(pdf_files))

    positive_set = _build_positive_set()
    logger.info("Positive seed set contains %d phrases", len(positive_set))

    extractor = KeyBERTExtractor(top_k=args.top_k)

    positives: list[dict] = []
    negatives: list[dict] = []

    for pdf_path in pdf_files:
        text = extract_pdf(pdf_path, cache_dir=args.cache_dir)
        if not text.strip():
            continue
        candidates = extractor.extract_candidates(text)
        for phrase, score in candidates:
            norm = normalize_phrase(phrase)
            record = {
                "text": norm,
                "source_pdf": pdf_path,
                "extractor_score": round(score, 4),
            }
            if norm.lower() in positive_set:
                record["label"] = 1
                positives.append(record)
            else:
                record["label"] = 0
                negatives.append(record)

    max_neg = int(len(positives) * args.neg_ratio)
    if len(negatives) > max_neg:
        negatives = random.sample(negatives, max_neg)

    all_records = positives + negatives
    random.shuffle(all_records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(
        "Wrote %d records (%d positive, %d negative) to %s",
        len(all_records), len(positives), len(negatives), args.out,
    )


if __name__ == "__main__":
    main()
