#!/usr/bin/env python3
"""Evaluate and compare extractors on a sample of PDFs.

Runs both regex and ML extractors on the same docs and outputs a diff
report showing top deltas (skills found by one but not the other).

Usage:
    python scripts/eval_extractor.py \
        --pdf_dirs Summer_2025_Co-op \
        --max_docs 20 \
        --model_dir models/skill_classifier
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobdistill.extractors.ml_extractor import MLSkillExtractor
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.pdf_text import extract_pdf
from jobdistill.pipeline import collect_pdf_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regex vs ML extractors")
    parser.add_argument(
        "--pdf_dirs", nargs="+",
        default=["Summer_2025_Co-op", "Fall_2025_Co-op", "Winter_2026_Co-op", "Summer_2026_Co-op"],
    )
    parser.add_argument("--max_docs", type=int, default=20)
    parser.add_argument("--model_dir", type=str, default="models/skill_classifier")
    parser.add_argument("--cache_dir", type=str, default=".cache/jobdistill")
    parser.add_argument("--out", type=str, default="eval_report.csv")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--min_confidence", type=float, default=0.60)
    args = parser.parse_args()

    pdf_files = collect_pdf_files(args.pdf_dirs, max_docs=args.max_docs)
    if not pdf_files:
        logger.error("No PDFs found.")
        sys.exit(1)

    regex_ext = RegexSkillExtractor()
    ml_ext = MLSkillExtractor(
        model_dir=args.model_dir,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )

    regex_counts: Counter = Counter()
    ml_counts: Counter = Counter()

    for pdf_path in pdf_files:
        text = extract_pdf(pdf_path, cache_dir=args.cache_dir)
        if not text.strip():
            continue

        regex_result = regex_ext.extract(text)
        for skill in regex_result.skills:
            regex_counts[skill] += 1

        ml_result = ml_ext.extract(text)
        for skill in ml_result.skills:
            ml_counts[skill] += 1

    all_skills = sorted(set(regex_counts.keys()) | set(ml_counts.keys()))

    rows = []
    for skill in all_skills:
        rc = regex_counts.get(skill, 0)
        mc = ml_counts.get(skill, 0)
        rows.append({
            "Skill": skill,
            "RegexCount": rc,
            "MLCount": mc,
            "Delta": mc - rc,
            "Source": "both" if rc > 0 and mc > 0 else ("regex_only" if rc > 0 else "ml_only"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Delta", key=abs, ascending=False)
    df.to_csv(args.out, index=False)

    print(f"\nEvaluation report written to {args.out}")
    print(f"  Total unique skills — regex: {len(regex_counts)}, ML: {len(ml_counts)}")
    print(f"  Skills found by both: {sum(1 for r in rows if r['Source'] == 'both')}")
    print(f"  Regex-only: {sum(1 for r in rows if r['Source'] == 'regex_only')}")
    print(f"  ML-only: {sum(1 for r in rows if r['Source'] == 'ml_only')}")
    print("\nTop 15 deltas:")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
