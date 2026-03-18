#!/usr/bin/env python3
"""JobDistill CLI entrypoint.

Thin wrapper that delegates to the jobdistill package.
Keeps `python main.py` working exactly as before.
"""

import logging
import os
import random
import sys

from jobdistill.cli import parse_args
from jobdistill.pipeline import build_extractor, collect_pdf_files, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="skill_analysis.log",
)
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    pdf_files = collect_pdf_files(args.pdf_dirs, max_docs=args.max_docs)
    if not pdf_files:
        logger.error("No PDF files found in any directory")
        print("Error: No PDF files found in any directory")
        return

    total = len(pdf_files)
    print(f"Found {total} PDF files to analyze")
    logger.info("Found %d PDF files across all directories", total)

    extractor = build_extractor(
        extractor_name=args.extractor,
        gemini_model=args.gemini_model,
    )
    print(f"Using extractor: {extractor.name}")

    metrics_out = args.metrics_out if args.metrics_out else None
    df, metrics, categories = run_pipeline(
        pdf_files=pdf_files,
        extractor=extractor,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        metrics_out=metrics_out,
        concurrency=args.concurrency,
    )

    print("\nSkill Counts:")
    for _, row in df.iterrows():
        print(f"{row['Skill']}: {row['Count']}")

    output_path = args.output
    if output_path is None:
        if args.pdf_dirs:
            parent = os.path.dirname(args.pdf_dirs[0])
            output_path = os.path.join(parent, "skill_analysis_results.csv") if parent else "skill_analysis_results.csv"
        else:
            output_path = "skill_analysis_results.csv"

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Persist categories for offline dashboard regeneration
    import json
    cats_path = output_path.replace(".csv", "_categories.json")
    if categories:
        with open(cats_path, "w") as f:
            json.dump(categories, f, indent=2)
        print(f"Categories saved to: {cats_path}")

    if args.dashboard:
        from jobdistill.dashboard import generate_dashboard

        metrics_dict = metrics.to_dict(
            top_skills=list(zip(df["Skill"].tolist(), df["Count"].tolist()))
        )
        generate_dashboard(df, metrics_dict, args.dashboard, num_pdfs=total, categories=categories)
        print(f"Dashboard saved to: {args.dashboard}")


if __name__ == "__main__":
    main()
