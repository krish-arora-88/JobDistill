"""CLI argument definitions for JobDistill."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JobDistill — extract and rank skills from job-posting PDFs",
    )

    parser.add_argument(
        "--pdf_dirs",
        nargs="+",
        default=["Summer_2025_Co-op", "Fall_2025_Co-op", "Winter_2026_Co-op", "Summer_2026_Co-op"],
        help="Directories containing PDF files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of PDFs to process in each batch (regex mode)",
    )

    parser.add_argument(
        "--extractor",
        choices=["ml", "regex"],
        default="ml",
        help="Extraction backend (default: ml)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/skill_classifier",
        help="Directory containing trained skill classifier (ML mode)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.75,
        help="Minimum P(skill) threshold for ML extractor (default: 0.75)",
    )
    parser.add_argument(
        "--top_k_phrases",
        type=int,
        default=30,
        help="Max candidate keyphrases per document before classifier (default: 30)",
    )
    parser.add_argument(
        "--boilerplate_df_threshold",
        type=float,
        default=0.05,
        help="Document-frequency threshold for corpus boilerplate removal (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--include_confidence_cols",
        action="store_true",
        default=False,
        help="Add AvgConfidence and ExampleMentions columns to output CSV",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/jobdistill",
        help="Directory for caching extracted text and results",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="metrics.json",
        help="Path to write pipeline metrics JSON (set '' to disable)",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Limit number of PDFs to process (for sampling/debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: skill_analysis_results.csv)",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=0,
        help="Log detailed extraction info for the first N docs (default: 0 = off)",
    )
    parser.add_argument(
        "--debug_dump_path",
        type=str,
        default=None,
        help="Write per-doc debug JSONL to this path (requires --debug_samples > 0)",
    )
    parser.add_argument(
        "--disable_boilerplate_removal",
        action="store_true",
        default=False,
        help="Skip corpus boilerplate removal (for debugging)",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
