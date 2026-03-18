"""Main processing pipeline: PDF ingestion, extraction, aggregation."""

from __future__ import annotations

import concurrent.futures
import glob
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from jobdistill.extractors.base import ExtractionResult, SkillExtractor
from jobdistill.extractors.gemini_extractor import GeminiSkillExtractor
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.metrics import PipelineMetrics
from jobdistill.pdf_text import extract_pdf
from jobdistill.skill_aliases import normalize_skill

logger = logging.getLogger(__name__)


def collect_pdf_files(pdf_dirs: List[str], max_docs: Optional[int] = None) -> List[str]:
    """Gather PDF paths from one or more directories."""
    pdf_files: list[str] = []
    for pdf_dir in pdf_dirs:
        if not os.path.exists(pdf_dir):
            logger.error("Directory not found: %s", pdf_dir)
            print(f"Error: Directory not found: {pdf_dir}")
            continue
        dir_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not dir_files:
            logger.error("No PDF files found in %s", pdf_dir)
            print(f"Error: No PDF files found in {pdf_dir}")
            continue
        pdf_files.extend(dir_files)

    if max_docs is not None and max_docs > 0:
        pdf_files = pdf_files[:max_docs]
    return pdf_files


def build_extractor(
    extractor_name: str,
    gemini_model: str = "gemini-2.5-flash",
) -> SkillExtractor:
    """Factory: create the right extractor from CLI args."""
    if extractor_name == "regex":
        return RegexSkillExtractor()
    elif extractor_name == "gemini":
        return GeminiSkillExtractor(model=gemini_model)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name!r}. Use 'gemini' or 'regex'.")


def run_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    batch_size: int = 20,
    cache_dir: Optional[str] = None,
    metrics_out: Optional[str] = None,
    concurrency: int = 10,
) -> Tuple[pd.DataFrame, PipelineMetrics, Dict[str, str]]:
    """Process PDFs, aggregate skill counts, return DataFrame + metrics + categories.

    For the regex extractor, we use its batch counting semantics.
    For the Gemini extractor, we do concurrent LLM calls and aggregate by doc frequency.
    """
    metrics = PipelineMetrics()
    metrics.start_timer()

    categories: Dict[str, str] = {}

    if isinstance(extractor, RegexSkillExtractor):
        df, metrics = _run_regex_pipeline(pdf_files, extractor, batch_size, cache_dir, metrics)
    else:
        df, metrics, categories = _run_gemini_pipeline(
            pdf_files, extractor, cache_dir, metrics, concurrency,
        )

    metrics.stop_timer()

    sorted_skills = list(zip(df["Skill"].tolist(), df["Count"].tolist()))
    metrics.log_summary(sorted_skills)

    if metrics_out:
        metrics.write_json(metrics_out, sorted_skills)

    return df, metrics, categories


def _run_regex_pipeline(
    pdf_files: List[str],
    extractor: RegexSkillExtractor,
    batch_size: int,
    cache_dir: Optional[str],
    metrics: PipelineMetrics,
) -> Tuple[pd.DataFrame, PipelineMetrics]:
    """Regex path: preserves original batching + dedup-per-PDF semantics."""
    all_counts: Counter = Counter()

    batches = [pdf_files[i : i + batch_size] for i in range(0, len(pdf_files), batch_size)]
    print(f"Processing {len(batches)} batches with batch size {batch_size}")

    def _process_batch(batch: List[str]) -> Counter:
        batch_text = ""
        for pdf in batch:
            text = extract_pdf(pdf, cache_dir=cache_dir)
            tokens = text.split()
            seen: set[str] = set()
            unique: list[str] = []
            for tok in tokens:
                if tok not in seen:
                    unique.append(tok)
                    seen.add(tok)
            batch_text += " ".join(unique) + " "
        return extractor.extract_counts(batch_text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(batches) or 1)) as pool:
        futures = {pool.submit(_process_batch, b): b for b in batches}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(batches),
            desc="Processing PDF batches",
        ):
            try:
                batch_counts = future.result()
                all_counts.update(batch_counts)
                for pdf in futures[future]:
                    text = extract_pdf(pdf, cache_dir=cache_dir)
                    metrics.record_pdf(text, candidates=0, skills=len(batch_counts))
            except Exception as e:
                logger.error("Error processing batch: %s", e)

    sorted_skills = sorted(all_counts.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame(sorted_skills, columns=["Skill", "Count"])
    return df, metrics


def _run_gemini_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    cache_dir: Optional[str],
    metrics: PipelineMetrics,
    concurrency: int = 10,
) -> Tuple[pd.DataFrame, PipelineMetrics, Dict[str, str]]:
    """Gemini path: extract text, concurrent LLM calls, aggregate by doc frequency."""

    # Pass 1: Extract flat text from all PDFs
    print(f"Pass 1: Extracting text from {len(pdf_files)} PDFs...")
    texts: dict[str, str] = {}
    for pdf_path in tqdm(pdf_files, desc="Extracting PDF text"):
        texts[pdf_path] = extract_pdf(pdf_path, cache_dir=cache_dir)

    # Pass 2: Concurrent Gemini extraction
    print(f"Pass 2: Extracting skills via Gemini ({concurrency} concurrent)...")
    skill_counts: Counter = Counter()
    categories: Dict[str, str] = {}
    results: dict[str, ExtractionResult] = {}

    def _extract_one(pdf_path: str) -> Tuple[str, ExtractionResult]:
        text = texts[pdf_path]
        result = extractor.extract(text)
        return pdf_path, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_extract_one, p): p for p in pdf_files}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Gemini extraction",
        ):
            pdf_path = futures[future]
            try:
                _, result = future.result()
                results[pdf_path] = result
                metrics.gemini_request_count += 1

                # Aggregate by doc frequency (each skill counts once per PDF)
                doc_skills: set = set()
                for skill in result.skills:
                    canonical = normalize_skill(skill)
                    if canonical and canonical not in doc_skills:
                        skill_counts[canonical] += 1
                        doc_skills.add(canonical)

                # Collect categories from debug_info
                if result.debug_info and "categories" in result.debug_info:
                    for skill, cat in result.debug_info["categories"].items():
                        canonical = normalize_skill(skill)
                        if canonical and canonical not in categories:
                            categories[canonical] = cat

                metrics.record_pdf(texts[pdf_path], result.candidates_considered, len(result.skills))
            except Exception as e:
                logger.error("Error processing %s: %s", pdf_path, e)
                metrics.gemini_error_count += 1

    sorted_skills = sorted(skill_counts.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame(sorted_skills, columns=["Skill", "Count"])
    return df, metrics, categories
