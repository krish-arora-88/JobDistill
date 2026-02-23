"""Main processing pipeline: PDF ingestion, extraction, aggregation, output."""

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
from jobdistill.extractors.ml_extractor import MLSkillExtractor
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.metrics import PipelineMetrics
from jobdistill.pdf_text import extract_pdf

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
    model_dir: Optional[str] = None,
    top_k: int = 30,
    min_confidence: float = 0.60,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> SkillExtractor:
    """Factory: create the right extractor from CLI args."""
    if extractor_name == "regex":
        return RegexSkillExtractor()
    elif extractor_name == "ml":
        return MLSkillExtractor(
            embedding_model=embedding_model,
            model_dir=model_dir,
            top_k=top_k,
            min_confidence=min_confidence,
        )
    else:
        raise ValueError(f"Unknown extractor: {extractor_name!r}. Use 'ml' or 'regex'.")


def _process_single_pdf(
    pdf_path: str,
    extractor: SkillExtractor,
    cache_dir: Optional[str],
) -> Tuple[str, str, ExtractionResult]:
    """Extract text and skills from one PDF. Returns (path, text, result)."""
    text = extract_pdf(pdf_path, cache_dir=cache_dir)
    result = extractor.extract(text)
    return pdf_path, text, result


def run_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    batch_size: int = 20,
    cache_dir: Optional[str] = None,
    include_confidence: bool = False,
    metrics_out: Optional[str] = None,
) -> Tuple[pd.DataFrame, PipelineMetrics]:
    """Process PDFs, aggregate skill counts, return DataFrame + metrics.

    For the regex extractor, we use its special batch counting semantics
    (dedup within each PDF, then count across the batch) to preserve
    backward compatibility.  For the ML extractor, we run per-doc
    extraction and aggregate.
    """
    metrics = PipelineMetrics()
    metrics.start_timer()

    if isinstance(extractor, RegexSkillExtractor):
        df, metrics = _run_regex_pipeline(pdf_files, extractor, batch_size, cache_dir, metrics)
    else:
        df, metrics = _run_ml_pipeline(
            pdf_files, extractor, batch_size, cache_dir, include_confidence, metrics
        )

    metrics.stop_timer()

    sorted_skills = list(zip(df["Skill"].tolist(), df["Count"].tolist()))
    metrics.log_summary(sorted_skills)

    if metrics_out:
        metrics.write_json(metrics_out, sorted_skills)

    return df, metrics


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


def _run_ml_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    batch_size: int,
    cache_dir: Optional[str],
    include_confidence: bool,
    metrics: PipelineMetrics,
) -> Tuple[pd.DataFrame, PipelineMetrics]:
    """ML path: per-doc extraction, then aggregate."""
    skill_counts: Counter = Counter()
    skill_confidences: Dict[str, List[float]] = defaultdict(list)
    skill_examples: Dict[str, List[str]] = defaultdict(list)

    print(f"Processing {len(pdf_files)} PDFs with ML extractor...")
    for pdf_path in tqdm(pdf_files, desc="Extracting skills"):
        try:
            text = extract_pdf(pdf_path, cache_dir=cache_dir)
            result = extractor.extract(text)

            doc_skills = set()
            for skill, conf in result.skills.items():
                if skill not in doc_skills:
                    skill_counts[skill] += 1
                    doc_skills.add(skill)
                skill_confidences[skill].append(conf)

            for skill, mentions in result.example_mentions.items():
                remaining = 3 - len(skill_examples[skill])
                if remaining > 0:
                    skill_examples[skill].extend(mentions[:remaining])

            metrics.record_pdf(text, result.candidates_considered, len(result.skills))
        except Exception as e:
            logger.error("Error processing %s: %s", pdf_path, e)

    sorted_skills = sorted(skill_counts.items(), key=lambda x: (-x[1], x[0]))

    rows: list[dict] = []
    for skill, count in sorted_skills:
        row: dict = {"Skill": skill, "Count": count}
        if include_confidence:
            confs = skill_confidences.get(skill, [])
            row["AvgConfidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
            row["ExampleMentions"] = " | ".join(skill_examples.get(skill, [])[:3])
        rows.append(row)

    columns = ["Skill", "Count"]
    if include_confidence:
        columns += ["AvgConfidence", "ExampleMentions"]
    df = pd.DataFrame(rows, columns=columns)
    return df, metrics
