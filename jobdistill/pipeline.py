"""Main processing pipeline: PDF ingestion, boilerplate removal, extraction, aggregation."""

from __future__ import annotations

import concurrent.futures
import glob
import json
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from jobdistill.boilerplate import BoilerplateStats, strip_boilerplate_corpus
from jobdistill.extractors.base import ExtractionResult, SkillExtractor
from jobdistill.extractors.ml_extractor import MLSkillExtractor
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.metrics import PipelineMetrics
from jobdistill.pdf_text import extract_pdf, extract_pdf_with_lines

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
    min_confidence: float = 0.75,
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


def run_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    batch_size: int = 20,
    cache_dir: Optional[str] = None,
    include_confidence: bool = False,
    metrics_out: Optional[str] = None,
    boilerplate_df_threshold: float = 0.05,
    disable_boilerplate: bool = False,
    debug_samples: int = 0,
    debug_dump_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, PipelineMetrics]:
    """Process PDFs, aggregate skill counts, return DataFrame + metrics.

    For the regex extractor, we use its batch counting semantics.
    For the ML extractor, we do a 2-pass approach:
      Pass 1: Extract text from all PDFs (preserving newlines), compute corpus boilerplate map.
      Pass 2: Strip boilerplate per doc, run skill extraction, aggregate.
    """
    metrics = PipelineMetrics()
    metrics.start_timer()

    if isinstance(extractor, RegexSkillExtractor):
        df, metrics = _run_regex_pipeline(pdf_files, extractor, batch_size, cache_dir, metrics)
    else:
        df, metrics = _run_ml_pipeline(
            pdf_files, extractor, batch_size, cache_dir,
            include_confidence, metrics, boilerplate_df_threshold,
            disable_boilerplate, debug_samples, debug_dump_path,
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


def _log_debug_doc(
    idx: int, pdf_path: str, text: str, result: ExtractionResult,
) -> None:
    """Log detailed debug info for a single document."""
    lines_count = len([ln for ln in text.split("\n") if ln.strip()])
    info = result.debug_info or {}
    logger.info(
        "[DEBUG doc %d] %s | chars=%d lines=%d | keybert=%d tfidf=%d tech=%d "
        "union=%d | threshold=%.2f relaxed=%s | final_skills=%d",
        idx,
        os.path.basename(pdf_path),
        len(text),
        lines_count,
        info.get("keybert_candidates", 0),
        info.get("tfidf_candidates", 0),
        info.get("tech_token_candidates", 0),
        info.get("total_union_candidates", 0),
        info.get("threshold_used", 0),
        info.get("threshold_relaxed", False),
        info.get("final_skills", 0),
    )
    print(
        f"  [DEBUG doc {idx}] {os.path.basename(pdf_path)}: "
        f"{len(text)} chars, {lines_count} lines | "
        f"candidates: keybert={info.get('keybert_candidates', 0)} "
        f"tfidf={info.get('tfidf_candidates', 0)} "
        f"tech={info.get('tech_token_candidates', 0)} | "
        f"union={info.get('total_union_candidates', 0)} | "
        f"threshold={info.get('threshold_used', 0):.2f} "
        f"(relaxed={info.get('threshold_relaxed', False)}) | "
        f"final={info.get('final_skills', 0)} skills"
    )
    top_cands = info.get("top_candidates", [])
    if top_cands:
        print("    Top candidates:")
        for c in top_cands[:10]:
            status = "KEPT" if c.get("kept") else "filtered"
            print(f"      {c['phrase']:30s}  prob={c['classifier_prob']:.4f}  {status}")


def _run_ml_pipeline(
    pdf_files: List[str],
    extractor: SkillExtractor,
    batch_size: int,
    cache_dir: Optional[str],
    include_confidence: bool,
    metrics: PipelineMetrics,
    boilerplate_df_threshold: float = 0.05,
    disable_boilerplate: bool = False,
    debug_samples: int = 0,
    debug_dump_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, PipelineMetrics]:
    """ML path: 2-pass boilerplate removal, per-doc extraction, aggregate by document frequency."""

    # --- Pass 1: Extract raw text from all PDFs (preserving newlines) ---
    print(f"Pass 1: Extracting text from {len(pdf_files)} PDFs...")
    raw_texts: list[str] = []
    for pdf_path in tqdm(pdf_files, desc="Extracting PDF text"):
        text = extract_pdf_with_lines(pdf_path, cache_dir=cache_dir)
        raw_texts.append(text)

    # --- Corpus boilerplate removal ---
    if disable_boilerplate:
        cleaned_texts = raw_texts
        total = sum(len([ln for ln in t.split("\n") if ln.strip()]) for t in raw_texts)
        bp_stats = BoilerplateStats(total_lines=total)
        print("  Boilerplate removal DISABLED")
    else:
        print("Stripping corpus-level boilerplate...")
        cleaned_texts, bp_stats = strip_boilerplate_corpus(
            raw_texts, df_threshold=boilerplate_df_threshold,
        )
        metrics.record_boilerplate(bp_stats)
    print(
        f"  Removed {bp_stats.removed_lines}/{bp_stats.total_lines} lines "
        f"({bp_stats.removed_ratio:.1%})"
    )

    # --- Pass 2: Per-document skill extraction + document-frequency aggregation ---
    skill_counts: Counter = Counter()
    skill_confidences: Dict[str, List[float]] = defaultdict(list)
    skill_examples: Dict[str, List[str]] = defaultdict(list)
    debug_rows: list[dict] = []

    print(f"Pass 2: Extracting skills from {len(pdf_files)} cleaned documents...")
    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Extracting skills")):
        try:
            text = cleaned_texts[idx]
            result = extractor.extract(text)

            if debug_samples > 0 and idx < debug_samples:
                _log_debug_doc(idx, pdf_path, text, result)
                if debug_dump_path and result.debug_info:
                    debug_rows.append({
                        "doc_idx": idx,
                        "pdf_path": pdf_path,
                        "chars": len(text),
                        **result.debug_info,
                    })

            doc_skills: set = set()
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

    # Write debug dump
    if debug_dump_path and debug_rows:
        with open(debug_dump_path, "w") as f:
            for row in debug_rows:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"  Debug dump written to {debug_dump_path}")

    # Record classifier floor trigger count
    if hasattr(extractor, "classifier_floor_triggered_count"):
        metrics.classifier_floor_triggered_count = extractor.classifier_floor_triggered_count

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
