"""Metrics collection and JSON/log output for pipeline health monitoring."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Accumulates stats during a pipeline run."""

    num_pdfs_total: int = 0
    num_pdfs_extracted_ok: int = 0
    num_pdfs_empty_text: int = 0
    chars_per_pdf: List[int] = field(default_factory=list)
    candidates_per_pdf: List[int] = field(default_factory=list)
    skills_per_pdf: List[int] = field(default_factory=list)
    extraction_start: float = 0.0
    extraction_end: float = 0.0
    rejected_phrases: List[str] = field(default_factory=list)

    def start_timer(self) -> None:
        self.extraction_start = time.time()

    def stop_timer(self) -> None:
        self.extraction_end = time.time()

    def record_pdf(self, text: str, candidates: int, skills: int) -> None:
        self.num_pdfs_total += 1
        n_chars = len(text)
        self.chars_per_pdf.append(n_chars)
        self.candidates_per_pdf.append(candidates)
        self.skills_per_pdf.append(skills)
        if n_chars > 0:
            self.num_pdfs_extracted_ok += 1
        else:
            self.num_pdfs_empty_text += 1

    def record_rejected(self, phrases: List[str], max_keep: int = 50) -> None:
        remaining = max_keep - len(self.rejected_phrases)
        if remaining > 0:
            self.rejected_phrases.extend(phrases[:remaining])

    def to_dict(self, top_skills: Optional[List[tuple]] = None) -> Dict[str, Any]:
        elapsed = self.extraction_end - self.extraction_start if self.extraction_end else 0
        chars = np.array(self.chars_per_pdf) if self.chars_per_pdf else np.array([0])
        cands = np.array(self.candidates_per_pdf) if self.candidates_per_pdf else np.array([0])
        skills = np.array(self.skills_per_pdf) if self.skills_per_pdf else np.array([0])

        total = self.num_pdfs_total or 1
        single_token_skills = 0
        lowercase_only_skills = 0
        if top_skills:
            for skill, _ in top_skills:
                if len(skill.split()) == 1:
                    single_token_skills += 1
                if skill == skill.lower():
                    lowercase_only_skills += 1

        result: Dict[str, Any] = {
            "num_pdfs_total": self.num_pdfs_total,
            "num_pdfs_extracted_ok": self.num_pdfs_extracted_ok,
            "num_pdfs_empty_text": self.num_pdfs_empty_text,
            "avg_chars_per_pdf": float(chars.mean()),
            "p95_chars_per_pdf": float(np.percentile(chars, 95)) if len(chars) > 0 else 0,
            "extraction_seconds_total": round(elapsed, 2),
            "pdfs_per_second": round(total / elapsed, 2) if elapsed > 0 else 0,
            "avg_candidates_per_pdf": float(cands.mean()),
            "avg_skills_per_pdf": float(skills.mean()),
            "diagnostics": {
                "pct_single_token_skills": round(
                    single_token_skills / max(len(top_skills or []), 1) * 100, 1
                ),
                "pct_lowercase_only_skills": round(
                    lowercase_only_skills / max(len(top_skills or []), 1) * 100, 1
                ),
                "top_rejected_phrases": self.rejected_phrases[:20],
            },
        }
        if top_skills:
            result["top20_skills"] = [
                {"skill": s, "count": c} for s, c in top_skills[:20]
            ]
        return result

    def write_json(self, path: str, top_skills: Optional[List[tuple]] = None) -> None:
        data = self.to_dict(top_skills)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Metrics written to %s", path)

    def log_summary(self, top_skills: Optional[List[tuple]] = None) -> None:
        d = self.to_dict(top_skills)
        logger.info(
            "Pipeline summary: %d PDFs total, %d OK, %d empty text, "
            "%.1f avg chars, %.2f PDFs/sec",
            d["num_pdfs_total"],
            d["num_pdfs_extracted_ok"],
            d["num_pdfs_empty_text"],
            d["avg_chars_per_pdf"],
            d["pdfs_per_second"],
        )
