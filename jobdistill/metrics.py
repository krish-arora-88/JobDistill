"""Metrics collection, quality guardrails, and JSON/log output."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Generic boilerplate markers for the quality guardrail.
# Kept intentionally small (<15 terms) and in one place for easy auditing.
BOILERPLATE_MARKERS = frozenset({
    "application", "submit", "posting", "students", "canada",
    "apply", "employment", "employer", "accommodation", "deadline",
    "applicant", "hire", "resume",
})


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

    boilerplate_lines_total: int = 0
    boilerplate_lines_removed_total: int = 0
    boilerplate_lines_removed_ratio: float = 0.0
    top_removed_lines: List[tuple] = field(default_factory=list)
    classifier_floor_triggered_count: int = 0

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

    def record_boilerplate(self, stats: Any) -> None:
        """Record boilerplate removal stats from BoilerplateStats."""
        self.boilerplate_lines_total = stats.total_lines
        self.boilerplate_lines_removed_total = stats.removed_lines
        self.boilerplate_lines_removed_ratio = stats.removed_ratio
        self.top_removed_lines = [(line, count) for line, count in stats.top_removed[:20]]

    def _compute_quality_check(self, top_skills: List[tuple]) -> Dict[str, Any]:
        """Run quality guardrail on top-50 skills."""
        top50 = top_skills[:50]
        if not top50:
            return {"quality_failed": False, "boilerplate_pct_top50": 0.0}

        flagged = 0
        for skill, _ in top50:
            tokens = set(skill.lower().split())
            if tokens & BOILERPLATE_MARKERS:
                flagged += 1

        pct = flagged / len(top50)
        failed = pct > 0.30

        if failed:
            logger.warning(
                "QUALITY CHECK FAILED: %.0f%% of top-50 skills contain boilerplate markers",
                pct * 100,
            )

        return {
            "quality_failed": failed,
            "boilerplate_pct_top50": round(pct * 100, 1),
            "flagged_count": flagged,
            "checked_count": len(top50),
        }

    def to_dict(self, top_skills: Optional[List[tuple]] = None) -> Dict[str, Any]:
        elapsed = self.extraction_end - self.extraction_start if self.extraction_end else 0
        chars = np.array(self.chars_per_pdf) if self.chars_per_pdf else np.array([0])
        cands = np.array(self.candidates_per_pdf) if self.candidates_per_pdf else np.array([0])
        skills = np.array(self.skills_per_pdf) if self.skills_per_pdf else np.array([0])

        total = self.num_pdfs_total or 1

        all_skills = top_skills or []
        multiword = sum(1 for s, _ in all_skills if " " in s)
        lowercase = sum(1 for s, _ in all_skills if s == s.lower())

        apply_terms = set()
        for phrase in self.rejected_phrases[:100]:
            for tok in phrase.lower().split():
                if tok in BOILERPLATE_MARKERS:
                    apply_terms.add(tok)
        apply_count = sum(
            1 for s, _ in all_skills
            if set(s.lower().split()) & BOILERPLATE_MARKERS
        )

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
            "boilerplate": {
                "lines_total": self.boilerplate_lines_total,
                "lines_removed_total": self.boilerplate_lines_removed_total,
                "lines_removed_ratio": round(self.boilerplate_lines_removed_ratio, 4),
                "top_removed_lines": [
                    {"line": line, "doc_freq": df} for line, df in self.top_removed_lines[:20]
                ],
            },
            "diagnostics": {
                "pct_skills_with_space": round(
                    multiword / max(len(all_skills), 1) * 100, 1
                ),
                "pct_skills_all_lowercase": round(
                    lowercase / max(len(all_skills), 1) * 100, 1
                ),
                "pct_skills_containing_apply_terms": round(
                    apply_count / max(len(all_skills), 1) * 100, 1
                ),
                "top_rejected_phrases": self.rejected_phrases[:20],
            },
        }

        result["classifier_floor_triggered_count"] = self.classifier_floor_triggered_count

        if top_skills:
            result["top20_skills"] = [
                {"skill": s, "count": c} for s, c in top_skills[:20]
            ]
            result["quality_guardrail"] = self._compute_quality_check(top_skills)

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
            "%.1f avg chars, %.2f PDFs/sec, "
            "%d boilerplate lines removed (%.1f%%)",
            d["num_pdfs_total"],
            d["num_pdfs_extracted_ok"],
            d["num_pdfs_empty_text"],
            d["avg_chars_per_pdf"],
            d["pdfs_per_second"],
            d["boilerplate"]["lines_removed_total"],
            d["boilerplate"]["lines_removed_ratio"] * 100,
        )
