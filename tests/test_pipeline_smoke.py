"""Smoke tests for the processing pipeline using synthetic data."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.metrics import PipelineMetrics
from jobdistill.pipeline import build_extractor, collect_pdf_files, run_pipeline


SAMPLE_TEXT_1 = """
Software Developer Position
Skills required: Python, JavaScript, React, Docker, AWS, SQL, Git
Experience with machine learning and TensorFlow preferred.
Agile methodology. CI/CD pipelines.
"""

SAMPLE_TEXT_2 = """
Data Engineer Role
Must know Python, SQL, Apache Spark, Kafka, Kubernetes.
Experience with PostgreSQL, Redis, and cloud platforms (AWS, Azure).
Understanding of ETL pipelines and data modeling.
"""


def _create_fake_pdf(directory: str, name: str, text: str) -> str:
    """Create a minimal PDF-like file for testing.

    Since pdfminer will fail on these, we test by mocking extract_pdf.
    """
    path = os.path.join(directory, name)
    Path(path).write_text(text)
    return path


class TestCollectPdfFiles:
    def test_finds_pdfs(self, tmp_path):
        d = tmp_path / "pdfs"
        d.mkdir()
        (d / "a.pdf").write_text("test")
        (d / "b.pdf").write_text("test")
        (d / "c.txt").write_text("test")
        result = collect_pdf_files([str(d)])
        assert len(result) == 2

    def test_max_docs(self, tmp_path):
        d = tmp_path / "pdfs"
        d.mkdir()
        for i in range(10):
            (d / f"{i}.pdf").write_text("test")
        result = collect_pdf_files([str(d)], max_docs=3)
        assert len(result) == 3

    def test_missing_dir(self, capsys):
        result = collect_pdf_files(["/nonexistent/path"])
        assert result == []

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = collect_pdf_files([str(d)])
        assert result == []


class TestBuildExtractor:
    def test_regex(self):
        ext = build_extractor("regex")
        assert ext.name == "regex"
        assert isinstance(ext, RegexSkillExtractor)

    def test_ml(self):
        ext = build_extractor("ml", model_dir=None)
        assert ext.name == "ml"

    def test_invalid(self):
        with pytest.raises(ValueError):
            build_extractor("invalid")


class TestRegexPipeline:
    def test_regex_extractor_output(self):
        ext = RegexSkillExtractor()
        result = ext.extract(SAMPLE_TEXT_1)
        assert "Python" in result.skills
        assert "JavaScript" in result.skills

    def test_regex_count_aggregation(self):
        ext = RegexSkillExtractor()
        counts = ext.extract_counts(SAMPLE_TEXT_1 + " " + SAMPLE_TEXT_2)
        assert counts["Python"] >= 2
        assert counts["SQL"] >= 2


class TestPipelineMetrics:
    def test_record_pdf(self):
        m = PipelineMetrics()
        m.record_pdf("hello world", candidates=10, skills=5)
        assert m.num_pdfs_total == 1
        assert m.num_pdfs_extracted_ok == 1
        assert m.num_pdfs_empty_text == 0

    def test_empty_pdf(self):
        m = PipelineMetrics()
        m.record_pdf("", candidates=0, skills=0)
        assert m.num_pdfs_empty_text == 1

    def test_to_dict(self):
        m = PipelineMetrics()
        m.start_timer()
        m.record_pdf("some text here", candidates=5, skills=3)
        m.stop_timer()
        d = m.to_dict(top_skills=[("Python", 10), ("Java", 8)])
        assert "num_pdfs_total" in d
        assert "extraction_seconds_total" in d
        assert "top20_skills" in d
        assert d["num_pdfs_total"] == 1

    def test_write_json(self, tmp_path):
        m = PipelineMetrics()
        m.start_timer()
        m.record_pdf("text", candidates=3, skills=2)
        m.stop_timer()
        out = str(tmp_path / "metrics.json")
        m.write_json(out, top_skills=[("Python", 5)])
        assert os.path.exists(out)


class TestOutputCSVColumns:
    def test_basic_columns(self):
        ext = RegexSkillExtractor()
        result = ext.extract(SAMPLE_TEXT_1)
        skills = sorted(result.skills.items(), key=lambda x: x[0])
        df = pd.DataFrame(
            [(s, 1) for s, _ in skills],
            columns=["Skill", "Count"],
        )
        assert "Skill" in df.columns
        assert "Count" in df.columns
