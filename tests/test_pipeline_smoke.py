"""Smoke tests for the processing pipeline using synthetic data."""

import os
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jobdistill.boilerplate import strip_boilerplate_corpus
from jobdistill.extractors.base import ExtractionResult
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

    def test_gemini(self):
        ext = build_extractor("gemini", gemini_model="gemini-2.5-flash")
        assert ext.name == "gemini"

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
        assert "boilerplate" in d
        assert "quality_guardrail" in d
        assert d["num_pdfs_total"] == 1
        assert d["quality_guardrail"]["quality_failed"] is False
        assert "lines_total" in d["boilerplate"]
        assert "gemini_request_count" in d
        assert "gemini_error_count" in d

    def test_quality_guardrail_fails_on_boilerplate(self):
        m = PipelineMetrics()
        m.start_timer()
        m.record_pdf("text", candidates=3, skills=2)
        m.stop_timer()
        bad_skills = [
            ("canada application", 50),
            ("submit posting", 45),
            ("students employer", 40),
            ("application deadline", 35),
            ("apply employer", 30),
            ("accommodation hire", 25),
            ("applicant resume", 20),
            ("submit deadline", 15),
            ("posting employer", 10),
            ("canada posting", 9),
            ("employer accommodation", 8),
            ("submit applicant", 7),
            ("application students", 6),
            ("deadline application", 5),
            ("employer submit", 4),
            ("hire applicant", 3),
            ("Python", 2),
            ("Git", 1),
        ]
        d = m.to_dict(top_skills=bad_skills)
        assert d["quality_guardrail"]["quality_failed"] is True

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


class TestGeminiPipelineSmoke:
    """Simulate 5 docs through the Gemini pipeline with mocked API."""

    def test_gemini_pipeline_produces_skills(self, tmp_path):
        fake_results = [
            ExtractionResult(
                skills={"Python": 1.0, "JavaScript": 1.0, "React": 1.0, "Docker": 1.0, "AWS": 1.0, "SQL": 1.0, "Git": 1.0},
                candidates_considered=7,
                debug_info={"categories": {"Python": "Language", "JavaScript": "Language", "React": "Framework", "Docker": "Tool", "AWS": "Cloud", "SQL": "Database", "Git": "Tool"}},
            ),
            ExtractionResult(
                skills={"Java": 1.0, "C++": 1.0, "Linux": 1.0, "Kubernetes": 1.0, "CI/CD": 1.0},
                candidates_considered=5,
                debug_info={"categories": {"Java": "Language", "C++": "Language", "Linux": "Platform", "Kubernetes": "Tool", "CI/CD": "Methodology"}},
            ),
            ExtractionResult(
                skills={"TypeScript": 1.0, "Angular": 1.0, "Node.js": 1.0, "Redis": 1.0, "PostgreSQL": 1.0},
                candidates_considered=5,
                debug_info={"categories": {"TypeScript": "Language", "Angular": "Framework", "Node.js": "Framework", "Redis": "Database", "PostgreSQL": "Database"}},
            ),
            ExtractionResult(
                skills={"Python": 1.0, "Docker": 1.0, "AWS": 1.0, "Jenkins": 1.0, "Terraform": 1.0},
                candidates_considered=5,
                debug_info={"categories": {"Python": "Language", "Docker": "Tool", "AWS": "Cloud", "Jenkins": "Tool", "Terraform": "Tool"}},
            ),
            ExtractionResult(
                skills={"JavaScript": 1.0, "React": 1.0, "SQL": 1.0, "MongoDB": 1.0},
                candidates_considered=4,
                debug_info={"categories": {"JavaScript": "Language", "React": "Framework", "SQL": "Database", "MongoDB": "Database"}},
            ),
        ]

        docs = [
            "Python JavaScript React Docker AWS SQL Git\n" * 3,
            "Java C++ Linux Kubernetes CI/CD\n" * 3,
            "TypeScript Angular Node.js Redis PostgreSQL\n" * 3,
            "Python Docker AWS Jenkins Terraform\n" * 3,
            "JavaScript React SQL MongoDB\n" * 3,
        ]

        d = tmp_path / "pdfs"
        d.mkdir()
        pdf_files = []
        for i in range(5):
            p = d / f"doc{i}.pdf"
            p.write_text("fake pdf")
            pdf_files.append(str(p))

        mock_extractor = MagicMock()
        mock_extractor.name = "gemini"
        mock_extractor.extract = MagicMock(side_effect=fake_results)
        mock_extractor.gemini_request_count = 0
        mock_extractor.gemini_error_count = 0

        with patch(
            "jobdistill.pipeline.extract_pdf",
            side_effect=lambda path, cache_dir=None: docs[pdf_files.index(path)],
        ):
            df, metrics, categories = run_pipeline(
                pdf_files=pdf_files,
                extractor=mock_extractor,
            )

        assert len(df) > 5, f"Expected >5 skills, got {len(df)}"

        skill_names = set(df["Skill"].tolist())
        real = {"Python", "AWS", "Docker", "Git", "JavaScript", "SQL", "React"}
        found = real & skill_names
        assert len(found) >= 4, f"Expected >=4 real skills, got: {found}"

        multi_count = df[df["Count"] > 1]
        assert len(multi_count) >= 3, "Expected >=3 skills with count > 1"

        assert df.iloc[0]["Count"] >= df.iloc[-1]["Count"], "Should be sorted desc"

        # Verify categories were collected
        assert len(categories) > 0
        assert "Python" in categories
