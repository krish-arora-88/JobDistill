"""Unit tests for corpus-level boilerplate removal."""

import pytest

from jobdistill.boilerplate import (
    BoilerplateStats,
    canonicalize_line,
    compute_line_df,
    strip_boilerplate,
    strip_boilerplate_corpus,
)


BOILERPLATE_LINE = "Apply now at jobs.example.com - Equal Opportunity Employer"
BOILERPLATE_FOOTER = "Students must submit applications before the deadline"

DOC_1 = f"""Senior Software Engineer
Requirements: Python, Git, AWS, Docker
Experience with CI/CD and Kubernetes
{BOILERPLATE_LINE}
{BOILERPLATE_FOOTER}"""

DOC_2 = f"""Data Scientist Position
Skills: Python, SQL, TensorFlow, ML
Experience with cloud platforms
{BOILERPLATE_LINE}
{BOILERPLATE_FOOTER}"""

DOC_3 = f"""Frontend Developer
Skills: JavaScript, React, TypeScript, CSS
Experience with REST APIs
{BOILERPLATE_LINE}
{BOILERPLATE_FOOTER}"""

DOC_4 = f"""DevOps Engineer
Skills: Docker, Kubernetes, Terraform, AWS
Familiarity with Linux and Git
{BOILERPLATE_LINE}
{BOILERPLATE_FOOTER}"""

CORPUS = [DOC_1, DOC_2, DOC_3, DOC_4]


class TestCanonicalizeLine:
    def test_lowercase_and_strip(self):
        assert canonicalize_line("  Hello World  ") == "hello world"

    def test_digit_replacement(self):
        assert canonicalize_line("Job posting 12345") == "job posting <NUM>"
        assert canonicalize_line("Apply by 2025-01-15") == "apply by <NUM>-<NUM>-<NUM>"

    def test_whitespace_collapse(self):
        assert canonicalize_line("  foo   bar  baz  ") == "foo bar baz"

    def test_truncation(self):
        long_line = "x" * 300
        assert len(canonicalize_line(long_line)) == 200

    def test_dates_canonicalized(self):
        line1 = "Posted on 2025-06-15 by admin"
        line2 = "Posted on 2024-12-01 by admin"
        assert canonicalize_line(line1) == canonicalize_line(line2)


class TestComputeLineDF:
    def test_boilerplate_has_high_df(self):
        df = compute_line_df(CORPUS)
        key = canonicalize_line(BOILERPLATE_LINE)
        assert df[key] == 4

    def test_unique_lines_have_low_df(self):
        df = compute_line_df(CORPUS)
        key = canonicalize_line("Requirements: Python, Git, AWS, Docker")
        assert df.get(key, 0) == 1

    def test_empty_lines_ignored(self):
        df = compute_line_df(["hello\n\nworld", "hello\n\nfoo"])
        assert "" not in df

    def test_single_doc(self):
        df = compute_line_df(["line a\nline b\nline a"])
        assert df.get("line a", 0) == 1


class TestStripBoilerplate:
    def test_removes_high_df_lines(self):
        df_map = compute_line_df(CORPUS)
        cleaned, removed = strip_boilerplate(DOC_1, df_map, num_docs=4, df_threshold=0.05)
        assert BOILERPLATE_LINE not in cleaned
        assert BOILERPLATE_FOOTER not in cleaned
        assert removed >= 2

    def test_keeps_unique_content(self):
        df_map = compute_line_df(CORPUS)
        cleaned, _ = strip_boilerplate(DOC_1, df_map, num_docs=4, df_threshold=0.05)
        assert "Python" in cleaned
        assert "Senior Software Engineer" in cleaned

    def test_removes_ui_like_lines(self):
        text = "Python\n---\n!!!\nAWS"
        df_map = compute_line_df([text])
        cleaned, removed = strip_boilerplate(text, df_map, num_docs=1, df_threshold=0.5)
        assert "---" not in cleaned
        assert "!!!" not in cleaned


class TestStripBoilerplateCorpus:
    def test_returns_same_count(self):
        cleaned, stats = strip_boilerplate_corpus(CORPUS, df_threshold=0.05)
        assert len(cleaned) == len(CORPUS)

    def test_boilerplate_removed_from_all_docs(self):
        cleaned, stats = strip_boilerplate_corpus(CORPUS, df_threshold=0.05)
        for doc in cleaned:
            assert BOILERPLATE_LINE not in doc
            assert BOILERPLATE_FOOTER not in doc

    def test_stats_populated(self):
        _, stats = strip_boilerplate_corpus(CORPUS, df_threshold=0.05)
        assert stats.removed_lines > 0
        assert stats.total_lines > 0
        assert stats.removed_ratio > 0
        assert len(stats.top_removed) > 0

    def test_real_content_preserved(self):
        cleaned, _ = strip_boilerplate_corpus(CORPUS, df_threshold=0.05)
        assert "Python" in cleaned[0]
        assert "JavaScript" in cleaned[2]

    def test_empty_corpus(self):
        cleaned, stats = strip_boilerplate_corpus([], df_threshold=0.05)
        assert cleaned == []
        assert stats.removed_lines == 0

    def test_high_threshold_removes_nothing(self):
        cleaned, stats = strip_boilerplate_corpus(CORPUS, df_threshold=1.0)
        for orig, clean in zip(CORPUS, cleaned):
            assert "Python" in clean or "JavaScript" in clean or "Docker" in clean


class TestCanonicalizationWithVaryingNumbers:
    """Lines differing only by IDs / dates must be treated as identical."""

    TEMPLATE_DOCS = [
        "Job posting 12345\nApply by 2025-01-15\nRequires Python and AWS",
        "Job posting 67890\nApply by 2025-02-28\nRequires JavaScript and React",
        "Job posting 11111\nApply by 2025-03-10\nRequires Docker and Linux",
    ]

    def test_boilerplate_lines_have_high_df(self):
        df = compute_line_df(self.TEMPLATE_DOCS)
        key = canonicalize_line("Job posting 99999")
        assert df[key] == 3

    def test_date_lines_have_high_df(self):
        df = compute_line_df(self.TEMPLATE_DOCS)
        key = canonicalize_line("Apply by 2099-12-31")
        assert df[key] == 3

    def test_boilerplate_removed_after_canonicalization(self):
        cleaned, stats = strip_boilerplate_corpus(self.TEMPLATE_DOCS, df_threshold=0.05)
        assert stats.removed_lines > 0
        for doc in cleaned:
            assert "Job posting" not in doc
            assert "Apply by" not in doc

    def test_real_skills_preserved(self):
        cleaned, _ = strip_boilerplate_corpus(self.TEMPLATE_DOCS, df_threshold=0.05)
        all_text = " ".join(cleaned)
        assert "Python" in all_text or "JavaScript" in all_text or "Docker" in all_text
