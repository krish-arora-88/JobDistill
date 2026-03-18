"""Tests for the HTML dashboard generator."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from jobdistill.dashboard import generate_dashboard


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Skill": "Python", "Count": 50},
            {"Skill": "Docker", "Count": 35},
            {"Skill": "AWS", "Count": 30},
            {"Skill": "SQL", "Count": 25},
            {"Skill": "Git", "Count": 20},
        ]
    )


def _sample_metrics() -> dict:
    return {
        "num_pdfs_total": 100,
        "extraction_seconds_total": 12.5,
    }


class TestDashboardGeneration:
    def test_creates_html_file(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100)
        assert os.path.exists(out)
        content = open(out).read()
        assert content.startswith("<!DOCTYPE html>")

    def test_contains_skill_names(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100)
        content = open(out).read()
        assert "Python" in content
        assert "Docker" in content
        assert "AWS" in content

    def test_contains_chartjs(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100)
        content = open(out).read()
        assert "chart.js" in content.lower() or "Chart" in content

    def test_summary_stats(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100)
        content = open(out).read()
        assert "100" in content  # num_pdfs
        assert "5" in content    # unique skills
        assert "12.5" in content  # processing time

    def test_category_breakdown(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        cats = {"Python": "Language", "Docker": "Tool", "AWS": "Cloud", "SQL": "Database", "Git": "Tool"}
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100, categories=cats)
        content = open(out).read()
        assert "Language" in content
        assert "Cloud" in content
        assert "Database" in content

    def test_empty_dataframe(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        empty_df = pd.DataFrame(columns=["Skill", "Count"])
        generate_dashboard(empty_df, _sample_metrics(), out, num_pdfs=0)
        assert os.path.exists(out)
        content = open(out).read()
        assert "JobDistill" in content

    def test_sortable_table(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        generate_dashboard(_sample_df(), _sample_metrics(), out, num_pdfs=100)
        content = open(out).read()
        assert "sortTable" in content
        assert "<table" in content
