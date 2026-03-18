# CLAUDE.md

## Project Overview
JobDistill v3.0 extracts and ranks in-demand technical skills from job posting PDFs using Gemini 2.5 Flash LLM. Skills are normalized via an alias mapping, categorized, and presented as a ranked CSV + interactive HTML dashboard.

## Running the Pipeline

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=<key>

# Quick test (3 PDFs)
python main.py --pdf_dirs Fall_2026_Co-op --dashboard dashboard.html

# Full run (all ~3261 PDFs)
python main.py --dashboard dashboard.html

# Regex fallback (no API key needed)
python main.py --extractor regex --pdf_dirs Fall_2026_Co-op
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests mock external APIs -- no real Gemini calls in tests.

## Architecture

- `main.py` -- CLI entry point, delegates to `jobdistill` package
- `jobdistill/cli.py` -- argument parsing (`--extractor`, `--gemini_model`, `--concurrency`, `--dashboard`)
- `jobdistill/pipeline.py` -- orchestration: PDF text extraction, concurrent Gemini calls, doc-frequency aggregation
- `jobdistill/extractors/` -- `gemini_extractor.py` (primary), `regex_extractor.py` (fallback), `base.py` (ABC)
- `jobdistill/skill_aliases.py` -- alias normalization + vague skill removal (maps variants like "GCP"/"Google Cloud" to canonical forms)
- `jobdistill/dashboard.py` -- standalone HTML dashboard (Chart.js bar + donut charts, sortable table)
- `jobdistill/pdf_text.py` -- PDF text extraction with SHA256 file caching
- `jobdistill/metrics.py` -- pipeline metrics and quality guardrails
- `tests/` -- pytest test suite

## Key Data Files

- `skill_alias_mapping.csv` -- canonical alias groups with categories (source of truth for dedup)
- `skill_categories.csv` -- every skill with its category assignment (source of truth for dashboard donut chart)
- `skill_analysis_results.csv` -- output: ranked skills by document frequency
- `dashboard.html` -- output: interactive HTML dashboard
- `metrics.json` -- output: pipeline metrics

## Conventions

- Python 3, pytest for testing
- All Gemini API calls must be mocked in tests
- Extractors implement `SkillExtractor` ABC in `jobdistill/extractors/base.py`
- New skill aliases go in `jobdistill/skill_aliases.py` (SKILL_ALIASES dict and VAGUE_SKILLS set)
- Output CSV: Skill and Count columns, sorted by count descending
- Gemini API costs ~$0.10-0.50 per full run -- avoid unnecessary re-runs; remap existing CSVs offline when possible

## Do Not Edit

The following directories contain raw PDF data and must never be modified:
- `Summer_2025_Co-op/`
- `Fall_2025_Co-op/`
- `Winter_2026_Co-op/`
- `Summer_2026_Co-op/`
- `Fall_2026_Co-op/`
