## JobDistill: LLM-Powered Skill Extraction from Job Postings

JobDistill extracts and ranks the most in-demand technical skills from job posting PDFs using Gemini LLM. Drop your PDFs in a folder, run the script, and get a ranked CSV of skills with an optional interactive HTML dashboard.

**v3.0** replaces the ML/NLP pipeline with Gemini 2.5 Flash for context-aware extraction with zero training, proper normalization, and automatic skill categorization.

---

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY=your-api-key-here

# Run with Gemini extractor (default)
python main.py --pdf_dirs Summer_2026_Co-op

# Run on all directories
python main.py

# Generate HTML dashboard
python main.py --pdf_dirs Summer_2026_Co-op --dashboard dashboard.html

# Use regex extractor (offline fallback, no API key needed)
python main.py --extractor regex --pdf_dirs Summer_2026_Co-op
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf_dirs` | All co-op dirs | Directories containing PDF files |
| `--extractor` | `gemini` | `gemini` or `regex` |
| `--gemini_model` | `gemini-2.5-flash` | Gemini model (`gemini-2.5-flash-lite` for budget) |
| `--concurrency` | `10` | Max concurrent Gemini API calls |
| `--dashboard` | None | Path for HTML dashboard output |
| `--output` | `skill_analysis_results.csv` | Output CSV path |
| `--metrics_out` | `metrics.json` | Metrics JSON path |
| `--cache_dir` | `.cache/jobdistill` | Cache directory for extracted text |
| `--max_docs` | None | Limit number of PDFs (for testing) |
| `--batch_size` | `20` | Batch size (regex mode only) |

### Output

- **`skill_analysis_results.csv`** — Ranked skills with document frequency counts
- **`metrics.json`** — Pipeline metrics, quality guardrail results, processing stats
- **`dashboard.html`** (optional) — Interactive dashboard with charts and sortable table

### How It Works

1. **PDF Text Extraction**: Extract text from all PDFs using pdfminer (with SHA256 file caching)
2. **Gemini Skill Extraction**: Send each document's text (truncated to 6000 chars) to Gemini with a structured prompt requesting JSON output
3. **Concurrent Processing**: Process multiple PDFs simultaneously via ThreadPoolExecutor
4. **Aggregation**: Count skills by document frequency (each skill counts once per PDF)
5. **Categorization**: Gemini categorizes each skill (Language, Framework, Tool, Platform, Database, Cloud, etc.)

### Architecture

```
main.py                          # CLI entrypoint
jobdistill/
  cli.py                         # Argument parsing
  pipeline.py                    # Orchestration (PDF collection, extraction, aggregation)
  pdf_text.py                    # PDF text extraction with caching
  dashboard.py                   # HTML dashboard generator
  metrics.py                     # Pipeline metrics and quality guardrails
  normalize.py                   # Text normalization utilities
  boilerplate.py                 # Corpus-level boilerplate removal
  extractors/
    base.py                      # SkillExtractor ABC + ExtractionResult
    gemini_extractor.py          # Gemini LLM extractor (primary)
    regex_extractor.py           # Regex extractor (offline fallback)
tests/                           # pytest test suite (all API calls mocked)
```

### Cost

Gemini 2.5 Flash is very cost-effective for this use case:
- ~$0.10-0.50 per run depending on corpus size
- `gemini-2.5-flash-lite` available as a budget option

### Testing

```bash
# Run all tests (no API key needed — all Gemini calls are mocked)
python -m pytest tests/ -v
```

### Requirements

- Python 3.10+
- `GEMINI_API_KEY` environment variable (for Gemini extractor)
- See `requirements.txt` for Python dependencies
