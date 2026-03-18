---
name: run-pipeline
description: Run the JobDistill skill extraction pipeline on PDF directories with Gemini or regex extractor
disable-model-invocation: true
---

# Run Pipeline

Run the JobDistill skill extraction pipeline.

## Usage

The user may specify:
- **PDF directory** (default: `Fall_2026_Co-op` for quick tests, all dirs for full runs)
- **Extractor**: `gemini` (default) or `regex`
- **Dashboard**: optionally generate HTML dashboard

## Steps

1. Verify `GEMINI_API_KEY` is set (if using gemini extractor):
   ```bash
   echo "GEMINI_API_KEY is ${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-NOT SET}"
   ```

2. Run the pipeline with the user's arguments:
   ```bash
   python main.py --pdf_dirs <dirs> [--extractor gemini|regex] [--dashboard dashboard.html] [--gemini_model gemini-2.5-flash]
   ```

3. After completion, check results:
   - Read `skill_analysis_results.csv` and show the top 20 skills
   - Read `metrics.json` and report any quality guardrail failures
   - If `--dashboard` was used, confirm the HTML file was generated

4. If the pipeline fails:
   - Check for API key issues (401/403 errors)
   - Check for rate limiting (429 errors)
   - Check for PDF read errors
   - Report the specific error and suggest fixes

## IMPORTANT: Avoid Unnecessary Gemini Calls

A full run costs ~$0.10-0.50 CAD. If the user only needs to re-apply alias mappings or fix categories:
```python
# Remap existing CSV offline (zero API calls)
python3 -c "
import csv, importlib, sys
sys.path.insert(0, '.')
from jobdistill.skill_aliases import normalize_skill
# ... remap and rewrite CSV
"
```

Only re-run Gemini when new PDFs are added or the extraction prompt changes.

## Defaults
- Quick test: `--pdf_dirs Fall_2026_Co-op` (only 3 PDFs)
- Full run: `--pdf_dirs Summer_2025_Co-op Fall_2025_Co-op Winter_2026_Co-op Summer_2026_Co-op Fall_2026_Co-op`
