# JobDistill v3.0 — Gemini LLM Skill Extractor

## Context
Both existing approaches are unsatisfactory: the regex extractor (main branch) can only find hardcoded skills, and the ML extractor (feat branch) produces noisy results without a trained classifier. Replacing the ML pipeline with Gemini 2.5 Flash LLM calls gives context-aware extraction with zero training, proper normalization, and skill categorization — all for ~$0.10-0.50 per run.

## Approach
On `feat/ml-skill-extraction` branch: **replace** the ML extractor with a Gemini LLM extractor, **keep** regex as offline fallback, **add** HTML dashboard output.

## SDK & Model
- Package: `google-genai` (new SDK, not deprecated `google-generativeai`)
- Model: `gemini-2.5-flash` (default), `gemini-2.5-flash-lite` (budget option)
- API key: `GEMINI_API_KEY` env var
- Structured output: Pydantic schema → `response_mime_type="application/json"`

---

## Files to Create

### 1. `jobdistill/extractors/gemini_extractor.py`
- `GeminiSkillExtractor(SkillExtractor)` — core LLM extractor
- Pydantic models: `ExtractedSkill(name, category)`, `SkillExtractionResponse(skills[])`
- System prompt instructs: extract tech skills only, normalize names, categorize as Language/Framework/Tool/Platform/Protocol/Database/Cloud/Other
- Truncate text to 6000 chars (job requirements are near the top)
- Retry with exponential backoff on 429/5xx errors
- Lazy client init, confidence always 1.0 per skill
- Categories stored in `debug_info` for dashboard

### 2. `jobdistill/dashboard.py`
- `generate_dashboard(df, metrics, output_path, num_pdfs)` → standalone HTML file
- Chart.js CDN for charts (no local dependencies)
- Horizontal bar chart: top 30 skills by count
- Donut chart: category breakdown (Languages vs Frameworks vs Tools etc.)
- Summary cards: total PDFs, unique skills, total mentions, processing time
- Full sortable table of all skills (inline JS)
- Clean inline CSS, no external stylesheets

### 3. `tests/test_gemini_extractor.py`
- All tests mock the Gemini API (no real calls)
- Tests: valid extraction, empty text, text truncation, API error handling, missing API key, duplicate dedup, categories in debug_info, prompt content checks, Pydantic schema parsing

### 4. `tests/test_dashboard.py`
- Tests: HTML file generated, contains skill names, contains Chart.js, summary stats present, handles empty DataFrame

---

## Files to Modify

### 5. `jobdistill/cli.py`
- `--extractor` choices: `gemini` (default), `regex`
- Add: `--gemini_model` (default `gemini-2.5-flash`), `--concurrency` (default 10), `--dashboard` (HTML output path)
- Remove ML flags: `--model_dir`, `--min_confidence`, `--top_k_phrases`, `--boilerplate_df_threshold`, `--include_confidence_cols`, `--disable_boilerplate_removal`

### 6. `jobdistill/pipeline.py`
- Replace `MLSkillExtractor` import with `GeminiSkillExtractor`
- `build_extractor()`: handle `"gemini"` → `GeminiSkillExtractor(model=gemini_model)`
- `run_pipeline()`: route to `_run_gemini_pipeline()` for Gemini extractor
- New `_run_gemini_pipeline()`: Pass 1 extract flat text (no boilerplate removal needed — LLM handles context), Pass 2 concurrent Gemini calls via `ThreadPoolExecutor`, aggregate by doc frequency. Collect categories for dashboard.
- Remove `_run_ml_pipeline()`, `_log_debug_doc()`
- Remove boilerplate imports (keep `extract_pdf` only, not `extract_pdf_with_lines`)

### 7. `main.py`
- Remove `numpy` import and `np.random.seed`
- Update `build_extractor()` call for new signature
- Pass `concurrency` to `run_pipeline()`
- Add dashboard generation when `--dashboard` is set
- Update output path logic

### 8. `requirements.txt`
- Remove: `numpy<2`, `keybert>=0.8.0`, `sentence-transformers>=2.2.0`, `scikit-learn>=1.3`, `spacy>=3.6`
- Add: `google-genai>=1.0.0`, `pydantic>=2.0`
- Keep: `pdfminer.six`, `pandas`, `tqdm`, `pytest`

### 9. `jobdistill/extractors/__init__.py`
- Export `GeminiSkillExtractor` instead of ML classes

### 10. `jobdistill/__init__.py`
- Version bump to `3.0.0`

### 11. `jobdistill/metrics.py`
- Remove `classifier_floor_triggered_count`
- Add optional `gemini_request_count`, `gemini_error_count`

### 12. `tests/test_end_to_end.py`
- Remove `from jobdistill.extractors.ml_extractor import extract_tech_tokens`
- Remove `TestTechTokenExtraction` class
- Keep all other test classes (boilerplate, candidate filter, doc-frequency, tech indicator, smoke)

### 13. `tests/test_pipeline_smoke.py`
- Replace `test_ml` with `test_gemini` in `TestBuildExtractor`
- Replace `TestMLPipelineSmoke` with mocked Gemini pipeline test

### 14. `README.md`
- Rewrite for v3.0: Gemini-powered, new CLI flags, dashboard usage, API key setup

---

## Files to Delete
- `jobdistill/extractors/ml_extractor.py`
- `jobdistill/extractors/keybert_extractor.py`
- `jobdistill/extractors/tfidf_extractor.py`
- `jobdistill/extractors/classifier.py`
- `scripts/build_training_data.py`
- `scripts/train_skill_classifier.py`
- `scripts/eval_extractor.py`
- `tests/test_ml_extractor_smoke.py`

---

## Implementation Order
1. Create `gemini_extractor.py` (no deps on other changes)
2. Create `dashboard.py`
3. Create test files
4. Update `requirements.txt`
5. Update `extractors/__init__.py`
6. Update `cli.py`
7. Update `pipeline.py`
8. Update `main.py`
9. Update `__init__.py`, `metrics.py`
10. Update existing tests
11. Delete ML files and scripts
12. Update `README.md`
13. Run `pytest tests/ -v`

## Verification
1. `export GEMINI_API_KEY=<key> && python main.py --pdf_dirs Fall_2026_Co-op` — should extract skills via Gemini (only 3 PDFs)
2. `python main.py --pdf_dirs Fall_2026_Co-op --dashboard dashboard.html` — should produce HTML
3. `python main.py --extractor regex --pdf_dirs Fall_2026_Co-op` — regex fallback still works
4. `pytest tests/ -v` — all tests pass with mocked API
5. Open `dashboard.html` in browser — charts and table render correctly
