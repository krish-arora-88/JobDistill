## JobDistill: ML-Powered Skill Extraction from Job Postings

JobDistill extracts and ranks the most in-demand technical skills from job posting PDFs. Drop your PDFs in a folder, run the script, and get a ranked CSV of skills.

**v2.2** fixes a critical regression where ML mode produced near-empty output ("cloud: 1") on 50+ PDFs. Root causes: boilerplate removal was inert (newlines collapsed before DF comparison) and the extractor had no fallback when KeyBERT returned few candidates. See [Changelog](#changelog) for details.

---

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, improves candidate quality)
python -m spacy download en_core_web_sm

# Run with ML extractor (default)
python main.py --pdf_dirs Summer_2025_Co-op Winter_2026_Co-op

# Run with legacy regex extractor (safe fallback)
python main.py --extractor regex --pdf_dirs Summer_2025_Co-op
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf_dirs` | `Summer_2025_Co-op Fall_2025_Co-op ...` | Directories containing PDF files |
| `--batch_size` | `20` | PDFs per batch (regex mode) |
| `--extractor` | `ml` | Extraction backend: `ml` or `regex` |
| `--model_dir` | `models/skill_classifier` | Trained classifier directory (ML mode) |
| `--min_confidence` | `0.75` | Minimum P(skill) threshold |
| `--top_k_phrases` | `30` | Max candidate keyphrases per doc |
| `--boilerplate_df_threshold` | `0.05` | Document-frequency threshold for boilerplate removal (5%) |
| `--disable_boilerplate_removal` | off | Skip boilerplate removal (for debugging) |
| `--include_confidence_cols` | off | Add AvgConfidence + ExampleMentions columns |
| `--cache_dir` | `.cache/jobdistill` | Cache for extracted text |
| `--metrics_out` | `metrics.json` | Pipeline metrics output |
| `--max_docs` | all | Limit PDFs processed (for debugging) |
| `--seed` | `42` | Random seed |
| `--output` | `skill_analysis_results.csv` | Output CSV path |
| `--debug_samples` | `0` | Log detailed extraction info for first N docs |
| `--debug_dump_path` | none | Write per-doc debug JSONL to this path |

### Debug Mode

When extraction results look wrong, use `--debug_samples N` to see what's happening inside the pipeline for the first N documents:

```bash
python main.py --extractor ml --max_docs 50 --debug_samples 3 \
    --boilerplate_df_threshold 0.05 --metrics_out metrics.json
```

This prints per-doc diagnostics showing:
- Character count and line count after boilerplate removal
- Number of candidates from each source (KeyBERT, TF-IDF, tech-token regex)
- Top 10 candidates with classifier probability and kept/filtered status
- Final skill count and whether threshold was auto-relaxed

Add `--debug_dump_path debug.jsonl` to save structured debug info as JSONL for offline analysis.

### Tuning

**Boilerplate DF threshold** (`--boilerplate_df_threshold`):
- Default `0.05` means lines appearing in ≥5% of PDFs are stripped.
- Raise to `0.10` if you're losing too many legitimate lines.
- Lower to `0.02` if boilerplate still leaks through.

**Confidence threshold** (`--min_confidence`):
- Default `0.75` keeps only phrases with high skill-likeness scores.
- Lower to `0.60` to discover more but noisier candidates.
- Raise to `0.85` for precision-focused results.
- The pipeline auto-relaxes the threshold (down to 0.3) for individual docs where the classifier is too aggressive, preventing empty output.

### Output

The CSV always contains `Skill, Count` columns (sorted by count descending, then skill name ascending). With `--include_confidence_cols`, two extra columns are added:

- **AvgConfidence** — average classifier confidence across documents
- **ExampleMentions** — 1–3 short context snippets from source PDFs

### How It Works

**ML mode** (default) uses a multi-stage pipeline:

1. **PDF text extraction** — pdfminer extracts text, with optional disk caching. Two versions are produced: newline-preserved (for boilerplate analysis) and flat (for ML extraction).
2. **Corpus boilerplate removal** — A 2-pass approach computes line document-frequency across all PDFs. Lines are **canonicalized** before comparison: lowercased, digit runs replaced with `<NUM>`, whitespace collapsed. This ensures lines that differ only by IDs, dates, or page numbers are treated as identical. Lines appearing in ≥5% of documents are stripped. Also removes short/UI-like lines. See [Why DF Canonicalization?](#why-df-canonicalization) below.
3. **Multi-source candidate extraction** — Three candidate sources are unioned:
   - **KeyBERT** keyphrases (primary, unsupervised embedding-based extraction)
   - **TF-IDF ngrams** (fallback when KeyBERT returns < 5 candidates)
   - **Tech-token regex** patterns (catches symbols like `C++`/`.NET`, acronyms like `AWS`/`SQL`, camelCase like `JavaScript`)
4. **Shape filtering** — Candidates must pass tech-indicator checks (symbols, uppercase acronyms, camelCase, capitalized proper nouns). Phrases with high stopword ratio or application-process vocabulary are rejected.
5. **Skill classification** — A trained logistic regression classifier (or anchor-phrase cosine similarity fallback) scores each candidate. Only phrases above `--min_confidence` are kept. Threshold auto-relaxes for docs where the classifier is too aggressive. Near-duplicate phrases are merged via embedding similarity.
6. **Document-frequency aggregation** — Each skill counts at most once per PDF. Output is sorted by count descending, then skill name ascending.

**Regex mode** (`--extractor regex`) uses the original hardcoded skills list with regex patterns and alias mappings — identical semantics to v1. Useful as a baseline for benchmarking.

### Why DF Canonicalization?

PDF job postings from platforms like WaterlooWorks contain repeated template text (navigation headers, footers, application instructions) that differs only by posting ID, date, or page number:

```
Job posting 12345 - Posted 2025-01-15
Job posting 67890 - Posted 2025-02-28
```

Without canonicalization, these appear as unique lines and escape document-frequency-based removal. By replacing digit runs with `<NUM>`:

```
job posting <NUM> - posted <NUM>-<NUM>-<NUM>
```

Both lines become identical, their DF reaches the corpus threshold, and they are stripped before skill extraction.

---

### Training / Improving the Classifier

The classifier bootstraps from weak labels (the legacy skills list is used ONLY as training seed data, never for ML inference):

```bash
# Step 1: Build weakly-labeled training data from your PDFs
python scripts/build_training_data.py \
    --pdf_dirs Summer_2025_Co-op Winter_2026_Co-op \
    --out data/training_data.jsonl \
    --max_docs 200

# Step 2: Train the classifier
python scripts/train_skill_classifier.py \
    --data data/training_data.jsonl \
    --model_dir models/skill_classifier \
    --eval_split 0.2

# Step 3: Run with the trained model
python main.py --extractor ml --model_dir models/skill_classifier
```

Without a trained model, the ML extractor falls back to an **anchor-phrase scorer** that computes cosine similarity between candidate phrases and generic tech anchor phrases (e.g., "programming language", "software framework", "cloud service"). This produces reasonable results without any training.

### Evaluating Extractors

Compare regex vs ML output side-by-side:

```bash
python scripts/eval_extractor.py \
    --pdf_dirs Summer_2025_Co-op \
    --max_docs 20 \
    --model_dir models/skill_classifier
```

### Metrics & Monitoring

Every run writes `metrics.json` (configurable with `--metrics_out`) containing:

- PDF processing stats (total, OK, empty, chars/PDF)
- Throughput (PDFs/second)
- Boilerplate removal stats (lines total, lines removed, ratio, top removed lines)
- Extraction quality indicators (candidates/PDF, skills/PDF)
- Classifier floor trigger count (how many docs required threshold relaxation)
- Diagnostics: % multiword skills, % lowercase-only, % containing application terms
- Top 20 skills by count
- **Quality guardrail**: if >30% of top-50 skills contain boilerplate markers, `quality_failed` is set to `true`

### Running Tests

```bash
# All tests (fast unit tests + boilerplate + filters + end-to-end)
pytest tests/ -v

# Just the fast tests (no ML model loading)
pytest tests/ -v --ignore=tests/test_ml_extractor_smoke.py

# Only ML smoke tests
pytest tests/test_ml_extractor_smoke.py -v
```

### Project Structure

```
JobDistill/
  main.py                       # CLI entrypoint
  requirements.txt
  jobdistill/
    __init__.py
    cli.py                      # Argument parsing
    pdf_text.py                 # PDF text extraction + caching
    normalize.py                # Text/phrase normalization + tech-shape filters
    boilerplate.py              # Corpus-level boilerplate removal with canonicalization
    pipeline.py                 # Orchestration: 2-pass extraction + aggregation
    metrics.py                  # Metrics collection + quality guardrails
    extractors/
      __init__.py
      base.py                   # SkillExtractor interface
      regex_extractor.py        # Legacy regex-based extraction
      keybert_extractor.py      # KeyBERT candidate extraction
      tfidf_extractor.py        # TF-IDF fallback candidate extraction
      classifier.py             # Skill classifier + anchor-phrase fallback
      ml_extractor.py           # ML pipeline: multi-source candidates + classification
  scripts/
    build_training_data.py      # Generate weak-label dataset
    train_skill_classifier.py   # Train and save classifier
    eval_extractor.py           # Compare extractors
  tests/
    test_normalize.py           # Normalization + filter unit tests
    test_keyphrase_filters.py   # Tech indicator + candidate filter tests
    test_boilerplate.py         # Boilerplate removal + canonicalization tests
    test_end_to_end.py          # Full pipeline smoke test with fixtures
    test_ml_extractor_smoke.py  # ML extractor integration tests
    test_pipeline_smoke.py      # Pipeline + metrics tests
```

---

### Changelog

#### v2.2 — Fix ML extraction regression: near-empty output on real PDFs

**Problem**: Running ML mode on 50 PDFs produced almost nothing ("cloud: 1").
Two root causes:

1. **Boilerplate removal inert** — `extract_pdf()` collapsed all whitespace
   (including newlines) to a single space. By the time text reached
   `strip_boilerplate_corpus()`, each document was one giant line. Line-level
   document-frequency comparison couldn't find repeated lines because there
   were none. Additionally, `_normalize_line()` didn't canonicalize digits, so
   template lines differing only by posting IDs or dates appeared unique.
2. **Extractor too fragile** — KeyBERT was the sole candidate source. If it
   returned few results, no fallback existed. The classifier threshold of 0.75
   filtered aggressively with no relaxation. Tech tokens like `C++`, `.NET`,
   and short acronyms like `AWS` were easily missed by KeyBERT alone.

**Fixes**:

- **Newline-preserved text extraction** (`pdf_text.py`): New
  `extract_pdf_with_lines()` normalizes within lines but keeps line breaks
  intact. Used by the ML pipeline for boilerplate analysis.
- **DF canonicalization** (`boilerplate.py`): New `canonicalize_line()` replaces
  digit runs with `<NUM>`, lowercases, collapses whitespace. Lines differing
  only by IDs/dates/page numbers are now treated as identical.
- **Multi-source candidate extraction** (`ml_extractor.py`):
  - **TF-IDF fallback** (`tfidf_extractor.py`): Activated when KeyBERT returns
    < 5 candidates. Uses sklearn TfidfVectorizer with ngram_range (1,3).
  - **Tech-token regex pass**: Pattern-based extraction catches `.NET`, `C++`,
    `C#`, uppercase acronyms, camelCase names, and capitalized tokens.
  - All three sources are unioned before classification.
- **Classifier auto-relaxation** (`ml_extractor.py`): If the classifier filters
  down to < 2 skills per doc, the threshold is lowered in 0.1 steps to a floor
  of 0.3. Absolute fallback keeps top N candidates by probability.
- **KeyBERT failure handling** (`keybert_extractor.py`): Import/load failures
  logged at WARNING level (not swallowed). Extractor marked unavailable,
  triggering TF-IDF fallback automatically.
- **Debug mode** (`--debug_samples`, `--debug_dump_path`): Per-doc diagnostics
  showing candidate counts, classifier scores, and filter decisions.
- **Expanded tech-shape filter** (`normalize.py`): `_APPLICATION_VERBS` expanded
  to reject more job-posting vocabulary ("position", "company",
  "requirements", etc.) from the capitalized-word heuristic.
- **Metrics** (`metrics.py`): Added `boilerplate_lines_total` and
  `classifier_floor_triggered_count`.
- **Tests**: Added canonicalization test with varying numbers, tech-token
  extraction test, ML pipeline smoke test with 5 mocked docs.

#### v2.1 — Fix ML extraction to produce real skills

**Problem**: The v2.0 ML pipeline output was dominated by platform boilerplate phrases
like "canada job posting", "students submit applications" instead of real technical
skills. Root causes:

1. **No boilerplate removal** — Job posting PDFs contain repeated platform text
   (navigation, footers, application instructions). KeyBERT treated these as
   high-relevance keyphrases since they appeared frequently.
2. **Weak candidate filtering** — The `is_valid_candidate` filter only checked basic
   properties (token count, stopword-only, junk phrase list). Generic English phrases
   like "submit applications soon" passed all checks.
3. **No classifier fallback** — Without a trained model at `models/skill_classifier`,
   the ML extractor returned ALL KeyBERT candidates unfiltered.
4. **Low default confidence** — The 0.60 threshold was too permissive.

**Fixes**:

- **Corpus boilerplate removal** (`boilerplate.py`): 2-pass approach computes line
  document-frequency across all PDFs, strips lines appearing in ≥5% of documents.
  Also removes short/UI-like/punctuation-only lines. Configurable via
  `--boilerplate_df_threshold`.
- **Tech-shape candidate filtering** (`normalize.py`): New `has_tech_indicator()`
  requires at least one tech marker (symbol, uppercase acronym, camelCase,
  capitalized proper noun, digit). Rejects all-lowercase multi-word phrases that
  look like regular English. Stopword ratio > 0.5 also rejected.
- **Anchor-phrase fallback scorer** (`classifier.py`): When no trained LR model
  exists, uses cosine similarity to generic tech anchor phrases instead of returning
  unfiltered candidates.
- **Raised default confidence** from 0.60 to 0.75.
- **Enhanced metrics** (`metrics.py`): Boilerplate removal stats, quality guardrail
  that flags if >30% of top-50 skills contain boilerplate markers.
- **148+ tests** covering boilerplate removal, candidate filtering, document-frequency
  counting, end-to-end pipeline smoke tests, and ML extractor integration.
