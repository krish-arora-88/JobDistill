## JobDistill: ML-Powered Skill Extraction from Job Postings

JobDistill extracts and ranks the most in-demand technical skills from job posting PDFs. Drop your PDFs in a folder, run the script, and get a ranked CSV of skills.

**v2.0** replaces the hardcoded skills list with an NLP/ML pipeline that *discovers* skill phrases automatically using KeyBERT + a trainable classifier. The legacy regex mode is still available as a fallback.

---

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, improves candidate quality)
python -m spacy download en_core_web_sm

# Run with ML extractor (default)
python main.py --pdf_dirs Summer_2025_Co-op Winter_2026_Co-op

# Run with legacy regex extractor
python main.py --extractor regex --pdf_dirs Summer_2025_Co-op
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf_dirs` | `Summer_2025_Co-op Fall_2025_Co-op ...` | Directories containing PDF files |
| `--batch_size` | `20` | PDFs per batch (regex mode) |
| `--extractor` | `ml` | Extraction backend: `ml` or `regex` |
| `--model_dir` | `models/skill_classifier` | Trained classifier directory (ML mode) |
| `--min_confidence` | `0.60` | Minimum P(skill) threshold |
| `--top_k_phrases` | `30` | Max candidate keyphrases per doc |
| `--include_confidence_cols` | off | Add AvgConfidence + ExampleMentions columns |
| `--cache_dir` | `.cache/jobdistill` | Cache for extracted text |
| `--metrics_out` | `metrics.json` | Pipeline metrics output |
| `--max_docs` | all | Limit PDFs processed (for debugging) |
| `--seed` | `42` | Random seed |
| `--output` | `skill_analysis_results.csv` | Output CSV path |

### Output

The CSV always contains `Skill, Count` columns (sorted by count descending, then skill name ascending). With `--include_confidence_cols`, two extra columns are added:

- **AvgConfidence** — average classifier confidence across documents
- **ExampleMentions** — 1–3 short context snippets from source PDFs

### How It Works

**ML mode** (default) uses a two-stage pipeline:

1. **Candidate extraction** — KeyBERT with sentence-transformers extracts keyphrases from each PDF. Documents are chunked to handle long text. Candidates are filtered by token count, stopword/junk checks, and optionally spaCy noun chunks.
2. **Skill classification** — A trained logistic regression classifier (over sentence-transformer embeddings) scores each candidate phrase as skill vs. not-skill. Only phrases above `--min_confidence` are kept. Near-duplicate phrases are merged via embedding cosine similarity.

**Regex mode** uses the original hardcoded skills list with regex patterns and alias mappings — identical semantics to v1.

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

To improve quality over time:
1. Manually review and correct labels in `data/training_data.jsonl`
2. Add more PDFs and regenerate training data
3. Retrain and compare using the eval script

### Evaluating Extractors

Compare regex vs ML output side-by-side:

```bash
python scripts/eval_extractor.py \
    --pdf_dirs Summer_2025_Co-op \
    --max_docs 20 \
    --model_dir models/skill_classifier
```

This produces `eval_report.csv` with columns: Skill, RegexCount, MLCount, Delta, Source.

### Metrics & Monitoring

Every run writes `metrics.json` (configurable with `--metrics_out`) containing:

- PDF processing stats (total, OK, empty, chars/PDF)
- Throughput (PDFs/second)
- Extraction quality indicators (candidates/PDF, skills/PDF)
- Diagnostics: % single-token skills, % lowercase-only, top rejected phrases
- Top 20 skills by count

### Running Tests

```bash
pytest tests/ -v
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
    normalize.py                # Text/phrase normalization
    pipeline.py                 # Orchestration, aggregation, output
    metrics.py                  # Metrics collection + JSON output
    extractors/
      __init__.py
      base.py                   # SkillExtractor interface
      regex_extractor.py        # Legacy regex-based extraction
      keybert_extractor.py      # KeyBERT candidate extraction
      classifier.py             # Skill-likeness classifier
      ml_extractor.py           # ML pipeline orchestrator
  scripts/
    build_training_data.py      # Generate weak-label dataset
    train_skill_classifier.py   # Train and save classifier
    eval_extractor.py           # Compare extractors
  tests/
    test_normalize.py
    test_keyphrase_filters.py
    test_ml_extractor_smoke.py
    test_pipeline_smoke.py
```
