---
name: check-migration
description: Verify v3 Gemini migration is complete — no stale ML imports, deleted files gone, tests pass
disable-model-invocation: true
---

# Check Migration

Verify the v3 Gemini migration is complete and clean.

## Checks to Run

### 1. Deleted files are gone
Verify these files no longer exist:
- `jobdistill/extractors/ml_extractor.py`
- `jobdistill/extractors/keybert_extractor.py`
- `jobdistill/extractors/tfidf_extractor.py`
- `jobdistill/extractors/classifier.py`
- `scripts/build_training_data.py`
- `scripts/train_skill_classifier.py`
- `scripts/eval_extractor.py`
- `tests/test_ml_extractor_smoke.py`

### 2. No stale ML imports
Search the entire `jobdistill/` and `tests/` directories for these patterns — none should exist:
- `MLSkillExtractor`
- `KeyBERTExtractor`
- `TFIDFCandidateExtractor`
- `SkillClassifier`
- `AnchorPhraseScorer`
- `from jobdistill.extractors.ml_extractor`
- `from jobdistill.extractors.keybert_extractor`
- `from jobdistill.extractors.tfidf_extractor`
- `from jobdistill.extractors.classifier`
- `import keybert`
- `import sentence_transformers`
- `from sklearn`
- `import spacy`

### 3. No stale CLI flags
Check `jobdistill/cli.py` does NOT contain:
- `--model_dir`
- `--min_confidence`
- `--top_k_phrases`
- `--boilerplate_df_threshold`
- `--include_confidence_cols`
- `--disable_boilerplate_removal`

### 4. New exports are correct
Verify `jobdistill/extractors/__init__.py` exports `GeminiSkillExtractor` (not ML classes).

### 5. Version bump
Verify `jobdistill/__init__.py` has version `3.0.0`.

### 6. Requirements are clean
Verify `requirements.txt`:
- Contains: `google-genai`, `pydantic`
- Does NOT contain: `keybert`, `sentence-transformers`, `scikit-learn`, `spacy`, `numpy`

### 7. Tests pass
```bash
python -m pytest tests/ -v
```

## Output
Report a checklist with pass/fail for each check. If any fail, list the specific files and lines that need fixing.
