---
name: add-alias
description: Add skill alias mappings to skill_aliases.py, then remap output files (zero API calls)
disable-model-invocation: true
---

# Add Alias

Add new alias mappings or vague skill removals to `jobdistill/skill_aliases.py`, then re-apply to output files.

## Steps

1. **Parse user request** — the user will specify either:
   - Aliases to merge: e.g. "merge Python 3 into Python"
   - Vague terms to remove: e.g. "remove Programming Languages"

2. **Edit `jobdistill/skill_aliases.py`**:
   - For aliases: add entries to `SKILL_ALIASES` dict (variant → canonical)
   - For vague terms: add entries to `VAGUE_SKILLS` set

3. **Update source CSVs if needed**:
   - If the user provides category corrections, update `skill_categories.csv`
   - If the user provides new alias groups, update `skill_alias_mapping.csv`

4. **Run `/remap-skills`** to re-apply aliases and regenerate `skill_analysis_results.csv` + `dashboard.html`.

5. **Verify** — show the affected skills and their new counts.

## Important
- NEVER re-run the Gemini pipeline for alias changes
- Always reload the module after editing (`importlib.reload`)
- The `normalize_skill()` function returns empty string for vague skills — the pipeline and remap script both skip empty returns
