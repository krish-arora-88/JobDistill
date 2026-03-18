---
name: remap-skills
description: Re-apply skill aliases and categories to existing CSV, regenerate dashboard (zero API calls)
disable-model-invocation: true
---

# Remap Skills

Re-apply `jobdistill/skill_aliases.py` normalization to the existing `skill_analysis_results.csv` and regenerate `dashboard.html`. Zero Gemini API calls.

## Steps

1. Run the remap script:
```python
import csv, json, importlib, sys
sys.path.insert(0, '.')
import jobdistill.skill_aliases
importlib.reload(jobdistill.skill_aliases)
from jobdistill.skill_aliases import normalize_skill

rows = []
with open('skill_analysis_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((row['Skill'], int(row['Count'])))

merged = {}
for skill, count in rows:
    canonical = normalize_skill(skill)
    if not canonical:
        continue
    merged[canonical] = merged.get(canonical, 0) + count

sorted_skills = sorted(merged.items(), key=lambda x: (-x[1], x[0]))

with open('skill_analysis_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Skill', 'Count'])
    for skill, count in sorted_skills:
        writer.writerow([skill, count])

print(f'{len(rows)} -> {len(sorted_skills)} skills ({len(rows) - len(sorted_skills)} removed/merged)')
```

2. Regenerate dashboard with categories from `skill_categories.csv`:
```python
import pandas as pd
from jobdistill.dashboard import generate_dashboard

categories = {}
with open('skill_categories.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        canonical = normalize_skill(row['Skill'].strip())
        if canonical:
            categories[canonical] = row['Category'].strip()

df = pd.read_csv('skill_analysis_results.csv')
with open('metrics.json') as f:
    metrics = json.load(f)

generate_dashboard(df, metrics, 'dashboard.html', num_pdfs=3261, categories=categories)
```

3. Report: show top 20 skills and confirm dashboard generated.
