"""Generate a standalone HTML dashboard for skill extraction results."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def generate_dashboard(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: str,
    num_pdfs: int,
    categories: Optional[Dict[str, str]] = None,
) -> None:
    """Write a self-contained HTML dashboard to *output_path*.

    Parameters
    ----------
    df : DataFrame with columns ``Skill`` and ``Count``.
    metrics : Dict from ``PipelineMetrics.to_dict()``.
    output_path : Where to write the HTML file.
    num_pdfs : Total PDFs processed.
    categories : Optional {skill_name: category} mapping for the donut chart.
    """
    categories = categories or {}

    top30 = df.head(30)
    labels_bar = top30["Skill"].tolist()
    values_bar = top30["Count"].tolist()

    # Category breakdown
    cat_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        cat = categories.get(row["Skill"], "Other")
        cat_counts[cat] = cat_counts.get(cat, 0) + int(row["Count"])
    cat_labels = list(cat_counts.keys())
    cat_values = list(cat_counts.values())

    unique_skills = len(df)
    total_mentions = int(df["Count"].sum()) if len(df) > 0 else 0
    elapsed = metrics.get("extraction_seconds_total", 0)

    # Build full table rows
    table_rows = ""
    for _, row in df.iterrows():
        cat = categories.get(row["Skill"], "Other")
        table_rows += (
            f"<tr><td>{row['Skill']}</td>"
            f"<td>{int(row['Count'])}</td>"
            f"<td>{cat}</td></tr>\n"
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JobDistill — Skill Analysis Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #222; padding: 2rem; }}
  h1 {{ margin-bottom: 1.5rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.25rem 1.5rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 160px; }}
  .card .value {{ font-size: 2rem; font-weight: 700; }}
  .card .label {{ font-size: .85rem; color: #666; margin-top: .25rem; }}
  .charts {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .chart-box {{ background: #fff; border-radius: 8px; padding: 1.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,.1); flex: 1; min-width: 320px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th, td {{ padding: .6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; cursor: pointer; user-select: none; }}
  th:hover {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>JobDistill — Skill Analysis Dashboard</h1>

<div class="cards">
  <div class="card"><div class="value">{num_pdfs}</div><div class="label">PDFs Processed</div></div>
  <div class="card"><div class="value">{unique_skills}</div><div class="label">Unique Skills</div></div>
  <div class="card"><div class="value">{total_mentions}</div><div class="label">Total Mentions</div></div>
  <div class="card"><div class="value">{elapsed:.1f}s</div><div class="label">Processing Time</div></div>
</div>

<div class="charts">
  <div class="chart-box" style="flex:2"><canvas id="barChart"></canvas></div>
  <div class="chart-box" style="flex:1"><canvas id="donutChart"></canvas></div>
</div>

<table id="skillTable">
<thead>
  <tr><th onclick="sortTable(0)">Skill</th><th onclick="sortTable(1)">Count</th><th onclick="sortTable(2)">Category</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<script>
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {labels_bar},
    datasets: [{{ label: 'Count', data: {values_bar},
      backgroundColor: 'rgba(59,130,246,.7)' }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'Top 30 Skills by Count' }} }},
    scales: {{ x: {{ beginAtZero: true }} }}
  }}
}});

new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: {cat_labels},
    datasets: [{{ data: {cat_values},
      backgroundColor: [
        '#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6',
        '#ec4899','#14b8a6','#f97316','#6366f1','#84cc16'
      ] }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Skills by Category' }} }}
  }}
}});

function sortTable(col) {{
  const table = document.getElementById('skillTable');
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = table.dataset.sortCol == col && table.dataset.sortDir === 'asc';
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent, vb = b.cells[col].textContent;
    if (col === 1) return asc ? vb - va : va - vb;
    return asc ? vb.localeCompare(va) : va.localeCompare(vb);
  }});
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? 'desc' : 'asc';
}}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
