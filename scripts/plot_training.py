"""
Training Visualization Dashboard
==================================
Reads logs/training_metrics.csv and generates a self-contained HTML file
with interactive Chart.js charts — zero extra pip installs required.

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --input logs/training_metrics.csv
    python scripts/plot_training.py --output logs/training_report.html

Charts generated:
    1. Loss Curve       — Train loss vs Val loss per epoch
    2. F1 Scores        — Sentiment F1 / Sarcasm F1 / Combined F1 per epoch
    3. Precision/Recall — Side-by-side bars for both tasks per epoch
    4. Metrics Table    — Full per-epoch table in HTML
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# CSV Reader
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(csv_path: Path) -> dict:
    """
    Read training_metrics.csv and return a dict keyed by epoch.
    Format: timestamp, epoch, step, metric_name, metric_value
    """
    if not csv_path.exists():
        print(f"ERROR: metrics file not found: {csv_path}")
        sys.exit(1)

    raw = defaultdict(dict)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
                raw[ep][row["metric_name"]] = float(row["metric_value"])
                raw[ep]["step"]      = int(row["step"])
                raw[ep]["timestamp"] = row["timestamp"]
            except (ValueError, KeyError):
                continue

    return dict(sorted(raw.items()))


def extract_series(metrics: dict):
    """Extract per-epoch series as lists for charting."""
    epochs   = sorted(metrics.keys())
    def get(ep, key, default=0.0):
        return metrics[ep].get(key, default)

    return {
        "epochs":    epochs,
        "train_loss":           [get(e, "train_loss")           for e in epochs],
        "val_loss":             [get(e, "val_loss")             for e in epochs],
        "sentiment_f1":         [get(e, "sentiment_f1")         for e in epochs],
        "sarcasm_f1":           [get(e, "sarcasm_f1")           for e in epochs],
        "combined_f1":          [get(e, "combined_f1")          for e in epochs],
        "sentiment_precision":  [get(e, "sentiment_precision")  for e in epochs],
        "sentiment_recall":     [get(e, "sentiment_recall")     for e in epochs],
        "sarcasm_precision":    [get(e, "sarcasm_precision")    for e in epochs],
        "sarcasm_recall":       [get(e, "sarcasm_recall")       for e in epochs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML Generator
# ─────────────────────────────────────────────────────────────────────────────

def build_html(series: dict, metrics: dict, csv_path: Path) -> str:
    epochs = series["epochs"]
    best_epoch = max(metrics, key=lambda e: metrics[e].get("combined_f1", 0.0))
    best_f1    = metrics[best_epoch].get("combined_f1", 0.0)
    num_steps  = metrics[max(metrics)].get("step", "?")

    # Metric table rows
    table_rows = ""
    for ep in epochs:
        m = metrics[ep]
        best_mark = " ★" if ep == best_epoch else ""
        table_rows += f"""
        <tr{'class="best-row"' if ep == best_epoch else ''}>
          <td>{ep}{best_mark}</td>
          <td>{m.get('train_loss', 0):.4f}</td>
          <td>{m.get('val_loss', 0):.4f}</td>
          <td>{m.get('sentiment_f1', 0):.4f}</td>
          <td>{m.get('sentiment_precision', 0):.4f}</td>
          <td>{m.get('sentiment_recall', 0):.4f}</td>
          <td>{m.get('sarcasm_f1', 0):.4f}</td>
          <td>{m.get('sarcasm_precision', 0):.4f}</td>
          <td>{m.get('sarcasm_recall', 0):.4f}</td>
          <td><strong>{m.get('combined_f1', 0):.4f}</strong></td>
        </tr>"""

    data_js = json.dumps(series, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Training Report — Multilingual Sentiment Model</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1117; --card: #1a1d2e; --accent: #6c63ff;
      --accent2: #00d4aa; --accent3: #ff6b6b; --accent4: #ffd93d;
      --text: #e2e8f0; --muted: #8892a4; --border: #2d3047;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh; padding: 24px;
    }}
    .hero {{
      background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px 40px;
      margin-bottom: 28px;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    }}
    .hero h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
    .hero p  {{ color: var(--muted); font-size: 14px; }}
    .hero .badge {{
      display: inline-flex; align-items: center; gap: 6px;
      background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.4);
      color: var(--accent); border-radius: 20px;
      padding: 4px 12px; font-size: 12px; font-weight: 600;
      margin-top: 12px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px; margin-bottom: 28px;
    }}
    .stat-card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 20px; text-align: center;
    }}
    .stat-card .value {{
      font-size: 32px; font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-card .label {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 24px; margin-bottom: 28px;
    }}
    .chart-card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 16px; padding: 24px;
    }}
    .chart-card h2 {{
      font-size: 15px; font-weight: 600; margin-bottom: 20px;
      color: var(--text); display: flex; align-items: center; gap: 8px;
    }}
    .chart-card h2::before {{
      content: ''; display: inline-block; width: 10px; height: 10px;
      border-radius: 50%; background: var(--accent);
    }}
    .chart-wrapper {{ position: relative; height: 260px; }}
    .table-card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 16px; padding: 24px; overflow-x: auto;
    }}
    .table-card h2 {{
      font-size: 15px; font-weight: 600; margin-bottom: 20px; color: var(--text);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{
      text-align: left; padding: 10px 14px;
      color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase;
      border-bottom: 1px solid var(--border);
    }}
    td {{ padding: 10px 14px; border-bottom: 1px solid rgba(45,48,71,0.5); }}
    tr.best-row td {{ background: rgba(108,99,255,0.08); color: var(--accent2); }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .footer {{ color: var(--muted); font-size: 11px; margin-top: 20px; text-align: center; }}
  </style>
</head>
<body>

<div class="hero">
  <h1>📊 Training Report</h1>
  <p>Multilingual Sentiment Analysis · Source: <code>{csv_path.name}</code></p>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <span class="badge">✦ Best Combined F1: {best_f1:.4f} at Epoch {best_epoch}</span>
</div>

<div class="stats-grid">
  <div class="stat-card">
    <div class="value">{len(epochs)}</div>
    <div class="label">Epochs Trained</div>
  </div>
  <div class="stat-card">
    <div class="value">{best_f1:.4f}</div>
    <div class="label">Best Combined F1</div>
  </div>
  <div class="stat-card">
    <div class="value">{max(s for s in series['sentiment_f1']):.4f}</div>
    <div class="label">Best Sentiment F1</div>
  </div>
  <div class="stat-card">
    <div class="value">{max(s for s in series['sarcasm_f1']):.4f}</div>
    <div class="label">Best Sarcasm F1</div>
  </div>
  <div class="stat-card">
    <div class="value">{min(series['val_loss']):.4f}</div>
    <div class="label">Min Val Loss</div>
  </div>
  <div class="stat-card">
    <div class="value">{num_steps}</div>
    <div class="label">Steps/Epoch</div>
  </div>
</div>

<div class="charts-grid">
  <div class="chart-card">
    <h2>Loss Curves</h2>
    <div class="chart-wrapper"><canvas id="lossChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h2>F1 Score Progress</h2>
    <div class="chart-wrapper"><canvas id="f1Chart"></canvas></div>
  </div>
  <div class="chart-card">
    <h2>Sentiment · Precision &amp; Recall</h2>
    <div class="chart-wrapper"><canvas id="sentPrChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h2>Sarcasm · Precision &amp; Recall</h2>
    <div class="chart-wrapper"><canvas id="sarcPrChart"></canvas></div>
  </div>
</div>

<div class="table-card">
  <h2>📋 Per-Epoch Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Epoch</th><th>Train Loss</th><th>Val Loss</th>
        <th>Sent F1</th><th>Sent Prec</th><th>Sent Rec</th>
        <th>Sarc F1</th><th>Sarc Prec</th><th>Sarc Rec</th>
        <th>Combined F1</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<p class="footer">Multilingual Sentiment Analysis · Training Report · {csv_path.resolve()}</p>

<script>
const data = {data_js};
const epochs = data.epochs.map(e => 'Epoch ' + e);

const COLORS = {{
  purple:  '#6c63ff', teal:   '#00d4aa',
  red:     '#ff6b6b', yellow: '#ffd93d',
  blue:    '#4a9eff', orange: '#ff8c42',
}};

const defaultFont = {{ family: 'Segoe UI, system-ui, sans-serif', size: 12 }};
Chart.defaults.font = defaultFont;
Chart.defaults.color = '#8892a4';

const gridOpts = {{
  color: 'rgba(255,255,255,0.05)',
}};

function makeChart(id, type, labels, datasets, yMin=null, yMax=null) {{
  const ctx = document.getElementById(id).getContext('2d');
  const scales = {{
    x: {{ grid: gridOpts }},
    y: {{ grid: gridOpts, beginAtZero: false }},
  }};
  if (yMin !== null) scales.y.min = yMin;
  if (yMax !== null) scales.y.max = yMax;
  return new Chart(ctx, {{
    type,
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ boxWidth: 12, padding: 15 }} }},
        tooltip: {{ backgroundColor: '#1a1d2e', borderColor: '#2d3047', borderWidth: 1 }},
      }},
      scales,
    }},
  }});
}}

// 1. Loss chart
makeChart('lossChart', 'line', epochs, [
  {{ label: 'Train Loss', data: data.train_loss, borderColor: COLORS.red,
     backgroundColor: 'rgba(255,107,107,0.1)', tension: 0.4, fill: true }},
  {{ label: 'Val Loss',   data: data.val_loss,   borderColor: COLORS.teal,
     backgroundColor: 'rgba(0,212,170,0.1)',   tension: 0.4, fill: true }},
]);

// 2. F1 chart
makeChart('f1Chart', 'line', epochs, [
  {{ label: 'Sentiment F1', data: data.sentiment_f1, borderColor: COLORS.purple,
     backgroundColor: 'rgba(108,99,255,0.1)', tension: 0.4, fill: true }},
  {{ label: 'Sarcasm F1',  data: data.sarcasm_f1,  borderColor: COLORS.yellow,
     backgroundColor: 'rgba(255,217,61,0.1)', tension: 0.4, fill: true }},
  {{ label: 'Combined F1', data: data.combined_f1, borderColor: COLORS.teal,
     backgroundColor: 'rgba(0,212,170,0.15)', tension: 0.4, fill: true, borderWidth: 2.5 }},
], 0, 1);

// 3. Sentiment Precision/Recall
makeChart('sentPrChart', 'bar', epochs, [
  {{ label: 'Precision', data: data.sentiment_precision, backgroundColor: 'rgba(108,99,255,0.75)' }},
  {{ label: 'Recall',    data: data.sentiment_recall,    backgroundColor: 'rgba(0,212,170,0.75)'  }},
], 0, 1);

// 4. Sarcasm Precision/Recall
makeChart('sarcPrChart', 'bar', epochs, [
  {{ label: 'Precision', data: data.sarcasm_precision, backgroundColor: 'rgba(255,107,107,0.75)' }},
  {{ label: 'Recall',    data: data.sarcasm_recall,    backgroundColor: 'rgba(255,217,61,0.75)'  }},
], 0, 1);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate training visualization HTML")
    p.add_argument("--input",  default=str(ROOT / "logs" / "training_metrics.csv"))
    p.add_argument("--output", default=str(ROOT / "logs" / "training_report.html"))
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input)
    out_path = Path(args.output)

    print(f"Reading: {csv_path}")
    metrics = load_metrics(csv_path)

    if not metrics:
        print("ERROR: No valid epoch data found in CSV.")
        sys.exit(1)

    series = extract_series(metrics)

    print(f"Epochs found: {sorted(metrics.keys())}")
    best = max(metrics, key=lambda e: metrics[e].get("combined_f1", 0))
    print(f"Best epoch: {best} | Combined F1: {metrics[best].get('combined_f1', 0):.4f}")

    html = build_html(series, metrics, csv_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Training dashboard saved → {out_path}")
    print(f"  Open in browser: file:///{out_path.resolve()}")


if __name__ == "__main__":
    main()
