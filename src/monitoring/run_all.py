from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import shutil

from src.utils import ensure_dir
from src.monitoring import run_evidently, run_shap, run_slices


def _latest_report_path(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for d in dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def _stage_latest(reports_root: Path, subdir: str, filename: str, dest_name: str) -> str:
    latest = _latest_report_path(reports_root / subdir, filename)
    if latest is None:
        return ""
    latest_dir = ensure_dir(reports_root / "latest")
    dest = Path(latest_dir) / dest_name
    shutil.copy2(latest, dest)
    return f"latest/{dest_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monitoring pipeline.")
    parser.add_argument("--ref_hours", type=int, default=24)
    parser.add_argument("--cur_hours", type=int, default=1)
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--sample_size", type=int, default=500)
    args = parser.parse_args()

    run_evidently.run(ref_hours=args.ref_hours, cur_hours=args.cur_hours, as_of=args.as_of)
    run_shap.run(
        ref_hours=args.ref_hours,
        cur_hours=args.cur_hours,
        as_of=args.as_of,
        sample_size=args.sample_size,
    )
    run_slices.run(hours=args.ref_hours, as_of=args.as_of)

    reports_root = Path("monitoring") / "reports"
    index_dir = ensure_dir(reports_root)
    index_path = Path(index_dir) / "index.html"

    latest_input = _stage_latest(reports_root, "evidently", "input_drift.html", "input_drift.html")
    if not latest_input:
        latest_input = _stage_latest(reports_root, "evidently", "input_drift.json", "input_drift.json")
    latest_feature = _stage_latest(reports_root, "evidently", "feature_drift.html", "feature_drift.html")
    if not latest_feature:
        latest_feature = _stage_latest(reports_root, "evidently", "feature_drift.json", "feature_drift.json")
    latest_pred = _stage_latest(reports_root, "evidently", "prediction_drift.html", "prediction_drift.html")
    if not latest_pred:
        latest_pred = _stage_latest(reports_root, "evidently", "prediction_drift.json", "prediction_drift.json")
    latest_shap = _stage_latest(reports_root, "shap", "shap_summary.png", "shap_summary.png")
    latest_slices = _stage_latest(reports_root, "slices", "slice_metrics.csv", "slice_metrics.csv")

    latest_dir = Path(index_dir) / "latest"
    ensure_dir(latest_dir)
    if latest_input and latest_input.endswith(".html"):
        _stage_latest(reports_root, "evidently", "input_drift.html", "evidently_drift.html")

    def _as_panel(path: str) -> str:
        if not path:
            return "<div class='empty'>No report yet</div>"
        return f"<div class='report' data-url='/reports/{path}'></div>"

    html = """
    <html>
      <head>
        <meta charset="UTF-8" />
        <title>Monitoring Reports</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          .tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
          .tab {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }}
          .tab.active {{ background: #111827; color: #fff; }}
          .panel {{ display: none; border: 1px solid #e5e7eb; padding: 12px; border-radius: 6px; }}
          .panel.active {{ display: block; }}
          .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
          .card {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; background: #f9fafb; }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ padding: 6px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
          th {{ background: #f3f4f6; }}
          details {{ margin-top: 12px; }}
        </style>
      </head>
      <body>
        <h2>Monitoring Reports</h2>
        <div class="tabs">
          <div class="tab active" data-tab="input">Input Drift</div>
          <div class="tab" data-tab="feature">Feature Drift</div>
          <div class="tab" data-tab="prediction">Prediction Drift</div>
          <div class="tab" data-tab="shap">SHAP</div>
          <div class="tab" data-tab="slices">Slices</div>
        </div>
        <div class="panel active" id="input">{input_panel}</div>
        <div class="panel" id="feature">{feature_panel}</div>
        <div class="panel" id="prediction">{pred_panel}</div>
        <div class="panel" id="shap">{shap_panel}</div>
        <div class="panel" id="slices">{slices_panel}</div>
        <script>
          const tabs = document.querySelectorAll('.tab');
          const panels = document.querySelectorAll('.panel');
          tabs.forEach(tab => {{
            tab.addEventListener('click', () => {{
              tabs.forEach(t => t.classList.remove('active'));
              panels.forEach(p => p.classList.remove('active'));
              tab.classList.add('active');
              document.getElementById(tab.dataset.tab).classList.add('active');
            }});
          }});

          function renderPanel(panel) {{
            const target = panel.querySelector('.report');
            if (!target) return;
            const url = target.dataset.url;
            if (url.endsWith('.json')) {{
              fetch(url).then(r => r.json()).then(data => {{
                if (data.metrics && Array.isArray(data.metrics)) {{
                  const driftCount = data.metrics.find(m => (m.metric_name || '').startsWith('DriftedColumnsCount'));
                  const valueDrift = data.metrics.filter(m => (m.metric_name || '').startsWith('ValueDrift'));
                  const getColumn = (m) => {{
                    if (m && m.config && m.config.column) return m.config.column;
                    const name = (m && m.metric_name) ? m.metric_name : '';
                    const parts = name.split('column=');
                    if (parts.length > 1) {{
                      return parts[1].split(',')[0] || 'unknown';
                    }}
                    return 'unknown';
                  }};
                  const getScore = (m) => {{
                    if (!m) return null;
                    if (m.value && m.value.score !== undefined) return m.value.score;
                    if (m.value && m.value.drift_score !== undefined) return m.value.drift_score;
                    return m.value ?? null;
                  }};
                  const top = valueDrift
                    .map(m => ({{ column: getColumn(m), score: getScore(m) }}))
                    .filter(r => r.score !== null)
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 8);

                  let html = `<div class="grid">`;
                  if (driftCount) {{
                    html += `<div class="card"><div>Drifted columns</div><strong>${{driftCount.value?.count ?? '-'}}</strong></div>`;
                    html += `<div class="card"><div>Drift share</div><strong>${{driftCount.value?.share ?? '-'}}</strong></div>`;
                  }}
                  html += `</div>`;

                  if (top.length) {{
                    html += `<h4>Top drifting columns</h4>`;
                    html += `<table><thead><tr><th>Column</th><th>Score</th></tr></thead><tbody>`;
                    top.forEach(row => {{
                      html += `<tr><td>${{row.column}}</td><td>${{row.score.toFixed ? row.score.toFixed(4) : row.score}}</td></tr>`;
                    }});
                    html += `</tbody></table>`;
                  }}

                  html += `<details><summary>Raw JSON</summary><pre style="white-space:pre-wrap;">${{JSON.stringify(data, null, 2)}}</pre></details>`;
                  target.innerHTML = html;
                  return;
                }}
                target.innerHTML = `<pre style="white-space:pre-wrap;">${{JSON.stringify(data, null, 2)}}</pre>`;
              }}).catch(() => {{
                target.innerHTML = `<a href="${{url}}" target="_blank">Open report</a>`;
              }});
            }} else if (url.endsWith('.png')) {{
              target.innerHTML = `<img src="${{url}}" style="max-width:100%;height:auto;" />`;
            }} else if (url.endsWith('.csv')) {{
              fetch(url).then(r => r.text()).then(text => {{
                const lines = text.trim().split('\\n').filter(Boolean);
                if (lines.length === 0) {{
                  target.innerHTML = `<div class="empty">No data</div>`;
                  return;
                }}
                const headers = lines[0].split(',');
                const rows = lines.slice(1).map(line => line.split(','));
                const idx = (name) => headers.indexOf(name);
                const f1Idx = idx('f1');
                if (f1Idx >= 0) {{
                  rows.sort((a, b) => parseFloat(b[f1Idx] || 0) - parseFloat(a[f1Idx] || 0));
                }}
                let html = `<table><thead><tr>${{headers.map(h => `<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
                rows.forEach(row => {{
                  html += `<tr>${{row.map((v, i) => {{
                    const key = headers[i];
                    const val = parseFloat(v);
                    const low = (key === 'precision' || key === 'recall') && !Number.isNaN(val) && val < 0.2;
                    const style = low ? " style='color:#b91c1c;font-weight:600;'" : "";
                    return `<td${{style}}>${{v}}</td>`;
                  }}).join('')}}</tr>`;
                }});
                html += `</tbody></table>`;
                target.innerHTML = html;
              }}).catch(() => {{
                target.innerHTML = `<a href="${{url}}" target="_blank">Open report</a>`;
              }});
            }} else {{
              target.innerHTML = `<a href="${{url}}" target="_blank">Open report</a>`;
            }}
          }}

          panels.forEach(renderPanel);
        </script>
      </body>
    </html>
    """.format(
        input_panel=_as_panel(latest_input),
        feature_panel=_as_panel(latest_feature),
        pred_panel=_as_panel(latest_pred),
        shap_panel=_as_panel(latest_shap),
        slices_panel=_as_panel(latest_slices),
    )
    index_path.write_text(html.strip(), encoding="utf-8")

    latest_index = Path(index_dir) / "latest" / "index.html"
    ensure_dir(latest_index.parent)
    latest_index.write_text(html.strip(), encoding="utf-8")


if __name__ == "__main__":
    main()
