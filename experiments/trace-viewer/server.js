const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3847;

function parseTSV(filepath) {
  const raw = fs.readFileSync(filepath, "utf-8");
  const lines = raw.split("\n").filter((l) => l.startsWith("T\t"));
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = lines[0].split("\t").slice(1); // skip "T" prefix
  const rows = lines.slice(1).map((line) => {
    const cols = line.split("\t").slice(1);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = parseFloat(cols[i]);
    });
    return obj;
  });
  return { headers, rows };
}

function listTraces() {
  const dirs = [
    path.join(__dirname, ".."),
    path.join(__dirname, "..", "logs"),
  ];
  const files = [];
  for (const dir of dirs) {
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir)) {
      if (f.endsWith(".tsv")) {
        files.push({
          name: f,
          path: path.join(dir, f),
          mtime: fs.statSync(path.join(dir, f)).mtimeMs,
        });
      }
    }
  }
  return files.sort((a, b) => b.mtime - a.mtime).slice(0, 20);
}

app.get("/api/traces", (req, res) => {
  res.json(listTraces().map((f) => f.name));
});

app.get("/api/trace/:name", (req, res) => {
  const traces = listTraces();
  const match = traces.find((t) => t.name === req.params.name);
  if (!match) return res.status(404).json({ error: "not found" });
  res.json(parseTSV(match.path));
});

app.get("/", (req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Flash-MoE Trace Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { font-size: 18px; margin-bottom: 12px; }
  select { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 6px 10px; border-radius: 4px; font-size: 14px; margin-bottom: 16px; }
  .charts { display: flex; flex-direction: column; gap: 20px; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  canvas { width: 100% !important; height: 900px !important; }
  .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  label { font-size: 13px; cursor: pointer; }
  label input { margin-right: 4px; }
</style>
</head>
<body>

<h1>Flash-MoE Trace Viewer</h1>
<div class="controls">
  <select id="file-select"></select>
</div>

<div class="controls" id="phase-toggles"></div>

<div class="charts">
  <div class="chart-box"><canvas id="lines"></canvas></div>
</div>

<script>
const COLORS = {
  def_wait:  '#636efa',
  def_cpu:   '#ab63fa',
  inorm:     '#00cc96',
  cmd1_sub:  '#19d3f3',
  cmd1_wait: '#ef553b',
  cpu_attn:  '#ffa15a',
  cmd2_enc:  '#ff6692',
  cmd2_wait: '#b6e880',
  route:     '#fecb52',
  expert_io: '#ff97ff',
  cmd3_enc:  '#e8e8e8',
  total:     '#ffffff',
};

let lineChart;
let currentData = null;
let enabledPhases = new Set();

async function loadTraces() {
  const resp = await fetch('/api/traces');
  const names = await resp.json();
  const sel = document.getElementById('file-select');
  sel.innerHTML = names.map(n => '<option value="'+n+'">'+n+'</option>').join('');
  sel.onchange = () => loadTrace(sel.value);
  if (names.length) loadTrace(names[0]);
}

function buildToggles(phases) {
  const div = document.getElementById('phase-toggles');
  div.innerHTML = phases.map(p => {
    const c = COLORS[p] || '#888';
    const checked = enabledPhases.has(p) ? 'checked' : '';
    return '<label><input type="checkbox" '+checked+' data-phase="'+p+'">'
      + '<span style="color:'+c+'">'+p+'</span></label>';
  }).join('');
  div.querySelectorAll('input').forEach(cb => {
    cb.onchange = () => {
      if (cb.checked) enabledPhases.add(cb.dataset.phase);
      else enabledPhases.delete(cb.dataset.phase);
      render(currentData);
    };
  });
}

async function loadTrace(name) {
  const resp = await fetch('/api/trace/' + encodeURIComponent(name));
  currentData = await resp.json();
  const phases = currentData.headers.filter(h => h !== 'token');
  enabledPhases = new Set();
  buildToggles(phases);
  render(currentData);
}

function render(data) {
  if (!data || !data.rows.length) return;
  const tokens = data.rows.map(r => r.token);
  const phases = data.headers.filter(h => h !== 'token' && enabledPhases.has(h));

  const lineData = {
    labels: tokens,
    datasets: phases.map(p => ({
      label: p,
      data: data.rows.map(r => r[p]),
      borderColor: COLORS[p] || '#888',
      backgroundColor: 'transparent',
      borderWidth: p === 'total' ? 2 : 1.5,
      pointRadius: 2,
      tension: 0.2,
      borderDash: p === 'total' ? [4, 4] : [],
    })),
  };

  if (lineChart) lineChart.destroy();
  lineChart = new Chart(document.getElementById('lines'), {
    type: 'line',
    data: lineData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Phase trends over tokens (ms, layer avg)', color: '#c9d1d9' },
                 legend: { labels: { color: '#c9d1d9', font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: 'token', color: '#8b949e' }, ticks: { color: '#8b949e' } },
        y: { title: { display: true, text: 'ms', color: '#8b949e' }, ticks: { color: '#8b949e' } },
      },
    },
  });
}

loadTraces();
</script>
</body>
</html>`);
});

app.listen(PORT, () => {
  console.log(`Trace viewer: http://localhost:${PORT}`);
});
