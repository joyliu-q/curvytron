"""Curvytron game replay viewer.

Serves a small FastAPI backend + single-page HTML frontend that lets you
step through recorded self-play games written by the rollout manager to
the shared `curvytron-data` Modal volume under `/data/game_traces/`.

Deploy:
    modal deploy slime/game_viewer.py

Then open the URL Modal prints.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

TRACE_DIR = Path("/data/game_traces")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.0")
)

data_volume = modal.Volume.from_name("curvytron-data", create_if_missing=True)

app = modal.App("curvytron-game-viewer")


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Curvytron Replay</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0; padding: 16px;
    background: #0d0f12; color: #e6e6e6;
  }
  h1 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0.02em; color: #9ecbff; }
  .row { display: flex; gap: 16px; align-items: flex-start; }
  .side { width: 320px; flex: 0 0 320px; }
  .main { flex: 1 1 auto; min-width: 0; }
  button, select, input {
    font: inherit; padding: 4px 10px; background: #1a1d22; color: #e6e6e6;
    border: 1px solid #2e333a; border-radius: 4px;
  }
  button:hover { background: #232830; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  select[size] { padding: 0; width: 100%; }
  pre.board {
    background: #000; color: #00ff99; padding: 10px;
    font: 11px/1.0 ui-monospace, SFMono-Regular, Consolas, monospace;
    overflow: auto; margin: 0; border: 1px solid #2e333a; border-radius: 4px;
    white-space: pre;
  }
  .controls { margin: 0 0 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .controls input[type=range] { flex: 1; accent-color: #7cb7ff; }
  .info { font-size: 13px; line-height: 1.55; background: #15181d; padding: 10px; border-radius: 4px; border: 1px solid #2e333a; margin-top: 10px; }
  .info code { background: #0d0f12; padding: 1px 5px; border-radius: 3px; }
  .badge { padding: 1px 7px; border-radius: 9px; background: #2a3342; color: #9ecbff; display: inline-block; font-size: 11px; letter-spacing: 0.04em; }
  .a { color: #ff7575; }
  .b { color: #7cb7ff; }
  .dim { color: #8a94a4; }
  .game-meta { font-size: 11px; color: #8a94a4; display: block; }
  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
</style>
</head>
<body>
<h1>Curvytron Replay Viewer</h1>
<div class="row">
  <div class="side">
    <div class="header">
      <strong>Games</strong>
      <button onclick="loadList()">&#x21bb; refresh</button>
    </div>
    <select id="games" size="28"></select>
    <div id="count" class="dim" style="font-size:11px; margin-top:6px;"></div>
  </div>
  <div class="main">
    <div class="controls">
      <button onclick="goStep(-1)">&laquo; prev</button>
      <button id="playbtn" onclick="togglePlay()">&#9654; play</button>
      <button onclick="goStep(1)">next &raquo;</button>
      <input type="range" id="slider" min="0" max="0" value="0" oninput="goto(this.value)">
      <span id="stepinfo" class="dim">0 / 0</span>
      <label class="dim">speed <input type="number" id="speed" min="30" max="2000" value="180" style="width:70px"> ms</label>
    </div>
    <pre id="board" class="board">Select a game on the left.</pre>
    <div id="info" class="info"></div>
  </div>
</div>
<script>
let trace = null;
let idx = 0;
let playing = false;
let timer = null;

async function loadList() {
  const r = await fetch('/api/games');
  const j = await r.json();
  const sel = document.getElementById('games');
  sel.innerHTML = '';
  for (const g of j.games) {
    const opt = document.createElement('option');
    opt.value = g.name;
    const dt = new Date(g.mtime * 1000).toISOString().replace('T',' ').slice(5,19);
    opt.textContent = `${dt}  ${g.outcome.padEnd(9)} ${g.total_steps}st  ${g.seed}`;
    sel.appendChild(opt);
  }
  document.getElementById('count').textContent = `${j.games.length} trace(s)`;
  sel.onchange = () => openGame(sel.value);
  if (j.games.length) { sel.value = j.games[0].name; openGame(j.games[0].name); }
}

async function openGame(name) {
  stopPlay();
  const r = await fetch('/api/games/' + encodeURIComponent(name));
  if (!r.ok) { alert('Failed to load: ' + r.status); return; }
  trace = await r.json();
  idx = 0;
  document.getElementById('slider').max = (trace.steps.length - 1);
  render();
}

function render() {
  if (!trace) return;
  const s = trace.steps[idx];
  document.getElementById('board').textContent = s.board || '(no board)';
  document.getElementById('slider').value = idx;
  document.getElementById('stepinfo').textContent = `${idx + 1} / ${trace.steps.length}`;
  const fmt = v => (typeof v === 'number') ? v.toFixed(4) : (v ?? '\u2014');
  const pstr = (s.players || []).map(p => {
    const cls = p.marker === trace.markers.a ? 'a' : (p.marker === trace.markers.b ? 'b' : '');
    const status = p.alive ? 'alive' : 'DEAD';
    return `<span class="${cls}">${p.marker} ${status} @ ${p.x?.toFixed?.(1)},${p.y?.toFixed?.(1)} \u2220${p.angle?.toFixed?.(2)}</span>`;
  }).join(' &nbsp;&middot;&nbsp; ');
  document.getElementById('info').innerHTML = `
    <div><span class="badge">${trace.outcome}</span>
      <span class="game-meta">seed=<code>${trace.seed}</code> &nbsp; recorded=<code>${trace.recorded_at}</code> &nbsp; total_steps=${trace.total_steps}</span></div>
    <div style="margin-top:6px"><strong>Step ${s.step}</strong> <span class="dim">(tick ${s.tick ?? '\u2014'})</span></div>
    <div><strong class="a">A (${trace.markers.a})</strong>: action=<code>${s.action_a ?? '\u2014'}</code> reward=${fmt(s.reward_a)}</div>
    <div><strong class="b">B (${trace.markers.b})</strong>: action=<code>${s.action_b ?? '\u2014'}</code> reward=${fmt(s.reward_b)}</div>
    <div style="margin-top:6px">${pstr}</div>
  `;
}

function goStep(d) {
  if (!trace) return;
  idx = Math.max(0, Math.min(trace.steps.length - 1, idx + d));
  render();
}

function goto(v) { if (!trace) return; idx = parseInt(v); render(); }

function togglePlay() {
  if (!trace) return;
  playing = !playing;
  document.getElementById('playbtn').innerHTML = playing ? '&#10073;&#10073; pause' : '&#9654; play';
  if (playing) {
    const tick = () => {
      if (idx >= trace.steps.length - 1) { stopPlay(); return; }
      idx++; render();
    };
    const ms = parseInt(document.getElementById('speed').value) || 180;
    timer = setInterval(tick, ms);
  } else { clearInterval(timer); timer = null; }
}

function stopPlay() { playing = false; clearInterval(timer); timer = null;
  document.getElementById('playbtn').innerHTML = '&#9654; play'; }

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') goStep(-1);
  else if (e.key === 'ArrowRight') goStep(1);
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
});

loadList();
</script>
</body>
</html>
"""


def _safe_name(name: str) -> str:
    """Allow only characters that appear in generated trace filenames."""
    return "".join(c for c in name if c.isalnum() or c in "-_.")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    min_containers=1,
    max_containers=1,
    timeout=60 * 60,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    api = FastAPI(title="Curvytron Game Viewer", docs_url=None, redoc_url=None)

    @api.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML_PAGE)

    @api.get("/api/games")
    def list_games():
        data_volume.reload()
        if not TRACE_DIR.exists():
            return {"games": []}

        items = []
        for p in TRACE_DIR.glob("*.json"):
            try:
                stat = p.stat()
                with p.open("r") as f:
                    head = json.load(f)
                items.append(
                    {
                        "name": p.stem,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "seed": head.get("seed"),
                        "outcome": head.get("outcome", "?"),
                        "total_steps": head.get("total_steps", 0),
                        "recorded_at": head.get("recorded_at"),
                    }
                )
            except Exception:
                continue

        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"games": items[:500]}

    @api.get("/api/games/{name}")
    def get_game(name: str):
        data_volume.reload()
        safe = _safe_name(name)
        path = TRACE_DIR / f"{safe}.json"
        if not path.exists():
            raise HTTPException(404, f"not found: {name}")
        with path.open("r") as f:
            return json.load(f)

    return api
