"""The run dashboard: a FastAPI app over the run store.

    GET /                                      the page
    GET /api/experiments                       every experiment with its jobs' progress
    GET /api/experiments/{name}                the experiment's cells with their rates
    GET /api/experiments/{name}/cells/{cell}   one cell, question by question
    GET /api/services                          the services of config/services.yaml
    GET /health

Experiments are summarised from ``load_runs`` and cached for
``cache_seconds``; the list of experiments reads only the job files.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from lakegen.config.settings import get_settings
from lakegen.experiments.analysis.jobs import read_progress
from lakegen.experiments.analysis.runs import load_runs
from lakegen.experiments.analysis.summary import cell_questions, cell_summaries
from lakegen.experiments.services import ServiceManager, load_services
from lakegen.experiments.store import CellRecord, RunStore

COOKIE = "lakegen_dashboard"


class Cache:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self._clock = clock
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, compute: Callable[[], Any]) -> Any:
        now = self._clock()
        entry = self._entries.get(key)
        if entry is not None and entry[0] > now:
            return entry[1]
        value = compute()
        self._entries[key] = (now + self.seconds, value)
        return value

    def clear(self) -> None:
        self._entries.clear()


def _progress_of(record: CellRecord) -> dict[str, Any]:
    progress = read_progress(Path(record.job_dir) / f"{record.job_id}.json") if record.job_dir and record.job_id else None
    return {
        "cell_id": record.cell_id,
        "experiment_id": record.experiment_id,
        "axes": record.axes,
        "pass": record.pass_name,
        "job_dir": record.job_dir,
        "job_id": record.job_id,
        "submitted_at": record.submitted_at,
        "status": progress.status if progress else ("missing" if record.job_id else "not submitted"),
        "question_count": progress.question_count if progress else 0,
        "processed": progress.processed if progress else 0,
        "failed": progress.failed if progress else 0,
        "error": progress.error if progress else "",
    }


def experiments_overview(store: RunStore) -> list[dict[str, Any]]:
    """Every experiment of the store with its cells' progress, from the job files alone."""
    overview = []
    if not store.root.is_dir():
        return overview
    for directory in sorted(path for path in store.root.iterdir() if path.is_dir()):
        cells = [_progress_of(record) for record in store.load_index(directory.name).values()]
        overview.append({
            "name": directory.name,
            "legacy": directory.name.startswith("legacy-"),
            "cells": len(cells),
            "by_status": {status: sum(cell["status"] == status for cell in cells) for status in sorted({c["status"] for c in cells})},
            "question_count": sum(cell["question_count"] for cell in cells),
            "processed": sum(cell["processed"] for cell in cells),
            "failed": sum(cell["failed"] for cell in cells),
        })
    return overview


def experiment_detail(store: RunStore, name: str) -> dict[str, Any]:
    index = store.load_index(name)
    if not index:
        raise HTTPException(status_code=404, detail=f"no experiment {name!r} in the run store")
    frame = load_runs(name, store=store)
    by_cell = {summary["cell_id"]: summary for summary in cell_summaries(frame)}
    cells = []
    for record in index.values():
        cell = _progress_of(record)
        cell["summary"] = by_cell.get(record.cell_id)
        cells.append(cell)
    return {"name": name, "cells": cells, "runs": int(len(frame))}


def cell_detail(store: RunStore, name: str, cell_id: str) -> dict[str, Any]:
    record = store.load_index(name).get(cell_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no cell {cell_id!r} in {name!r}")
    frame = load_runs(name, store=store)
    frame = frame[frame.cell_id == cell_id]
    summaries = cell_summaries(frame)
    return {
        "experiment": name,
        "cell": _progress_of(record),
        "summary": summaries[0] if summaries else None,
        "questions": cell_questions(frame),
    }


def create_app(
    *,
    store: RunStore | None = None,
    services: ServiceManager | None = None,
    token: str | None = None,
    cache_seconds: float = 30.0,
) -> FastAPI:
    store = store or RunStore()
    services = services or ServiceManager()
    required = get_settings().dashboard_token if token is None else token
    cache = Cache(cache_seconds)
    app = FastAPI(title="LakeGen runs")

    def guard(request: Request) -> None:
        if required and request.query_params.get("token") != required and request.cookies.get(COOKIE) != required:
            raise HTTPException(status_code=401, detail="token required")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/experiments", dependencies=[Depends(guard)])
    def experiments() -> list[dict[str, Any]]:
        return cache.get("experiments", lambda: experiments_overview(store))

    @app.get("/api/experiments/{name}", dependencies=[Depends(guard)])
    def experiment(name: str) -> dict[str, Any]:
        return cache.get(f"experiment:{name}", lambda: experiment_detail(store, name))

    @app.get("/api/experiments/{name}/cells/{cell_id}", dependencies=[Depends(guard)])
    def cell(name: str, cell_id: str) -> dict[str, Any]:
        return cache.get(f"cell:{name}:{cell_id}", lambda: cell_detail(store, name, cell_id))

    @app.get("/api/services", dependencies=[Depends(guard)])
    def service_statuses() -> list[dict[str, Any]]:
        return [
            {"name": service.name, "up": services.status(service).up, "health": service.health,
             "description": service.description}
            for service in load_services().values()
        ]

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(guard)])
    def index(request: Request) -> HTMLResponse:
        response = HTMLResponse(PAGE)
        if required and request.query_params.get("token") == required:
            response.set_cookie(COOKIE, required, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
        return response

    app.state.cache = cache
    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LakeGen runs</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;margin:0;color:#222;background:#fafafa}
header{padding:10px 16px;background:#263238;color:#fff;display:flex;gap:16px;align-items:baseline}
header h1{font-size:16px;margin:0}header small{color:#b0bec5}
main{display:grid;grid-template-columns:260px 1fr;gap:0;min-height:calc(100vh - 42px)}
nav{border-right:1px solid #ddd;background:#fff;overflow:auto}
nav a{display:block;padding:8px 12px;text-decoration:none;color:#222;border-bottom:1px solid #eee}
nav a.on{background:#e3f2fd}nav a small{color:#777;display:block}
section{padding:12px 16px;overflow:auto}
table{border-collapse:collapse;width:100%;background:#fff;margin-bottom:16px}
th,td{padding:4px 8px;border-bottom:1px solid #e5e5e5;text-align:right;white-space:nowrap}
th{background:#eceff1;cursor:pointer;position:sticky;top:0}th.l,td.l{text-align:left}
td.q{white-space:normal;max-width:520px}tr.row:hover{background:#f1f8e9;cursor:pointer}
.ok{color:#2e7d32}.bad{color:#c62828}.muted{color:#888}
#services span{display:inline-block;margin-right:12px}
</style></head><body>
<header><h1>LakeGen runs</h1><small id="where"></small><small id="services"></small></header>
<main><nav id="nav"></nav><section id="view"><p class="muted">Loading…</p></section></main>
<script>
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=v=>v==null?'–':(100*v).toFixed(0)+'%';const num=v=>v==null?'–':Math.round(v).toLocaleString();
async function get(u){const r=await fetch(u);if(!r.ok)throw new Error(r.status+' '+u);return r.json()}
let current=null,sortKey=null,sortDir=1;
function sortable(rows,key,dir){return rows.slice().sort((a,b)=>{const x=a[key],y=b[key];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*dir})}
async function loadNav(){const xs=await get('/api/experiments');const nav=document.getElementById('nav');
 nav.innerHTML=xs.map(x=>`<a href="#${esc(x.name)}" class="${x.name===current?'on':''}">${esc(x.name)}<small>${x.cells} cells · ${x.processed}/${x.question_count} questions · ${Object.entries(x.by_status).map(([k,v])=>v+' '+k).join(', ')}</small></a>`).join('');
 if(!current&&xs.length)location.hash=xs[0].name}
async function loadServices(){try{const s=await get('/api/services');document.getElementById('services').innerHTML=s.map(x=>`<span class="${x.up?'ok':'bad'}" title="${esc(x.description)}">${x.up?'●':'○'} ${esc(x.name)}</span>`).join('')}catch(e){}}
function cellRow(c){const s=c.summary||{};return `<tr class="row" data-cell="${esc(c.cell_id)}"><td class="l">${esc(c.experiment_id)}</td><td class="l">${esc(c.status)}</td><td>${c.processed}/${c.question_count}</td><td>${c.failed}</td><td>${pct(s.solved)}</td><td>${pct(s.exact)}</td><td>${pct(s.hit1)}</td><td>${pct(s.recall10)}</td><td>${pct(s.in_context)}</td><td>${num(s.elapsed_mean)}</td><td>${num(s.tokens_mean)}</td><td class="l">${esc((s.effective_models||[]).join(', '))}</td></tr>`}
const COLS=[['experiment_id','cell','l'],['status','status','l'],['processed','done',''],['failed','failed',''],['solved','solved',''],['exact','exact sel.',''],['hit1','H@1',''],['recall10','R@10',''],['in_context','ctx@20',''],['elapsed_mean','s',''],['tokens_mean','tokens',''],['effective_models','ran on','l']];
async function loadExperiment(name){current=name;sortKey=null;const view=document.getElementById('view');view.innerHTML='<p class="muted">Loading '+esc(name)+'…</p>';
 const x=await get('/api/experiments/'+encodeURIComponent(name));x.cells.forEach(c=>Object.assign(c,{solved:(c.summary||{}).solved,exact:(c.summary||{}).exact,hit1:(c.summary||{}).hit1,recall10:(c.summary||{}).recall10,in_context:(c.summary||{}).in_context,elapsed_mean:(c.summary||{}).elapsed_mean,tokens_mean:(c.summary||{}).tokens_mean}));
 const render=()=>{const rows=sortKey?sortable(x.cells,sortKey,sortDir):x.cells;view.innerHTML=`<h2>${esc(name)} <small class="muted">${x.runs} runs</small></h2><table id="cells"><tr>${COLS.map(([k,l,c])=>`<th class="${c}" data-key="${k}">${l}</th>`).join('')}</tr>${rows.map(cellRow).join('')}</table><div id="cell"></div>`;
  view.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.key;sortDir=sortKey===k?-sortDir:1;sortKey=k;render()});
  view.querySelectorAll('tr.row').forEach(tr=>tr.onclick=()=>loadCell(name,tr.dataset.cell))};
 render();document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+name))}
async function loadCell(name,cellId){const d=await get('/api/experiments/'+encodeURIComponent(name)+'/cells/'+encodeURIComponent(cellId));const s=d.summary||{};
 document.getElementById('cell').innerHTML=`<h3>${esc(d.cell.experiment_id)} <small class="muted">${esc(d.cell.job_id||'')} · ${esc(d.cell.job_dir||'')}</small></h3>
 <table><tr><th class="l">#</th><th class="l">question</th><th class="l">status</th><th>solved</th><th>exact</th><th>ctx</th><th class="l">gold (rank)</th><th class="l">selected</th><th>s</th><th>tokens</th><th class="l">ran on</th><th class="l">error</th></tr>
 ${d.questions.map(q=>`<tr><td>${q.index}</td><td class="l q" title="${esc(q.question_id)}">${esc(q.question)}</td><td class="l">${esc(q.status)}</td><td class="${q.solved?'ok':''}">${q.solved?'yes':''}</td><td>${q.exact?'yes':''}</td><td>${q.in_context?'yes':''}</td><td class="l">${q.gold.map(g=>esc(g)+' ('+(q.gold_ranks[g]==null?'–':q.gold_ranks[g])+')').join(', ')}</td><td class="l">${esc(q.tables.join(', '))}</td><td>${num(q.elapsed_seconds)}</td><td>${num(q.tokens)}</td><td class="l">${esc(q.effective_model)}</td><td class="l q">${esc((q.error||'').slice(0,160))}</td></tr>`).join('')}</table>`}
window.onhashchange=()=>{const n=decodeURIComponent(location.hash.slice(1));if(n)loadExperiment(n)};
document.getElementById('where').textContent=location.host;
loadNav().then(()=>{const n=decodeURIComponent(location.hash.slice(1));if(n)loadExperiment(n)});loadServices();
setInterval(loadNav,60000);setInterval(loadServices,60000);
</script></body></html>
"""
