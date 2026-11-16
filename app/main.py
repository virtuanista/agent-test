import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, health as mh, runs as bg
from .config import PROVIDER_SEP, settings
from .llm import available_targets, split_target

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.mark_interrupted()  # runs left 'running' by a previous process are dead
    yield


app = FastAPI(title="agent-test", lifespan=lifespan)
app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")  # marked + DOMPurify: works offline


@app.middleware("http")
async def require_app_token(request: Request, call_next):
    """Desktop mode: the launcher sets APP_TOKEN and gives the same value to its own window as a cookie.

    Any other local process or web page that can reach 127.0.0.1 (CSRF, DNS rebinding) has no cookie and is refused.
    Plain `start.bat` (no APP_TOKEN) is unchanged."""
    token = os.getenv("APP_TOKEN")
    if token and not hmac.compare_digest(request.cookies.get("agent_token", ""), token):
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return await call_next(request)


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)
    language: str = Field(default="Spanish", max_length=40)
    max_loops: int = Field(default=6, ge=1, le=settings.max_loops)
    model: str | None = Field(default=None, max_length=200)  # None/"" = automatic (pool order)


def _sse(event: dict, idx: int) -> str:
    return f"id: {idx}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/run")
async def run(req: RunRequest) -> dict:
    """Start a run in the background and return its id; follow it with GET /api/runs/{id}/stream."""
    if req.model and req.model not in available_targets():
        raise HTTPException(422, "modelo no disponible: " + req.model)
    if req.model and (why := mh.is_blocked(req.model)):
        raise HTTPException(422, f"ese modelo no está disponible ahora ({why}); elige otro o usa Automático")
    try:
        started = bg.start(req.goal, req.language, req.max_loops, model=req.model or None)
    except bg.Busy as e:
        raise HTTPException(409, {"message": "Ya hay una ejecución en curso", "run_id": e.run_id})
    return {"run_id": started.id}


@app.get("/api/runs/active")
def active_run() -> dict | None:
    r = bg.active()
    return {"run_id": r.id, "goal": r.goal, "language": r.language, "created": r.created, "max_loops": r.max_loops, "model": r.model} if r else None


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str, from_: int = Query(0, alias="from", ge=0)) -> StreamingResponse:
    """SSE replay from event index `from`, then live. Closing the connection does NOT stop the run."""
    r = bg.get(run_id)
    if not r:
        raise HTTPException(404, "run not in memory (finished long ago or server restarted): read /api/runs/{id}")

    async def gen() -> AsyncIterator[str]:
        async for i, ev in r.follow(from_):
            yield ": keepalive\n\n" if ev is None else _sse(ev, i)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    return {"stopped": bg.stop(run_id)}


@app.get("/api/health")
def health() -> dict:
    """Preflight for the UI: tells the user what is missing before they start a run."""
    return {
        "llm": {"ok": bool(settings.llm_api_key), "model": settings.model, "strong_model": settings.strong_model},
        "backup": {"ok": bool(settings.openrouter_api_key), "models": sum(m.startswith("openrouter" + PROVIDER_SEP) for m in settings.model_pool)},
        "max_loops": settings.max_loops,
        "active_run": (bg.active().id if bg.active() else None),
    }


@app.get("/api/models")
async def models(check: bool = False, manual: bool = False) -> dict:
    """Models the user can pin, each with a verified status (ok / limited / error / unknown).

    `check=1` first verifies the ones without a fresh verdict with a real tiny request. Probing is skipped while
    a run is active so it does not compete with the run for the free-tier quota."""
    targets = available_targets()
    paused = bool(check and bg.active())
    if check and not paused:
        await mh.probe_stale(targets, manual=manual)
    labels = {"default": "Principal", "openrouter": "OpenRouter"}
    items = [
        {"id": t, "provider": (p := split_target(t)[0]), "label": f"{labels.get(p, p)} · {split_target(t)[1]}",
         **mh.view(t)}
        for t in targets
    ]
    return {"models": items, "paused": paused}


@app.get("/api/runs")
def runs() -> list[dict]:
    return db.list_runs()


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    found = db.get_run(run_id)
    if not found:
        raise HTTPException(404, "run not found")
    return found


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
