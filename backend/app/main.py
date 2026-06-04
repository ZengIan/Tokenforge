"""Tokenforge backend — FastAPI app entrypoint.

Forge tokens from silicon. 🔥
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import estimate, gpus, models

app = FastAPI(
    title="Tokenforge API",
    description="异构 GPU 推理吞吐与显存估算器",
    version="1.0.0",
)

# CORS — restrict via env in production (PRD §6.4)
_origins = os.getenv("TOKENFORGE_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gpus.router)
app.include_router(models.router)
app.include_router(estimate.router)

# Serve frontend static files (only when dist exists, e.g. in Docker)
_dist = os.path.join(os.path.dirname(__file__), "../dist")
if os.path.isdir(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{path:path}")
    async def spa_catchall(path: str) -> FileResponse:
        return FileResponse(os.path.join(_dist, "index.html"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "tokenforge", "version": "1.0.0"}
