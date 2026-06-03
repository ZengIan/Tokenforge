from typing import Any

from fastapi import APIRouter, Query

from ..services import modelscope

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/search")
async def search(q: str = Query(..., min_length=1), limit: int = 10) -> dict[str, Any]:
    """Search ModelScope models (PRD §2.1.1)."""
    results = await modelscope.search_models(q, limit=limit)
    return {"query": q, "results": results}


@router.get("/{model_id:path}")
async def detail(model_id: str) -> dict[str, Any]:
    """Fetch architecture metadata for a model (PRD §2.1.2)."""
    return await modelscope.get_model_config(model_id)
