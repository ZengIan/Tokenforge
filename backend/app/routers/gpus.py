from fastapi import APIRouter

from ..models.schemas import GpuSpec
from ..services.gpu_db import load_gpus

router = APIRouter(prefix="/api", tags=["gpus"])


@router.get("/gpus", response_model=list[GpuSpec])
def list_gpus() -> list[GpuSpec]:
    """Return the preset GPU database (PRD §2.2.1)."""
    return load_gpus()
