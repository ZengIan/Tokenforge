from fastapi import APIRouter

from ..models.schemas import (
    CompareRequest,
    EstimateRequest,
    EstimateResponse,
)
from ..services.estimator import estimate

router = APIRouter(prefix="/api", tags=["estimate"])


@router.post("/estimate", response_model=EstimateResponse)
def post_estimate(req: EstimateRequest) -> EstimateResponse:
    """Estimate TPS + memory for one configuration (PRD §2.4)."""
    return estimate(req)


@router.post("/compare", response_model=list[EstimateResponse])
def post_compare(req: CompareRequest) -> list[EstimateResponse]:
    """Estimate multiple scenarios for side-by-side comparison (PRD §2.4.4)."""
    return [estimate(s) for s in req.scenarios]
