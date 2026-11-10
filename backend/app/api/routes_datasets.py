"""Dataset endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..logging_config import get_logger
from ..schemas.dataset import DatasetUploadResponse
from ..services.csv_validation import DatasetValidationError
from ..services.dataset_service import ingest_dataset

logger = get_logger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and profile a CSV dataset",
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    raw = await file.read()
    try:
        response = ingest_dataset(raw, file.filename or "upload.csv")
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    logger.info("Uploaded dataset %s (%s)", response.dataset_id, response.filename)
    return response
