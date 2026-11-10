"""Dataset ingestion service — validate, persist, and profile an upload.

Keeps the composition of validation + storage + profiling out of the route so it
can be unit-tested without HTTP.
"""
from __future__ import annotations

from ..schemas.dataset import DatasetUploadResponse
from ..tools.dataset_tools import profile_dataframe
from .csv_validation import validate_and_parse
from .storage import save_dataset


def ingest_dataset(raw: bytes, filename: str) -> DatasetUploadResponse:
    """Validate the raw upload, store it, and return its profile.

    Raises :class:`DatasetValidationError` if the file is not a usable CSV.
    """
    df = validate_and_parse(raw, filename)
    dataset_id, _ = save_dataset(df, filename)
    info = profile_dataframe(df)
    return DatasetUploadResponse(dataset_id=dataset_id, filename=filename, info=info)
