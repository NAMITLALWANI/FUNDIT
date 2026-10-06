"""
Ingestion package: mutual fund domain models, loaders, chunking, metric calculation and pipeline.
"""

from app.ingestion.models import (
    DataQualityReport,
    DocumentChunk,
    FundMetricsRecord,
    FundRecord,
    NavPoint,
    RawDocument,
    SourceMetadata,
)

__all__ = [
    "DataQualityReport",
    "DocumentChunk",
    "FundMetricsRecord",
    "FundRecord",
    "NavPoint",
    "RawDocument",
    "SourceMetadata",
]
