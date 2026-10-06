"""
Document loaders module.
"""

from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.document_loader import MarkdownDocumentLoader
from app.ingestion.loaders.fund_data_loader import FundDataLoader

__all__ = ["DocumentLoader", "MarkdownDocumentLoader", "FundDataLoader"]
