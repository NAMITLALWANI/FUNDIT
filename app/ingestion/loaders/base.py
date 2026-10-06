"""
Abstract document loader interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from app.ingestion.models import RawDocument


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source_path: Union[str, Path]) -> List[RawDocument]:
        """Load and parse raw documents from the specified file or directory."""
        pass
