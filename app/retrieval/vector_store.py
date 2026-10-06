"""
Qdrant vector database interface supporting both client-server and in-memory execution.

Each point stores the full chunk provenance as payload so that search results can be
reconstructed into ``DocumentChunk`` objects without a second lookup, and so that dense search
can be bounded to the SQL candidate set via a ``fund_id`` payload filter.
"""

import uuid
from typing import List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.ingestion.models import DocumentChunk, SourceMetadata

logger = get_logger(__name__)


def build_fund_filter(fund_ids: Optional[Sequence[str]], include_global: bool = True) -> Optional[qmodels.Filter]:
    """Restrict dense search to chunks of the candidate funds (plus universe-wide documents)."""
    if fund_ids is None:
        return None
    should: List[qmodels.Condition] = [
        qmodels.FieldCondition(key="fund_id", match=qmodels.MatchAny(any=list(fund_ids)))
    ]
    if include_global:
        should.append(qmodels.IsNullCondition(is_null=qmodels.PayloadField(key="fund_id")))
    return qmodels.Filter(should=should)


class QdrantVectorStore:
    """Abstraction for Qdrant vector database operations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        in_memory: bool = False,
        client: Optional[QdrantClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.qdrant_collection
        self.mode = "server"

        if client is not None:
            self.client = client
            self.mode = "injected"
        elif in_memory or not self.settings.qdrant_url:
            logger.info("Initializing in-memory Qdrant instance")
            self.client = QdrantClient(location=":memory:")
            self.mode = "in-memory"
        else:
            try:
                logger.info(f"Connecting to Qdrant at {self.settings.qdrant_url}")
                self.client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    prefer_grpc=self.settings.qdrant_prefer_grpc,
                    timeout=5,
                    check_compatibility=False,
                )
                self.client.get_collections()
            except Exception as e:  # noqa: BLE001 - connectivity failures are handled by policy below
                if not self.settings.qdrant_fallback_in_memory:
                    raise VectorStoreError(f"Could not connect to Qdrant at {self.settings.qdrant_url}: {e}")
                logger.warning(
                    f"Could not connect to Qdrant server ({e}). Falling back to in-memory Qdrant; "
                    "the index must be rebuilt at startup."
                )
                self.client = QdrantClient(location=":memory:")
                self.mode = "in-memory"

    def health_check(self) -> bool:
        """Verify vector store connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        col_name = collection_name or self.collection_name
        try:
            return col_name in {c.name for c in self.client.get_collections().collections}
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Failed to list Qdrant collections: {e}")

    def count(self, collection_name: Optional[str] = None) -> int:
        col_name = collection_name or self.collection_name
        if not self.collection_exists(col_name):
            return 0
        try:
            return int(self.client.count(collection_name=col_name, exact=True).count)
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Failed to count points in '{col_name}': {e}")

    def create_collection_if_not_exists(
        self,
        dimension: int,
        collection_name: Optional[str] = None,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> None:
        """Create Qdrant collection (and payload indexes) if it does not already exist."""
        col_name = collection_name or self.collection_name
        try:
            if not self.collection_exists(col_name):
                logger.info(f"Creating Qdrant collection '{col_name}' with dimension {dimension}")
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=qmodels.VectorParams(size=dimension, distance=distance),
                )
                self.client.create_payload_index(
                    collection_name=col_name,
                    field_name="fund_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
        except VectorStoreError:
            raise
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Failed to create collection '{col_name}': {e}")

    def recreate_collection(self, dimension: int, collection_name: Optional[str] = None) -> None:
        """Drop and recreate the collection (used by scripts/build_index.py --rebuild)."""
        col_name = collection_name or self.collection_name
        try:
            if self.collection_exists(col_name):
                self.client.delete_collection(collection_name=col_name)
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete collection '{col_name}': {e}")
        self.create_collection_if_not_exists(dimension=dimension, collection_name=col_name)

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> None:
        """Upsert document chunks and their vector embeddings into Qdrant."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        col_name = collection_name or self.collection_name
        self.create_collection_if_not_exists(dimension=len(embeddings[0]), collection_name=col_name)

        points: List[qmodels.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(mode="json"),
                "fund_id": chunk.metadata.fund_id,
                "document_type": chunk.metadata.document_type,
            }
            points.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))

        try:
            self.client.upsert(collection_name=col_name, points=points)
            logger.info(f"Upserted {len(points)} points into Qdrant collection '{col_name}'")
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Failed to upsert chunks into '{col_name}': {e}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        collection_name: Optional[str] = None,
        query_filter: Optional[qmodels.Filter] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search nearest vector neighbors in Qdrant returning chunks and cosine scores."""
        col_name = collection_name or self.collection_name
        if not self.collection_exists(col_name):
            logger.warning(f"Qdrant collection '{col_name}' does not exist; dense retrieval returns no results")
            return []
        try:
            response = self.client.query_points(
                collection_name=col_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            scored_points = response.points
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"Vector search failed on collection '{col_name}': {e}")

        results: List[Tuple[DocumentChunk, float]] = []
        for pt in scored_points:
            payload = pt.payload or {}
            metadata = SourceMetadata.model_validate(payload.get("metadata", {}))
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", str(pt.id)),
                document_id=payload.get("document_id", metadata.document_id),
                text=payload.get("text", ""),
                metadata=metadata,
            )
            results.append((chunk, float(pt.score)))
        return results
