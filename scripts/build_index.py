"""
Embed document chunks from the database and index them into Qdrant (dense) and BM25 (sparse).

Usage:
    python scripts/build_index.py              # upsert into the configured Qdrant collection
    python scripts/build_index.py --rebuild    # drop and recreate the collection first

Requires `scripts/ingest_funds.py` to have populated the database. Safe to re-run: point IDs are
derived deterministically from chunk IDs. If Qdrant is unreachable and QDRANT_FALLBACK_IN_MEMORY
is true the script still validates the pipeline, but the in-memory index is discarded on exit;
the API rebuilds it at startup in that mode.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.repository import FundRepository  # noqa: E402
from app.db.session import create_db_engine, get_session_factory  # noqa: E402
from app.retrieval.retriever import EvidenceRetriever  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the Qdrant collection")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, json_format=settings.log_json)

    engine = create_db_engine(settings=settings)
    with get_session_factory(engine)() as session:
        chunks = FundRepository(session).list_chunks()
    if not chunks:
        print("No document chunks found in the database. Run scripts/ingest_funds.py first.")
        return 1

    retriever = EvidenceRetriever(settings=settings)
    report = retriever.build_index(chunks, rebuild=args.rebuild)

    print("\n=== Index build report ===")
    print(f"Qdrant mode / collection:   {retriever.vector_store.mode} / {report.collection_name}")
    print(f"Chunks indexed:             {report.chunks_indexed}")
    print(f"Embedding backend / dim:    {report.embedding_backend} / {report.embedding_dimension}")
    print(f"Model load (cold start):    {retriever.embedding_service.load_time_ms} ms")
    print(f"Embedding time:             {report.embed_time_ms} ms")
    print(f"Upsert time:                {report.upsert_time_ms} ms")
    print(f"BM25 documents:             {report.bm25_documents}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if retriever.vector_store.mode == "in-memory":
        print("WARNING: Qdrant server unreachable; in-memory index was built for validation only and is not persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
