"""
Citation tracker: converts verified EvidenceRef objects into numbered Citations,
then validates that every chunk ID the LLM claims to have cited actually exists
in the retrieved set (programmatic citation validation).
"""

from typing import Dict, List, Set, Tuple

from app.decision.models import EvidenceRef
from app.generation.models import Citation


class CitationTracker:
    """Builds numbered Citations from EvidenceRefs and validates LLM-claimed chunk IDs."""

    def build_citations(self, evidence_refs: List[EvidenceRef]) -> List[Citation]:
        """Convert ordered EvidenceRef list into numbered Citation objects."""
        citations: List[Citation] = []
        for idx, ref in enumerate(evidence_refs, start=1):
            snippet = ref.snippet[:400] if len(ref.snippet) > 400 else ref.snippet
            citations.append(
                Citation(
                    citation_id=f"[{idx}]",
                    document_id=ref.document_id,
                    chunk_id=ref.chunk_id,
                    fund_id=None,  # EvidenceRef does not carry fund_id; look up via document_id if needed
                    source=ref.title,
                    source_url=ref.source_url,
                    title=ref.title,
                    document_type=ref.document_type,
                    publication_date=ref.publication_date,
                    snippet=snippet,
                )
            )
        return citations

    def validate_cited_ids(
        self,
        claimed_chunk_ids: List[str],
        available_citations: List[Citation],
    ) -> Tuple[List[str], List[str]]:
        """Check LLM-claimed chunk IDs against the retrieved set.

        Returns:
            (verified_ids, unverified_ids) — unverified IDs are hallucinated references
            that must be stripped from the final response.
        """
        available: Set[str] = {c.chunk_id for c in available_citations}
        verified: List[str] = []
        unverified: List[str] = []
        for cid in claimed_chunk_ids:
            (verified if cid in available else unverified).append(cid)
        return verified, unverified

    def filter_citations(
        self,
        citations: List[Citation],
        verified_chunk_ids: List[str],
    ) -> List[Citation]:
        """Return only citations whose chunk_id appears in the verified set."""
        if not verified_chunk_ids:
            return citations  # if LLM produced no cited_chunk_ids, keep all
        verified_set: Set[str] = set(verified_chunk_ids)
        return [c for c in citations if c.chunk_id in verified_set]

    def build_chunk_id_map(self, citations: List[Citation]) -> Dict[str, Citation]:
        """Return chunk_id -> Citation lookup for fast validation."""
        return {c.chunk_id: c for c in citations}
