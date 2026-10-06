"""
Evidence verification: binds reranked chunks to candidate funds and checks them against
structured facts before the decision engine or the generator may use them.
"""

from app.evidence.validator import EvidenceBundle, EvidenceValidator, ValidationReport

__all__ = ["EvidenceBundle", "EvidenceValidator", "ValidationReport"]
