"""
Latency tracking, execution context, and diagnostic instrumentation.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    """Metadata for an individual execution span."""

    name: str
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LatencyTracker:
    """Measures component timings across the decision pipeline."""

    def __init__(self) -> None:
        self.spans: List[SpanRecord] = []
        self._start_time = time.perf_counter()

    @contextmanager
    def measure(self, name: str, **metadata: Any) -> Generator[None, None, None]:
        """Context manager to measure the execution time of a code block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.spans.append(
                SpanRecord(name=name, duration_ms=round(elapsed_ms, 2), metadata=metadata)
            )

    @property
    def total_duration_ms(self) -> float:
        """Total elapsed time since tracker initialization in milliseconds."""
        return round((time.perf_counter() - self._start_time) * 1000.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert recorded spans to a dictionary representation."""
        return {
            "total_latency_ms": self.total_duration_ms,
            "spans": {span.name: span.duration_ms for span in self.spans},
            "detailed_spans": [span.model_dump() for span in self.spans],
        }


class ExecutionContext:
    """Carries request tracing state and diagnostics throughout pipeline execution."""

    def __init__(self, request_id: Optional[str] = None, debug: bool = False) -> None:
        import uuid

        self.request_id: str = request_id or str(uuid.uuid4())
        self.debug: bool = debug
        self.tracker = LatencyTracker()
        self.diagnostics: Dict[str, Any] = {}

    def log_diagnostic(self, stage: str, data: Any) -> None:
        """Store debug diagnostics if debug mode is active."""
        if self.debug:
            self.diagnostics[stage] = data
