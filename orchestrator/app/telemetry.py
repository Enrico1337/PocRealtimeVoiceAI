"""
Telemetry module for latency tracking and metrics.
No content logging - only timing and status metrics.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Metrics for a single voice session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Counters
    turn_count: int = 0
    stt_calls: int = 0
    llm_calls: int = 0
    tts_calls: int = 0
    rag_queries: int = 0
    barge_in_count: int = 0

    # Latency accumulators (in ms)
    stt_latency_total: float = 0.0
    llm_first_token_total: float = 0.0
    llm_total_latency_total: float = 0.0
    tts_latency_total: float = 0.0
    rag_latency_total: float = 0.0

    # Error counters
    stt_errors: int = 0
    llm_errors: int = 0
    tts_errors: int = 0
    rag_errors: int = 0

    def log_summary(self) -> None:
        """Log session summary metrics."""
        duration = (datetime.utcnow() - self.created_at).total_seconds()

        avg_stt = self.stt_latency_total / max(self.stt_calls, 1)
        avg_llm_first = self.llm_first_token_total / max(self.llm_calls, 1)
        avg_tts = self.tts_latency_total / max(self.tts_calls, 1)
        avg_rag = self.rag_latency_total / max(self.rag_queries, 1)

        logger.info(
            f"Session {self.session_id} summary: "
            f"duration={duration:.1f}s, turns={self.turn_count}, "
            f"barge_ins={self.barge_in_count}, "
            f"avg_stt={avg_stt:.0f}ms, avg_llm_first={avg_llm_first:.0f}ms, "
            f"avg_tts={avg_tts:.0f}ms, avg_rag={avg_rag:.0f}ms, "
            f"errors(stt={self.stt_errors},llm={self.llm_errors},"
            f"tts={self.tts_errors},rag={self.rag_errors})"
        )


class LatencyTracker:
    """Context manager for tracking operation latency."""

    def __init__(self, operation: str, session_id: str):
        self.operation = operation
        self.session_id = session_id
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0
        self.success: bool = True
        self.status_code: int = 200

    def __enter__(self) -> "LatencyTracker":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self.success = False
            self.status_code = 500
            logger.warning(
                f"[{self.session_id}] {self.operation} failed: "
                f"duration={self.duration_ms:.0f}ms, error={exc_type.__name__}"
            )
        else:
            logger.debug(
                f"[{self.session_id}] {self.operation}: "
                f"duration={self.duration_ms:.0f}ms, status={self.status_code}"
            )

        return False  # Don't suppress exceptions


@contextmanager
def track_latency(
    operation: str,
    session_metrics: SessionMetrics
) -> Generator[LatencyTracker, None, None]:
    """
    Context manager to track latency of an operation.

    Usage:
        with track_latency("stt", session_metrics) as tracker:
            result = await stt_call()
            tracker.status_code = 200
    """
    tracker = LatencyTracker(operation, session_metrics.session_id)

    try:
        with tracker:
            yield tracker
    finally:
        # Update session metrics based on operation
        if operation == "stt":
            session_metrics.stt_calls += 1
            session_metrics.stt_latency_total += tracker.duration_ms
            if not tracker.success:
                session_metrics.stt_errors += 1

        elif operation == "llm_first_token":
            session_metrics.llm_calls += 1
            session_metrics.llm_first_token_total += tracker.duration_ms
            if not tracker.success:
                session_metrics.llm_errors += 1

        elif operation == "llm_total":
            session_metrics.llm_total_latency_total += tracker.duration_ms

        elif operation == "tts":
            session_metrics.tts_calls += 1
            session_metrics.tts_latency_total += tracker.duration_ms
            if not tracker.success:
                session_metrics.tts_errors += 1

        elif operation == "rag":
            session_metrics.rag_queries += 1
            session_metrics.rag_latency_total += tracker.duration_ms
            if not tracker.success:
                session_metrics.rag_errors += 1


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.INFO)
