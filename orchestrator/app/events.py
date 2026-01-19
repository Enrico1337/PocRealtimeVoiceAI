"""
Event streaming system for real-time session monitoring.
Provides Server-Sent Events (SSE) for frontend logging.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types for session monitoring."""
    CONNECTION = "connection"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    RAG = "rag"
    TRANSCRIPT = "transcript"
    ERROR = "error"
    VAD = "vad"


@dataclass
class SessionEvent:
    """Represents an event in a session."""
    type: EventType
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_sse(self) -> str:
        """Format event for SSE transmission."""
        event_data = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data
        }
        return f"data: {json.dumps(event_data)}\n\n"


class SessionEventManager:
    """Manages SSE connections and event distribution for sessions."""

    def __init__(self):
        # session_id -> set of asyncio.Queue
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to events for a session.

        Returns an asyncio.Queue that will receive events.
        """
        async with self._lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = set()

            queue = asyncio.Queue(maxsize=100)
            self._subscribers[session_id].add(queue)
            logger.debug(f"New subscriber for session {session_id}, total: {len(self._subscribers[session_id])}")
            return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from session events."""
        async with self._lock:
            if session_id in self._subscribers:
                self._subscribers[session_id].discard(queue)
                if not self._subscribers[session_id]:
                    del self._subscribers[session_id]
                logger.debug(f"Removed subscriber for session {session_id}")

    async def emit(self, session_id: str, event: SessionEvent) -> None:
        """Emit an event to all subscribers of a session."""
        async with self._lock:
            if session_id not in self._subscribers:
                return

            subscribers = list(self._subscribers[session_id])

        # Send to all subscribers outside the lock
        sse_data = event.to_sse()
        for queue in subscribers:
            try:
                queue.put_nowait(sse_data)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for session {session_id}, dropping event")

    async def emit_connection(self, session_id: str, status: str, details: Optional[dict] = None) -> None:
        """Emit a connection status event."""
        data = {"status": status}
        if details:
            data.update(details)
        await self.emit(session_id, SessionEvent(EventType.CONNECTION, data))

    async def emit_stt(self, session_id: str, status: str, text: Optional[str] = None,
                       duration_ms: Optional[int] = None) -> None:
        """Emit an STT event."""
        data = {"status": status}
        if text:
            data["text"] = text
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        await self.emit(session_id, SessionEvent(EventType.STT, data))

    async def emit_llm(self, session_id: str, status: str, text: Optional[str] = None,
                       tokens: Optional[int] = None) -> None:
        """Emit an LLM event."""
        data = {"status": status}
        if text:
            data["text"] = text
        if tokens is not None:
            data["tokens"] = tokens
        await self.emit(session_id, SessionEvent(EventType.LLM, data))

    async def emit_tts(self, session_id: str, status: str, duration_ms: Optional[int] = None) -> None:
        """Emit a TTS event."""
        data = {"status": status}
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        await self.emit(session_id, SessionEvent(EventType.TTS, data))

    async def emit_rag(self, session_id: str, status: str, docs_count: Optional[int] = None) -> None:
        """Emit a RAG event."""
        data = {"status": status}
        if docs_count is not None:
            data["docs_count"] = docs_count
        await self.emit(session_id, SessionEvent(EventType.RAG, data))

    async def emit_transcript(self, session_id: str, role: str, text: str) -> None:
        """Emit a transcript event (user or bot message)."""
        await self.emit(session_id, SessionEvent(EventType.TRANSCRIPT, {
            "role": role,
            "text": text
        }))

    async def emit_vad(self, session_id: str, status: str) -> None:
        """Emit a VAD event (speaking started/stopped)."""
        await self.emit(session_id, SessionEvent(EventType.VAD, {"status": status}))

    async def emit_error(self, session_id: str, message: str, component: Optional[str] = None) -> None:
        """Emit an error event."""
        data = {"message": message}
        if component:
            data["component"] = component
        await self.emit(session_id, SessionEvent(EventType.ERROR, data))

    def get_active_sessions(self) -> list:
        """Get list of session IDs with active subscribers."""
        return list(self._subscribers.keys())


# Global event manager instance
event_manager = SessionEventManager()
