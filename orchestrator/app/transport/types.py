"""
Shared types for transport module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TransportMode(str, Enum):
    """Transport mode selection."""
    DAILY = "daily"
    LOCAL = "local"


@dataclass
class DailyRoomInfo:
    """Information about a Daily.co room."""
    room_url: str
    token: str
    room_name: str


@dataclass
class TransportConfig:
    """Configuration for transport factory."""
    mode: TransportMode
    vad_stop_secs: float = 0.8

    # Daily.co settings (only used when mode=DAILY)
    daily_api_key: str = ""
    daily_bot_name: str = "Voice Assistant"
    daily_room_expiry: int = 3600
