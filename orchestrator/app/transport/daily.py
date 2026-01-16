"""
Daily.co WebRTC transport manager.
Handles room creation and Daily transport configuration.
"""

import logging
from typing import Optional

import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

from .types import DailyRoomInfo

logger = logging.getLogger(__name__)


class DailyTransportManager:
    """Manages Daily.co rooms and transports."""

    def __init__(self, api_key: str, api_url: str = "https://api.daily.co/v1"):
        """Initialize Daily.co manager.

        Args:
            api_key: Daily.co API key
            api_url: Daily.co API URL (default: production)
        """
        if not api_key:
            raise ValueError("Daily.co API key is required")

        self.api_key = api_key
        self.api_url = api_url
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._rest_helper: Optional[DailyRESTHelper] = None

    async def _get_rest_helper(self) -> DailyRESTHelper:
        """Get or create REST helper (lazy initialization)."""
        if self._aiohttp_session is None:
            self._aiohttp_session = aiohttp.ClientSession()

        if self._rest_helper is None:
            self._rest_helper = DailyRESTHelper(
                daily_api_key=self.api_key,
                daily_api_url=self.api_url,
                aiohttp_session=self._aiohttp_session,
            )
        return self._rest_helper

    async def create_room(self, expiry_time: int = 3600) -> DailyRoomInfo:
        """Create a new Daily.co room.

        Args:
            expiry_time: Room expiry time in seconds

        Returns:
            DailyRoomInfo with room URL, token, and name
        """
        import time

        rest_helper = await self._get_rest_helper()

        # Create room with public privacy (no knock required)
        room = await rest_helper.create_room(
            DailyRoomParams(
                privacy="public",
            )
        )

        # Get token for the bot to join
        token = await rest_helper.get_token(room.url, expiry_time)

        logger.info(f"Created Daily room: {room.name} (expires in {expiry_time}s)")

        return DailyRoomInfo(
            room_url=room.url,
            token=token,
            room_name=room.name,
        )

    def create_transport(
        self,
        room_info: DailyRoomInfo,
        bot_name: str,
        vad_stop_secs: float,
    ) -> DailyTransport:
        """Create Daily transport for a room.

        Args:
            room_info: Room information from create_room()
            bot_name: Display name for the bot in the room
            vad_stop_secs: Silence duration (seconds) to finalize utterance

        Returns:
            Configured DailyTransport instance
        """
        return DailyTransport(
            room_url=room_info.room_url,
            token=room_info.token,
            bot_name=bot_name,
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=vad_stop_secs)
                ),
            ),
        )

    async def close(self):
        """Clean up resources."""
        self._rest_helper = None
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()
            self._aiohttp_session = None
