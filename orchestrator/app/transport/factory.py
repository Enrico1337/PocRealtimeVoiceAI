"""
Transport factory for creating Daily.co or local WebRTC transports.
"""

import logging
from typing import Optional, Union

from pipecat.transports.daily.transport import DailyTransport
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from .types import TransportMode, TransportConfig, DailyRoomInfo
from .daily import DailyTransportManager
from .local import create_local_transport, get_local_ice_servers

logger = logging.getLogger(__name__)


class TransportFactory:
    """Factory for creating WebRTC transports based on configuration."""

    def __init__(self, config: TransportConfig):
        """Initialize transport factory.

        Args:
            config: Transport configuration
        """
        self.config = config
        self._daily_manager: Optional[DailyTransportManager] = None

        if config.mode == TransportMode.DAILY:
            if not config.daily_api_key:
                raise ValueError(
                    "Daily API key is required when transport_mode=daily. "
                    "Set DAILY_API_KEY environment variable."
                )
            self._daily_manager = DailyTransportManager(config.daily_api_key)
            logger.info("Transport mode: daily (Daily.co hosted WebRTC)")
        else:
            logger.info("Transport mode: local (SmallWebRTC with STUN only)")

    @property
    def is_daily(self) -> bool:
        """Check if running in Daily mode."""
        return self.config.mode == TransportMode.DAILY

    @property
    def is_local(self) -> bool:
        """Check if running in local mode."""
        return self.config.mode == TransportMode.LOCAL

    async def create_daily_room(self) -> DailyRoomInfo:
        """Create a new Daily.co room.

        Returns:
            DailyRoomInfo with room URL and token

        Raises:
            RuntimeError: If not in Daily mode
        """
        if not self.is_daily or self._daily_manager is None:
            raise RuntimeError("Cannot create Daily room - not in Daily mode")

        return await self._daily_manager.create_room(
            expiry_time=self.config.daily_room_expiry
        )

    def create_daily_transport(self, room_info: DailyRoomInfo) -> DailyTransport:
        """Create Daily.co transport for a room.

        Args:
            room_info: Room information from create_daily_room()

        Returns:
            Configured DailyTransport

        Raises:
            RuntimeError: If not in Daily mode
        """
        if not self.is_daily or self._daily_manager is None:
            raise RuntimeError("Cannot create Daily transport - not in Daily mode")

        return self._daily_manager.create_transport(
            room_info=room_info,
            bot_name=self.config.daily_bot_name,
            vad_stop_secs=self.config.vad_stop_secs,
        )

    def create_local_transport(
        self,
        connection: SmallWebRTCConnection,
    ) -> SmallWebRTCTransport:
        """Create local SmallWebRTC transport.

        Args:
            connection: WebRTC connection from request handler

        Returns:
            Configured SmallWebRTCTransport

        Raises:
            RuntimeError: If not in local mode
        """
        if not self.is_local:
            raise RuntimeError("Cannot create local transport - not in local mode")

        return create_local_transport(
            connection=connection,
            vad_stop_secs=self.config.vad_stop_secs,
        )

    def get_ice_servers(self):
        """Get ICE servers for server-side (aiortc).

        Only used in local mode.
        """
        if not self.is_local:
            raise RuntimeError("ICE servers only used in local mode")
        return get_local_ice_servers()

    async def close(self):
        """Clean up resources."""
        if self._daily_manager is not None:
            await self._daily_manager.close()
