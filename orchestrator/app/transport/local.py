"""
Local WebRTC transport (SmallWebRTC) without TURN.
For local development or LAN usage where direct UDP works.
"""

import logging
from aiortc import RTCIceServer

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

logger = logging.getLogger(__name__)


def get_local_ice_servers() -> list[RTCIceServer]:
    """Get ICE servers for local mode - only STUN, no TURN needed for LAN."""
    return [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
    ]


def get_local_ice_servers_for_client() -> list[dict]:
    """Get ICE servers configuration for JavaScript client in local mode."""
    return [
        {"urls": "stun:stun.l.google.com:19302"},
    ]


def create_local_transport(
    connection: SmallWebRTCConnection,
    vad_stop_secs: float,
) -> SmallWebRTCTransport:
    """Create SmallWebRTC transport for local mode.

    Args:
        connection: WebRTC connection from SmallWebRTCRequestHandler
        vad_stop_secs: Silence duration (seconds) to finalize utterance

    Returns:
        Configured SmallWebRTCTransport instance
    """
    return SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=vad_stop_secs)
            ),
            audio_in_passthrough=True,
        ),
    )
