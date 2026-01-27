"""
Twilio telephony transport via FastAPI WebSocket.

Uses Pipecat's FastAPIWebsocketTransport with TwilioFrameSerializer
to bridge Twilio Media Streams (mu-law 8kHz) into the Pipecat pipeline.
"""

import logging

from fastapi import WebSocket

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from .types import TwilioCallInfo

logger = logging.getLogger(__name__)


def create_twilio_transport(
    websocket: WebSocket,
    call_info: TwilioCallInfo,
    vad_stop_secs: float,
    account_sid: str = "",
    auth_token: str = "",
) -> FastAPIWebsocketTransport:
    """Create a FastAPIWebsocketTransport for Twilio Media Streams.

    Args:
        websocket: FastAPI WebSocket connection from Twilio
        call_info: Twilio call info (stream_sid, call_sid)
        vad_stop_secs: Silence duration (seconds) to finalize utterance
        account_sid: Twilio Account SID (for auto hang-up)
        auth_token: Twilio Auth Token (for auto hang-up)

    Returns:
        Configured FastAPIWebsocketTransport instance
    """
    serializer = TwilioFrameSerializer(
        stream_sid=call_info.stream_sid,
        call_sid=call_info.call_sid,
        account_sid=account_sid,
        auth_token=auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=vad_stop_secs)
            ),
            serializer=serializer,
        ),
    )

    logger.info(
        f"Twilio transport created: stream_sid={call_info.stream_sid}, "
        f"call_sid={call_info.call_sid}"
    )

    return transport
