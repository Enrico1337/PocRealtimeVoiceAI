#!/usr/bin/env python3
"""Quick import test - run locally before deploying.

Usage:
    pip install pipecat-ai[webrtc,silero]==0.0.99
    python orchestrator/test_imports.py
"""

def test_imports():
    errors = []

    # Test all pipecat imports used in the orchestrator
    imports = [
        ("pipecat.audio.vad.silero", "SileroVADAnalyzer"),
        ("pipecat.audio.vad.vad_analyzer", "VADParams"),
        ("pipecat.pipeline.pipeline", "Pipeline"),
        ("pipecat.pipeline.runner", "PipelineRunner"),
        ("pipecat.pipeline.task", "PipelineParams, PipelineTask"),
        ("pipecat.processors.aggregators.llm_context", "LLMContext"),
        ("pipecat.processors.aggregators.llm_response_universal", "LLMContextAggregatorPair"),
        ("pipecat.services.openai.llm", "OpenAILLMService"),
        ("pipecat.transports.smallwebrtc.transport", "SmallWebRTCTransport"),
        ("pipecat.transports.base_transport", "TransportParams"),
        ("pipecat.transports.smallwebrtc.request_handler", "SmallWebRTCRequestHandler, SmallWebRTCRequest"),
        ("pipecat.transports.smallwebrtc.connection", "SmallWebRTCConnection"),
        ("pipecat.services.tts_service", "TTSService"),
        ("pipecat.frames.frames", "TextFrame, Frame, EndFrame, AudioRawFrame"),
        ("pipecat.processors.frame_processor", "FrameDirection, FrameProcessor"),
    ]

    print("Testing pipecat imports...\n")

    for module, classes in imports:
        try:
            exec(f"from {module} import {classes}")
            print(f"  OK: {module}")
        except ImportError as e:
            print(f"FAIL: {module} - {e}")
            errors.append((module, str(e)))

    # Test other dependencies
    print("\nTesting other dependencies...\n")
    other_imports = [
        ("aiortc", "RTCIceServer"),
        ("fastapi", "FastAPI"),
        ("qdrant_client", "QdrantClient"),
        ("sentence_transformers", "SentenceTransformer"),
    ]

    for module, classes in other_imports:
        try:
            exec(f"from {module} import {classes}")
            print(f"  OK: {module}")
        except ImportError as e:
            print(f"FAIL: {module} - {e}")
            errors.append((module, str(e)))

    print()
    if errors:
        print(f"{len(errors)} import(s) failed!")
        for module, error in errors:
            print(f"  - {module}: {error}")
        return False

    print("All imports OK!")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_imports() else 1)
