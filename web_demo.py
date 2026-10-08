#!/usr/bin/env python3
"""Corvix voice agent — browser-based web demo (no Twilio/phone needed).

Open the demo page in a browser, tap the mic button, and talk to the AI
receptionist of "Dehli Darbar Kabab House" — straight from the browser.

Audio path (all 16 kHz PCM):
    browser mic (AudioWorklet) -> websocket /ws-voice ->
    Deepgram Nova-3 STT (language="ur") -> Gemini 2.0 Flash ->
    ElevenLabs TTS (eleven_flash_v2_5) -> websocket -> browser playback.

Live transcripts (user + agent turns) are pushed to the browser as JSON
messages over the same websocket.

Run:  python web_demo.py        (needs .env; see README.md)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Reuse the restaurant-booking persona from the Twilio agent (single source
# of truth for the demo script).
from agent import SYSTEM_PROMPT

load_dotenv(override=True)

# ---------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
# HF Spaces serves Docker apps on port 7860. Local runs keep the 8765 default
# (PORT env var overrides in both cases).
PORT = int(os.getenv("PORT", "7860"))

# Voice fallback: "George — Warm, Captivating Storyteller". Verified present in
# this ElevenLabs account's voice list (2026-10-08). Override any time with
# ELEVENLABS_VOICE_ID in .env.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "") or "JBFqnCBsd6RMkjVDRZzb"

REQUIRED = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
}

missing = [k for k, v in REQUIRED.items() if not v]
if missing:
    logger.warning(
        f"Missing .env values: {', '.join(missing)}. "
        "Server will start, but the voice pipeline will fail until filled in."
    )

SAMPLE_RATE = 16000  # browser <-> server audio, 16 kHz mono PCM16
DEMO_HTML = Path(__file__).parent / "demo.html"


# ------------------------------------------------- websocket <-> pipecat
class BrowserAudioSerializer(FrameSerializer):
    """Minimal protocol for the web demo page.

    Browser -> server:  binary  = 16 kHz mono PCM16 mic audio
                        text    = JSON control, e.g. {"type": "end"}
    Server -> browser:  binary  = 16 kHz mono PCM16 agent audio
                        text    = JSON, e.g. {"type": "transcript", ...}
    """

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return bytes(frame.audio)
        if isinstance(
            frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)
        ):
            message = frame.message
            if isinstance(message, dict) and message.get("label") == "corvix.transcript":
                return json.dumps(
                    {
                        "type": "transcript",
                        "role": message["role"],
                        "text": message["text"],
                    }
                )
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            if not data:
                return None
            return InputAudioRawFrame(
                audio=data, sample_rate=SAMPLE_RATE, num_channels=1
            )
        try:
            msg = json.loads(data)
        except Exception:
            return None
        if isinstance(msg, dict) and msg.get("type") == "end":
            return EndFrame()
        return None


class TranscriptBridge(FrameProcessor):
    """Watches pipeline text frames and forwards them to the browser.

    - Final user transcripts (TranscriptionFrame) -> {"role": "user", ...}
    - Agent reply text (LLMTextFrame chunks, flushed at LLMFullResponseEndFrame)
      -> {"role": "agent", ...}
    Emitted as OutputTransportMessageFrame so the serializer turns them into
    JSON text messages on the websocket.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._assistant_text: list[str] = []

    async def _emit(self, role: str, text: str, direction: FrameDirection):
        text = (text or "").strip()
        if not text:
            return
        await self.push_frame(
            OutputTransportMessageFrame(
                message={
                    "label": "corvix.transcript",
                    "role": role,
                    "text": text,
                }
            ),
            direction,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Deepgram pushes TranscriptionFrame only for final segments
            # (interims are InterimTranscriptionFrame, a sibling class).
            await self._emit("user", frame.text, direction)
        elif isinstance(frame, LLMTextFrame):
            self._assistant_text.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._assistant_text)
            self._assistant_text = []
            await self._emit("agent", text, direction)
        elif isinstance(frame, InterruptionFrame):
            # Reply was cut off — don't show the partial text as complete.
            self._assistant_text = []

        await self.push_frame(frame, direction)


# ------------------------------------------------------------ FastAPI
app = FastAPI(title="Corvix Voice Agent — Web Demo")


@app.get("/", response_class=FileResponse)
async def demo_page():
    return DEMO_HTML


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "corvix-web-demo up"


async def run_web_bot(websocket: WebSocket):
    """One pipeline per browser connection."""
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            add_wav_header=False,
            serializer=BrowserAudioSerializer(),
        ),
    )

    # NOTE: `task` is referenced before its source-order definition below,
    # but the handlers only run after the pipeline starts, so the closure
    # resolves fine (standard pipecat pattern).
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Browser client connected — agent greets first")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Browser client disconnected — cancelling pipeline")
        try:
            await task.cancel()
        except Exception:
            pass

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        live_options=LiveOptions(
            language="ur",  # explicit Urdu — never "multi"
            model="nova-3",
            interim_results=True,
            smart_format=True,
        ),
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    # NOTE: `model=` is deprecated in pipecat 1.12 in favour of
    # GoogleLLMService.Settings(model=...); it still works on pinned 1.12.0.
    llm = GoogleLLMService(
        api_key=GEMINI_API_KEY,
        model="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    # NOTE: `voice_id=` / `model=` kwargs are deprecated in pipecat 1.12 in
    # favour of ElevenLabsTTSService.Settings(...); they still work on 1.12.0.
    # eleven_flash_v2_5 = lowest-latency streaming multilingual model.
    # sample_rate=16000 -> ElevenLabs pcm_16000 output, matching the browser.
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_flash_v2_5",
        sample_rate=SAMPLE_RATE,
    )

    user_tap = TranscriptBridge()
    assistant_tap = TranscriptBridge()

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_tap,
            aggregators.user(),
            llm,
            assistant_tap,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)


@app.websocket("/ws-voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket /ws-voice accepted")
    try:
        await run_web_bot(websocket)
    except Exception as e:
        logger.error(f"Web demo error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting corvix web demo on port {PORT} — open http://localhost:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
