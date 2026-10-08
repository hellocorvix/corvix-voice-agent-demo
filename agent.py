#!/usr/bin/env python3
"""Corvix voice agent — inbound call flow (Tareeqa 1).

Visitor calls the Twilio number from their own phone
  -> Twilio Media Streams (websocket) -> Deepgram Nova-3 STT (language="ur")
  -> Gemini (gemini-2.5-flash) with the Dehli Darbar Kabab House booking prompt
  -> ElevenLabs TTS (Urdu voice) -> back to the caller.

Run:  python agent.py        (needs .env; see README.md)
"""

import json
import os
from datetime import date

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

# ---------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_HOSTNAME = os.getenv("PUBLIC_HOSTNAME", "")
PORT = int(os.getenv("PORT", "8765"))

REQUIRED = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
    "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
    "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
    "TWILIO_PHONE_NUMBER": TWILIO_PHONE_NUMBER,
}

missing = [k for k, v in REQUIRED.items() if not v]
if missing:
    logger.warning(
        f"Missing .env values: {', '.join(missing)}. "
        "Server will start, but calls will fail until they are filled in."
    )
if not PUBLIC_HOSTNAME:
    logger.warning(
        "PUBLIC_HOSTNAME is not set — the /incoming-call TwiML will contain "
        "a placeholder wss:// URL. Set it to your ngrok/cloudflared hostname."
    )

# ------------------------------------------------------- system prompt
# Built from demo-script-restaurant.md (Dehli Darbar Kabab House booking flow).
SYSTEM_PROMPT = f"""You are the AI receptionist for "Dehli Darbar Kabab House", a restaurant in Pakistan. You answer incoming phone calls and help customers book tables.

LANGUAGE: Speak Urdu by default, mixing in simple English words naturally the way Pakistanis speak. If the caller speaks English, reply in English. Keep every reply SHORT — one or two short sentences, like a real receptionist on the phone. Never lecture.

BOOKING FLOW — collect these, in order, one question at a time:
1. Date — "aaj" means today ({date.today().isoformat()}), "kal" means tomorrow. Resolve day names against today's date.
2. Time — confirm morning/evening clearly. This is an evening restaurant; if the caller names a morning time, gently double-check.
3. Party size — number of people.
4. Name.
5. Phone number — the caller's mobile, for the booking confirmation.

Then read back ALL details (date, time, party size, name, phone) and ask for confirmation. Only after the caller confirms, say the table is booked and thank them.

RULES:
- Never invent the restaurant's phone number, address, timings, or menu items. If asked something you do not know, say the staff will call them back shortly.
- For anything complex (catering orders, complaints, jobs), offer to connect the caller to the staff.
- Be warm and polite. Confirm details back explicitly ("toh {date.today().isoformat()} shaam 8 bajay, 4 afraad — theek hai?") rather than assuming.
- When the call is done, end politely ("Allah Hafiz, Dehli Darbar aanay ka shukriya!").
"""

# ------------------------------------------------------------ FastAPI
app = FastAPI(title="Corvix Voice Agent")


@app.get("/", response_class=PlainTextResponse)
async def health() -> str:
    return "corvix-voice-agent up"


@app.post("/incoming-call", response_class=PlainTextResponse)
async def incoming_call() -> str:
    """Twilio hits this when someone calls the Twilio number.

    It returns TwiML telling Twilio to open a Media Stream to our websocket.
    """
    host = PUBLIC_HOSTNAME or "REPLACE-WITH-YOUR-PUBLIC-HOSTNAME"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws" />
  </Connect>
</Response>"""
    logger.info("Incoming call — returning TwiML with stream URL "
                f"wss://{host}/ws")
    return twiml


async def run_bot(websocket: WebSocket, stream_sid: str, call_sid: str | None):
    """One pipeline per call."""
    serializer = TwilioFrameSerializer(
        stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID or None,
        auth_token=TWILIO_AUTH_TOKEN or None,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # NOTE: LiveOptions is deprecated in pipecat 1.12 (removal in 2.0.0) in
    # favour of DeepgramSTTService.Settings, but it still works on the pinned
    # 1.12.0. Migrate to Settings when upgrading to pipecat 2.x.
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        encoding="mulaw",      # Twilio sends 8 kHz mulaw
        sample_rate=8000,
        live_options=LiveOptions(
            language="ur",     # explicit Urdu — never "multi"
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
    # eleven_flash_v2_5 = lowest-latency streaming model; pick any Urdu-capable
    # voice in the ElevenLabs dashboard and put its ID in ELEVENLABS_VOICE_ID.
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_flash_v2_5",
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,   # caller can cut the agent off
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        messages = websocket.iter_text()
        await messages.__anext__()                      # "connected" event
        start_msg = json.loads(await messages.__anext__())  # "start" event
        stream_sid = start_msg["start"]["streamSid"]
        call_sid = start_msg["start"].get("callSid")
        logger.info(f"Media stream started: stream_sid={stream_sid} "
                    f"call_sid={call_sid}")
    except Exception as e:
        logger.error(f"Failed to read Twilio stream start: {e}")
        await websocket.close()
        return

    try:
        await run_bot(websocket, stream_sid, call_sid)
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting corvix-voice-agent on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
