# Corvix Voice Agent — browser-based demo, packaged for Hugging Face Spaces.
FROM python:3.12-slim

# onnxruntime (Silero VAD) needs libgomp on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + demo page. (agent.py is imported by web_demo.py for SYSTEM_PROMPT.)
COPY web_demo.py agent.py demo.html ./

# HF Spaces routes to port 7860 for Docker SDK spaces.
ENV PORT=7860
EXPOSE 7860

# Secrets (GEMINI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY) come from
# Space Settings -> Secrets; see SECRETS.md. They are read via python-dotenv.
CMD ["python", "web_demo.py"]
