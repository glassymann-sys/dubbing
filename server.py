"""
Uzbek TTS — FastAPI сервер с Microsoft Edge TTS
Голос: uz-UZ-MadinaNeural / uz-UZ-SardorNeural

Запуск: python3 server.py
Порт:   http://localhost:8000
"""

import asyncio
import io
import edge_tts
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Uzbek TTS API — Edge TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Доступные голоса
VOICES = {
    "madina": "uz-UZ-MadinaNeural",   # женский
    "sardor": "uz-UZ-SardorNeural",   # мужской
}

DEFAULT_VOICE = "madina"


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE   # "madina" или "sardor"
    rate: str = "-5%"            # скорость: -10%, 0%, +10%
    pitch: str = "+0Hz"          # высота: -5Hz, 0Hz, +5Hz


async def synthesize_edge(text: str, voice_key: str, rate: str, pitch: str) -> bytes:
    voice = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])
    buf = io.BytesIO()
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


@app.post("/tts")
async def tts(req: TTSRequest):
    audio = await synthesize_edge(req.text, req.voice, req.rate, req.pitch)
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")


@app.get("/voices")
def list_voices():
    return {"voices": list(VOICES.keys()), "default": DEFAULT_VOICE}


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    print("🎤 Узбекский TTS сервер запущен!")
    print("📡 http://localhost:8000")
    print(f"🎙️  Голоса: {', '.join(VOICES.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
