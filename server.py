"""
Uzbek TTS — FastAPI + Edge TTS + SSML
Голос: uz-UZ-MadinaNeural / uz-UZ-SardorNeural

Запуск: python3 server.py → http://localhost:8000
"""

import asyncio
import io
import edge_tts
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Uzbek TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICES = {
    "madina": "uz-UZ-MadinaNeural",
    "sardor": "uz-UZ-SardorNeural",
}


def build_ssml(text: str, voice: str, rate: str, pitch: str, style: str, styledegree: float) -> str:
    return f"""<speak version="1.0"
  xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:mstts="https://www.w3.org/2001/mstts"
  xml:lang="uz-UZ">
  <voice name="{voice}">
    <mstts:express-as style="{style}" styledegree="{styledegree}">
      <prosody rate="{rate}" pitch="{pitch}">
        {text}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>"""


class TTSRequest(BaseModel):
    text: str
    voice: str = "madina"
    rate: str = "-8%"
    pitch: str = "+1Hz"
    style: str = "friendly"
    styledegree: float = 1.5


async def synth(req: TTSRequest) -> bytes:
    voice = VOICES.get(req.voice, VOICES["madina"])
    ssml  = build_ssml(req.text, voice, req.rate, req.pitch, req.style, req.styledegree)
    buf   = io.BytesIO()
    communicate = edge_tts.Communicate(text=ssml, voice=voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


@app.post("/tts")
async def tts(req: TTSRequest):
    audio = await synth(req)
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")


@app.get("/voices")
def voices():
    return {"voices": list(VOICES.keys())}


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    print("🎤 Uzbek TTS сервер запущен!")
    print("📡 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
