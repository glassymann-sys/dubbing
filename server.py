"""
Uzbek TTS — простой FastAPI сервер
Запуск: python server.py
Порт: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import scipy.io.wavfile as wavfile
import io, os

app = FastAPI(title="Uzbek TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель загружается один раз при старте
print("⏳ Загружаю TTS модель...")
from transformers import VitsModel, AutoTokenizer
import torch

MODEL_ID = "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID)
model.eval()
SAMPLE_RATE = model.config.sampling_rate
print(f"✅ Модель готова! Sample rate: {SAMPLE_RATE} Hz")


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def synthesize(req: TTSRequest):
    inputs = tokenizer(req.text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy()
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform_int16 = (waveform * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, waveform_int16)
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
