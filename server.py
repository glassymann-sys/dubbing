"""
Uzbek TTS — FastAPI сервер
Запуск: python server.py
Порт: http://localhost:8000

Поддерживает: Mac M1/M2 (MPS), NVIDIA (CUDA), CPU
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import scipy.io.wavfile as wavfile
import io

app = FastAPI(title="Uzbek TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Определяем устройство автоматически ---
import torch

def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Apple M1/M2 — используется MPS (Neural Engine)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("⚡ NVIDIA GPU — используется CUDA")
        return torch.device("cuda")
    else:
        print("💻 GPU не найден — используется CPU")
        return torch.device("cpu")

# Модель загружается один раз при старте
print("⏳ Загружаю TTS модель...")
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic"
DEVICE = get_device()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
SAMPLE_RATE = model.config.sampling_rate
print(f"✅ Модель готова! Устройство: {DEVICE} | Sample rate: {SAMPLE_RATE} Hz")


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def synthesize(req: TTSRequest):
    inputs = tokenizer(req.text, return_tensors="pt")
    # Переносим inputs на то же устройство что и модель
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

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
