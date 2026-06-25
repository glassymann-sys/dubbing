# -*- coding: utf-8 -*-
"""
Uzbek AI Dubbing Server
FastAPI сервер: TTS + видео дублирование

Запуск: python3 server.py → http://localhost:8000
"""

import asyncio
import io
import os
import uuid
from pathlib import Path

import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from text_normalizer import normalize
from dubber import dub_video, VOICE_FEMALE, VOICE_MALE

app = FastAPI(title="Uzbek AI Dubbing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папки
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Хранилище статусов задач
tasks: dict = {}

VOICES = {
    "madina": VOICE_FEMALE,
    "sardor": VOICE_MALE,
}


# ─────────────────────────────────────────────
# TTS endpoint (как раньше)
# ─────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = "madina"
    rate: str = "-8%"
    pitch: str = "+0Hz"


@app.post("/tts")
async def tts(req: TTSRequest):
    voice      = VOICES.get(req.voice, VOICE_FEMALE)
    clean_text = normalize(req.text)
    buf        = io.BytesIO()
    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=voice,
        rate=req.rate,
        pitch=req.pitch,
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


# ─────────────────────────────────────────────
# Дублирование видео
# ─────────────────────────────────────────────

async def run_dubbing(task_id: str, video_path: str, output_path: str,
                      groq_api_key: str, voice: str, src_language: str):
    try:
        steps = [
            "📢 Извлекаю аудио...",
            "📝 Транскрибирую речь (AssemblyAI)...",
            "👥 Определяю пол спикеров...",
            "🌐 Перевожу на узбекский (Groq)...",
            "🎤 Генерирую голос (TTS)...",
            "🎬 Собираю финальное видео...",
        ]
        tasks[task_id]["status"]  = "processing"
        tasks[task_id]["message"] = steps[0]
        tasks[task_id]["step"]    = 1
        tasks[task_id]["total"]   = len(steps)

        import dubber as db

        # Патчим print чтобы обновлять статус
        orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

        result = await dub_video(
            video_path=video_path,
            output_path=output_path,
            groq_api_key=groq_api_key,
            voice=voice,
            src_language=src_language if src_language != "auto" else None,
        )

        tasks[task_id]["status"]        = "done"
        tasks[task_id]["message"]       = "✅ Tayyor!"
        tasks[task_id]["step"]          = len(steps)
        tasks[task_id]["output"]        = str(output_path)
        tasks[task_id]["segments"]      = result["segments"]
        tasks[task_id]["segment_count"] = result["segment_count"]

    except Exception as e:
        tasks[task_id]["status"]  = "error"
        tasks[task_id]["message"] = str(e)
        print(f"❌ Ошибка: {e}")


@app.post("/dub")
async def dub(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    voice: str = Form("madina"),
    language: str = Form("auto"),
    groq_key: str = Form(...),
):
    # Сохраняем видео
    task_id    = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{task_id}_{video.filename}"
    output_path = OUTPUT_DIR / f"{task_id}_dubbed.mp4"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Создаём задачу
    tasks[task_id] = {
        "status":  "queued",
        "message": "В очереди...",
        "output":  None,
    }

    # Запускаем в фоне
    background_tasks.add_task(
        run_dubbing,
        task_id, str(video_path), str(output_path),
        groq_key, VOICES.get(voice, VOICE_FEMALE), language
    )

    return JSONResponse({"task_id": task_id})


@app.get("/status/{task_id}")
def status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(task)


@app.get("/download/{task_id}")
def download(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "done":
        return JSONResponse({"error": "Not ready"}, status_code=400)
    return FileResponse(
        task["output"],
        media_type="video/mp4",
        filename=f"dubbed_{task_id}.mp4"
    )


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    groq_key = os.environ.get("GROQ_API_KEY", "")
    print("🎬 Uzbek AI Dubbing Server")
    print("📡 http://localhost:8000")
    print(f"🔑 Groq API: {'✅' if groq_key else '❌ не найден'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
