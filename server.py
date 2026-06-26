# -*- coding: utf-8 -*-
"""
Uzbek AI Dubbing Server v7
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
from pydantic import BaseModel

from text_normalizer import normalize
from dubber import transcribe_video, dub_video, VOICE_FEMALE, VOICE_MALE

app = FastAPI(title="Uzbek AI Dubbing v7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tasks: dict = {}


# ─────────────────────────────────────────────
# TTS тест
# ─────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = "sardor"
    rate: str  = "-8%"
    pitch: str = "+0Hz"

VOICES = {"madina": "uz-UZ-MadinaNeural", "sardor": "uz-UZ-SardorNeural"}

@app.post("/tts")
async def tts(req: TTSRequest):
    voice = VOICES.get(req.voice, VOICES["sardor"])
    clean = normalize(req.text)
    buf   = io.BytesIO()
    comm  = edge_tts.Communicate(text=clean, voice=voice, rate=req.rate, pitch=req.pitch)
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


# ─────────────────────────────────────────────
# Шаг 1: Загрузить видео + транскрибировать
# ─────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form("auto"),
    groq_key: str = Form(""),
):
    task_id    = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{task_id}_{video.filename}"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    tasks[task_id] = {
        "status":     "transcribing",
        "message":    "📝 Транскрибирую...",
        "video_path": str(video_path),
        "segments":   [],
        "output":     None,
    }

    async def run():
        try:
            segs = await transcribe_video(
                str(video_path),
                groq_key,
                src_language=language if language != "auto" else None,
            )
            tasks[task_id]["status"]   = "ready_to_dub"
            tasks[task_id]["message"]  = f"✅ {len(segs)} сегментов — проверь перевод!"
            tasks[task_id]["segments"] = segs
        except Exception as e:
            tasks[task_id]["status"]  = "error"
            tasks[task_id]["message"] = str(e)

    background_tasks.add_task(run)
    return JSONResponse({"task_id": task_id})


# ─────────────────────────────────────────────
# Шаг 2: Дублировать с готовыми переводами
# ─────────────────────────────────────────────

class DubRequest(BaseModel):
    task_id:  str
    segments: list   # [{start, end, text, translated, gender, speaker, duration}]

@app.post("/dub")
async def dub_endpoint(req: DubRequest, background_tasks: BackgroundTasks):
    task = tasks.get(req.task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    video_path  = task["video_path"]
    output_path = str(OUTPUT_DIR / f"{req.task_id}_dubbed.mp4")

    tasks[req.task_id]["status"]  = "dubbing"
    tasks[req.task_id]["message"] = "🎤 Генерирую голос..."

    async def run():
        try:
            result = await dub_video(
                video_path=video_path,
                output_path=output_path,
                segments=req.segments,
            )
            tasks[req.task_id]["status"]  = "done"
            tasks[req.task_id]["message"] = "🎉 Tayyor!"
            tasks[req.task_id]["output"]  = output_path
        except Exception as e:
            tasks[req.task_id]["status"]  = "error"
            tasks[req.task_id]["message"] = str(e)

    background_tasks.add_task(run)
    return JSONResponse({"ok": True})


# ─────────────────────────────────────────────
# Статус / Скачать
# ─────────────────────────────────────────────

@app.get("/status/{task_id}")
def status(task_id: str):
    t = tasks.get(task_id)
    if not t:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(t)

@app.get("/download/{task_id}")
def download(task_id: str):
    t = tasks.get(task_id)
    if not t or t["status"] != "done":
        return JSONResponse({"error": "Not ready"}, status_code=400)
    return FileResponse(t["output"], media_type="video/mp4",
                        filename=f"dubbed_{task_id}.mp4")

@app.get("/")
def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    print("🎬 Uzbek AI Dubbing v7")
    print("📡 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
