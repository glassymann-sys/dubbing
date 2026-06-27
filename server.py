# -*- coding: utf-8 -*-
"""
Uzbek AI Dubbing Server v8
Запуск: python3 server.py → http://localhost:8000
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dubber import dub_video

app = FastAPI(title="Uzbek AI Dubbing v8")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tasks: dict = {}


@app.post("/dub")
async def dub(
    background_tasks: BackgroundTasks,
    video:       UploadFile = File(...),
    translation: str        = Form(...),
    language:    str        = Form("auto"),
    groq_key:    str        = Form(""),
):
    task_id     = str(uuid.uuid4())[:8]
    video_path  = UPLOAD_DIR / f"{task_id}_{video.filename}"
    output_path = OUTPUT_DIR / f"{task_id}_dubbed.mp4"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    tasks[task_id] = {"status": "processing", "message": "⏳ Boshlanmoqda...", "output": None}

    async def run():
        try:
            tasks[task_id]["message"] = "📢 1/5 Audio ajratilmoqda..."
            await dub_video(
                video_path   = str(video_path),
                output_path  = str(output_path),
                translation  = translation,
                groq_api_key = groq_key,
                src_language = language if language != "auto" else None,
                status_cb    = lambda msg: tasks[task_id].update({"message": msg}),
            )
            tasks[task_id].update({"status": "done", "message": "🎉 Tayyor!", "output": str(output_path)})
        except Exception as e:
            tasks[task_id].update({"status": "error", "message": str(e)})
            print(f"❌ {e}")

    background_tasks.add_task(run)
    return JSONResponse({"task_id": task_id})


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
    print("🎬 Uzbek AI Dubbing v8 → http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
