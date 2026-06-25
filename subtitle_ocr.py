# -*- coding: utf-8 -*-
"""
Subtitle OCR — извлекает субтитры с видео через OCR
Анализирует нижнюю часть каждого кадра и находит текст субтитров
"""

import subprocess
import os
import json
from PIL import Image
import pytesseract
import tempfile


def extract_subtitle_frames(video_path: str, temp_dir: str,
                             fps: float = 3.0) -> list:
    """
    Извлекает кадры из средне-нижней части видео где субтитры.
    """
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", video_path],
        capture_output=True, text=True
    )
    data = json.loads(probe.stdout)
    width, height = 1920, 1080
    duration = 30.0

    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            width  = s.get("width", 1920)
            height = s.get("height", 1080)

    for s in data.get("streams", []):
        if "duration" in s:
            duration = float(s["duration"])
            break

    print(f"  Видео: {width}x{height}, {duration:.1f} сек")

    # Берём среднюю нижнюю часть — от 50% до 85% высоты
    crop_y = int(height * 0.50)
    crop_h = int(height * 0.35)
    out_pattern = os.path.join(frames_dir, "frame_%04d.png")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},crop={width}:{crop_h}:0:{crop_y}",
        "-y", out_pattern
    ]
    subprocess.run(cmd, capture_output=True)

    frames = []
    frame_files = sorted([
        f for f in os.listdir(frames_dir) if f.endswith(".png")
    ])

    for i, fname in enumerate(frame_files):
        timestamp = i / fps
        frames.append({
            "path":      os.path.join(frames_dir, fname),
            "timestamp": timestamp
        })

    print(f"  Извлечено {len(frames)} кадров")
    return frames, duration


def ocr_frame(image_path: str) -> str:
    """OCR одного кадра — извлекает белый текст субтитров"""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        import numpy as np
        arr = np.array(img)

        # Белый текст: все каналы > 180
        white_mask = (arr[:,:,0] > 180) & (arr[:,:,1] > 180) & (arr[:,:,2] > 180)

        # Белый текст → чёрный на белом фоне
        result = np.ones_like(arr) * 255
        result[white_mask] = 0

        img_bw = Image.fromarray(result.astype(np.uint8))

        # Увеличиваем для лучшего OCR
        w, h = img_bw.size
        img_bw = img_bw.resize((w*2, h*2), Image.LANCZOS)

        text = pytesseract.image_to_string(
            img_bw,
            lang="eng",
            config="--psm 6 --oem 3"
        )
        # Убираем строки с мусором (меньше 3 букв)
        lines = [l.strip() for l in text.split('\n')
                 if len(l.strip()) > 3 and any(c.isalpha() for c in l)]
        return " ".join(lines).strip()
    except:
        return ""


def extract_subtitles_ocr(video_path: str) -> list:
    """
    Извлекает субтитры из видео через OCR.
    Возвращает: [{"start": 0.0, "end": 2.5, "text": "Hello"}]
    """
    print("📺 Извлекаю субтитры через OCR...")

    subtitles = []

    with tempfile.TemporaryDirectory() as tmp:
        frames, duration = extract_subtitle_frames(video_path, tmp, fps=3.0)

        prev_text  = ""
        seg_start  = 0.0
        seg_text   = ""

        for frame in frames:
            text = ocr_frame(frame["path"])
            # Убираем шум и пустые строки
            text = " ".join(text.split())

            if text and len(text) > 3:
                if text != prev_text:
                    # Сохраняем предыдущий сегмент
                    if seg_text and len(seg_text) > 3:
                        subtitles.append({
                            "start": seg_start,
                            "end":   frame["timestamp"],
                            "text":  seg_text
                        })
                    seg_start = frame["timestamp"]
                    seg_text  = text
                    prev_text = text
            else:
                if seg_text:
                    subtitles.append({
                        "start": seg_start,
                        "end":   frame["timestamp"],
                        "text":  seg_text
                    })
                    seg_text  = ""
                    prev_text = ""

        # Последний сегмент
        if seg_text:
            subtitles.append({
                "start": seg_start,
                "end":   duration,
                "text":  seg_text
            })

    # Убираем дубликаты
    unique = []
    seen   = set()
    for s in subtitles:
        key = s["text"][:30]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"  Найдено {len(unique)} субтитров:")
    for s in unique:
        print(f"    [{s['start']:.1f}s → {s['end']:.1f}s] {s['text'][:60]}")

    return unique


if __name__ == "__main__":
    import sys
    import glob

    files = sorted(glob.glob("uploads/*.mov") + glob.glob("uploads/*.mp4"))
    video = files[-1] if files else None

    if not video:
        print("Нет видео в папке uploads/")
        sys.exit(1)

    print(f"Тестирую на: {video}")
    subs = extract_subtitles_ocr(video)
    print(f"\nИтого: {len(subs)} субтитров")
