# -*- coding: utf-8 -*-
"""
AI Video Dubber — полный пайплайн
Видео → Whisper → Groq перевод → Edge TTS → ffmpeg → готовое видео
"""

import os
import asyncio
import tempfile
import subprocess
import json
from pathlib import Path

import whisper
import edge_tts
from groq import Groq
from text_normalizer import normalize

# ─────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────
VOICE_FEMALE = "uz-UZ-MadinaNeural"
VOICE_MALE   = "uz-UZ-SardorNeural"

GROQ_SYSTEM_PROMPT = """Sen professional tarjimon va dublyaj mutaxassisissan. 
Matnni zamonaviy o'zbek tiliga tarjima qil, xuddi Toshkentlik 20-25 yoshli yigit yoki qiz gapirgandek.
Misol uchun bunday gapirasan:
- "Yaxshimisiz? Bugun zo'r kun!"
- "Bro, bu juda muhim!"
- "Tushundim, davom et"
- "Haqiqatan ham shunaqami?"

Qoidalar:
1. Faqat tarjima matnini yoz — hech qanday izoh, tushuntirish yo'q
2. Rasmiy so'zlar ishlatma
3. Jonli, tabiiy til
4. Har bir gapni alohida qator qil
5. Tinish belgilarini saqlagan holda tarjima qil"""


# ─────────────────────────────────────────────
# Шаг 1: Извлечь аудио из видео
# ─────────────────────────────────────────────

def extract_audio(video_path: str, output_path: str) -> str:
    """Извлекает аудио из видео файла"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                    # без видео
        "-acodec", "pcm_s16le",   # WAV формат
        "-ar", "16000",           # 16kHz для Whisper
        "-ac", "1",               # моно
        "-y",                     # перезаписать
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg error: {result.stderr}")
    return output_path


# ─────────────────────────────────────────────
# Шаг 2: Транскрипция через Whisper
# ─────────────────────────────────────────────

def transcribe(audio_path: str, language: str = None) -> list:
    """
    Транскрибирует аудио и возвращает сегменты с временными метками
    Возвращает: [{"start": 0.0, "end": 2.5, "text": "Hello world"}, ...]
    """
    print("⏳ Загружаю Whisper модель...")
    model = whisper.load_model("base")  # base — быстро и точно
    print("🎙️  Транскрибирую...")

    options = {"task": "transcribe"}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"].strip()
        })
        print(f"  [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text'].strip()}")

    print(f"✅ Транскрипция готова: {len(segments)} сегментов")
    return segments


# ─────────────────────────────────────────────
# Шаг 3: Перевод через Groq
# ─────────────────────────────────────────────

def translate_segments(segments: list, groq_api_key: str) -> list:
    """Переводит каждый сегмент на узбекский"""
    client = Groq(api_key=groq_api_key)
    translated = []

    print("🌐 Переводю на узбекский...")
    # Переводим весь текст одним запросом для контекста
    full_text = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(segments)])

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Tarjima qil (har bir raqamli qatorni alohida tarjima qil, raqamlarni saqlagan holda):\n{full_text}"}
        ]
    )

    lines = r.choices[0].message.content.strip().split("\n")
    # Парсим пронумерованные строки
    translated_texts = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # "1. Salom dunyo" → {1: "Salom dunyo"}
        if line[0].isdigit() and ". " in line:
            num, text = line.split(". ", 1)
            translated_texts[int(num)] = text.strip()

    for i, seg in enumerate(segments):
        uz_text = translated_texts.get(i + 1, seg["text"])
        translated.append({
            "start": seg["start"],
            "end":   seg["end"],
            "original": seg["text"],
            "translated": uz_text
        })
        print(f"  [{seg['start']:.1f}s] {seg['text'][:30]}... → {uz_text[:40]}...")

    print(f"✅ Перевод готов!")
    return translated


# ─────────────────────────────────────────────
# Шаг 4: Генерация аудио через Edge TTS
# ─────────────────────────────────────────────

async def generate_tts_segment(text: str, voice: str, output_path: str, rate: str = "-8%"):
    """Генерирует один аудио сегмент"""
    clean = normalize(text)
    tts = edge_tts.Communicate(text=clean, voice=voice, rate=rate)
    await tts.save(output_path)


async def generate_all_tts(segments: list, voice: str, temp_dir: str) -> list:
    """Генерирует аудио для всех сегментов"""
    print("🎤 Генерирую узбекский голос...")
    audio_segments = []

    for i, seg in enumerate(segments):
        out = os.path.join(temp_dir, f"seg_{i:04d}.mp3")
        await generate_tts_segment(seg["translated"], voice, out)
        audio_segments.append({
            **seg,
            "audio_file": out
        })
        print(f"  [{i+1}/{len(segments)}] ✅ {seg['translated'][:40]}...")

    print(f"✅ TTS готов!")
    return audio_segments


# ─────────────────────────────────────────────
# Шаг 5: Собрать финальное видео через ffmpeg
# ─────────────────────────────────────────────

def get_audio_duration(audio_file: str) -> float:
    """Возвращает длительность аудио файла"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    return 0.0


def create_dubbed_video(
    original_video: str,
    segments: list,
    output_path: str,
    temp_dir: str,
    original_volume: float = 0.15
) -> str:
    """
    Собирает финальное видео:
    - Оригинальный голос тихо на фоне (15%)
    - Узбекский голос поверх (100%)
    """
    print("🎬 Собираю финальное видео...")

    # Сначала соединяем все TTS сегменты в один аудио файл с паузами
    concat_list = os.path.join(temp_dir, "concat.txt")
    merged_tts  = os.path.join(temp_dir, "merged_tts.mp3")

    # Получаем длину оригинального видео
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", original_video],
        capture_output=True, text=True
    )
    video_duration = float(json.loads(probe.stdout)["format"]["duration"])

    # Строим filter_complex
    input_args = ["-i", original_video]
    for seg in segments:
        input_args += ["-i", seg["audio_file"]]

    n_segs = len(segments)

    # Оригинальный звук — тише
    filters = [f"[0:a]volume={original_volume}[orig]"]

    # Каждый TTS сегмент с задержкой по времени
    seg_labels = []
    for i, seg in enumerate(segments):
        delay_ms = int(seg["start"] * 1000)
        label = f"s{i}"
        filters.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=1.8[{label}]")
        seg_labels.append(f"[{label}]")

    # Смешиваем всё
    all_inputs = "[orig]" + "".join(seg_labels)
    filters.append(
        f"{all_inputs}amix=inputs={n_segs + 1}:normalize=0:dropout_transition=0[aout]"
    )

    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(video_duration),
        "-y",
        output_path
    ]

    print(f"  ffmpeg команда готова, запускаю...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg error: {result.stderr[-500:]}")

    print(f"✅ Видео готово: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────

async def dub_video(
    video_path: str,
    output_path: str,
    groq_api_key: str,
    voice: str = VOICE_FEMALE,
    src_language: str = None,
    original_volume: float = 0.1
) -> dict:
    """
    Полный пайплайн дублирования видео
    Returns: {"output": path, "segments": [...], "duration": float}
    """
    print(f"\n🎬 Начинаю дублирование: {video_path}")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as temp_dir:

        # Шаг 1: Извлечь аудио
        print("\n📢 Шаг 1/5: Извлекаю аудио...")
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(video_path, audio_path)

        # Шаг 2: Транскрипция
        print("\n📝 Шаг 2/5: Транскрибирую речь...")
        segments = transcribe(audio_path, language=src_language)

        if not segments:
            raise Exception("Речь не найдена в видео")

        # Шаг 3: Перевод
        print("\n🌐 Шаг 3/5: Перевожу на узбекский...")
        translated = translate_segments(segments, groq_api_key)

        # Шаг 4: TTS
        print("\n🎤 Шаг 4/5: Генерирую узбекский голос...")
        audio_segs = await generate_all_tts(translated, voice, temp_dir)

        # Шаг 5: Собрать видео
        print("\n🎬 Шаг 5/5: Собираю финальное видео...")
        create_dubbed_video(video_path, audio_segs, output_path, temp_dir, original_volume)

    print("\n" + "=" * 55)
    print(f"🎉 ГОТОВО! Видео сохранено: {output_path}")
    print("=" * 55)

    return {
        "output": output_path,
        "segments": translated,
        "segment_count": len(translated)
    }
