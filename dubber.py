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

TTS_RATE   = "-5%"    # немного быстрее чем раньше (-8%)
TTS_VOLUME = 2.2      # громче чем раньше (было 1.8)

# ─────────────────────────────────────────────
# Промпт перевода — точный, правильный, живой
# ─────────────────────────────────────────────
GROQ_SYSTEM_PROMPT = """Sen professional dublyaj tarjimonisin. Vazifang — matnni to'g'ri va aniq o'zbek tiliga tarjima qilish.

Qoidalar:
1. Matnni SO'ZMA-SO'Z to'g'ri tarjima qil — ma'noni o'zgartirma
2. Adabiy o'zbek tilidan foydalan — televideniye diktoridek
3. Hech qanday ko'cha sleng ishlatma (masalan "bro", "zo'r-da", "bem sayil" kabi so'zlar YO'Q)
4. Raqamlar, ismlar, joy nomlarini to'g'ri talaffuz shaklida yoz
5. Tinish belgilarini saqlagan holda tarjima qil
6. Faqat tarjima matnini yoz — izoh, tushuntirish MUTLAQO YO'Q
7. Har bir raqamli qatorni alohida tarjima qil, raqamni saqlagan holda

Misol:
Kirdi: "1. Hello everyone, welcome to our channel!"
Chiqdi: "1. Salom hammaga, kanalimizga xush kelibsiz!"

Kirdi: "2. Today we will talk about science."
Chiqdi: "2. Bugun biz fan haqida gaplashamiz."
"""

# Промпт для определения пола спикера
GENDER_PROMPT = """Quyidagi matn parchalarini tahlil qil va har bir segment uchun spikerni jinsi (erkak yoki ayol) ni aniqlash. 

Faqat JSON formatda javob ber:
{"segments": [{"index": 1, "gender": "male"}, {"index": 2, "gender": "female"}, ...]}

Agar aniqlab bo'lmasa — "male" deb yoz.
Matn:
"""


# ─────────────────────────────────────────────
# Шаг 1: Извлечь аудио из видео
# ─────────────────────────────────────────────

def extract_audio(video_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
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
    print("⏳ Загружаю Whisper модель...")
    model = whisper.load_model("base")
    print("🎙️  Транскрибирую...")

    options = {"task": "transcribe"}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start":  seg["start"],
            "end":    seg["end"],
            "text":   seg["text"].strip(),
            "gender": "male"   # default, потом определим
        })
        print(f"  [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text'].strip()}")

    print(f"✅ Транскрипция готова: {len(segments)} сегментов")
    return segments


# ─────────────────────────────────────────────
# Шаг 2b: Определение пола спикера через Groq
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Шаг 2b: Определение пола по АУДИО (не тексту)
# ─────────────────────────────────────────────

def detect_gender_by_audio(segments: list, audio_path: str) -> list:
    """
    Определяет пол спикера по частоте голоса (pitch).
    Мужской голос: 85-180 Hz
    Женский голос: 165-255 Hz
    """
    print("👥 Определяю пол по голосу (аудио анализ)...")

    try:
        import numpy as np
        import wave
        import struct

        # Читаем WAV файл
        with wave.open(audio_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw_data   = wf.readframes(n_frames)

        # Конвертируем в numpy array
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        else:
            audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128

        if n_channels > 1:
            audio_data = audio_data[::n_channels]

        # Для каждого сегмента определяем среднюю частоту
        for seg in segments:
            start_sample = int(seg["start"] * framerate)
            end_sample   = int(seg["end"]   * framerate)
            chunk = audio_data[start_sample:end_sample]

            if len(chunk) < 100:
                seg["gender"] = "male"
                continue

            # FFT для определения доминантной частоты
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / framerate)

            # Ищем пик в диапазоне голоса 80-300 Hz
            voice_mask = (freqs >= 80) & (freqs <= 300)
            if not np.any(voice_mask):
                seg["gender"] = "male"
                continue

            voice_fft   = fft[voice_mask]
            voice_freqs = freqs[voice_mask]
            peak_freq   = voice_freqs[np.argmax(voice_fft)]

            # Женский голос выше 160 Hz
            gender = "female" if peak_freq > 160 else "male"
            seg["gender"] = gender
            icon = "👩" if gender == "female" else "👨"
            print(f"  {icon} [{seg['start']:.1f}s] {peak_freq:.0f}Hz → {gender} — {seg['text'][:35]}")

    except Exception as e:
        print(f"  ⚠️ Аудио анализ не удался: {e}")
        # Фолбэк: определяем по тексту через Groq
        for seg in segments:
            seg["gender"] = "male"

    return segments


# ─────────────────────────────────────────────
# Шаг 3: Перевод через Groq
# ─────────────────────────────────────────────

def translate_segments(segments: list, groq_api_key: str) -> list:
    client = Groq(api_key=groq_api_key)

    print("🌐 Перевожу на узбекский...")
    full_text = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(segments)])

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Tarjima qil:\n{full_text}"}
        ]
    )

    lines = r.choices[0].message.content.strip().split("\n")
    translated_texts = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and ". " in line:
            num, text = line.split(". ", 1)
            try:
                translated_texts[int(num)] = text.strip()
            except:
                pass

    translated = []
    for i, seg in enumerate(segments):
        uz_text = translated_texts.get(i + 1, seg["text"])
        translated.append({
            "start":      seg["start"],
            "end":        seg["end"],
            "original":   seg["text"],
            "translated": uz_text,
            "gender":     seg.get("gender", "male")
        })
        print(f"  [{seg['start']:.1f}s] {seg['text'][:35]} → {uz_text[:40]}")

    print("✅ Перевод готов!")
    return translated


# ─────────────────────────────────────────────
# Шаг 4: TTS — голос по полу спикера
# ─────────────────────────────────────────────

async def generate_tts_segment(text: str, voice: str, output_path: str):
    clean = normalize(text)
    tts = edge_tts.Communicate(text=clean, voice=voice, rate=TTS_RATE)
    await tts.save(output_path)


async def generate_all_tts(segments: list, default_voice: str, temp_dir: str) -> list:
    print("🎤 Генерирую узбекский голос...")
    audio_segments = []

    for i, seg in enumerate(segments):
        out = os.path.join(temp_dir, f"seg_{i:04d}.mp3")

        # Выбираем голос по полу спикера
        gender = seg.get("gender", "male")
        if gender == "female":
            voice = VOICE_FEMALE
        else:
            voice = VOICE_MALE

        await generate_tts_segment(seg["translated"], voice, out)
        icon = "👩" if gender == "female" else "👨"
        audio_segments.append({**seg, "audio_file": out})
        print(f"  [{i+1}/{len(segments)}] {icon} {seg['translated'][:45]}...")

    print("✅ TTS готов!")
    return audio_segments


# ─────────────────────────────────────────────
# Шаг 5: Собрать финальное видео через ffmpeg
# ─────────────────────────────────────────────

def create_dubbed_video(
    original_video: str,
    segments: list,
    output_path: str,
    temp_dir: str,
    original_volume: float = 0.15
) -> str:
    print("🎬 Собираю финальное видео...")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", original_video],
        capture_output=True, text=True
    )
    video_duration = float(json.loads(probe.stdout)["format"]["duration"])

    input_args = ["-i", original_video]
    for seg in segments:
        input_args += ["-i", seg["audio_file"]]

    n_segs = len(segments)
    filters = [f"[0:a]volume={original_volume}[orig]"]

    seg_labels = []
    for i, seg in enumerate(segments):
        delay_ms = int(seg["start"] * 1000)
        label = f"s{i}"
        filters.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume={TTS_VOLUME}[{label}]")
        seg_labels.append(f"[{label}]")

    all_inputs = "[orig]" + "".join(seg_labels)
    filters.append(
        f"{all_inputs}amix=inputs={n_segs+1}:normalize=0:dropout_transition=0[aout]"
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
    voice: str = VOICE_MALE,
    src_language: str = None,
    original_volume: float = 0.15
) -> dict:
    print(f"\n🎬 Начинаю дублирование: {video_path}")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as temp_dir:

        # 1. Извлечь аудио
        print("\n📢 Шаг 1/5: Извлекаю аудио...")
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(video_path, audio_path)

        # 2. Транскрипция
        print("\n📝 Шаг 2/5: Транскрибирую речь...")
        segments = transcribe(audio_path, language=src_language)
        if not segments:
            raise Exception("Речь не найдена в видео")

        # 2b. Определение пола по аудио
        segments = detect_gender_by_audio(segments, audio_path)

        # 3. Перевод
        print("\n🌐 Шаг 3/5: Перевожу на узбекский...")
        translated = translate_segments(segments, groq_api_key)

        # 4. TTS
        print("\n🎤 Шаг 4/5: Генерирую узбекский голос...")
        audio_segs = await generate_all_tts(translated, voice, temp_dir)

        # 5. Собрать видео
        print("\n🎬 Шаг 5/5: Собираю финальное видео...")
        create_dubbed_video(video_path, audio_segs, output_path, temp_dir, original_volume)

    print("\n" + "=" * 55)
    print(f"🎉 ГОТОВО! Видео: {output_path}")
    print("=" * 55)

    return {
        "output":        output_path,
        "segments":      translated,
        "segment_count": len(translated)
    }
