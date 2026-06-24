# -*- coding: utf-8 -*-
"""
AI Video Dubber v2
Видео → Whisper → Groq перевод → Edge TTS (синхронизированный) → ffmpeg
"""

import os
import asyncio
import tempfile
import subprocess
import json
import numpy as np
import wave

import whisper
import edge_tts
from groq import Groq
from text_normalizer import normalize

# ─────────────────────────────────────────────
VOICE_FEMALE = "uz-UZ-MadinaNeural"
VOICE_MALE   = "uz-UZ-SardorNeural"
ORIG_VOLUME  = 0.12   # оригинал тихо на фоне
TTS_VOLUME   = 2.2    # дублёр громко

# ─────────────────────────────────────────────
# Промпт — точный правильный перевод
# ─────────────────────────────────────────────
TRANSLATE_PROMPT = """Sen professional dublyaj tarjimonisin.

QOIDALAR (majburiy):
1. Matnni SO'ZMA-SO'Z aniq tarjima qil — hech narsa qo'shma, hech narsa o'chirma
2. Adabiy o'zbek tilidan foydalan — TV diktoridek, lekin tabiiy
3. Ko'cha sleng YO'Q: "bro", "zo'r-da", "bem sayil" kabi so'zlar ishlatma
4. Ismlar, joylar, raqamlarni to'g'ri yoz
5. Tinish belgilarini saqlagan holda tarjima qil
6. Faqat tarjima matnini yoz — HECH QANDAY izoh yoki tushuntirish yo'q
7. Har bir raqamli qatorni alohida tarjima qil

Misol:
Kirdi:  "1. Good morning everyone, today we discuss climate change."
Chiqdi: "1. Xayrli tong, bugun biz iqlim o'zgarishi haqida gaplashamiz."

Kirdi:  "2. Scientists have found new evidence."
Chiqdi: "2. Olimlar yangi dalillar topishdi."
"""


# ─────────────────────────────────────────────
# Шаг 1: Извлечь аудио
# ─────────────────────────────────────────────

def extract_audio(video_path: str, out_path: str) -> str:
    cmd = ["ffmpeg", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-y", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f"ffmpeg audio extract error: {r.stderr[-300:]}")
    return out_path


# ─────────────────────────────────────────────
# Шаг 2: Транскрипция Whisper
# ─────────────────────────────────────────────

def transcribe(audio_path: str, language: str = None) -> list:
    print("⏳ Загружаю Whisper...")
    model = whisper.load_model("base")
    opts = {"task": "transcribe"}
    if language:
        opts["language"] = language
    result = model.transcribe(audio_path, **opts)
    segs = []
    for s in result["segments"]:
        segs.append({
            "start":  s["start"],
            "end":    s["end"],
            "text":   s["text"].strip(),
            "gender": "male"
        })
        print(f"  [{s['start']:.1f}→{s['end']:.1f}s] {s['text'].strip()[:50]}")
    print(f"✅ {len(segs)} сегментов")
    return segs


# ─────────────────────────────────────────────
# Шаг 2b: Определение пола по частоте голоса
# ─────────────────────────────────────────────

def detect_gender_by_pitch(segments: list, audio_path: str) -> list:
    """
    Определяет пол по среднему pitch (F0) каждого сегмента.
    Мужской голос: 85-165 Hz
    Женский голос: 165-300 Hz
    """
    print("👥 Анализирую пол по голосу...")
    try:
        with wave.open(audio_path, 'rb') as wf:
            framerate = wf.getframerate()
            raw       = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        for seg in segments:
            s = int(seg["start"] * framerate)
            e = int(seg["end"]   * framerate)
            chunk = audio[s:e]

            if len(chunk) < framerate * 0.3:  # меньше 300ms — пропускаем
                continue

            # Автокорреляция для определения pitch
            chunk = chunk - chunk.mean()
            corr  = np.correlate(chunk, chunk, mode='full')
            corr  = corr[len(corr)//2:]

            # Ищем период в диапазоне голоса
            min_lag = int(framerate / 300)  # 300 Hz max
            max_lag = int(framerate / 80)   # 80 Hz min
            if max_lag >= len(corr):
                continue

            peak = np.argmax(corr[min_lag:max_lag]) + min_lag
            if peak == 0:
                continue

            pitch = framerate / peak

            # Порог: выше 165 Hz → женский
            gender = "female" if pitch > 165 else "male"
            seg["gender"] = gender
            icon = "👩" if gender == "female" else "👨"
            print(f"  {icon} [{seg['start']:.1f}s] {pitch:.0f}Hz → {gender}")

    except Exception as e:
        print(f"  ⚠️ Pitch анализ не удался: {e} — используем male")

    return segments


# ─────────────────────────────────────────────
# Шаг 3: Перевод через Groq
# ─────────────────────────────────────────────

def translate_segments(segments: list, groq_api_key: str) -> list:
    print("🌐 Перевожу на узбекский...")
    client    = Groq(api_key=groq_api_key)
    full_text = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(segments)])

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user",   "content": f"Tarjima qil:\n{full_text}"}
        ]
    )

    # Парсим ответ
    translated_map = {}
    for line in r.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            num, txt = line.split(". ", 1)
            try:
                translated_map[int(num)] = txt.strip()
            except:
                pass

    result = []
    for i, seg in enumerate(segments):
        uz = translated_map.get(i + 1, seg["text"])
        result.append({
            "start":      seg["start"],
            "end":        seg["end"],
            "original":   seg["text"],
            "translated": uz,
            "gender":     seg.get("gender", "male"),
            "duration":   seg["end"] - seg["start"],
        })
        icon = "👩" if seg.get("gender") == "female" else "👨"
        print(f"  {icon} {seg['text'][:30]} → {uz[:40]}")

    print("✅ Перевод готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 4: TTS с синхронизацией по времени
# ─────────────────────────────────────────────

def get_audio_duration_ffprobe(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True
    )
    try:
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if "duration" in s:
                return float(s["duration"])
    except:
        pass
    return 1.0


async def generate_tts_synced(seg: dict, out_path: str):
    """
    Генерирует TTS и подстраивает скорость под длину оригинала.
    Если TTS длиннее оригинала — ускоряем.
    Если короче — оставляем как есть (пауза естественна).
    """
    voice = VOICE_FEMALE if seg["gender"] == "female" else VOICE_MALE
    text  = normalize(seg["translated"])
    orig_dur = seg["duration"]

    # Сначала генерируем с нормальной скоростью
    tmp_path = out_path + "_tmp.mp3"
    tts = edge_tts.Communicate(text=text, voice=voice, rate="-5%")
    await tts.save(tmp_path)

    tts_dur = get_audio_duration_ffprobe(tmp_path)

    if tts_dur <= 0:
        os.rename(tmp_path, out_path)
        return

    # Вычисляем нужный коэффициент скорости
    ratio = tts_dur / orig_dur

    if ratio > 1.15:
        # TTS длиннее оригинала → ускоряем через ffmpeg atempo
        # atempo принимает значения 0.5–2.0
        tempo = min(ratio, 2.0)
        tempo = max(tempo, 0.5)
        cmd = [
            "ffmpeg", "-i", tmp_path,
            "-filter:a", f"atempo={tempo:.3f}",
            "-y", out_path
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            os.rename(tmp_path, out_path)
        else:
            os.remove(tmp_path)
    else:
        # TTS короче или совпадает — оставляем как есть
        os.rename(tmp_path, out_path)


async def generate_all_tts(segments: list, temp_dir: str) -> list:
    print("🎤 Генерирую голос (синхронизирую с оригиналом)...")
    result = []
    for i, seg in enumerate(segments):
        out = os.path.join(temp_dir, f"seg_{i:04d}.mp3")
        await generate_tts_synced(seg, out)
        icon = "👩" if seg["gender"] == "female" else "👨"
        print(f"  [{i+1}/{len(segments)}] {icon} {seg['translated'][:45]}")
        result.append({**seg, "audio_file": out})
    print("✅ TTS готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 5: Сборка видео через ffmpeg
# ─────────────────────────────────────────────

def create_dubbed_video(original_video: str, segments: list,
                        output_path: str, temp_dir: str) -> str:
    print("🎬 Собираю финальное видео...")

    # Длительность оригинала
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", original_video],
        capture_output=True, text=True
    )
    video_dur = float(json.loads(probe.stdout)["format"]["duration"])

    # ffmpeg inputs
    inputs = ["-i", original_video]
    for seg in segments:
        inputs += ["-i", seg["audio_file"]]

    n = len(segments)

    # Filters
    filters = [f"[0:a]volume={ORIG_VOLUME}[orig]"]
    labels  = []
    for i, seg in enumerate(segments):
        delay = int(seg["start"] * 1000)
        lbl   = f"s{i}"
        filters.append(
            f"[{i+1}:a]adelay={delay}|{delay},volume={TTS_VOLUME}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    all_in = "[orig]" + "".join(labels)
    filters.append(
        f"{all_in}amix=inputs={n+1}:normalize=0:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(video_dur),
        "-y", output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f"ffmpeg: {r.stderr[-400:]}")

    print(f"✅ Видео: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────

async def dub_video(video_path: str, output_path: str, groq_api_key: str,
                    voice: str = VOICE_MALE, src_language: str = None,
                    original_volume: float = ORIG_VOLUME) -> dict:
    print(f"\n🎬 Дублирую: {video_path}")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:

        print("\n📢 1/5 Извлекаю аудио...")
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        print("\n📝 2/5 Транскрибирую (Whisper)...")
        segs = transcribe(audio, language=src_language)
        if not segs:
            raise Exception("Речь не найдена")

        print("\n👥 2b Определяю пол спикеров...")
        segs = detect_gender_by_pitch(segs, audio)

        print("\n🌐 3/5 Перевожу (Groq)...")
        translated = translate_segments(segs, groq_api_key)

        print("\n🎤 4/5 Генерирую голос (TTS)...")
        audio_segs = await generate_all_tts(translated, tmp)

        print("\n🎬 5/5 Собираю видео (ffmpeg)...")
        create_dubbed_video(video_path, audio_segs, output_path, tmp)

    print(f"\n🎉 ГОТОВО: {output_path}")
    return {"output": output_path, "segments": translated, "segment_count": len(translated)}
