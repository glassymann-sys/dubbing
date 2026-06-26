# -*- coding: utf-8 -*-
"""
AI Video Dubber v7
1. AssemblyAI → транскрипция + временные метки
2. Пользователь редактирует перевод в UI
3. Gemini TTS генерирует голос для каждого сегмента
4. ffmpeg собирает финальное видео
"""

import os
import re
import asyncio
import tempfile
import subprocess
import json
import wave
import time

import assemblyai as aai
from google import genai
from google.genai import types
from text_normalizer import normalize

# ─────────────────────────────────────────────
VOICE_FEMALE = "Leda"
VOICE_MALE   = "Puck"
ORIG_VOLUME  = 0.08
TTS_VOLUME   = 2.2
GEMINI_TTS   = "gemini-3.1-flash-tts-preview"
GEMINI_MODEL = "gemini-2.5-flash"


# ─────────────────────────────────────────────
# Шаг 1: Извлечь аудио
# ─────────────────────────────────────────────

def extract_audio(video_path: str, out_path: str) -> str:
    cmd = ["ffmpeg", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-y", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f"ffmpeg: {r.stderr[-200:]}")
    return out_path


# ─────────────────────────────────────────────
# Шаг 2: Транскрипция
# ─────────────────────────────────────────────

def transcribe(audio_path: str, api_key: str, language: str = None) -> list:
    print("🎙️  AssemblyAI транскрибирует...")
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=not bool(language),
        language_code=language if language else None,
    )

    t = aai.Transcriber().transcribe(audio_path, config=config)
    if t.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI: {t.error}")

    segments = []

    # Если мало сегментов — разбиваем по предложениям
    utterances = t.utterances or []
    if len(utterances) <= 2 and utterances:
        for utt in utterances:
            sents    = re.split(r'(?<=[.!?,])\s+', utt.text.strip())
            dur      = (utt.end - utt.start) / 1000.0
            per      = dur / max(len(sents), 1)
            for j, s in enumerate(sents):
                s = s.strip()
                if not s: continue
                segments.append({
                    "start":   round(utt.start / 1000.0 + j * per, 3),
                    "end":     round(utt.start / 1000.0 + (j+1) * per, 3),
                    "text":    s,
                    "speaker": utt.speaker,
                    "gender":  "male",
                    "translated": ""
                })
    else:
        for utt in utterances:
            segments.append({
                "start":   round(utt.start / 1000.0, 3),
                "end":     round(utt.end   / 1000.0, 3),
                "text":    utt.text.strip(),
                "speaker": utt.speaker,
                "gender":  "male",
                "translated": ""
            })

    print(f"✅ {len(segments)} сегментов")
    return segments


# ─────────────────────────────────────────────
# Шаг 2b: Определить пол + авто-перевод через Gemini
# ─────────────────────────────────────────────

def detect_and_translate(segments: list, groq_api_key: str) -> list:
    """Gemini определяет пол И переводит за один раз"""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return segments

    client   = genai.Client(api_key=gemini_key)
    dialog   = "\n".join([f"Speaker {s['speaker']}: {s['text']}" for s in segments])
    numbered = "\n".join([f"{i+1}. [{s['speaker']}] {s['text']}" for i, s in enumerate(segments)])
    speakers = list(set(s['speaker'] for s in segments))

    prompt = f"""Vazifang ikkita:

1. Har bir spiker jinsi: {speakers} uchun JSON yoz {{"A": "male", "B": "female"}}

2. Har bir gapni SO'ZMA-SO'Z o'zbek tiliga tarjima qil:
- Hech narsa qo'shma, hech narsa o'chirma
- O'zbek tiliga ekvivalenti bo'lmagan so'zlarni aslida qoldur
- Faqat tarjima, izoh yo'q

Dialog:
{dialog}

Tarjima qil:
{numbered}

Javob formati:
GENDERS: {{"A": "male", "B": "female"}}
1. [tarjima]
2. [tarjima]
..."""

    try:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = r.text.strip()

        # Парсим пол
        genders = {}
        gender_match = re.search(r'GENDERS:\s*(\{[^}]+\})', text)
        if gender_match:
            try:
                genders = json.loads(gender_match.group(1))
            except:
                pass

        # Парсим переводы
        tmap = {}
        for line in text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                try:
                    num, txt = line.split(". ", 1)
                    # Убираем [speaker] если есть
                    txt = re.sub(r'^\[[A-Z]\]\s*', '', txt).strip()
                    tmap[int(num)] = txt
                except:
                    pass

        for i, seg in enumerate(segments):
            seg["gender"]     = genders.get(seg["speaker"], "male")
            seg["translated"] = tmap.get(i + 1, seg["text"])
            icon = "👩" if seg["gender"] == "female" else "👨"
            print(f"  {icon} {seg['text'][:30]:30} → {seg['translated'][:35]}")

    except Exception as e:
        print(f"  ⚠️ Gemini: {e}")
        # Фолбэк — оставляем оригинальный текст
        for seg in segments:
            seg["translated"] = seg["text"]

    return segments


# ─────────────────────────────────────────────
# Шаг 3: Gemini TTS — генерация голоса
# ─────────────────────────────────────────────

def gemini_tts_sync(text: str, voice: str, gemini_key: str) -> bytes:
    """Синхронная генерация одного сегмента"""
    client = genai.Client(api_key=gemini_key)

    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=GEMINI_TTS,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )
            return r.candidates[0].content.parts[0].inline_data.data

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 7 * (attempt + 1)
                print(f"  ⏳ Rate limit — жду {wait}с...")
                time.sleep(wait)
            else:
                raise e

    raise Exception("Gemini TTS лимит исчерпан")


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True
    )
    try:
        for s in json.loads(r.stdout).get("streams", []):
            if "duration" in s:
                return float(s["duration"])
    except:
        pass
    return 1.0


def fit_to_duration(in_path: str, out_path: str, target: float):
    """Подгоняет длину аудио под оригинал"""
    tts_dur = get_duration(in_path)
    if tts_dur <= 0 or target <= 0:
        try: os.rename(in_path, out_path)
        except: pass
        return

    ratio = tts_dur / target

    if 0.9 <= ratio <= 1.1:
        try: os.rename(in_path, out_path)
        except: pass
        return

    # Применяем atempo (0.5-2.0)
    if ratio > 2.0:
        # Двойной atempo
        mid = in_path + "_mid.wav"
        subprocess.run(["ffmpeg", "-i", in_path, "-filter:a", "atempo=2.0", "-y", mid],
                       capture_output=True)
        subprocess.run(["ffmpeg", "-i", mid, "-filter:a", f"atempo={ratio/2.0:.3f}", "-y", out_path],
                       capture_output=True)
        try: os.remove(in_path); os.remove(mid)
        except: pass
    elif ratio < 0.5:
        mid = in_path + "_mid.wav"
        subprocess.run(["ffmpeg", "-i", in_path, "-filter:a", "atempo=0.5", "-y", mid],
                       capture_output=True)
        subprocess.run(["ffmpeg", "-i", mid, "-filter:a", f"atempo={ratio/0.5:.3f}", "-y", out_path],
                       capture_output=True)
        try: os.remove(in_path); os.remove(mid)
        except: pass
    else:
        tempo = max(0.5, min(ratio, 2.0))
        subprocess.run(["ffmpeg", "-i", in_path, "-filter:a", f"atempo={tempo:.3f}", "-y", out_path],
                       capture_output=True)
        try: os.remove(in_path)
        except: pass


async def generate_tts_all(segments: list, temp_dir: str) -> list:
    print("🎤 Генерирую голос (Gemini TTS)...")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise Exception("GEMINI_API_KEY не найден!")

    result = []
    for i, seg in enumerate(segments):
        text = normalize(seg.get("translated") or seg["text"])
        if not text.strip():
            continue

        voice    = VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE
        raw_path = os.path.join(temp_dir, f"seg_{i:04d}_raw.wav")
        out_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")

        try:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, gemini_tts_sync, text, voice, gemini_key
            )
            with wave.open(raw_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)

            # Подгоняем под длину оригинала
            fit_to_duration(raw_path, out_path, seg["duration"])

            icon = "👩" if seg.get("gender") == "female" else "👨"
            print(f"  [{i+1}/{len(segments)}] {icon} {voice}: {text[:45]}")
            result.append({**seg, "audio_file": out_path})

            # Пауза 6 сек между запросами
            if i < len(segments) - 1:
                await asyncio.sleep(6)

        except Exception as e:
            print(f"  ❌ Сегмент {i}: {e}")

    print(f"✅ TTS готов! {len(result)}/{len(segments)} сегментов")
    return result


# ─────────────────────────────────────────────
# Шаг 4: Сборка видео
# ─────────────────────────────────────────────

def create_dubbed_video(original_video: str, segments: list,
                        output_path: str) -> str:
    print("🎬 Собираю видео...")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", original_video],
        capture_output=True, text=True
    )
    video_dur = float(json.loads(probe.stdout)["format"]["duration"])

    # Только сегменты у которых есть аудио файл
    valid = [s for s in segments if os.path.exists(s.get("audio_file", ""))]
    if not valid:
        raise Exception("Нет аудио сегментов!")

    inputs  = ["-i", original_video]
    filters = [f"[0:a]volume={ORIG_VOLUME}[orig]"]
    labels  = []

    for idx, seg in enumerate(valid):
        inputs += ["-i", seg["audio_file"]]
        delay   = int(seg["start"] * 1000)
        lbl     = f"s{idx}"
        n_input = idx + 1
        filters.append(f"[{n_input}:a]adelay={delay}|{delay},volume={TTS_VOLUME}[{lbl}]")
        labels.append(f"[{lbl}]")

    all_in = "[orig]" + "".join(labels)
    filters.append(f"{all_in}amix=inputs={len(labels)+1}:normalize=0:dropout_transition=0[aout]")

    cmd = [
        "ffmpeg", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(video_dur), "-y", output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f"ffmpeg: {r.stderr[-300:]}")

    print(f"✅ Видео готово: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Главные функции
# ─────────────────────────────────────────────

async def transcribe_video(video_path: str, groq_api_key: str,
                            src_language: str = None) -> list:
    """Шаг 1: Транскрипция — возвращает сегменты для редактирования в UI"""
    assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not assemblyai_key:
        raise Exception("ASSEMBLYAI_API_KEY не найден!")

    with tempfile.TemporaryDirectory() as tmp:
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        lang = src_language if src_language and src_language != "auto" else None
        segs = transcribe(audio, assemblyai_key, lang)
        if not segs:
            raise Exception("Речь не найдена")

        # Авто-определение пола + авто-перевод
        print("🌐 Авто-перевод через Gemini...")
        segs = detect_and_translate(segs, groq_api_key)

    return segs


async def dub_video(video_path: str, output_path: str, segments: list,
                    groq_api_key: str = "", **kwargs) -> dict:
    """Шаг 2: Дублирование с готовыми переводами"""
    print(f"\n🎬 Дублирую: {video_path}")

    # Добавляем duration если нет
    for seg in segments:
        if "duration" not in seg:
            seg["duration"] = seg["end"] - seg["start"]

    with tempfile.TemporaryDirectory() as tmp:
        print("🎤 Генерирую голос...")
        audio_segs = await generate_tts_all(segments, tmp)

        print("🎬 Собираю видео...")
        create_dubbed_video(video_path, audio_segs, output_path)

    return {
        "output":        output_path,
        "segments":      segments,
        "segment_count": len(segments)
    }
