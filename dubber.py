# -*- coding: utf-8 -*-
"""
AI Video Dubber v3
Видео → AssemblyAI (транскрипция + диаризация) → Groq перевод → Edge TTS → ffmpeg
"""

import os
import asyncio
import tempfile
import subprocess
import json
import numpy as np
import wave

import assemblyai as aai
import edge_tts
from groq import Groq
from text_normalizer import normalize
from prosody import load_audio, analyze_segment_prosody, apply_prosody

# ─────────────────────────────────────────────
VOICE_FEMALE = "uz-UZ-MadinaNeural"
VOICE_MALE   = "uz-UZ-SardorNeural"
ORIG_VOLUME  = 0.12
TTS_VOLUME   = 2.2
TTS_RATE = "+5%"  # чуть быстрее

# ─────────────────────────────────────────────
# Промпт перевода
# ─────────────────────────────────────────────
TRANSLATE_PROMPT = """Sen professional dublyaj tarjimonisin. Har qanday matnni aniq va to'g'ri o'zbek tiliga tarjima qilish.

MUHIM QOIDALAR:
1. Matnni SO'ZMA-SO'Z tarjima qil — ma'noni o'zgartirma, hech narsa qo'shma
2. Adabiy o'zbek tilidan foydalan — TV diktoridek ravon va tabiiy
3. Har qanday so'zni ham to'g'ri tarjima qil, senzura qilma
4. Savol bo'lsa — savol shaklida, his-tuyg'u bo'lsa — his bilan tarjima qil
5. Faqat tarjima matnini yoz — HECH QANDAY izoh yo'q
6. Har bir raqamli qatorni raqami bilan yoz
7. MUHIM: Agar so'zning o'zbek tilida to'liq ekvivalenti bo'lmasa — 
   o'sha so'zni ASLIDA qoldur. Lekin agar to'liq tarjimasi mavjud bo'lsa — 
   albatta tarjima qil. Masalan:
   - "gay" → o'zbek tilida to'liq ekvivalenti yo'q → "gay" qoldir
   - "subscribe" → "obuna bo'lish" bor → "obuna bo'ling" deb tarjima qil
   - "like" (ijtimoiy tarmoq) → "layk" deb qoldir
   - "like" (yoqtirmoq) → "yoqtirmoq" deb tarjima qil
   Agar ikkilansang — tarjima qil

MISOL:
Kirdi:  "1. Would you go on a date with me?"
Chiqdi: "1. Men bilan uchrashuvga borarmidingiz?"

Kirdi:  "2. That's so gay, no way!"
Chiqdi: "2. Bu juda gay, yo'q!"

Kirdi:  "3. Subscribe to my channel!"
Chiqdi: "3. Kanalimga obuna bo'ling!"
"""


# ─────────────────────────────────────────────
# Шаг 1: Извлечь аудио из видео
# ─────────────────────────────────────────────

def extract_audio(video_path: str, out_path: str) -> str:
    cmd = ["ffmpeg", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-y", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f"ffmpeg error: {r.stderr[-300:]}")
    return out_path


# ─────────────────────────────────────────────
# Шаг 2: AssemblyAI — транскрипция + диаризация
# ─────────────────────────────────────────────

def transcribe_with_diarization(audio_path: str, api_key: str,
                                 language: str = None) -> list:
    """
    AssemblyAI транскрибирует речь И определяет спикеров (A, B, C...).
    Возвращает сегменты с speaker label.
    """
    print("🎙️  AssemblyAI: транскрипция + определение спикеров...")

    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,        # ← диаризация спикеров!
        speakers_expected=2,        # ожидаем 2 спикера
        language_detection=True if not language else False,
        language_code=language if language else None,
    )

    transcriber = aai.Transcriber()
    transcript  = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI error: {transcript.error}")

    # Собираем сегменты с метками спикеров
    segments = []
    for utt in transcript.utterances:
        segments.append({
            "start":   utt.start / 1000.0,   # ms → sec
            "end":     utt.end   / 1000.0,
            "text":    utt.text.strip(),
            "speaker": utt.speaker,           # "A", "B", "C"...
            "gender":  "male"                 # определим ниже
        })
        print(f"  Speaker {utt.speaker} [{utt.start/1000:.1f}→{utt.end/1000:.1f}s]: {utt.text[:50]}")

    print(f"✅ {len(segments)} сегментов, спикеры: {set(s['speaker'] for s in segments)}")
    return segments


# ─────────────────────────────────────────────
# Шаг 2b: Определяем пол каждого спикера
# по частоте голоса (pitch)
# ─────────────────────────────────────────────

def detect_speaker_genders_ai(segments: list, groq_api_key: str) -> dict:
    """
    Groq анализирует весь разговор и определяет пол каждого спикера.
    Смотрит на контекст, имена, местоимения, стиль речи.
    """
    print("👥 Groq определяет пол спикеров по контексту разговора...")

    client = Groq(api_key=groq_api_key)

    # Строим полный диалог с метками спикеров
    dialog = "\n".join([
        f"Speaker {s['speaker']}: {s['text']}"
        for s in segments
    ])

    speakers = list(set(s['speaker'] for s in segments))

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at detecting speaker gender from conversation context.
Analyze the conversation and determine the gender of each speaker.
Look for: pronouns (he/she/his/her/I/me), names, topics discussed, speaking style.
Reply ONLY with valid JSON: {"A": "male", "B": "female"}
If unsure, make your best guess based on context."""
                },
                {
                    "role": "user",
                    "content": f"Determine gender for each speaker in this conversation:\n\n{dialog}\n\nSpeakers to identify: {speakers}"
                }
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(r.choices[0].message.content)
        # Нормализуем ответ
        genders = {}
        for spk in speakers:
            gender = result.get(spk, "male").lower()
            if "female" in gender or "woman" in gender or "girl" in gender:
                genders[spk] = "female"
            else:
                genders[spk] = "male"
            icon = "👩" if genders[spk] == "female" else "👨"
            print(f"  Speaker {spk} → {icon} {genders[spk]}")

        return genders

    except Exception as e:
        print(f"  ⚠️ Groq gender detection failed: {e}")
        # Фолбэк: первый спикер мужчина, второй женщина
        genders = {}
        for i, spk in enumerate(sorted(speakers)):
            genders[spk] = "female" if i % 2 == 1 else "male"
        return genders


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
            "speaker":    seg.get("speaker", "A"),
            "gender":     seg.get("gender", "male"),
            "duration":   seg["end"] - seg["start"],
        })
        icon = "👩" if seg.get("gender") == "female" else "👨"
        print(f"  {icon} Spk{seg.get('speaker','A')}: {seg['text'][:30]} → {uz[:40]}")

    print("✅ Перевод готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 4: TTS с правильным голосом
# ─────────────────────────────────────────────

async def generate_tts_segment(seg: dict, out_path: str):
    voice = VOICE_FEMALE if seg["gender"] == "female" else VOICE_MALE
    text  = normalize(seg["translated"])
    tts   = edge_tts.Communicate(text=text, voice=voice, rate=TTS_RATE)
    await tts.save(out_path)


def get_duration(path: str) -> float:
    """Длительность аудио файла через ffprobe"""
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


def fit_audio_to_duration(in_path: str, out_path: str, target_dur: float):
    """
    Подгоняет аудио под нужную длительность через atempo.
    Если TTS длиннее оригинала — ускоряем чтобы вписаться.
    Если короче — оставляем (пауза естественна).
    """
    tts_dur = get_duration(in_path)
    if tts_dur <= 0:
        os.rename(in_path, out_path)
        return

    ratio = tts_dur / target_dur

    if ratio > 1.1:
        # TTS длиннее оригинала → ускоряем но не более чем 1.8x
        tempo = min(ratio, 1.8)
        tempo = max(tempo, 0.8)
        cmd = [
            "ffmpeg", "-i", in_path,
            "-filter:a", f"atempo={tempo:.3f}",
            "-y", out_path
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            try: os.remove(in_path)
            except: pass
            return
    # Оставляем как есть
    try:
        os.rename(in_path, out_path)
    except:
        pass


async def generate_all_tts(segments: list, temp_dir: str,
                           orig_audio_path: str = None) -> list:
    print("🎤 Генерирую голос (с переносом интонации)...")

    orig_data, orig_sr = None, 16000
    if orig_audio_path and os.path.exists(orig_audio_path):
        try:
            orig_data, orig_sr = load_audio(orig_audio_path)
            print("  ✅ Оригинал загружен для анализа интонации")
        except Exception as e:
            print(f"  ⚠️ Не удалось загрузить оригинал: {e}")

    result = []
    for i, seg in enumerate(segments):
        raw_path = os.path.join(temp_dir, f"seg_{i:04d}_raw.mp3")
        out_path = os.path.join(temp_dir, f"seg_{i:04d}.mp3")

        voice = VOICE_FEMALE if seg["gender"] == "female" else VOICE_MALE
        text  = normalize(seg["translated"])

        # Мужской голос — делаем чуть ниже через pitch
        if seg["gender"] == "male":
            tts = edge_tts.Communicate(text=text, voice=voice,
                                        rate=TTS_RATE, pitch="-4Hz")
        else:
            tts = edge_tts.Communicate(text=text, voice=voice,
                                        rate=TTS_RATE, pitch="+0Hz")
        await tts.save(raw_path)

        if orig_data is not None:
            try:
                orig_stats = analyze_segment_prosody(
                    orig_data, orig_sr, seg["start"], seg["end"]
                )
                apply_prosody(raw_path, out_path, orig_stats)
            except Exception as e:
                print(f"  ⚠️ Просодия не применена: {e}")
                try: os.rename(raw_path, out_path)
                except: pass
        else:
            try: os.rename(raw_path, out_path)
            except: pass

        icon = "👩" if seg["gender"] == "female" else "👨"
        print(f"  [{i+1}/{len(segments)}] {icon} {seg['translated'][:45]}")
        result.append({**seg, "audio_file": out_path})

    print("✅ TTS готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 5: Сборка видео
# ─────────────────────────────────────────────

def create_dubbed_video(original_video: str, segments: list,
                        output_path: str) -> str:
    print("🎬 Собираю финальное видео...")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", original_video],
        capture_output=True, text=True
    )
    video_dur = float(json.loads(probe.stdout)["format"]["duration"])

    inputs  = ["-i", original_video]
    filters = [f"[0:a]volume={ORIG_VOLUME}[orig]"]
    labels  = []

    for i, seg in enumerate(segments):
        inputs += ["-i", seg["audio_file"]]
        # +150ms задержка чтобы голос не опережал видео
        delay = int(seg["start"] * 1000) + 150
        lbl   = f"s{i}"
        filters.append(
            f"[{i+1}:a]adelay={delay}|{delay},volume={TTS_VOLUME}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    all_in = "[orig]" + "".join(labels)
    filters.append(
        f"{all_in}amix=inputs={len(segments)+1}:normalize=0:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(video_dur), "-y", output_path
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

    assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not assemblyai_key:
        raise Exception("ASSEMBLYAI_API_KEY не найден! Запусти: export ASSEMBLYAI_API_KEY='твой_ключ'")

    print(f"\n🎬 Дублирую: {video_path}")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:

        # 1. Аудио
        print("\n📢 1/5 Извлекаю аудио...")
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        # 2. Транскрипция + диаризация
        print("\n📝 2/5 AssemblyAI: транскрипция + спикеры...")
        lang = src_language if src_language and src_language != "auto" else None
        segs = transcribe_with_diarization(audio, assemblyai_key, lang)
        if not segs:
            raise Exception("Речь не найдена")

        # 2b. Пол каждого спикера через Groq AI
        print("\n👥 Groq определяет пол спикеров...")
        speaker_genders = detect_speaker_genders_ai(segs, groq_api_key)
        for seg in segs:
            seg["gender"] = speaker_genders.get(seg["speaker"], "male")

        # 3. Перевод
        print("\n🌐 3/5 Перевожу (Groq)...")
        translated = translate_segments(segs, groq_api_key)

        # 4. TTS с переносом интонации
        print("\n🎤 4/5 Генерирую голос (интонация оригинала)...")
        audio_segs = await generate_all_tts(translated, tmp, audio)

        # 5. Видео
        print("\n🎬 5/5 Собираю видео (ffmpeg)...")
        create_dubbed_video(video_path, audio_segs, output_path)

    print(f"\n🎉 ГОТОВО: {output_path}")
    return {
        "output":        output_path,
        "segments":      translated,
        "segment_count": len(translated)
    }
