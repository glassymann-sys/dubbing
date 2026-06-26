# -*- coding: utf-8 -*-
"""
AI Video Dubber v4 — улучшенная версия
Видео → AssemblyAI → Groq (умный перевод) → Edge TTS (синхронизация) → ffmpeg
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
ORIG_VOLUME  = 0.08   # оригинал очень тихо на фоне
TTS_VOLUME   = 2.5    # дублёр громче
TTS_RATE     = "-12%"  # медленнее — более человечно

# ─────────────────────────────────────────────
# Промпт перевода — с контекстом
# ─────────────────────────────────────────────
def build_translate_prompt(full_context: str) -> str:
    return f"""Sen professional dublyaj tarjimonisin. Quyidagi video matnini aniq va tabiiy o'zbek tiliga tarjima qil.

VIDEO KONTEKSTI (umumiy tushunish uchun):
{full_context}

QOIDALAR:
1. Har bir gapni SO'ZMA-SO'Z tarjima qil — ma'noni o'zgartirma
2. Adabiy o'zbek tili — TV diktoridek, lekin tabiiy va jonli
3. His-tuyg'ularni saqlagan holda tarjima qil (hayrat, kulgu, g'azab va h.k.)
4. Savol → savol, undov → undov shaklida tarjima qil
5. O'zbek tilida to'liq ekvivalenti bo'lmagan so'zlarni aslida qoldur (gay, OK, wow va h.k.)
6. Faqat tarjima matnini yoz — hech qanday izoh yo'q
7. Har bir raqamli qatorni raqami bilan yoz
8. Tinish belgilarini saqlagan holda tarjima qil

MISOL:
Kirdi:  "1. Oh wow, would you really go on a date with me?!"
Chiqdi: "1. Voy, haqiqatan ham men bilan uchrashuvga borarmidingiz?!"

Kirdi:  "2. That's crazy, no way!"
Chiqdi: "2. Bu aqldan tashqari, yo'q!"
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
        raise Exception(f"ffmpeg error: {r.stderr[-300:]}")
    return out_path


# ─────────────────────────────────────────────
# Шаг 2: AssemblyAI — транскрипция + диаризация
# ─────────────────────────────────────────────

def transcribe_with_diarization(audio_path: str, api_key: str,
                                 language: str = None) -> list:
    print("🎙️  AssemblyAI: транскрипция + спикеры...")
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=not bool(language),
        language_code=language if language else None,
    )

    transcript = aai.Transcriber().transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI error: {transcript.error}")

    segments = []
    for utt in transcript.utterances:
        segments.append({
            "start":   utt.start / 1000.0,
            "end":     utt.end   / 1000.0,
            "text":    utt.text.strip(),
            "speaker": utt.speaker,
            "gender":  "male"
        })
        print(f"  Spk {utt.speaker} [{utt.start/1000:.1f}→{utt.end/1000:.1f}s]: {utt.text[:55]}")

    print(f"✅ {len(segments)} сегментов, спикеры: {set(s['speaker'] for s in segments)}")
    return segments


# ─────────────────────────────────────────────
# Шаг 2b: Определение пола через Groq
# ─────────────────────────────────────────────

def detect_speaker_genders_ai(segments: list, groq_api_key: str) -> dict:
    print("👥 Определяю пол спикеров...")
    client   = Groq(api_key=groq_api_key)
    dialog   = "\n".join([f"Speaker {s['speaker']}: {s['text']}" for s in segments])
    speakers = list(set(s['speaker'] for s in segments))

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Detect gender for each speaker from conversation.
Look for pronouns, names, topics, speaking style.
Reply ONLY with JSON: {"A": "male", "B": "female"}
Make best guess if unsure."""
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{dialog}\n\nIdentify gender for: {speakers}"
                }
            ],
            response_format={"type": "json_object"}
        )

        result  = json.loads(r.choices[0].message.content)
        genders = {}
        for spk in speakers:
            g = result.get(spk, "male").lower()
            genders[spk] = "female" if any(w in g for w in ["female", "woman", "girl"]) else "male"
            icon = "👩" if genders[spk] == "female" else "👨"
            print(f"  Speaker {spk} → {icon} {genders[spk]}")
        return genders

    except Exception as e:
        print(f"  ⚠️ {e} — используем alternating")
        return {spk: ("female" if i % 2 == 1 else "male")
                for i, spk in enumerate(sorted(speakers))}


# ─────────────────────────────────────────────
# Шаг 3: Умный перевод с контекстом
# ─────────────────────────────────────────────

def translate_segments(segments: list, groq_api_key: str) -> list:
    print("🌐 Перевожу на узбекский (с контекстом)...")
    client = Groq(api_key=groq_api_key)

    # Строим полный контекст из всех реплик
    full_context = " ".join([s["text"] for s in segments])
    prompt = build_translate_prompt(full_context)

    # Переводим всё одним запросом
    numbered = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(segments)])

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"Tarjima qil:\n{numbered}"}
        ]
    )

    tmap = {}
    for line in r.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            try:
                num, txt = line.split(". ", 1)
                tmap[int(num)] = txt.strip()
            except:
                pass

    result = []
    for i, seg in enumerate(segments):
        uz = tmap.get(i + 1, seg["text"])
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
        print(f"  {icon} {seg['text'][:30]:30} → {uz[:40]}")

    print("✅ Перевод готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 4: TTS с синхронизацией и просодией
# ─────────────────────────────────────────────

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
    """
    Подгоняет TTS под длину оригинала.
    МЯГКИЙ режим — ускоряем максимум на 20%, не больше.
    Если TTS намного длиннее — просто обрезаем конец (пауза).
    """
    tts_dur = get_duration(in_path)
    if tts_dur <= 0:
        try: os.rename(in_path, out_path)
        except: pass
        return

    ratio = tts_dur / target

    if ratio <= 1.2:
        # TTS короче или немного длиннее — не трогаем, звучит естественно
        try: os.rename(in_path, out_path)
        except: pass
        return

    # TTS длиннее более чем на 20% — слегка ускоряем max 1.35x
    tempo = min(ratio * 0.9, 1.35)
    tempo = max(tempo, 0.95)

    cmd = ["ffmpeg", "-i", in_path,
           "-filter:a", f"atempo={tempo:.3f}",
           "-y", out_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0:
        try: os.remove(in_path)
        except: pass
    else:
        try: os.rename(in_path, out_path)
        except: pass


async def generate_all_tts(segments: list, temp_dir: str,
                            orig_audio_path: str = None) -> list:
    print("🎤 Генерирую голос (синхронизация + просодия)...")

    orig_data, orig_sr = None, 16000
    if orig_audio_path and os.path.exists(orig_audio_path):
        try:
            orig_data, orig_sr = load_audio(orig_audio_path)
            print("  ✅ Аудио оригинала загружено")
        except Exception as e:
            print(f"  ⚠️ {e}")

    result = []
    for i, seg in enumerate(segments):
        raw  = os.path.join(temp_dir, f"seg_{i:04d}_raw.mp3")
        fit  = os.path.join(temp_dir, f"seg_{i:04d}_fit.mp3")
        out  = os.path.join(temp_dir, f"seg_{i:04d}.mp3")

        voice = VOICE_FEMALE if seg["gender"] == "female" else VOICE_MALE
        text  = normalize(seg["translated"])
        pitch = "-5Hz" if seg["gender"] == "male" else "+2Hz"

        # 1. Генерируем TTS
        tts = edge_tts.Communicate(text=text, voice=voice,
                                    rate=TTS_RATE, pitch=pitch)
        await tts.save(raw)

        # 2. Подгоняем под длину оригинала
        fit_to_duration(raw, fit, seg["duration"])

        # 3. Переносим просодию из оригинала
        if orig_data is not None:
            try:
                stats = analyze_segment_prosody(orig_data, orig_sr,
                                                 seg["start"], seg["end"])
                apply_prosody(fit, out, stats)
            except:
                try: os.rename(fit, out)
                except: pass
        else:
            try: os.rename(fit, out)
            except: pass

        icon = "👩" if seg["gender"] == "female" else "👨"
        print(f"  [{i+1}/{len(segments)}] {icon} {seg['translated'][:50]}")
        result.append({**seg, "audio_file": out})

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
        # Точная задержка по времени сегмента (без искусственного сдвига)
        delay = int(seg["start"] * 1000)
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
        raise Exception("ASSEMBLYAI_API_KEY топилмади! export ASSEMBLYAI_API_KEY='...'")

    print(f"\n🎬 Дублирую: {video_path}")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:

        print("\n📢 1/5 Извлекаю аудио...")
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        print("\n📝 2/5 Транскрибирую (AssemblyAI)...")
        lang = src_language if src_language and src_language != "auto" else None
        segs = transcribe_with_diarization(audio, assemblyai_key, lang)
        if not segs:
            raise Exception("Речь не найдена в видео")

        print("\n👥 Определяю пол спикеров (Groq)...")
        genders = detect_speaker_genders_ai(segs, groq_api_key)
        for seg in segs:
            seg["gender"] = genders.get(seg["speaker"], "male")

        print("\n🌐 3/5 Перевожу (Groq + контекст)...")
        translated = translate_segments(segs, groq_api_key)

        print("\n🎤 4/5 Генерирую голос (TTS + синхронизация)...")
        audio_segs = await generate_all_tts(translated, tmp, audio)

        print("\n🎬 5/5 Собираю видео (ffmpeg)...")
        create_dubbed_video(video_path, audio_segs, output_path)

    print(f"\n🎉 ГОТОВО: {output_path}")
    return {
        "output":        output_path,
        "segments":      translated,
        "segment_count": len(translated)
    }
