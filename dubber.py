# -*- coding: utf-8 -*-
"""
AI Video Dubber v6
Видео → AssemblyAI → Gemini (перевод) → Gemini TTS (Puck/Leda) → ffmpeg
"""

import os
import re
import asyncio
import tempfile
import subprocess
import json
import numpy as np
import wave

import assemblyai as aai
from google import genai
from google.genai import types
from groq import Groq
from text_normalizer import normalize
from prosody import load_audio, analyze_segment_prosody, apply_prosody

# ─────────────────────────────────────────────
VOICE_FEMALE = "Leda"   # Gemini TTS женский
VOICE_MALE   = "Puck"   # Gemini TTS мужской
ORIG_VOLUME  = 0.08
TTS_VOLUME   = 2.5
GEMINI_MODEL = "gemini-3.1-flash-tts-preview"

# ─────────────────────────────────────────────
# Промпт перевода — СТРОГО слово в слово
# ─────────────────────────────────────────────
TRANSLATE_PROMPT = """Sen professional dublyaj tarjimonisin.

MUHIM: Matnni SO'ZMA-SO'Z, AYNAN tarjima qil!
- Biror so'z qo'shma, biror so'z o'chirma
- Ma'noni o'zgartirma, gapni qayta yozma
- Faqat o'zbek tiliga mos shaklga o'tkaz
- His-tuyg'u, intonatsiya, savol belgilarini saqlagan holda tarjima qil
- O'zbek tilida ekvivalenti bo'lmagan so'zlarni aslida qoldur (gay, OK, wow va h.k.)
- Faqat tarjima matnini yoz — hech qanday izoh yo'q
- Har bir raqamli qatorni raqami bilan yoz

MISOL TO'G'RI:
Kirdi:  "1. I was watching you from over there with binoculars"
Chiqdi: "1. Men sizni u tomondan durbin bilan kuzatib turgan edim"

MISOL NOTO'G'RI (bunday qilma!):
Kirdi:  "1. I was watching you from over there with binoculars"
Chiqdi: "1. Men sizni kuzatmoqda edim" — noto'g'ri, qisqartirilgan!"""


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

    # Если мало сегментов — разбиваем по предложениям
    if len(transcript.utterances) <= 2:
        print("  ⚠️ Мало сегментов — разбиваю по предложениям...")
        for utt in transcript.utterances:
            sentences = re.split(r'(?<=[.!?,])\s+', utt.text.strip())
            total_dur = (utt.end - utt.start) / 1000.0
            per_sent  = total_dur / max(len(sentences), 1)

            for j, sent in enumerate(sentences):
                sent = sent.strip()
                if not sent:
                    continue
                seg_start = utt.start / 1000.0 + j * per_sent
                seg_end   = seg_start + per_sent
                segments.append({
                    "start":   round(seg_start, 3),
                    "end":     round(seg_end, 3),
                    "text":    sent,
                    "speaker": utt.speaker,
                    "gender":  "male"
                })
                print(f"  Spk {utt.speaker} [{seg_start:.1f}→{seg_end:.1f}s]: {sent[:55]}")
    else:
        for utt in transcript.utterances:
            segments.append({
                "start":   round(utt.start / 1000.0, 3),
                "end":     round(utt.end   / 1000.0, 3),
                "text":    utt.text.strip(),
                "speaker": utt.speaker,
                "gender":  "male"
            })
            print(f"  Spk {utt.speaker} [{utt.start/1000:.1f}→{utt.end/1000:.1f}s]: {utt.text[:55]}")

    print(f"✅ {len(segments)} сегментов, спикеры: {set(s['speaker'] for s in segments)}")
    return segments


# ─────────────────────────────────────────────
# Шаг 2b: Gemini определяет пол спикеров
# ─────────────────────────────────────────────

def detect_speaker_genders_ai(segments: list, groq_api_key: str) -> dict:
    print("👥 Gemini определяет пол спикеров...")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    dialog     = "\n".join([f"Speaker {s['speaker']}: {s['text']}" for s in segments])
    speakers   = list(set(s['speaker'] for s in segments))

    try:
        client = genai.Client(api_key=gemini_key)
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Analyze this conversation and detect the gender of each speaker.
Look for pronouns (he/she/his/her), names, topics, context clues.
Conversation:
{dialog}
Speakers: {speakers}
Reply ONLY with valid JSON like: {{"A": "male", "B": "female"}}"""
        )
        text = r.text.strip().replace("```json", "").replace("```", "").strip()
        result  = json.loads(text)
        genders = {}
        for spk in speakers:
            g = result.get(spk, "male").lower()
            genders[spk] = "female" if any(w in g for w in ["female","woman","girl"]) else "male"
            icon = "👩" if genders[spk] == "female" else "👨"
            print(f"  Speaker {spk} → {icon} {genders[spk]}")
        return genders
    except Exception as e:
        print(f"  ⚠️ Gemini gender error: {e} — Groq fallback")
        try:
            groq = Groq(api_key=groq_api_key)
            r2 = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Detect gender for each speaker. Reply ONLY with JSON: {\"A\": \"male\", \"B\": \"female\"}"},
                    {"role": "user", "content": f"{dialog}\n\nIdentify: {speakers}"}
                ],
                response_format={"type": "json_object"}
            )
            result  = json.loads(r2.choices[0].message.content)
            genders = {}
            for spk in speakers:
                g = result.get(spk, "male").lower()
                genders[spk] = "female" if "female" in g else "male"
                icon = "👩" if genders[spk] == "female" else "👨"
                print(f"  Speaker {spk} → {icon} {genders[spk]}")
            return genders
        except:
            return {spk: ("female" if i % 2 == 1 else "male")
                    for i, spk in enumerate(sorted(speakers))}


# ─────────────────────────────────────────────
# Шаг 3: Перевод через Gemini (строго)
# ─────────────────────────────────────────────

def translate_segments(segments: list, groq_api_key: str) -> list:
    print("🌐 Перевожу через Gemini (строго слово в слово)...")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    numbered   = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(segments)])

    try:
        client = genai.Client(api_key=gemini_key)
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{TRANSLATE_PROMPT}\n\nTarjima qil:\n{numbered}"
        )
        response_text = r.text
    except Exception as e:
        print(f"  ⚠️ Gemini error: {e} — Groq fallback")
        groq = Groq(api_key=groq_api_key)
        r2 = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user",   "content": f"Tarjima qil:\n{numbered}"}
            ]
        )
        response_text = r2.choices[0].message.content

    tmap = {}
    for line in response_text.strip().split("\n"):
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
            "start":    seg["start"],
            "end":      seg["end"],
            "original": seg["text"],
            "translated": uz,
            "speaker":  seg.get("speaker", "A"),
            "gender":   seg.get("gender", "male"),
            "duration": seg["end"] - seg["start"],
        })
        icon = "👩" if seg.get("gender") == "female" else "👨"
        print(f"  {icon} {seg['text'][:30]:30} → {uz[:40]}")

    print("✅ Перевод готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 4: Gemini TTS — синхронизация
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
    """Подгоняет TTS строго под длину оригинала"""
    tts_dur = get_duration(in_path)
    if tts_dur <= 0:
        try: os.rename(in_path, out_path)
        except: pass
        return

    ratio = tts_dur / target

    if 0.92 <= ratio <= 1.08:
        try: os.rename(in_path, out_path)
        except: pass
        return

    # atempo диапазон 0.5-2.0 — если больше применяем дважды
    if ratio > 1.8:
        mid = in_path + "_mid.wav"
        r1 = subprocess.run(
            ["ffmpeg", "-i", in_path, "-filter:a", "atempo=1.8", "-y", mid],
            capture_output=True
        )
        if r1.returncode == 0:
            r2 = subprocess.run(
                ["ffmpeg", "-i", mid, "-filter:a", f"atempo={ratio/1.8:.3f}", "-y", out_path],
                capture_output=True
            )
            try: os.remove(in_path); os.remove(mid)
            except: pass
            if r2.returncode == 0:
                return

    tempo = max(0.5, min(ratio, 2.0))
    cmd = ["ffmpeg", "-i", in_path, "-filter:a", f"atempo={tempo:.3f}", "-y", out_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0:
        try: os.remove(in_path)
        except: pass
    else:
        try: os.rename(in_path, out_path)
        except: pass


async def generate_all_tts(segments: list, temp_dir: str,
                            orig_audio_path: str = None) -> list:
    print("🎤 Генерирую голос (Gemini TTS Puck/Leda)...")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise Exception("GEMINI_API_KEY не найден!")

    client = genai.Client(api_key=gemini_key)

    orig_data, orig_sr = None, 16000
    if orig_audio_path and os.path.exists(orig_audio_path):
        try:
            orig_data, orig_sr = load_audio(orig_audio_path)
        except Exception as e:
            print(f"  ⚠️ {e}")

    result = []
    for i, seg in enumerate(segments):
        raw = os.path.join(temp_dir, f"seg_{i:04d}_raw.wav")
        fit = os.path.join(temp_dir, f"seg_{i:04d}_fit.wav")
        out = os.path.join(temp_dir, f"seg_{i:04d}.wav")

        voice = VOICE_FEMALE if seg["gender"] == "female" else VOICE_MALE
        text  = normalize(seg["translated"])

        try:
            # Retry до 3 раз с паузой при 429
            audio_data = None
            for attempt in range(3):
                try:
                    r = client.models.generate_content(
                        model=GEMINI_MODEL,
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
                    audio_data = r.candidates[0].content.parts[0].inline_data.data
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = 6 + attempt * 5  # 6, 11, 16 сек
                        print(f"  ⏳ Rate limit — жду {wait} сек... (попытка {attempt+1}/3)")
                        await asyncio.sleep(wait)
                    else:
                        raise e

            if audio_data is None:
                print(f"  ❌ Пропускаю сегмент {i} — лимит исчерпан")
                continue

            with wave.open(raw, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            # Небольшая пауза между запросами чтобы не превысить лимит
            await asyncio.sleep(6)
        except Exception as e:
            print(f"  ⚠️ TTS error seg {i}: {e}")
            continue

        # Подгоняем под длину оригинала
        fit_to_duration(raw, fit, seg["duration"])

        # Переносим просодию
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
        print(f"  [{i+1}/{len(segments)}] {icon} {voice}: {seg['translated'][:45]}")
        result.append({**seg, "audio_file": out})

    print("✅ TTS готов!")
    return result


# ─────────────────────────────────────────────
# Шаг 5: Сборка видео без перекрытий
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
        if not os.path.exists(seg.get("audio_file", "")):
            continue
        inputs += ["-i", seg["audio_file"]]
        # Точная задержка по времени сегмента
        delay = int(seg["start"] * 1000)
        lbl   = f"s{i}"
        filters.append(
            f"[{len(labels)+1}:a]adelay={delay}|{delay},volume={TTS_VOLUME}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    if not labels:
        raise Exception("Нет аудио сегментов!")

    all_in = "[orig]" + "".join(labels)
    filters.append(
        f"{all_in}amix=inputs={len(labels)+1}:normalize=0:dropout_transition=0[aout]"
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
        raise Exception("ASSEMBLYAI_API_KEY топилмади!")

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

        print("\n👥 Определяю пол спикеров (Gemini)...")
        genders = detect_speaker_genders_ai(segs, groq_api_key)
        for seg in segs:
            seg["gender"] = genders.get(seg["speaker"], "male")

        print("\n🌐 3/5 Перевожу (Gemini)...")
        translated = translate_segments(segs, groq_api_key)

        print("\n🎤 4/5 Генерирую голос (Gemini TTS)...")
        audio_segs = await generate_all_tts(translated, tmp, audio)

        print("\n🎬 5/5 Собираю видео (ffmpeg)...")
        create_dubbed_video(video_path, audio_segs, output_path)

    print(f"\n🎉 ГОТОВО: {output_path}")
    return {
        "output":        output_path,
        "segments":      translated,
        "segment_count": len(translated)
    }
