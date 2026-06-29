# -*- coding: utf-8 -*-
"""
AI Video Dubber v10
Архитектура (по совету ИИ):
  1. AssemblyAI  → сегменты с точными timestamps
  2. Gemini      → перевод + пол каждого сегмента
  3. Gemini TTS  → один вызов = один сегмент (не монолитный WAV!)
  4. loudnorm    → нормализация громкости каждого сегмента
  5. atempo≤1.15 → растяжка ≤15%, иначе укорачиваем текст
  6. adelay      → каждый сегмент на своё точное время
  7. amix        → финальная сборка
"""

import os, re, json, time, asyncio, tempfile, subprocess, wave
from typing import Callable

import assemblyai as aai
from google import genai
from google.genai import types
from text_normalizer import normalize

# ── Настройки ────────────────────────────────
VOICE_MALE   = "Puck"
VOICE_FEMALE = "Leda"
GEMINI_TTS   = "gemini-3.1-flash-tts-preview"
GEMINI_MODEL = "gemini-2.5-flash"
ORIG_VOL     = 0.08
DUB_VOL      = 2.3
MAX_STRETCH  = 1.5   # максимальная растяжка atempo (≤50%)


# ── 1. Аудио из видео ────────────────────────

def extract_audio(video: str, out: str):
    r = subprocess.run(
        ["ffmpeg", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-y", out],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise Exception(f"ffmpeg extract: {r.stderr[-200:]}")


# ── 2. Транскрипция ──────────────────────────

def transcribe(audio: str, language: str = None) -> list:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        raise Exception("ASSEMBLYAI_API_KEY топилмади!")
    aai.settings.api_key = key
    cfg = aai.TranscriptionConfig(
        speaker_labels     = True,
        speakers_expected  = 2,
        language_detection = not bool(language),
        language_code      = language if language else None,
    )
    t = aai.Transcriber().transcribe(audio, cfg)
    if t.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI: {t.error}")
    segs = [
        {"start": round(u.start/1000, 3), "end": round(u.end/1000, 3),
         "text": u.text.strip(), "speaker": u.speaker}
        for u in (t.utterances or [])
    ]
    print(f"  ✅ {len(segs)} сегментов, спикеры: {set(s['speaker'] for s in segs)}")
    return segs


# ── 3. Gemini: перевод + пол ─────────────────

def align_and_detect(segs: list, translation: str) -> list:
    key         = os.environ.get("GEMINI_API_KEY", "")
    trans_lines = [l.strip() for l in translation.strip().split("\n") if l.strip()]

    if not key:
        for i, seg in enumerate(segs):
            seg["translated"] = trans_lines[i] if i < len(trans_lines) else seg["text"]
            seg["gender"]     = "male"
        return segs

    orig  = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s] Spk {s['speaker']}: {s['text']}"
        for s in segs
    )
    trans = "\n".join(f"{i+1}. {l}" for i, l in enumerate(trans_lines))

    prompt = f"""Vazifang ikkita:
1. O'zbek tarjima qatorlarini original replikalar bilan vaqt tartibida moslashtir
2. Har bir spiker jinsi aniqla (erkak/ayol) dialog kontekstidan

ORIGINAL:
{orig}

TARJIMA:
{trans}

Faqat JSON:
{{"segments":[{{"index":0,"translated":"...","gender":"male"}},{{"index":1,"translated":"...","gender":"female"}}]}}

gender: faqat "male" yoki "female". Barcha {len(segs)} segment uchun yoz."""

    try:
        # Создаём новый клиент на каждый вызов
        client = genai.Client(api_key=key)
        r    = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt)
        m    = re.search(r'\{.*\}', r.text.strip(), re.DOTALL)
        if not m: raise ValueError("No JSON")
        data = json.loads(m.group(0))
        for item in data.get("segments", []):
            idx = item.get("index", 0)
            if 0 <= idx < len(segs):
                segs[idx]["translated"] = item.get("translated", segs[idx]["text"])
                g = item.get("gender", "male").lower()
                segs[idx]["gender"] = "female" if any(
                    w in g for w in ["female", "ayol", "woman", "girl"]
                ) else "male"
    except Exception as e:
        print(f"  ⚠️ Gemini align error: {e}")
        for i, seg in enumerate(segs):
            seg["translated"] = trans_lines[i] if i < len(trans_lines) else seg["text"]
            # Фолбэк по спикеру: A=male, B=female (обычно так в диалогах)
            seg["gender"] = "female" if seg.get("speaker", "A") == "B" else "male"

    for s in segs:
        s.setdefault("translated", s["text"])
        s.setdefault("gender", "male")
        icon = "👩" if s["gender"] == "female" else "👨"
        print(f"  {icon} [{s['start']:.1f}s] {s['translated'][:45]}")
    return segs


# ── 4. Утилиты аудио ─────────────────────────

def save_wav(data: bytes, path: str):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(24000); wf.writeframes(data)


def get_dur(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True
    )
    try:
        for s in json.loads(r.stdout).get("streams", []):
            if "duration" in s: return float(s["duration"])
    except: pass
    return 0.0


def loudnorm(src: str, dst: str):
    """Нормализует громкость — loudnorm=I=-16:TP=-1.5:LRA=11"""
    subprocess.run([
        "ffmpeg", "-i", src,
        "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "24000", "-y", dst
    ], capture_output=True)


def stretch_audio(src: str, dst: str, ratio: float):
    """
    Растягивает/сжимает аудио через atempo (≤15%).
    ratio = tts_dur / target_dur
    Если ratio > MAX_STRETCH — не применяем (текст уже укорочен).
    """
    if abs(ratio - 1.0) < 0.03:
        subprocess.run(["cp", src, dst])
        return

    tempo = max(0.5, min(ratio, 2.0))

    if ratio > 2.0:
        # Двойной atempo
        mid = src + "_mid.wav"
        subprocess.run(["ffmpeg", "-i", src, "-filter:a", "atempo=2.0", "-y", mid],
                       capture_output=True)
        subprocess.run(["ffmpeg", "-i", mid, "-filter:a", f"atempo={ratio/2.0:.3f}",
                        "-y", dst], capture_output=True)
        try: os.remove(mid)
        except: pass
    else:
        subprocess.run(["ffmpeg", "-i", src, "-filter:a", f"atempo={tempo:.3f}",
                        "-y", dst], capture_output=True)


# ── 5. Gemini TTS — per-segment ──────────────

def tts_one_sync(text: str, voice: str, key: str) -> bytes:
    """Один сегмент = один запрос. Retry при 429."""
    client = genai.Client(api_key=key)
    for attempt in range(5):
        try:
            r = client.models.generate_content(
                model    = GEMINI_TTS,
                contents = text,
                config   = types.GenerateContentConfig(
                    response_modalities = ["AUDIO"],
                    speech_config = types.SpeechConfig(
                        voice_config = types.VoiceConfig(
                            prebuilt_voice_config = types.PrebuiltVoiceConfig(
                                voice_name = voice
                            )
                        )
                    ),
                ),
            )
            # Проверяем что ответ не пустой
            candidates = r.candidates
            if not candidates or not candidates[0].content.parts:
                raise ValueError("Empty TTS response")
            data = candidates[0].content.parts[0].inline_data
            if data is None:
                raise ValueError("TTS inline_data is None")
            return data.data
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                m    = re.search(r'retry in (\d+)', str(e))
                wait = int(m.group(1)) + 3 if m else 15 * (attempt + 1)
                print(f"  ⏳ Rate limit — wait {wait}s")
                time.sleep(wait)
            else:
                raise
    raise Exception("Gemini TTS: rate limit exceeded")


def clean_text(text: str) -> str:
    """Убирает метки спикера и эмодзи из текста"""
    text = re.sub(r'^[^:：]+[:：]\s*', '', text).strip()
    text = re.sub(r'[^\w\s.,!?\'\-—]', '', text).strip()
    return normalize(text)


def shorten_text(text: str, ratio: float) -> str:
    """
    Если ratio > MAX_STRETCH — укорачиваем текст пропорционально.
    Убираем слова с конца чтобы уложиться в duration.
    """
    if ratio <= MAX_STRETCH:
        return text
    words     = text.split()
    target_n  = max(1, int(len(words) / ratio))
    shortened = " ".join(words[:target_n])
    print(f"  ✂️  Текст укорочен: {len(words)} → {target_n} слов (ratio={ratio:.2f})")
    return shortened


async def generate_all_tts(segs: list, tmp: str) -> list:
    """
    Per-segment TTS:
    1. Генерируем каждый сегмент отдельно
    2. loudnorm — стабилизируем громкость
    3. atempo ≤ 15% — подгоняем под duration
    4. Возвращаем список с audio_file и start/end
    """
    key  = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise Exception("GEMINI_API_KEY не найден!")
    loop = asyncio.get_event_loop()
    result = []

    for i, seg in enumerate(segs):
        text = clean_text(seg.get("translated") or seg["text"])
        if not text: continue

        voice    = VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE
        duration = seg["end"] - seg["start"]
        icon     = "👩" if seg.get("gender") == "female" else "👨"

        # НЕ укорачиваем текст — пусть Gemini говорит естественно
        # atempo подгонит если нужно

        raw  = os.path.join(tmp, f"{i:04d}_raw.wav")
        norm = os.path.join(tmp, f"{i:04d}_norm.wav")
        out  = os.path.join(tmp, f"{i:04d}.wav")

        # Генерируем TTS
        try:
            audio = await loop.run_in_executor(
                None, tts_one_sync, text, voice, key
            )
            save_wav(audio, raw)
        except Exception as e:
            print(f"  ❌ Seg {i}: {e}")
            continue

        # loudnorm — стабилизируем громкость
        loudnorm(raw, norm)
        try: os.remove(raw)
        except: pass

        # atempo — подгоняем под duration (≤ MAX_STRETCH)
        tts_dur = get_dur(norm)
        ratio   = tts_dur / duration if duration > 0 else 1.0

        if ratio > 2.0:
            print(f"  ⚠️  Seg {i}: ratio={ratio:.2f} — применяем двойной atempo")
        elif ratio > MAX_STRETCH:
            print(f"  ⚠️  Seg {i}: ratio={ratio:.2f} — применяем atempo")

        stretch_audio(norm, out, ratio)
        try: os.remove(norm)
        except: pass

        final_dur = get_dur(out)
        print(f"  [{i+1}/{len(segs)}] {icon} [{seg['start']:.1f}s] "
              f"{voice}: {text[:35]} ({final_dur:.1f}s/{duration:.1f}s)")

        result.append({**seg, "audio_file": out})

        # Пауза между запросами (лимит 10/мин)
        if i < len(segs) - 1:
            await asyncio.sleep(6)

    print(f"✅ TTS готов: {len(result)}/{len(segs)} сегментов")
    return result


# ── 6. Сборка видео ──────────────────────────

def build_video(video: str, segs: list, out: str):
    """
    adelay — каждый сегмент на своё точное время (start_ms).
    amix   — смешиваем все сегменты в один трек.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video],
        capture_output=True, text=True
    )
    dur   = float(json.loads(probe.stdout)["format"]["duration"])
    valid = [s for s in segs if os.path.exists(s.get("audio_file", ""))]
    if not valid:
        raise Exception("Нет аудио сегментов!")

    inputs  = ["-i", video]
    filters = [f"[0:a]volume={ORIG_VOL}[orig]"]
    labels  = []

    for i, s in enumerate(valid):
        inputs += ["-i", s["audio_file"]]
        delay   = int(s["start"] * 1000)  # adelay в миллисекундах
        lbl     = f"v{i}"
        # loudnorm уже применён, дополнительно нормализуем volume
        filters.append(
            f"[{i+1}:a]adelay={delay}|{delay},volume={DUB_VOL}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    all_in = "[orig]" + "".join(labels)
    filters.append(
        f"{all_in}amix=inputs={len(labels)+1}:normalize=0:dropout_transition=0[aout]"
    )

    r = subprocess.run([
        "ffmpeg", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(dur), "-y", out
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"ffmpeg build: {r.stderr[-300:]}")
    print(f"✅ Видео: {out}")


# ── Главная функция ──────────────────────────

async def dub_video(
    video_path:   str,
    output_path:  str,
    translation:  str,
    groq_api_key: str      = "",
    src_language: str      = None,
    status_cb:    Callable = None,
):
    def cb(msg):
        print(msg)
        if status_cb: status_cb(msg)

    with tempfile.TemporaryDirectory() as tmp:

        cb("📢 1/5 Audio ajratilmoqda...")
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        cb("📝 2/5 Nutq tanib olinmoqda (AssemblyAI)...")
        segs = transcribe(audio, src_language)
        if not segs:
            raise Exception("Речь не найдена!")

        cb("🤝 3/5 Tarjima va jins aniqlanmoqda (Gemini)...")
        segs = align_and_detect(segs, translation)

        cb(f"🎤 4/5 Gemini TTS — {len(segs)} segment (har biri alohida)...")
        audio_segs = await generate_all_tts(segs, tmp)

        cb("🎬 5/5 Video yig'ilmoqda (adelay sync)...")
        build_video(video_path, audio_segs, output_path)

    return {"output": output_path, "segments": segs}
