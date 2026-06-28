# -*- coding: utf-8 -*-
"""
AI Video Dubber v9 — чистая архитектура
Поток:
  1. AssemblyAI  → транскрипция + спикеры + временные метки
  2. Gemini      → сопоставляет перевод + определяет пол
  3. Gemini Vision → смотрит видео → строит скрипт с паузами
  4. Gemini TTS  → 1 запрос, multi-speaker (Puck/Leda)
  5. ffmpeg      → накладывает аудио поверх видео
"""

import os, re, json, time, asyncio, tempfile, subprocess, wave
from typing import Callable

import assemblyai as aai
from google import genai
from google.genai import types
from text_normalizer import normalize

# ─── Настройки ───────────────────────────────
VOICE_MALE   = "Puck"
VOICE_FEMALE = "Leda"
GEMINI_TTS   = "gemini-3.1-flash-tts-preview"
GEMINI_MODEL = "gemini-2.5-flash"
ORIG_VOL     = 0.08
DUB_VOL      = 2.3


# ─── 1. Аудио из видео ───────────────────────

def extract_audio(video: str, out: str):
    r = subprocess.run(
        ["ffmpeg", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-y", out],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise Exception(f"ffmpeg: {r.stderr[-200:]}")


# ─── 2. Транскрипция ─────────────────────────

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


# ─── 3. Gemini: сопоставление перевода + пол ─

def align_and_detect(segs: list, translation: str) -> list:
    key = os.environ.get("GEMINI_API_KEY", "")
    trans_lines = [l.strip() for l in translation.strip().split("\n") if l.strip()]

    if not key:
        for i, seg in enumerate(segs):
            seg["translated"] = trans_lines[i] if i < len(trans_lines) else seg["text"]
            seg["gender"]     = "male"
        return segs

    orig = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s] Spk {s['speaker']}: {s['text']}"
        for s in segs
    )
    trans = "\n".join(f"{i+1}. {l}" for i, l in enumerate(trans_lines))

    prompt = f"""Vazifang ikkita:
1. O'zbek tarjima qatorlarini original replikalar bilan vaqt tartibida moslashtir
2. Har bir spiker jinsi aniqla

ORIGINAL (vaqt bilan):
{orig}

O'ZBEK TARJIMA:
{trans}

Faqat JSON javob ber:
{{"segments":[{{"index":0,"translated":"...","gender":"male"}},{{"index":1,"translated":"...","gender":"female"}}]}}

QOIDALAR:
- translated: foydalanuvchi tarjimasidan hech o'zgartirmasdan ol
- gender: faqat "male" yoki "female"
- Barcha {len(segs)} segment uchun yoz"""

    try:
        r    = genai.Client(api_key=key).models.generate_content(
            model=GEMINI_MODEL, contents=prompt)
        m    = re.search(r'\{.*\}', r.text.strip(), re.DOTALL)
        data = json.loads(m.group(0))
        for item in data.get("segments", []):
            idx = item.get("index", 0)
            if 0 <= idx < len(segs):
                segs[idx]["translated"] = item.get("translated", segs[idx]["text"])
                segs[idx]["gender"]     = item.get("gender", "male")
    except Exception as e:
        print(f"  ⚠️ Gemini align error: {e}")
        for i, seg in enumerate(segs):
            seg["translated"] = trans_lines[i] if i < len(trans_lines) else seg["text"]
            seg["gender"]     = "male"

    for s in segs:
        s.setdefault("translated", s["text"])
        s.setdefault("gender", "male")
        icon = "👩" if s["gender"] == "female" else "👨"
        print(f"  {icon} [{s['start']:.1f}s] {s['translated'][:45]}")

    return segs


# ─── 4. Gemini Vision → TTS скрипт с паузами ─

def build_tts_script(segs: list, video_path: str, key: str) -> tuple:
    """
    Gemini смотрит видео и строит скрипт с точными паузами.
    Возвращает (script, speakers_dict).
    """
    # Определяем спикеров
    speakers = {}
    for seg in segs:
        spk = seg["speaker"]
        if spk not in speakers:
            speakers[spk] = {
                "voice": VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE,
                "label": f"Speaker{len(speakers)+1}"
            }

    # Базовый скрипт с временными метками
    base = "\n".join(
        f"[{s['start']:.1f}s→{s['end']:.1f}s] {speakers[s['speaker']]['label']}: {normalize(re.sub(chr(8203), '', re.sub(r'^[^:]+[:] *', '', s.get('translated', s['text'])))).strip()}"
        for s in segs if s.get("translated", s["text"]).strip()
    )

    print("  🎬 Загружаю видео в Gemini Vision...")

    try:
        client     = genai.Client(api_key=key)
        video_file = client.files.upload(file=video_path)

        # Ждём обработки
        for _ in range(30):
            if video_file.state.name != "PROCESSING":
                break
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception("Video upload failed")

        prompt = f"""Sen professional dublyaj rejissorisin.
Video va o'zbek tarjimasini ko'r.

REPLIKALAR (vaqt bilan):
{base}

Har bir replika orasidagi TOK PAUZA VAQTINI videodagi original pauza bo'yicha aniqla.
Natijani faqat TTS skript shaklida yoz:

{speakers[list(speakers.keys())[0]]['label']}: [birinchi replika]
<break time="X.Xs"/>
{speakers[list(speakers.keys())[1]]['label'] if len(speakers) > 1 else speakers[list(speakers.keys())[0]]['label']}: [ikkinchi replika]
<break time="X.Xs"/>
...

Qoidalar:
- Har bir replika ORASIDA faqat 1 ta <break> qo'y — uning vaqti original videodagi PAUZA ga teng bo'lsin
- Replikani o'zgartirma
- Faqat skriptni yoz, izoh yo'q"""

        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [video_file, prompt],
        )
        try: client.files.delete(name=video_file.name)
        except: pass

        script = response.text.strip()
        print(f"  ✅ Скрипт готов!")
        return script, speakers

    except Exception as e:
        print(f"  ⚠️ Vision failed: {e} — строю скрипт вручную")
        # Фолбэк: паузы вычисляем из временных меток
        lines = []
        for i, seg in enumerate(segs):
            raw   = seg.get("translated", seg["text"])
            raw   = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
            raw   = re.sub(r'[^\w\s.,!?\'-]', '', raw).strip()
            text  = normalize(raw)
            label = speakers[seg["speaker"]]["label"]
            if text.strip():
                lines.append(f"{label}: {text}")
                # Добавляем паузу между сегментами
                if i < len(segs) - 1:
                    gap = round(segs[i+1]["start"] - seg["end"], 1)
                    if gap > 0.2:
                        lines.append(f'<break time="{min(gap, 3.0):.1f}s"/>')
        return "\n".join(lines), speakers


# ─── 5. Gemini TTS → аудио ───────────────────

def run_tts(script: str, speakers: dict, key: str) -> bytes:
    """Один запрос → полное аудио для всего видео"""
    voice_configs = [
        types.SpeakerVoiceConfig(
            speaker      = data["label"],
            voice_config = types.VoiceConfig(
                prebuilt_voice_config = types.PrebuiltVoiceConfig(
                    voice_name = data["voice"]
                )
            )
        )
        for data in speakers.values()
    ]

    client = genai.Client(api_key=key)
    r = client.models.generate_content(
        model    = GEMINI_TTS,
        contents = script,
        config   = types.GenerateContentConfig(
            response_modalities = ["AUDIO"],
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config = types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs = voice_configs
                )
            ),
        ),
    )
    return r.candidates[0].content.parts[0].inline_data.data


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


# ─── 6. Сборка видео ─────────────────────────

def build_video(video: str, dub_wav: str, out: str):
    """Накладывает дублированный аудио поверх оригинального видео"""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video],
        capture_output=True, text=True
    )
    dur = float(json.loads(probe.stdout)["format"]["duration"])

    r = subprocess.run([
        "ffmpeg",
        "-i", video,          # оригинальное видео
        "-i", dub_wav,        # дублированный аудио
        "-filter_complex",
        f"[0:a]volume={ORIG_VOL}[orig];[1:a]volume={DUB_VOL}[dub];[orig][dub]amix=inputs=2:normalize=0[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(dur),
        "-y", out
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"ffmpeg: {r.stderr[-300:]}")
    print(f"✅ Видео: {out}")


# ─── Главная функция ─────────────────────────

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

    key  = os.environ.get("GEMINI_API_KEY", "")
    loop = asyncio.get_event_loop()

    with tempfile.TemporaryDirectory() as tmp:

        # 1. Аудио
        cb("📢 1/5 Audio ajratilmoqda...")
        audio = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, audio)

        # 2. Транскрипция
        cb("📝 2/5 Nutq tanib olinmoqda (AssemblyAI)...")
        segs = transcribe(audio, src_language)
        if not segs:
            raise Exception("Речь не найдена!")

        # 3. Перевод + пол
        cb("🤝 3/5 Tarjima va jins aniqlanmoqda...")
        segs = align_and_detect(segs, translation)

        # 4. Vision анализ + TTS скрипт
        cb("🎬 4/5 Gemini video ko'rib, skript yaratmoqda...")
        script, speakers = await loop.run_in_executor(
            None, build_tts_script, segs, video_path, key
        )
        print(f"  📝 Скрипт:\n{script[:300]}...")

        # 5. TTS генерация
        cb("🎤 5/5 Gemini TTS ovoz yaratmoqda (1 ta so'rov)...")
        try:
            audio_data = await loop.run_in_executor(
                None, run_tts, script, speakers, key
            )
        except Exception as e:
            raise Exception(f"Gemini TTS error: {e}")

        dub_wav = os.path.join(tmp, "dubbed.wav")
        save_wav(audio_data, dub_wav)
        dur = get_dur(dub_wav)
        print(f"  ✅ Аудио: {dur:.1f}с")

        # 6. Сборка
        cb("🎬 Видео yig'ilmoqda...")
        build_video(video_path, dub_wav, output_path)

    return {"output": output_path, "segments": segs}
