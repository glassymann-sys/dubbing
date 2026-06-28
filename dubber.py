# -*- coding: utf-8 -*-
"""
AI Video Dubber v8
1. AssemblyAI  — транскрипция с временными метками
2. Gemini      — определяет кто говорит когда + пол каждого спикера
3. Gemini TTS  — генерирует голос (Puck=мужчина, Leda=женщина)
4. ffmpeg      — собирает финальное видео
"""

import os, re, json, time, asyncio, tempfile, subprocess, wave
from typing import Callable

import assemblyai as aai
from google import genai
from google.genai import types
from text_normalizer import normalize

# ── Голоса ──────────────────────────────────
VOICE_MALE   = "Puck"
VOICE_FEMALE = "Leda"
GEMINI_TTS   = "gemini-3.1-flash-tts-preview"
GEMINI_MODEL = "gemini-2.5-flash"
ORIG_VOL     = 0.08   # оригинал тихо на фоне
DUB_VOL      = 2.3    # дублёр громко


# ── 1. Извлечь аудио ────────────────────────

def extract_audio(video: str, out: str):
    r = subprocess.run(
        ["ffmpeg", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-y", out],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise Exception(f"ffmpeg extract: {r.stderr[-200:]}")


# ── 2. Транскрипция AssemblyAI ───────────────

def transcribe(audio: str, language: str = None) -> list:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        raise Exception("ASSEMBLYAI_API_KEY не найден!")

    aai.settings.api_key = key
    cfg = aai.TranscriptionConfig(
        speaker_labels    = True,
        speakers_expected = 2,
        language_detection = not bool(language),
        language_code      = language if language else None,
    )
    t = aai.Transcriber().transcribe(audio, cfg)
    if t.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI: {t.error}")

    segs = []
    for u in (t.utterances or []):
        segs.append({
            "start":   round(u.start / 1000, 3),
            "end":     round(u.end   / 1000, 3),
            "text":    u.text.strip(),
            "speaker": u.speaker,
        })
    print(f"  ✅ {len(segs)} сегментов, спикеры: {set(s['speaker'] for s in segs)}")
    return segs


# ── 3. Gemini: выравнивание перевода + пол ───

def align_and_detect(segs: list, translation: str) -> list:
    """
    Gemini получает:
    - оригинальные реплики с временными метками и спикерами
    - перевод пользователя (построчно)
    
    Возвращает каждому сегменту:
    - translated: соответствующая строка перевода
    - gender: male/female
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        # Фолбэк: просто разбиваем перевод по строкам
        lines = [l.strip() for l in translation.strip().split("\n") if l.strip()]
        for i, seg in enumerate(segs):
            seg["translated"] = lines[i] if i < len(lines) else seg["text"]
            seg["gender"]     = "male"
        return segs

    client = genai.Client(api_key=key)

    orig_dialog = "\n".join([
        f"[{s['start']:.1f}s-{s['end']:.1f}s] Speaker {s['speaker']}: {s['text']}"
        for s in segs
    ])
    trans_lines = "\n".join([
        f"{i+1}. {l.strip()}"
        for i, l in enumerate(l for l in translation.strip().split("\n") if l.strip())
    ])

    prompt = f"""Vazifang:
1. Quyidagi O'ZBEK TARJIMA qatorlarini original replikalar bilan vaqt bo'yicha moslashtir
2. Har bir spiker uchun jins aniqla (erkak/ayol) — dialogdan, ismlardan, zamirlardан

ORIGINAL REPLIKALAR (vaqt bilan):
{orig_dialog}

O'ZBEK TARJIMA (foydalanuvchi yozgan):
{trans_lines}

Javob faqat JSON formatda:
{{
  "segments": [
    {{"index": 0, "translated": "...", "gender": "male"}},
    {{"index": 1, "translated": "...", "gender": "female"}}
  ]
}}

Qoidalar:
- Tarjima qatorlarini original replikalarga vaqt tartibida moslashtir
- Agar tarjima qatorlari soni bilan replikalar soni mos kelmasa — eng yaqin ma'noni ishlat
- gender: faqat "male" yoki "female"
- translated: foydalanuvchi tarjimasidan oling, o'zgartirmang"""

    try:
        r    = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = r.text.strip()
        # Вырезаем JSON
        m    = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            raise ValueError("No JSON in response")
        data = json.loads(m.group(0))

        for item in data.get("segments", []):
            idx = item.get("index", 0)
            if 0 <= idx < len(segs):
                segs[idx]["translated"] = item.get("translated", segs[idx]["text"])
                segs[idx]["gender"]     = item.get("gender", "male")

        for s in segs:
            if "translated" not in s: s["translated"] = s["text"]
            if "gender"     not in s: s["gender"]     = "male"
            icon = "👩" if s["gender"] == "female" else "👨"
            print(f"  {icon} [{s['start']:.1f}s] {s['translated'][:45]}")

    except Exception as e:
        print(f"  ⚠️ Gemini align error: {e} — используем простое разбиение")
        lines = [l.strip() for l in translation.strip().split("\n") if l.strip()]
        for i, seg in enumerate(segs):
            seg["translated"] = lines[i] if i < len(lines) else seg["text"]
            seg["gender"]     = "male"

    return segs


# ── 4. Gemini TTS ────────────────────────────

def tts_one(text: str, voice: str, gemini_key: str) -> bytes:
    """Генерирует один аудио сегмент, retry при 429"""
    client = genai.Client(api_key=gemini_key)
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
            return r.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 10 * (attempt + 1)  # 10, 20, 30, 40, 50 сек
                print(f"  ⏳ Rate limit — wait {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise
    raise Exception("Gemini TTS: rate limit exceeded after 5 attempts")


def save_wav(data: bytes, path: str):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(data)


def get_dur(path: str) -> float:
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
    return 0.0


def fit_duration(src: str, dst: str, target: float):
    """Подгоняет аудио под длину оригинала через atempo"""
    dur = get_dur(src)
    if dur <= 0 or target <= 0:
        os.rename(src, dst); return

    ratio = dur / target
    if 0.88 <= ratio <= 1.12:
        os.rename(src, dst); return

    def apply(inp, out, tempo):
        subprocess.run(["ffmpeg", "-i", inp, "-filter:a", f"atempo={tempo:.3f}",
                        "-y", out], capture_output=True)

    if ratio > 2.0:
        tmp = src + "_t.wav"
        apply(src, tmp, 2.0)
        apply(tmp, dst, min(ratio / 2.0, 2.0))
        try: os.remove(src); os.remove(tmp)
        except: pass
    elif ratio < 0.5:
        tmp = src + "_t.wav"
        apply(src, tmp, 0.5)
        apply(tmp, dst, max(ratio / 0.5, 0.5))
        try: os.remove(src); os.remove(tmp)
        except: pass
    else:
        apply(src, dst, max(0.5, min(ratio, 2.0)))
        try: os.remove(src)
        except: pass


async def analyze_video_and_build_script(
    video_path: str, segs: list, key: str
) -> str:
    """
    Gemini Vision смотрит видео и строит оптимальный TTS скрипт с паузами.
    Знает где нужна длинная/короткая пауза, где эмоция, где ускорить.
    """
    client = genai.Client(api_key=key)
    loop   = asyncio.get_event_loop()

    # Строим базовый скрипт
    speakers = {}
    for seg in segs:
        spk = seg.get("speaker", "A")
        if spk not in speakers:
            speakers[spk] = {
                "voice": VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE,
                "label": f"Speaker{len(speakers)+1}"
            }

    lines = []
    for seg in segs:
        spk   = seg.get("speaker", "A")
        label = speakers[spk]["label"]
        raw   = seg.get("translated") or seg["text"]
        raw   = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
        raw   = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
        text  = normalize(raw)
        if text.strip():
            # Добавляем временную метку для Gemini
            lines.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {label}: {text}")

    base_script = "\n".join(lines)

    # Загружаем видео в Gemini
    print("  🎬 Загружаю видео в Gemini для анализа...")

    def upload_and_analyze():
        # Загружаем видео файл
        video_file = client.files.upload(file=video_path)

        # Ждём пока обработается
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception("Video upload failed")

        prompt = f"""Sen dublyaj rejissorisin. Quyidagi video va uzbek tarjimasini ko'rasan.

Video faylini analiz qil va TTS uchun optimal skript yarat:
1. Har bir replika orasida qancha pauza kerakligini aniqla (original videodagi pauza vaqtiga qarab)
2. His-tuyg'ularni belgilagan holda (hayrat, kulgu, savol va h.k.)
3. Speaker1 = erkak ovoz (Puck), Speaker2 = ayol ovoz (Leda)

ORIGINAL REPLIKALAR (vaqt bilan):
{base_script}

NATIJA FORMATI — faqat skriptni yoz, izoh yo'q:
Speaker1: [birinchi replikaning tarjimasi]
<break time="1.5s"/>
Speaker2: [ikkinchi replikaning tarjimasi]
<break time="0.8s"/>
...

QOIDALAR:
- Original videodagi pauza vaqtiga mos <break time="Xs"/> qo'y
- Savol ohangida "?" qo'y
- Hayrat uchun "!" ishlatish mumkin
- Faqat tayyor skriptni yoz"""

        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [video_file, prompt],
        )

        # Удаляем загруженный файл
        try:
            client.files.delete(name=video_file.name)
        except:
            pass

        return response.text.strip()

    try:
        script = await loop.run_in_executor(None, upload_and_analyze)
        print(f"  ✅ Gemini создал скрипт с паузами!")
        print(f"  📝 {script[:200]}...")
        return script, speakers
    except Exception as e:
        print(f"  ⚠️ Video analysis failed: {e} — используем базовый скрипт")
        # Фолбэк — простой скрипт без видео анализа
        simple_lines = []
        for seg in segs:
            spk   = seg.get("speaker", "A")
            label = speakers[spk]["label"]
            raw   = seg.get("translated") or seg["text"]
            raw   = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
            raw   = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
            text  = normalize(raw)
            if text.strip():
                # Добавляем паузу между репликами на основе разницы времён
                simple_lines.append(f"{label}: {text}")
        return "\n".join(simple_lines), speakers
    """
    1 запрос на всё видео через Gemini Multi-Speaker TTS.
    Весь дублированный аудио кладём поверх видео целиком — без нарезки.
    Самый экономный вариант: 1 запрос = 1 видео.
    """
    key  = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise Exception("GEMINI_API_KEY не найден!")

    client = genai.Client(api_key=key)
    loop   = asyncio.get_event_loop()

    # Определяем спикеров
    speakers = {}
    for seg in segs:
        spk = seg.get("speaker", "A")
        if spk not in speakers:
            gender = seg.get("gender", "male")
            speakers[spk] = {
                "voice": VOICE_FEMALE if gender == "female" else VOICE_MALE,
                "label": f"Speaker{len(speakers)+1}"
            }

    # Строим полный скрипт с паузами по времени оригинала
    lines = []
    prev_end = 0.0
    for seg in segs:
        spk   = seg.get("speaker", "A")
        label = speakers[spk]["label"]
        raw   = seg.get("translated") or seg["text"]
        raw   = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
        raw   = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
        text  = normalize(raw)
        if text.strip():
            lines.append(f"{label}: {text}")
        prev_end = seg["end"]

    full_script = "\n".join(lines)
    print(f"  📝 {len(lines)} реплик → 1 запрос Gemini Multi-Speaker TTS")
    print(f"  🎭 {[(spk, d['voice']) for spk, d in speakers.items()]}")

    # Конфигурация голосов
    voice_configs = [
        types.SpeakerVoiceConfig(
            speaker      = data["label"],
            voice_config = types.VoiceConfig(
                prebuilt_voice_config = types.PrebuiltVoiceConfig(
                    voice_name = data["voice"]
                )
            )
        )
        for spk, data in speakers.items()
    ]

    full_wav = os.path.join(tmp, "dubbed_full.wav")

    def call_multi():
        return client.models.generate_content(
            model    = GEMINI_TTS,
            contents = full_script,
            config   = types.GenerateContentConfig(
                response_modalities = ["AUDIO"],
                speech_config = types.SpeechConfig(
                    multi_speaker_voice_config = types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs = voice_configs
                    )
                ),
            ),
        )

    try:
        r     = await loop.run_in_executor(None, call_multi)
        audio = r.candidates[0].content.parts[0].inline_data.data
        save_wav(audio, full_wav)
        print(f"  ✅ Аудио готово: {get_dur(full_wav):.1f}с")

        # Возвращаем один "сегмент" — всё аудио с delay=0
        return [{
            "start":      0.0,
            "end":        get_dur(full_wav),
            "audio_file": full_wav,
            "speaker":    "ALL",
            "gender":     "male",
            "translated": full_script,
        }]

    except Exception as e:
        print(f"  ⚠️ Multi-speaker failed: {e}")
        print(f"  🔄 Fallback: per-segment...")
        result = []
        for i, seg in enumerate(segs):
            raw  = seg.get("translated") or seg["text"]
            raw  = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
            raw  = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
            text = normalize(raw)
            if not text.strip(): continue
            voice = VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE
            out   = os.path.join(tmp, f"{i:04d}.wav")
            try:
                audio2 = await loop.run_in_executor(None, tts_one, text, voice, key)
                save_wav(audio2, out)
                result.append({**seg, "audio_file": out})
                if i < len(segs) - 1:
                    await asyncio.sleep(6)
            except Exception as e2:
                print(f"  ❌ Seg {i}: {e2}")
        return result


# ── 4b. Генерация TTS ────────────────────────

async def generate_all_tts(segs: list, tmp: str) -> list:
    """Анализирует видео через Gemini Vision + генерирует multi-speaker TTS"""
    key  = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise Exception("GEMINI_API_KEY не найден!")
    loop = asyncio.get_event_loop()

    video_path = segs[0].get("_video_path", "") if segs else ""
    script, speakers = await analyze_video_and_build_script(video_path, segs, key)

    print(f"  🎭 Спикеры: {[(spk, d['voice']) for spk, d in speakers.items()]}")

    voice_configs = [
        types.SpeakerVoiceConfig(
            speaker      = data["label"],
            voice_config = types.VoiceConfig(
                prebuilt_voice_config = types.PrebuiltVoiceConfig(
                    voice_name = data["voice"]
                )
            )
        )
        for spk, data in speakers.items()
    ]

    full_wav = os.path.join(tmp, "dubbed_full.wav")

    def call_multi():
        # Создаём новый клиент для TTS запроса
        tts_client = genai.Client(api_key=key)
        return tts_client.models.generate_content(
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

    try:
        r     = await loop.run_in_executor(None, call_multi)
        audio = r.candidates[0].content.parts[0].inline_data.data
        save_wav(audio, full_wav)
        dur = get_dur(full_wav)
        print(f"  ✅ Аудио готово: {dur:.1f}с")
        return [{"start": 0.0, "end": dur, "audio_file": full_wav,
                 "speaker": "ALL", "gender": "male", "translated": script}]
    except Exception as e:
        print(f"  ⚠️ Multi-speaker failed: {e} — fallback per-segment...")
        result = []
        for i, seg in enumerate(segs):
            raw  = seg.get("translated") or seg["text"]
            raw  = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
            raw  = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
            text = normalize(raw)
            if not text.strip(): continue
            voice = VOICE_FEMALE if seg.get("gender") == "female" else VOICE_MALE
            out   = os.path.join(tmp, f"{i:04d}.wav")
            try:
                audio2 = await loop.run_in_executor(None, tts_one, text, voice, key)
                save_wav(audio2, out)
                result.append({**seg, "audio_file": out})
                if i < len(segs) - 1:
                    await asyncio.sleep(6)
            except Exception as e2:
                print(f"  ❌ Seg {i}: {e2}")
        return result


# ── 5. Сборка видео ──────────────────────────

def build_video(video: str, segs: list, out: str):
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video],
        capture_output=True, text=True
    )
    dur    = float(json.loads(probe.stdout)["format"]["duration"])
    valid  = [s for s in segs if os.path.exists(s.get("audio_file", ""))]
    if not valid:
        raise Exception("Нет аудио сегментов!")

    inputs  = ["-i", video]
    filters = [f"[0:a]volume={ORIG_VOL}[orig]"]
    labels  = []

    for i, s in enumerate(valid):
        inputs  += ["-i", s["audio_file"]]
        delay    = int(s["start"] * 1000)
        lbl      = f"v{i}"
        filters.append(f"[{i+1}:a]adelay={delay}|{delay},volume={DUB_VOL}[{lbl}]")
        labels.append(f"[{lbl}]")

    all_in = "[orig]" + "".join(labels)
    filters.append(f"{all_in}amix=inputs={len(labels)+1}:normalize=0:dropout_transition=0[aout]")

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
    groq_api_key: str  = "",
    src_language: str  = None,
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
            raise Exception("Речь не найдена в видео")

        cb("🤝 3/5 Tarjima va jins aniqlanmoqda (Gemini)...")
        segs = align_and_detect(segs, translation)

        cb(f"🎤 4/5 Ovoz yaratilmoqda — Gemini video analiz qilmoqda...")
        # Передаём путь к видео через сегменты для анализа
        for seg in segs:
            seg["_video_path"] = video_path
        audio_segs = await generate_all_tts(segs, tmp)

        cb("🎬 5/5 Video yig'ilmoqda...")
        build_video(video_path, audio_segs, output_path)

    return {"output": output_path, "segments": segs}
