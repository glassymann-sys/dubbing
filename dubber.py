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


async def generate_all_tts(segs: list, tmp: str) -> list:
    """
    Gemini Multi-Speaker TTS — ОДИН запрос на всё видео!
    Текст передаётся с метками Speaker1/Speaker2, Gemini генерирует
    всё аудио за один раз с разными голосами.
    Потом нарезаем по временным меткам оригинала.
    """
    key  = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise Exception("GEMINI_API_KEY не найден!")

    client = genai.Client(api_key=key)

    # Определяем спикеров и их голоса
    speakers  = {}
    for seg in segs:
        spk = seg.get("speaker", "A")
        if spk not in speakers:
            gender = seg.get("gender", "male")
            speakers[spk] = {
                "voice":  VOICE_FEMALE if gender == "female" else VOICE_MALE,
                "gender": gender,
                "label":  f"Speaker{len(speakers)+1}"
            }

    print(f"  🎭 Спикеры: {[(spk, d['voice']) for spk, d in speakers.items()]}")

    # Строим текст с метками спикеров
    script_parts = []
    for seg in segs:
        spk   = seg.get("speaker", "A")
        label = speakers[spk]["label"]
        raw   = seg.get("translated") or seg["text"]
        raw   = re.sub(r'^[^:：]+[:：]\s*', '', raw).strip()
        raw   = re.sub(r'[^\w\s.,!?\'"-]', '', raw).strip()
        text  = normalize(raw)
        if text.strip():
            script_parts.append(f"{label}: {text}")

    full_script = "\n".join(script_parts)
    print(f"  📝 Скрипт ({len(script_parts)} реплик) → 1 запрос к Gemini TTS")

    # Настраиваем голоса для каждого спикера
    voice_configs = []
    for spk, data in speakers.items():
        voice_configs.append(
            types.SpeakerVoiceConfig(
                speaker       = data["label"],
                voice_config  = types.VoiceConfig(
                    prebuilt_voice_config = types.PrebuiltVoiceConfig(
                        voice_name = data["voice"]
                    )
                )
            )
        )

    # Один запрос для всего видео!
    full_wav = os.path.join(tmp, "full_audio.wav")
    try:
        loop  = asyncio.get_event_loop()

        def call_tts():
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

        r          = await loop.run_in_executor(None, call_tts)
        audio_data = r.candidates[0].content.parts[0].inline_data.data
        save_wav(audio_data, full_wav)
        print(f"  ✅ Gemini multi-speaker аудио готово!")

    except Exception as e:
        print(f"  ⚠️ Multi-speaker failed: {e}")
        print(f"  🔄 Fallback: генерирую по одному сегменту...")
        # Фолбэк — по одному
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
                audio = await loop.run_in_executor(
                    None, tts_one, text, voice, key
                )
                save_wav(audio, out)
                result.append({**seg, "audio_file": out})
                if i < len(segs) - 1:
                    await asyncio.sleep(6)
            except Exception as e2:
                print(f"  ❌ Seg {i}: {e2}")
        return result

    # Нарезаем полное аудио по паузам между репликами
    full_dur = get_dur(full_wav)
    print(f"  ✂️  Нарезаю аудио ({full_dur:.1f}с) на {len(segs)} сегментов по паузам...")

    # Читаем аудио и находим паузы через энергию сигнала
    with wave.open(full_wav, 'rb') as wf:
        fr   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    # Находим паузы (тишина < порога)
    frame_ms  = 20  # 20ms фреймы
    frame_sz  = int(fr * frame_ms / 1000)
    energies  = []
    for j in range(0, len(audio_np) - frame_sz, frame_sz):
        chunk = audio_np[j:j+frame_sz]
        energies.append(float(np.sqrt(np.mean(chunk**2))))

    # Порог тишины
    max_e     = max(energies) if energies else 1
    threshold = max_e * 0.02  # 2% от максимума = тишина

    # Находим границы реплик (переходы тишина→речь)
    boundaries = []
    in_speech  = False
    for j, e in enumerate(energies):
        t = j * frame_ms / 1000.0
        if not in_speech and e > threshold:
            boundaries.append(("start", t))
            in_speech = True
        elif in_speech and e <= threshold:
            boundaries.append(("end", t))
            in_speech = False
    if in_speech:
        boundaries.append(("end", full_dur))

    # Собираем отрезки речи
    speech_segments = []
    for j in range(0, len(boundaries)-1, 2):
        if boundaries[j][0] == "start" and boundaries[j+1][0] == "end":
            speech_segments.append((boundaries[j][1], boundaries[j+1][1]))

    print(f"  🔍 Найдено {len(speech_segments)} речевых отрезков в аудио")

    result = []
    # Сопоставляем найденные отрезки с оригинальными сегментами
    n_match = min(len(speech_segments), len(segs))

    for i, seg in enumerate(segs):
        out_seg = os.path.join(tmp, f"{i:04d}.wav")

        if i < len(speech_segments):
            s_start, s_end = speech_segments[i]
            s_dur = s_end - s_start
            subprocess.run([
                "ffmpeg", "-i", full_wav,
                "-ss", f"{s_start:.3f}",
                "-t",  f"{s_dur:.3f}",
                "-y",  out_seg
            ], capture_output=True)
        else:
            # Если отрезков меньше чем сегментов — берём по пропорции
            video_dur = segs[-1]["end"] if segs else full_dur
            scale     = full_dur / video_dur if video_dur > 0 else 1.0
            adj_start = seg["start"] * scale
            adj_dur   = (seg["end"] - seg["start"]) * scale
            subprocess.run([
                "ffmpeg", "-i", full_wav,
                "-ss", f"{adj_start:.3f}",
                "-t",  f"{adj_dur:.3f}",
                "-y",  out_seg
            ], capture_output=True)

        result.append({**seg, "audio_file": out_seg})

    print(f"✅ TTS готов! 1 запрос → {len(result)} сегментов")
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

        cb(f"🎤 4/5 Ovoz yaratilmoqda ({len(segs)} segment × 6s)...")
        audio_segs = await generate_all_tts(segs, tmp)

        cb("🎬 5/5 Video yig'ilmoqda...")
        build_video(video_path, audio_segs, output_path)

    return {"output": output_path, "segments": segs}
