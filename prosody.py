# -*- coding: utf-8 -*-
"""
Prosody Transfer — переносит интонацию из оригинала в TTS аудио
Анализирует pitch (F0) оригинала и применяет к дублёру через ffmpeg
"""

import subprocess
import numpy as np
import wave
import os


def get_audio_chunk(audio_data: np.ndarray, start: float,
                    end: float, framerate: int) -> np.ndarray:
    s = int(start * framerate)
    e = int(end   * framerate)
    return audio_data[s:e]


def estimate_pitch_stats(chunk: np.ndarray, framerate: int) -> dict:
    """
    Оценивает статистику pitch сегмента:
    - средний pitch
    - мин/макс pitch (диапазон интонации)
    - энергия (громкость)
    - скорость изменения pitch (динамика)
    """
    if len(chunk) < framerate * 0.1:
        return {"mean": 150, "min": 100, "max": 200, "energy": 0.5, "dynamic": 0.3}

    chunk = chunk - chunk.mean()

    # Автокорреляция для pitch
    corr = np.correlate(chunk, chunk, mode='full')
    corr = corr[len(corr)//2:]

    min_lag = int(framerate / 400)  # 400 Hz max
    max_lag = int(framerate / 60)   # 60 Hz min

    if max_lag >= len(corr) or min_lag >= len(corr):
        return {"mean": 150, "min": 100, "max": 200, "energy": 0.5, "dynamic": 0.3}

    # Находим pitch по фреймам
    frame_size   = int(framerate * 0.025)  # 25ms фреймы
    frame_step   = int(framerate * 0.010)  # 10ms шаг
    pitches      = []

    for i in range(0, len(chunk) - frame_size, frame_step):
        frame = chunk[i:i + frame_size]
        if np.max(np.abs(frame)) < 100:  # тишина
            continue
        c = np.correlate(frame, frame, mode='full')
        c = c[len(c)//2:]
        if max_lag >= len(c):
            continue
        peak = np.argmax(c[min_lag:max_lag]) + min_lag
        if peak > 0:
            p = framerate / peak
            if 60 <= p <= 400:
                pitches.append(p)

    if not pitches:
        return {"mean": 150, "min": 100, "max": 200, "energy": 0.5, "dynamic": 0.3}

    pitches = np.array(pitches)
    energy  = float(np.sqrt(np.mean(chunk**2))) / 32768.0

    # Динамика = насколько сильно меняется pitch
    if len(pitches) > 1:
        dynamic = float(np.std(np.diff(pitches)) / (np.mean(pitches) + 1e-6))
    else:
        dynamic = 0.3

    return {
        "mean":    float(np.mean(pitches)),
        "min":     float(np.min(pitches)),
        "max":     float(np.max(pitches)),
        "energy":  min(energy * 10, 1.0),
        "dynamic": min(dynamic, 1.0),
        "count":   len(pitches)
    }


def load_audio(path: str) -> tuple:
    """Загружает WAV файл, возвращает (data, framerate)"""
    with wave.open(path, 'rb') as wf:
        framerate = wf.getframerate()
        raw       = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return data, framerate


def apply_prosody(tts_path: str, out_path: str, orig_stats: dict) -> str:
    """
    Применяет просодию к TTS аудио.
    ТОЛЬКО высота тона — НЕ меняем скорость (она уже настроена в fit_to_duration).
    Это убирает роботизированность через лёгкое изменение pitch.
    """
    orig_mean = orig_stats.get("mean", 150)

    # Базовый pitch для голосов
    if orig_mean < 165:
        base_pitch = 120  # мужской
    else:
        base_pitch = 210  # женский

    # Смещение в полутонах — очень мягко ±2 полутона max
    if orig_mean > 0 and base_pitch > 0:
        pitch_ratio = orig_mean / base_pitch
        semitones   = 12 * np.log2(max(pitch_ratio, 0.1))
        semitones   = max(-2, min(semitones, 2))
    else:
        semitones = 0

    # Применяем только если смещение значимое
    if abs(semitones) > 0.3:
        rate = 2 ** (semitones / 12)
        cmd  = [
            "ffmpeg", "-i", tts_path,
            "-filter:a", f"asetrate=24000*{rate:.4f},aresample=24000",
            "-y", out_path
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            try: os.remove(tts_path)
            except: pass
            return out_path

    try:
        os.rename(tts_path, out_path)
    except:
        pass
    return out_path


def analyze_segment_prosody(audio_data: np.ndarray, framerate: int,
                             start: float, end: float) -> dict:
    """Анализирует просодию конкретного сегмента"""
    chunk = get_audio_chunk(audio_data, start, end, framerate)
    return estimate_pitch_stats(chunk, framerate)
