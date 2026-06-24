"""
Uzbek TTS — тестовая генерация аудио
Модель: MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic

Запуск:
    python3 generate_audio.py

Поддерживает: Mac M1/M2 (MPS), NVIDIA (CUDA), CPU
"""

import os
import json
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import VitsModel, AutoTokenizer


def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Apple M1/M2 — используется MPS (Neural Engine)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("⚡ NVIDIA GPU — используется CUDA")
        return torch.device("cuda")
    else:
        print("💻 GPU не найден — используется CPU")
        return torch.device("cpu")


MODEL_ID = "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic"
DEVICE = get_device()

print("⏳ Загружаю модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
SAMPLE_RATE = model.config.sampling_rate
print(f"✅ Модель загружена! Устройство: {DEVICE}\n")

# ─────────────────────────────────────────────
# 🎛️  НАСТРОЙКИ КАЧЕСТВА — меняй здесь
# ─────────────────────────────────────────────
# speaking_rate: скорость речи
#   0.8 = медленнее (чётче)   1.0 = норма   1.2 = быстрее
SPEAKING_RATE = 0.82

# noise_scale: случайность интонации
#   0.1 = монотонно/чисто     0.667 = норма     1.0 = разнообразно
NOISE_SCALE = 0.3

# noise_scale_duration: вариация длительности звуков
#   0.1 = стабильно/чётко     0.8 = норма       1.0 = гибко
NOISE_SCALE_DURATION = 0.2
# ─────────────────────────────────────────────


def generate(text: str, output_path: str):
    print(f"🎙️  Текст: {text}")

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(
            **inputs,
            speaking_rate=SPEAKING_RATE,
            noise_scale=NOISE_SCALE,
            noise_scale_duration=NOISE_SCALE_DURATION,
        ).waveform

    waveform = output.squeeze().cpu().numpy()
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform_int16 = (waveform * 32767).astype(np.int16)

    wavfile.write(output_path, SAMPLE_RATE, waveform_int16)

    duration = len(waveform_int16) / SAMPLE_RATE
    print(f"✅ Сохранено: {output_path}  ({duration:.2f} сек)")


if __name__ == "__main__":
    samples = [
        "Салом! Мен сунъий интеллект ёрдамида яратилган овозман.",
        "Бугун об-ҳаво жуда яхши, осмон очиқ.",
        "Хуш келибсиз! Биз сизга ёрдам беришга тайёрмиз.",
        "Ўзбекистон — гўзал ва тарихий юрт.",
        "Саломатликни асранг ва бахтли бўлинг!",
    ]

    os.makedirs("audio_samples", exist_ok=True)
    results = []

    for i, text in enumerate(samples, 1):
        out = f"audio_samples/sample_{i}.wav"
        print(f"[{i}/{len(samples)}]")
        generate(text, out)
        results.append({"text": text, "file": out})
        print()

    print("=" * 55)
    print("🎉 Готово! Все аудио файлы сгенерированы:")
    for r in results:
        print(f"  📁 {r['file']}")
    print("\n👉 Открой  index.html  в браузере чтобы послушать!")
    print("=" * 55)

    with open("audio_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("📄 audio_data.json обновлён.")
