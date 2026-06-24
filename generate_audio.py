"""
Uzbek TTS — тестовая генерация аудио
Модель: MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic

Запуск:
    pip install transformers torch scipy
    python generate_audio.py
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile


def generate(text: str, output_path: str):
    from transformers import VitsModel, AutoTokenizer
    import torch

    model_id = "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic"

    print(f"⏳ Загружаю модель (первый раз ~500MB, потом кэш)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    print("✅ Модель загружена!")

    print(f"🎙️  Текст: {text}")
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy()
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform_int16 = (waveform * 32767).astype(np.int16)

    sr = model.config.sampling_rate
    wavfile.write(output_path, sr, waveform_int16)

    duration = len(waveform_int16) / sr
    print(f"✅ Сохранено: {output_path}  ({duration:.2f} сек)")
    return sr


if __name__ == "__main__":
    # Тестовые фразы на узбекском (кириллица)
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
        print(f"\n[{i}/{len(samples)}]")
        generate(text, out)
        results.append({"text": text, "file": out})

    print("\n" + "=" * 55)
    print("🎉 Готово! Все аудио файлы сгенерированы:")
    for r in results:
        print(f"  📁 {r['file']}")
    print("\n👉 Открой  index.html  в браузере чтобы послушать!")
    print("=" * 55)

    # Автоматически обновляем index.html с реальными файлами
    import json
    with open("audio_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("📄 audio_data.json обновлён для веб-плеера.")
