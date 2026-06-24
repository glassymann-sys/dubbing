"""
Uzbek TTS — Microsoft Edge TTS
Голос: uz-UZ-MadinaNeural (живой узбекский голос)

Запуск:
    pip3 install edge-tts
    python3 generate_audio.py

Не требует API ключей, работает бесплатно!
"""

import asyncio
import os
import json
import edge_tts

# ─────────────────────────────────────────────
# 🎛️  НАСТРОЙКИ ГОЛОСА
# ─────────────────────────────────────────────
VOICE = "uz-UZ-MadinaNeural"       # Женский голос (узбекский)
# VOICE = "uz-UZ-SardorNeural"     # Мужской голос (раскомментируй если нужен)

RATE  = "-5%"    # Скорость: -10% медленнее, 0% норма, +10% быстрее
PITCH = "+0Hz"   # Высота: +5Hz выше, -5Hz ниже
VOLUME = "+0%"   # Громкость
# ─────────────────────────────────────────────

SAMPLES = [
    "Salom! Men sun'iy intellekt yordamida yaratilgan ovozman.",
    "Bugun ob-havo juda yaxshi, osmon ochiq.",
    "Xush kelibsiz! Biz sizga yordam berishga tayyormiz.",
    "O'zbekiston — go'zal va tarixiy yurt.",
    "Sog'liqni asrang va baxtli bo'ling!",
]


async def generate(text: str, output_path: str):
    print(f"🎙️  Текст: {text}")
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
        volume=VOLUME,
    )
    await communicate.save(output_path)
    print(f"✅ Сохранено: {output_path}")


async def main():
    os.makedirs("audio_samples", exist_ok=True)
    results = []

    print(f"🎤 Голос: {VOICE}")
    print(f"⚙️  Скорость: {RATE} | Высота: {PITCH}\n")

    for i, text in enumerate(SAMPLES, 1):
        out = f"audio_samples/sample_{i}.mp3"
        print(f"[{i}/{len(SAMPLES)}]")
        await generate(text, out)
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


if __name__ == "__main__":
    asyncio.run(main())
