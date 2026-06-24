"""
Uzbek TTS — Microsoft Edge TTS + SSML разметка
SSML позволяет добавить паузы, ударения, эмоции на уровне слов

Запуск:
    pip3 install edge-tts
    python3 generate_audio.py
"""

import asyncio
import os
import json
import edge_tts

# ─────────────────────────────────────────────
VOICE_FEMALE = "uz-UZ-MadinaNeural"
VOICE_MALE   = "uz-UZ-SardorNeural"

VOICE = VOICE_FEMALE   # Поменяй на VOICE_MALE если нужен мужской
# ─────────────────────────────────────────────

# SSML шаблон — добавляем паузы, ударения, изменения темпа
def make_ssml(text: str, voice: str) -> str:
    return f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="https://www.w3.org/2001/mstts"
    xml:lang="uz-UZ">
  <voice name="{voice}">
    <mstts:express-as style="friendly" styledegree="1.5">
      <prosody rate="-8%" pitch="+1Hz">
        {text}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>"""


# Тексты с SSML паузами и ударениями
# <break time="300ms"/> — пауза
# <emphasis level="strong"> — ударение
# <prosody pitch="+10Hz"> — повысить тон на слово
SAMPLES_SSML = [
    (
        "Salom! Men sun'iy intellekt yordamida yaratilgan ovozman.",
        """<emphasis level="strong">Salom!</emphasis>
        <break time="250ms"/>
        Men <prosody pitch="+8Hz">sun'iy intellekt</prosody> yordamida
        <break time="150ms"/>
        yaratilgan ovozman."""
    ),
    (
        "Bugun ob-havo juda yaxshi, osmon ochiq.",
        """Bugun ob-havo
        <prosody pitch="+6Hz" rate="-5%"><emphasis level="moderate">juda yaxshi</emphasis></prosody>,
        <break time="200ms"/>
        osmon <prosody pitch="+4Hz">ochiq</prosody>."""
    ),
    (
        "Xush kelibsiz! Biz sizga yordam berishga tayyormiz.",
        """<prosody pitch="+8Hz"><emphasis level="strong">Xush kelibsiz!</emphasis></prosody>
        <break time="300ms"/>
        Biz sizga
        <prosody rate="-10%" pitch="+5Hz">yordam berishga</prosody>
        <break time="150ms"/>
        <emphasis level="moderate">tayyormiz</emphasis>."""
    ),
    (
        "O'zbekiston — go'zal va tarixiy yurt.",
        """<emphasis level="strong"><prosody pitch="+6Hz">O'zbekiston</prosody></emphasis>
        <break time="300ms"/>
        go'zal
        <break time="100ms"/>
        va
        <prosody pitch="+4Hz" rate="-8%"><emphasis level="moderate">tarixiy yurt</emphasis></prosody>."""
    ),
    (
        "Sog'liqni asrang va baxtli bo'ling!",
        """<prosody rate="-5%">Sog'liqni <emphasis level="moderate">asrang</emphasis></prosody>
        <break time="200ms"/>
        va
        <prosody pitch="+10Hz" rate="-10%"><emphasis level="strong">baxtli bo'ling!</emphasis></prosody>"""
    ),
]


async def generate_ssml(ssml_text: str, voice: str, output_path: str):
    full_ssml = make_ssml(ssml_text, voice)
    communicate = edge_tts.Communicate(text=full_ssml, voice=voice)
    await communicate.save(output_path)


async def main():
    os.makedirs("audio_samples", exist_ok=True)
    results = []

    print(f"🎤 Голос: {VOICE}")
    print(f"🎭 Режим: SSML (паузы + ударения + эмоции)\n")

    for i, (plain_text, ssml_text) in enumerate(SAMPLES_SSML, 1):
        out = f"audio_samples/sample_{i}.mp3"
        print(f"[{i}/{len(SAMPLES_SSML)}] 🎙️  {plain_text}")
        await generate_ssml(ssml_text, VOICE, out)
        print(f"✅ Сохранено: {out}\n")
        results.append({"text": plain_text, "file": out})

    print("=" * 55)
    print("🎉 Готово!")
    for r in results:
        print(f"  📁 {r['file']}")
    print("\n👉 Открой  index.html  в браузере чтобы послушать!")
    print("=" * 55)

    with open("audio_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
