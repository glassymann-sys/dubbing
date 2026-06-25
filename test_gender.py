import os
import assemblyai as aai
import subprocess
from groq import Groq
import json

aai.settings.api_key = os.environ['ASSEMBLYAI_API_KEY']
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Последний файл
import glob
files = sorted(glob.glob('uploads/*.mov') + glob.glob('uploads/*.mp4'))
video = files[-1]
print(f"Видео: {video}")

# Извлекаем аудио
subprocess.run(['ffmpeg', '-i', video, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', 'test_gender.wav'],
               capture_output=True)

# Транскрипция
print("Транскрибирую...")
config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
t = aai.Transcriber().transcribe('test_gender.wav', config=config)

print("\n=== ДИАЛОГ ===")
for u in t.utterances:
    print(f"Speaker {u.speaker} [{u.start/1000:.1f}s]: {u.text}")

speakers = list(set(u.speaker for u in t.utterances))
print(f"\nСпикеры: {speakers}")

# Groq определяет пол
dialog = "\n".join([f"Speaker {u.speaker}: {u.text}" for u in t.utterances])

r = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "Determine gender for each speaker. Reply ONLY with JSON like {\"A\": \"male\", \"B\": \"female\"}"
        },
        {
            "role": "user",
            "content": f"Dialog:\n{dialog}\n\nSpeakers to identify: {speakers}"
        }
    ],
    response_format={"type": "json_object"}
)

result = json.loads(r.choices[0].message.content)
print(f"\n=== РЕЗУЛЬТАТ ===")
for spk, gender in result.items():
    icon = "👩" if gender == "female" else "👨"
    print(f"Speaker {spk} → {icon} {gender}")
