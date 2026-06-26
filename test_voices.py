import os
import wave
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

voices = ['Kore', 'Puck', 'Charon', 'Fenrir', 'Aoede', 'Leda']

for voice in voices:
    print(f"Генерирую {voice}...")
    r = client.models.generate_content(
        model='gemini-3.1-flash-tts-preview',
        contents='Salom! Men sizga yordam berishga tayyorman. Bugun havo juda yaxshi.',
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    audio = r.candidates[0].content.parts[0].inline_data.data
    with wave.open(f'voice_{voice}.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio)
    print(f"  OK voice_{voice}.wav")

print("Готово!")
