# 🎙️ Uzbek TTS — O'zbek ovoz generatori

## Tezkor boshlash / Быстрый старт

### 1. Зависимости
```bash
pip install transformers torch scipy fastapi uvicorn
```

### 2. Тестовое аудио (5 примеров)
```bash
python generate_audio.py
```
После этого открой `index.html` в браузере — там будут все 5 аудио с плеером.

### 3. Полноценный сервер (API + веб-интерфейс)
```bash
python server.py
```
Открой http://localhost:8000 — там можно вводить любой текст и слушать.

---

## 📁 Структура файлов
```
uzbek_tts_test/
├── generate_audio.py   # Генерирует 5 тестовых WAV файлов
├── server.py           # FastAPI сервер с REST API
├── index.html          # Веб-плеер для прослушивания
├── audio_samples/      # Папка с WAV файлами (создаётся автоматически)
└── README.md
```

## 🔌 API
```
POST /tts
{"text": "Салом дунё!"}
→ audio/wav
```

## 📝 Модель
- **Название:** MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic
- **Архитектура:** VITS (Fine-tuned от Facebook MMS-TTS)
- **Язык:** Узбекский (кириллица)
- **Размер:** ~36M параметров
- **Лицензия:** CC BY 4.0
