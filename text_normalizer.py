"""
Uzbek TTS — Text Normalizer
Нормализует текст перед отправкой в Edge TTS:
- Раскрывает аббревиатуры
- Числа → слова
- Спецсимволы → паузы
- Латиница → кириллица/узбекская латиница
"""

import re


# ─────────────────────────────────────────────────────────────
# Словари замен
# ─────────────────────────────────────────────────────────────

# Аббревиатуры и сокращения → произношение
ABBREVIATIONS = {
    # Общие
    "TTS":   "Ти-Ти-Эс",
    "AI":    "Эй-Ай",
    "API":   "Эй-Пи-Ай",
    "URL":   "Ю-Эр-Эл",
    "HTTP":  "Эйч-Ти-Ти-Пи",
    "HTTPS": "Эйч-Ти-Ти-Пи-Эс",
    "ID":    "Ай-Ди",
    "SMS":   "Эс-Эм-Эс",
    "UZS":   "сўм",
    "USD":   "доллар",
    "EUR":   "евро",
    "RUB":   "рубль",
    "kg":    "килограмм",
    "km":    "километр",
    "cm":    "сантиметр",
    "mm":    "миллиметр",
    "ml":    "миллилитр",
    "l":     "литр",
    "g":     "грамм",
    "т.е.":  "яъни",
    "т.д.":  "ва ҳоказо",
    "и т.д.":"ва ҳоказо",
    "и т.п.":"ва шу кабилар",
    "др.":   "ва бошқалар",
    "пр.":   "масалан",
    "ул.":   "кўча",
    "пр-т":  "проспект",
    "г.":    "йил",
    "кг":    "килограмм",
    "шт":    "дона",
    "шт.":   "дона",
    "руб":   "рубль",
    "руб.":  "рубль",
    "сум":   "сўм",
    "тыс":   "минг",
    "тыс.":  "минг",
    "млн":   "миллион",
    "млрд":  "миллиард",
}

# Числа до 20
ONES = {
    "0": "нол", "1": "бир", "2": "икки", "3": "уч", "4": "тўрт",
    "5": "беш", "6": "олти", "7": "етти", "8": "саккиз", "9": "тўqqиз",
    "10": "ўн", "11": "ўн бир", "12": "ўн икки", "13": "ўн уч",
    "14": "ўн тўрт", "15": "ўн беш", "16": "ўн олти", "17": "ўн етти",
    "18": "ўн саккиз", "19": "ўн тўqqиз",
}

TENS = {
    "2": "йигирма", "3": "ўттиз", "4": "қирқ", "5": "эллик",
    "6": "олтмиш", "7": "етмиш", "8": "саксон", "9": "тўqсон",
}


def number_to_uzbek(n: int) -> str:
    """Число → узбекские слова (до миллиарда)"""
    if n < 0:
        return "минус " + number_to_uzbek(-n)
    if n <= 19:
        return ONES[str(n)]
    if n < 100:
        t = TENS[str(n // 10)]
        o = ONES[str(n % 10)] if n % 10 != 0 else ""
        return (t + " " + o).strip()
    if n < 1000:
        h = ONES[str(n // 100)] + " юз"
        rest = number_to_uzbek(n % 100) if n % 100 != 0 else ""
        return (h + " " + rest).strip()
    if n < 1_000_000:
        th = number_to_uzbek(n // 1000) + " минг"
        rest = number_to_uzbek(n % 1000) if n % 1000 != 0 else ""
        return (th + " " + rest).strip()
    if n < 1_000_000_000:
        m = number_to_uzbek(n // 1_000_000) + " миллион"
        rest = number_to_uzbek(n % 1_000_000) if n % 1_000_000 != 0 else ""
        return (m + " " + rest).strip()
    b = number_to_uzbek(n // 1_000_000_000) + " миллиард"
    rest = number_to_uzbek(n % 1_000_000_000) if n % 1_000_000_000 != 0 else ""
    return (b + " " + rest).strip()


def expand_numbers(text: str) -> str:
    """Заменяет числа в тексте на узбекские слова"""
    def replace(m):
        raw = m.group(0).replace(" ", "").replace(",", "")
        try:
            return number_to_uzbek(int(raw))
        except:
            return m.group(0)
    # Большие числа с разделителями (1 000 000 / 1,000,000)
    text = re.sub(r'\b\d{1,3}(?:[ ,]\d{3})+\b', replace, text)
    # Обычные числа
    text = re.sub(r'\b\d+\b', replace, text)
    return text


def expand_abbreviations(text: str) -> str:
    """Заменяет аббревиатуры на произношение"""
    for abbr, reading in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        text = pattern.sub(reading, text)
    return text


def normalize_punctuation(text: str) -> str:
    """Нормализует знаки препинания для TTS"""
    # Тире → пауза (запятая)
    text = re.sub(r'\s*[-–—]\s*', ', ', text)
    # Слэш → или
    text = re.sub(r'\s*/\s*', ' ёки ', text)
    # Подчёркивание → пробел
    text = text.replace('_', ' ')
    # Многоточие → пауза
    text = re.sub(r'\.{2,}', '...', text)
    # Несколько пробелов → один
    text = re.sub(r' +', ' ', text)
    # Скобки — убираем
    text = re.sub(r'[(){}\[\]]', ', ', text)
    # Кавычки — убираем
    text = re.sub(r'[«»"\'`]', '', text)
    # Символы которые TTS читает странно
    text = re.sub(r'[#@$%^&*+=|\\<>~]', ' ', text)
    return text.strip()


def spell_out_mixed(text: str) -> str:
    """
    Читает смешанные коды по буквам/цифрам.
    Например: BV3.org2001 → Би-Ви-три орг две тысячи один
    """
    # Домены и URL
    text = re.sub(r'https?://\S+', 'ссылка', text)
    text = re.sub(r'www\.\S+',     'сайт',   text)
    text = re.sub(r'\S+\.(com|org|net|uz|ru|io)\b', 'сайт', text, flags=re.IGNORECASE)

    # Email
    text = re.sub(r'\S+@\S+\.\S+', 'электрон почта', text)

    # Артикулы вида ABC-123, XY_456 (буквы+цифры вместе)
    def spell_code(m):
        code = m.group(0)
        parts = []
        for ch in code:
            if ch.isdigit():
                parts.append(number_to_uzbek(int(ch)))
            elif ch.isalpha():
                parts.append(ch.upper())
            else:
                parts.append(' ')
        return '-'.join(p for p in parts if p.strip())

    text = re.sub(r'\b[A-Za-z]+\d+\w*\b', spell_code, text)
    text = re.sub(r'\b\d+[A-Za-z]+\w*\b', spell_code, text)

    return text


def add_natural_pauses(text: str) -> str:
    """Добавляет запятые там, где нужны паузы"""
    # После «однако», «поэтому», «таким образом» и т.д.
    connectors = ["аммо", "лекин", "шунинг учун", "шундай қилиб", "бироқ", "демак"]
    for word in connectors:
        text = re.sub(r'\b' + word + r'\b', f'{word},', text, flags=re.IGNORECASE)
    return text


def normalize(text: str, expand_nums: bool = True) -> str:
    """
    Главная функция нормализации.
    Прогоняет текст через все этапы.
    """
    text = spell_out_mixed(text)
    text = expand_abbreviations(text)
    if expand_nums:
        text = expand_numbers(text)
    text = normalize_punctuation(text)
    text = add_natural_pauses(text)
    # Убираем двойные запятые
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Тест
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Mahsulot ID: BV3.org2001 narxi 150000 UZS",
        "HTTPS://shop.uz/product_123 orqali 2 ta buyurtma bering",
        "SMS-kod: 4521, telefon: +998901234567",
        "Og'irligi: 1.5 kg, hajmi: 250 ml",
        "API kaliti: XY_456-ABC ni server.py ga kiriting",
        "Bugun 15.06.2025 da 1000000 so'm to'landi",
        "Mijoz ID #4521 — VIP status — 30% chegirma",
    ]

    print("=" * 60)
    print("🧪 TEXT NORMALIZER — TEST")
    print("=" * 60)
    for t in test_cases:
        result = normalize(t)
        print(f"\n📥 Кирди:  {t}")
        print(f"📤 Чиқди: {result}")
    print("\n" + "=" * 60)
