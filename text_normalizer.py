"""
Uzbek TTS — Text Normalizer v2
Нормализует текст перед отправкой в Edge TTS
"""

import re


# ─────────────────────────────────────────────
# Числа → узбекские слова
# ─────────────────────────────────────────────

ONES = [
    "", "bir", "ikki", "uch", "to'rt", "besh",
    "olti", "yetti", "sakkiz", "to'qqiz"
]
TEENS = [
    "o'n", "o'n bir", "o'n ikki", "o'n uch", "o'n to'rt",
    "o'n besh", "o'n olti", "o'n yetti", "o'n sakkiz", "o'n to'qqiz"
]
TENS = [
    "", "o'n", "yigirma", "o'ttiz", "qirq", "ellik",
    "oltmish", "yetmish", "sakson", "to'qson"
]


def number_to_uzbek(n: int) -> str:
    if n < 0:
        return "minus " + number_to_uzbek(-n)
    if n == 0:
        return "nol"
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        t = TENS[n // 10]
        o = ONES[n % 10]
        return (t + " " + o).strip()
    if n < 1000:
        h = ONES[n // 100] + " yuz"
        rest = number_to_uzbek(n % 100) if n % 100 else ""
        return (h + " " + rest).strip()
    if n < 1_000_000:
        th = number_to_uzbek(n // 1000) + " ming"
        rest = number_to_uzbek(n % 1000) if n % 1000 else ""
        return (th + " " + rest).strip()
    if n < 1_000_000_000:
        m = number_to_uzbek(n // 1_000_000) + " million"
        rest = number_to_uzbek(n % 1_000_000) if n % 1_000_000 else ""
        return (m + " " + rest).strip()
    b = number_to_uzbek(n // 1_000_000_000) + " milliard"
    rest = number_to_uzbek(n % 1_000_000_000) if n % 1_000_000_000 else ""
    return (b + " " + rest).strip()


# ─────────────────────────────────────────────
# Аббревиатуры (только целые слова)
# ─────────────────────────────────────────────

ABBREVIATIONS = {
    "UZS": "so'm",
    "USD": "dollar",
    "EUR": "evro",
    "RUB": "rubl",
    "TTS": "te-te-es",
    "API": "a-pi-ay",
    "SMS": "es-em-es",
    "VIP": "vi-ay-pi",
    "ID":  "aydiy",
    "URL": "yu-ar-el",
    "AI":  "sun'iy intellekt",
    "HTTP":  "ey-ti-ti-pi",
    "HTTPS": "ey-ti-ti-pi-es",
    "kg": "kilogramm",
    "ml": "millilitr",
    "km": "kilometr",
    "cm": "santimetr",
    "mm": "millimetr",
    "kW": "kilovat",
    "GB": "gigabayt",
    "MB": "megabayt",
    "KB": "kilobayt",
}


def expand_abbreviations(text: str) -> str:
    for abbr, reading in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', reading, text)
    return text


# ─────────────────────────────────────────────
# Спецсимволы и форматирование
# ─────────────────────────────────────────────

def clean_special(text: str) -> str:
    # URL → "havola"
    text = re.sub(r'https?://\S+', 'havola', text)
    text = re.sub(r'www\.\S+', 'sayt', text)
    # Email → "elektron pochta"
    text = re.sub(r'\S+@\S+\.\S+', 'elektron pochta', text)
    # Телефон +998... → цифры по одной
    def phone_to_words(m):
        digits = re.sub(r'\D', '', m.group(0))
        return ' '.join(number_to_uzbek(int(d)) for d in digits)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}', phone_to_words, text)
    # Дата ДД.ММ.ГГГГ → "15-iyun 2025-yil"
    def date_replace(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        months = {
            "01":"yanvar","02":"fevral","03":"mart","04":"aprel",
            "05":"may","06":"iyun","07":"iyul","08":"avgust",
            "09":"sentabr","10":"oktabr","11":"noyabr","12":"dekabr"
        }
        month = months.get(mo, mo)
        return f"{number_to_uzbek(int(d))}-{month} {number_to_uzbek(int(y))}-yil"
    text = re.sub(r'\b(\d{1,2})\.(\d{2})\.(\d{4})\b', date_replace, text)
    # Десятичные числа 1.5 → "bir butun besh"
    def decimal_replace(m):
        left, right = m.group(1), m.group(2)
        return number_to_uzbek(int(left)) + " butun " + number_to_uzbek(int(right))
    text = re.sub(r'\b(\d+)\.(\d+)\b', decimal_replace, text)
    # % → "foiz"
    text = re.sub(r'(\d+)\s*%', lambda m: number_to_uzbek(int(m.group(1))) + " foiz", text)
    # № или # → "raqam"
    text = re.sub(r'[#№]\s*(\d+)', lambda m: "raqam " + number_to_uzbek(int(m.group(1))), text)
    # Тире/дефис между словами → пауза (запятая)
    text = re.sub(r'\s*[-–—]\s*', ', ', text)
    # Слэш → "yoki"
    text = re.sub(r'\s*/\s*', ' yoki ', text)
    # Подчёркивание → пробел
    text = text.replace('_', ' ')
    # Скобки → убираем
    text = re.sub(r'[(){}\[\]]', ', ', text)
    # Кавычки → убираем
    text = re.sub(r'[«»""\'`]', '', text)
    # Остальные спецсимволы
    text = re.sub(r'[@$^&*+=|\\<>~]', ' ', text)
    return text


# ─────────────────────────────────────────────
# Числа → слова (только чистые целые)
# ─────────────────────────────────────────────

def expand_numbers(text: str) -> str:
    # Числа с разделителями: 1 000 000 или 1,000,000
    def big_num(m):
        raw = re.sub(r'[\s,]', '', m.group(0))
        return number_to_uzbek(int(raw))
    text = re.sub(r'\b\d{1,3}(?:[,\s]\d{3})+\b', big_num, text)
    # Обычные числа
    text = re.sub(r'\b\d+\b', lambda m: number_to_uzbek(int(m.group(0))), text)
    return text


# ─────────────────────────────────────────────
# Смешанные коды: ABC123, XY-456
# Читаем по буквам
# ─────────────────────────────────────────────

LETTER_READ = {
    'A':'a','B':'be','C':'se','D':'de','E':'e','F':'ef',
    'G':'ge','H':'ha','I':'i','J':'jey','K':'ka','L':'el',
    'M':'em','N':'en','O':'o','P':'pe','Q':'qu','R':'ar',
    'S':'es','T':'te','U':'u','V':'ve','W':'double-yu',
    'X':'iks','Y':'ey','Z':'zed',
}


def spell_code(code: str) -> str:
    parts = []
    i = 0
    while i < len(code):
        if code[i].isdigit():
            # Собираем всю цифровую последовательность
            j = i
            while j < len(code) and code[j].isdigit():
                j += 1
            parts.append(number_to_uzbek(int(code[i:j])))
            i = j
        elif code[i].isalpha():
            parts.append(LETTER_READ.get(code[i].upper(), code[i]))
            i += 1
        else:
            i += 1
    return '-'.join(parts)


def expand_mixed_codes(text: str) -> str:
    # Код = минимум 2 символа, смесь букв и цифр
    def replace(m):
        token = m.group(0)
        has_letter = bool(re.search(r'[A-Za-z]', token))
        has_digit  = bool(re.search(r'\d', token))
        if has_letter and has_digit:
            return spell_code(token)
        return token
    text = re.sub(r'\b[A-Za-z0-9]{2,}\b', replace, text)
    return text


# ─────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    text = clean_special(text)          # 1. URL, email, дата, %, #, тире
    text = expand_abbreviations(text)   # 2. UZS→so'm, SMS→es-em-es
    text = expand_mixed_codes(text)     # 3. BV3→Be-Ve-uch
    text = expand_numbers(text)         # 4. 150000 → bir yuz ellik ming
    # Финальная чистка
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    return text.strip()


# ─────────────────────────────────────────────
# Тест
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Mahsulot ID: BV3.org2001 narxi 150000 UZS",
         "Mahsulot aydiy: Be-Ve-uch ... bir yuz ellik ming so'm"),

        ("HTTPS://shop.uz/product_123 orqali 2 ta buyurtma bering",
         "havola ... ikki ta buyurtma bering"),

        ("SMS-kod: 4521, telefon: +998901234567",
         "es-em-es, kod: to'rt ming besh yuz yigirma bir ..."),

        ("Og'irligi: 1.5 kg, hajmi: 250 ml",
         "... bir butun besh kilogramm, ikki yuz ellik millilitr"),

        ("API kaliti: XY_456-ABC ni server.py ga kiriting",
         "a-pi-ay kaliti: iks-ey to'rt yuz ellik olti ..."),

        ("Bugun 15.06.2025 da 1000000 so'm to'landi",
         "o'n besh-iyun ikki ming yigirma besh-yil ... bir million ..."),

        ("Mijoz ID #4521 — VIP status — 30% chegirma",
         "... aydiy raqam to'rt ming ... vi-ay-pi ... o'ttiz foiz chegirma"),
    ]

    print("=" * 65)
    print("🧪 TEXT NORMALIZER v2 — TEST")
    print("=" * 65)

    all_ok = True
    for text, expected_hint in tests:
        result = normalize(text)
        print(f"\n📥 Кирди:   {text}")
        print(f"📤 Чиқди:  {result}")

    print("\n" + "=" * 65)
    print("✅ Тест якунланди!")
