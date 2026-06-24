"""
Uzbek TTS — Text Normalizer v3
"""

import re

# ─────────────────────────────────────────────
# Числа → узбекские слова (латиница)
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
        return (TENS[n // 10] + " " + ONES[n % 10]).strip()
    if n < 1000:
        rest = (" " + number_to_uzbek(n % 100)) if n % 100 else ""
        return ONES[n // 100] + " yuz" + rest
    if n < 1_000_000:
        rest = (" " + number_to_uzbek(n % 1000)) if n % 1000 else ""
        return number_to_uzbek(n // 1000) + " ming" + rest
    if n < 1_000_000_000:
        rest = (" " + number_to_uzbek(n % 1_000_000)) if n % 1_000_000 else ""
        return number_to_uzbek(n // 1_000_000) + " million" + rest
    rest = (" " + number_to_uzbek(n % 1_000_000_000)) if n % 1_000_000_000 else ""
    return number_to_uzbek(n // 1_000_000_000) + " milliard" + rest


def digits_spaced(digits: str) -> str:
    """Цифры по одной с пробелом: 998 → to'qqiz to'qqiz sakkiz"""
    return " ".join(number_to_uzbek(int(d)) for d in digits)


# ─────────────────────────────────────────────
# Аббревиатуры
# ─────────────────────────────────────────────

ABBREVIATIONS = {
    "UZS":   "so'm",
    "USD":   "dollar",
    "EUR":   "evro",
    "RUB":   "rubl",
    "TTS":   "te-te-es",
    "API":   "a-pi-ay",
    "SMS":   "es-em-es",
    "VIP":   "vi-ay-pi",
    "ID":    "aydiy",
    "URL":   "yu-ar-el",
    "AI":    "sun'iy intellekt",
    "HTTP":  "ey-ti-ti-pi",
    "HTTPS": "ey-ti-ti-pi-es",
    "kg":    "kilogramm",
    "ml":    "millilitr",
    "km":    "kilometr",
    "cm":    "santimetr",
    "mm":    "millimetr",
    "kW":    "kilovat",
    "GB":    "gigabayt",
    "MB":    "megabayt",
    "KB":    "kilobayt",
}


def expand_abbreviations(text: str) -> str:
    for abbr, reading in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'(?<![A-Za-z])' + re.escape(abbr) + r'(?![A-Za-z])', reading, text)
    return text


# ─────────────────────────────────────────────
# Буквы для чтения кодов
# ─────────────────────────────────────────────

LETTER_READ = {
    'A': 'a',   'B': 'be',  'C': 'se',  'D': 'de',  'E': 'e',
    'F': 'ef',  'G': 'ge',  'H': 'ha',  'I': 'i',   'J': 'jey',
    'K': 'ka',  'L': 'el',  'M': 'em',  'N': 'en',  'O': 'o',
    'P': 'pe',  'Q': 'qu',  'R': 'ar',  'S': 'es',  'T': 'te',
    'U': 'u',   'V': 've',  'W': 'double-yu', 'X': 'iks',
    'Y': 'ey',  'Z': 'zed',
}


def spell_code(code: str) -> str:
    """ABC123 → a-be-se-bir-ikki-uch"""
    parts = []
    i = 0
    while i < len(code):
        if code[i].isdigit():
            j = i
            while j < len(code) and code[j].isdigit():
                j += 1
            parts.append(number_to_uzbek(int(code[i:j])))
            i = j
        elif code[i].isalpha():
            parts.append(LETTER_READ.get(code[i].upper(), code[i].lower()))
            i += 1
        else:
            i += 1
    return "-".join(p for p in parts if p)


# ─────────────────────────────────────────────
# Главный пайплайн
# ─────────────────────────────────────────────

def normalize(text: str) -> str:

    # 1. URL полностью → "havola"
    text = re.sub(r'https?://\S+', 'havola', text)
    text = re.sub(r'www\.\S+',     'sayt',   text)

    # 2. Email → "elektron pochta"
    text = re.sub(r'\S+@\S+\.\S+', 'elektron pochta', text)

    # 3. Телефон +998XXXXXXXXX → цифры по одной
    def phone_replace(m):
        digits = re.sub(r'\D', '', m.group(0))
        return digits_spaced(digits)
    text = re.sub(r'\+\d[\d\s\-\(\)]{6,}', phone_replace, text)

    # 4. Дата ДД.ММ.ГГГГ
    MONTHS = {
        "01":"yanvar","02":"fevral","03":"mart","04":"aprel",
        "05":"may","06":"iyun","07":"iyul","08":"avgust",
        "09":"sentabr","10":"oktabr","11":"noyabr","12":"dekabr"
    }
    def date_replace(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        month = MONTHS.get(mo, mo)
        return f"{number_to_uzbek(int(d))}-{month}, {number_to_uzbek(int(y))}-yil"
    text = re.sub(r'\b(\d{1,2})\.(\d{2})\.(\d{4})\b', date_replace, text)

    # 5. Десятичные числа  1.5 → "bir butun besh"
    def decimal_replace(m):
        return number_to_uzbek(int(m.group(1))) + " butun " + number_to_uzbek(int(m.group(2)))
    text = re.sub(r'\b(\d+)\.(\d+)\b', decimal_replace, text)

    # 6. Проценты  30% → "o'ttiz foiz"
    text = re.sub(
        r'(\d+)\s*%',
        lambda m: number_to_uzbek(int(m.group(1))) + " foiz",
        text
    )

    # 7. № и # → "raqam N"
    text = re.sub(
        r'[#№]\s*(\d+)',
        lambda m: "raqam " + number_to_uzbek(int(m.group(1))),
        text
    )

    # 8. Аббревиатуры (до кода — чтобы SMS не читался как S-M-S)
    text = expand_abbreviations(text)

    # 9. Смешанные коды (буквы+цифры) — только если рядом есть и то и то
    def mixed_replace(m):
        token = m.group(0)
        if re.search(r'[A-Za-z]', token) and re.search(r'\d', token):
            return spell_code(token)
        return token
    text = re.sub(r'\b[A-Za-z0-9]{2,}\b', mixed_replace, text)

    # 10. Числа с разделителями: 1 000 000
    text = re.sub(
        r'\b\d{1,3}(?:\s\d{3})+\b',
        lambda m: number_to_uzbek(int(re.sub(r'\s', '', m.group(0)))),
        text
    )

    # 11. Обычные числа
    text = re.sub(r'\b\d+\b', lambda m: number_to_uzbek(int(m.group(0))), text)

    # 12. Знаки препинания → естественные паузы
    text = re.sub(r'\s*[-–—]\s*', ', ',  text)   # тире → запятая
    text = re.sub(r'\s*/\s*',     ' yoki ', text) # / → yoki
    text = text.replace('_', ' ')
    text = re.sub(r'[(){}\[\]]',  ', ',    text)
    text = re.sub(r'[«»""\'`]',   '',      text)
    text = re.sub(r'[@$^&*+=|\\<>~]', ' ', text)

    # 13. Финальная чистка
    text = re.sub(r',\s*,+',   ',',  text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    text = re.sub(r'\s+',       ' ',   text)

    return text.strip()


# ─────────────────────────────────────────────
# Тест
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "Mahsulot ID: BV3.org2001 narxi 150000 UZS",
        "HTTPS://shop.uz/product_123 orqali 2 ta buyurtma bering",
        "SMS-kod: 4521, telefon: +998901234567",
        "Og'irligi: 1.5 kg, hajmi: 250 ml",
        "API kaliti: XY_456-ABC ni server.py ga kiriting",
        "Bugun 15.06.2025 da 1000000 so'm to'landi",
        "Mijoz ID #4521 — VIP status — 30% chegirma",
    ]

    expected = [
        "Mahsulot aydiy: be-ve-uch ... bir yuz ellik ming so'm",
        "havola ... ikki ta buyurtma bering",
        "es-em-es, kod: to'rt ming ... to'qqiz to'qqiz sakkiz ...",
        "bir butun besh kilogramm, ikki yuz ellik millilitr",
        "a-pi-ay kaliti: iks-ey to'rt yuz ellik olti, a-be-se ...",
        "o'n besh-iyun, ikki ming yigirma besh-yil ... bir million ...",
        "aydiy raqam to'rt ming ... vi-ay-pi ... o'ttiz foiz chegirma",
    ]

    print("=" * 65)
    print("🧪 TEXT NORMALIZER v3 — TEST")
    print("=" * 65)

    for text, hint in zip(tests, expected):
        result = normalize(text)
        print(f"\n📥 Кирди:   {text}")
        print(f"📤 Чиқди:  {result}")
        print(f"🎯 Кутилган: {hint}")

    print("\n" + "=" * 65)
    print("✅ Тест якунланди!")
