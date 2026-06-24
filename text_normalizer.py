# -*- coding: utf-8 -*-
"""
Uzbek TTS — Text Normalizer v5
Ключевое изменение: аббревиатуры и коды сохраняют дефисы.
Порядок шагов жёстко фиксирован чтобы шаги не ломали друг друга.
"""

import re

APO = "'"  # ASCII апостроф — безопасен для TTS

# ── Числа ───────────────────────────────────────────────────
ONES  = ["", "bir", "ikki", "uch", f"to{APO}rt", "besh",
         "olti", "yetti", "sakkiz", f"to{APO}qqiz"]
TEENS = [f"o{APO}n", f"o{APO}n bir", f"o{APO}n ikki", f"o{APO}n uch",
         f"o{APO}n to{APO}rt", f"o{APO}n besh", f"o{APO}n olti",
         f"o{APO}n yetti", f"o{APO}n sakkiz", f"o{APO}n to{APO}qqiz"]
TENS  = ["", f"o{APO}n", "yigirma", f"o{APO}ttiz", "qirq", "ellik",
         "oltmish", "yetmish", "sakson", f"to{APO}qson"]


def num(n: int) -> str:
    if n < 0:          return "minus " + num(-n)
    if n == 0:         return "nol"
    if n < 10:         return ONES[n]
    if n < 20:         return TEENS[n - 10]
    if n < 100:        return (TENS[n // 10] + " " + ONES[n % 10]).strip()
    if n < 1_000:
        r = (" " + num(n % 100)) if n % 100 else ""
        return ONES[n // 100] + " yuz" + r
    if n < 1_000_000:
        r = (" " + num(n % 1_000)) if n % 1_000 else ""
        return num(n // 1_000) + " ming" + r
    if n < 1_000_000_000:
        r = (" " + num(n % 1_000_000)) if n % 1_000_000 else ""
        return num(n // 1_000_000) + " million" + r
    r = (" " + num(n % 1_000_000_000)) if n % 1_000_000_000 else ""
    return num(n // 1_000_000_000) + " milliard" + r


def digits_one_by_one(s: str) -> str:
    return " ".join(num(int(d)) for d in s if d.isdigit())


# ── Аббревиатуры ────────────────────────────────────────────
# ВАЖНО: значения НЕ содержат дефисов — используем пробелы.
# Дефис в "es-em-es" шаг 13 превращал в запятую.
# Решение: заменяем аббревиатуры словами через ПРОБЕЛ, не дефис.
ABBR = {
    "HTTPS": "ey ti ti pi es",
    "HTTP":  "ey ti ti pi",
    "UZS":   f"so{APO}m",
    "USD":   "dollar",
    "EUR":   "evro",
    "RUB":   "rubl",
    "TTS":   "te te es",
    "API":   "a pi ay",
    "SMS":   "es em es",
    "VIP":   "vi ay pi",
    "ID":    "aydiy",
    "URL":   "yu ar el",
    "AI":    f"sun{APO}iy intellekt",
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


def expand_abbr(text: str) -> str:
    for k, v in sorted(ABBR.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'(?<![A-Za-z])' + re.escape(k) + r'(?![A-Za-z])', v, text)
    return text


# ── Чтение кодов ────────────────────────────────────────────
LETTERS = {
    'A':'a',   'B':'be',  'C':'se',  'D':'de',  'E':'e',   'F':'ef',
    'G':'ge',  'H':'ha',  'I':'i',   'J':'jey', 'K':'ka',  'L':'el',
    'M':'em',  'N':'en',  'O':'o',   'P':'pe',  'Q':'qu',  'R':'ar',
    'S':'es',  'T':'te',  'U':'u',   'V':'ve',  'W':'ve',  'X':'iks',
    'Y':'ey',  'Z':'zed',
}


def spell(code: str) -> str:
    """BV3 → be ve uch  (пробелы, не дефисы)"""
    parts, i = [], 0
    while i < len(code):
        if code[i].isdigit():
            j = i
            while j < len(code) and code[j].isdigit():
                j += 1
            parts.append(num(int(code[i:j])))
            i = j
        elif code[i].isalpha():
            parts.append(LETTERS.get(code[i].upper(), code[i].lower()))
            i += 1
        else:
            i += 1
    return " ".join(p for p in parts if p)


# ── Месяцы ───────────────────────────────────────────────────
MONTHS = {
    "01":"yanvar", "02":"fevral", "03":"mart",    "04":"aprel",
    "05":"may",    "06":"iyun",   "07":"iyul",    "08":"avgust",
    "09":"sentabr","10":"oktabr", "11":"noyabr",  "12":"dekabr",
}


# ── Главная функция ──────────────────────────────────────────

def normalize(text: str) -> str:

    # 1. URL → havola  (до expand_abbr, иначе HTTPS разберётся)
    text = re.sub(r'https?://\S+', 'havola', text, flags=re.IGNORECASE)
    text = re.sub(r'www\.\S+',     'sayt',   text, flags=re.IGNORECASE)

    # 2. Email → elektron pochta
    text = re.sub(r'\S+@\S+\.\S+', 'elektron pochta', text)

    # 3. Телефон +998... → цифры по одной
    text = re.sub(
        r'\+\d[\d\s\-\(\)]{6,}',
        lambda m: digits_one_by_one(re.sub(r'\D', '', m.group(0))),
        text,
    )

    # 4. Дата ДД.ММ.ГГГГ — ОБЯЗАТЕЛЬНО до замены точек!
    def fmt_date(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{num(int(d))} {MONTHS.get(mo, mo)}, {num(int(y))} yil"
    text = re.sub(r'\b(\d{1,2})\.(\d{2})\.(\d{4})\b', fmt_date, text)

    # 5. Десятичные  1.5 → bir butun besh — ОБЯЗАТЕЛЬНО до замены точек!
    text = re.sub(
        r'\b(\d+)\.(\d+)\b',
        lambda m: num(int(m.group(1))) + " butun " + num(int(m.group(2))),
        text,
    )

    # 5b. Точка между буквами → пробел  (после дат и десятичных!)
    #     BV3.org2001 → BV3 org2001  |  server.py → server py
    text = re.sub(r'(?<=[A-Za-z0-9])\.(?=[A-Za-z])', ' ', text)

    # 6. Проценты  30% → o'ttiz foiz
    text = re.sub(r'(\d+)\s*%', lambda m: num(int(m.group(1))) + " foiz", text)

    # 7. № # → raqam N
    text = re.sub(r'[#№]\s*(\d+)', lambda m: "raqam " + num(int(m.group(1))), text)

    # 8. Аббревиатуры  SMS→es em es  (до кодов, чтобы SMS не читался по буквам)
    text = expand_abbr(text)

    # 9. Подчёркивание → пробел  (до кодов, чтобы XY_456 → XY 456)
    text = text.replace('_', ' ')

    # 10. Дефис между словами → пробел  (до кодов, чтобы SMS-kod → es em es kod)
    #     Но только если с обеих сторон буквы/цифры (не минус числа)
    text = re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])', ' ', text)

    # 11. Коды → по буквам
    #     а) смешанные буквы+цифры: BV3, XY456
    #     б) чисто буквенные капслоком 2+ букв: XY, ABC, VIP (если не в словаре)
    def mixed(m):
        tok = m.group(0)
        has_letter = bool(re.search(r'[A-Za-z]', tok))
        has_digit  = bool(re.search(r'\d', tok))
        # Смешанный код
        if has_letter and has_digit:
            return spell(tok)
        # Все заглавные буквы 2-5 символов (артикулы, коды)
        if has_letter and not has_digit and tok == tok.upper() and 2 <= len(tok) <= 5:
            return spell(tok)
        return tok
    text = re.sub(r'\b[A-Za-z0-9]{2,}\b', mixed, text)

    # 12. Большие числа с пробелами  1 000 000
    text = re.sub(
        r'\b\d{1,3}(?:\s\d{3})+\b',
        lambda m: num(int(re.sub(r'\s', '', m.group(0)))),
        text,
    )

    # 13. Обычные числа
    text = re.sub(r'\b\d+\b', lambda m: num(int(m.group(0))), text)

    # 14. Оставшиеся тире/длинные тире → запятая
    text = re.sub(r'\s*[–—]\s*', ', ', text)

    # 15. Прочие спецсимволы
    text = re.sub(r'\s*/\s*',    ' yoki ', text)
    text = re.sub(r'[(){}\[\]]', ', ',     text)
    text = re.sub(r'[«»""\u2018\u201C\u201D]', '', text)
    text = re.sub(r'[@$^&*+=|\\<>~`]', ' ', text)

    # 16. Финальная чистка
    text = re.sub(r',\s*,+',    ',',  text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    text = re.sub(r'\s+',        ' ', text)

    return text.strip()


# ── Тест ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Mahsulot ID: BV3.org2001 narxi 150000 UZS",
         "Mahsulot aydiy: be ve uch ... bir yuz ellik ming so'm"),

        ("HTTPS://shop.uz/product_123 orqali 2 ta buyurtma bering",
         "havola orqali ikki ta buyurtma bering"),

        ("SMS-kod: 4521, telefon: +998901234567",
         "es em es kod: to'rt ming ... to'qqiz to'qqiz sakkiz ..."),

        ("Og'irligi: 1.5 kg, hajmi: 250 ml",
         "Og'irligi: bir butun besh kilogramm, ikki yuz ellik millilitr"),

        ("API kaliti: XY_456-ABC ni server.py ga kiriting",
         "a pi ay kaliti: iks ey to'rt yuz ellik olti a be se ..."),

        ("Bugun 15.06.2025 da 1000000 so'm to'landi",
         "Bugun o'n besh iyun, ikki ming yigirma besh yil da bir million ..."),

        ("Mijoz ID #4521 — VIP status — 30% chegirma",
         "Mijoz aydiy raqam to'rt ming ... vi ay pi status, o'ttiz foiz chegirma"),
    ]

    print("=" * 65)
    print("🧪 TEXT NORMALIZER v5")
    print("=" * 65)
    for text, hint in tests:
        result = normalize(text)
        print(f"\n📥  {text}")
        print(f"📤  {result}")
        print(f"🎯  {hint}")
    print("\n" + "=" * 65)
    print("✅ Tест yakullandi!")
