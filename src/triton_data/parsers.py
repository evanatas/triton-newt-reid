"""Чистые парсеры имён файлов/подпапок и нормализация дат.

Без обращений к диску — поэтому покрываются юнит-тестами целиком.
Вся «грязь» исходных данных (аномальные имена, опечатки) разбирается здесь.
"""
import re
from datetime import date as _date

# Допустимый диапазон лет (защита от опечаток вроде «2002» — урок triton 2.0).
_MIN_YEAR = 2023
_MAX_YEAR = 2026


def normalize_mmyy(token: str) -> str | None:
    """MMYY (4 цифры) -> "YYYY-MM".

    Чинит известную 5-значную опечатку (74-02-01025 -> "0125" -> Jan 2025): лишний
    символ в позиции 2 отбрасывается. Возвращает None, если вход не цифровой/не той
    длины или результат не проходит валидацию (месяц 01..12, год 2023..2026).
    """
    if not token or not token.isdigit():
        return None
    if len(token) == 5:
        token = token[:2] + token[3:]
    if len(token) != 4:
        return None
    month = int(token[:2])
    year = 2000 + int(token[2:])
    if not (1 <= month <= 12):
        return None
    if not (_MIN_YEAR <= year <= _MAX_YEAR):
        return None
    return f"{year:04d}-{month:02d}"


_TK_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.jpe?g$", re.IGNORECASE)


def parse_tk_filename(name: str) -> tuple[int, str, str | None] | None:
    """"{id}-{session}-{MMYY}.JPG" -> (local_id, session, date|None).

    None, если имя не датированный TK-формат (IMG_xxxx.JPG, Иллюстрация.jpg) —
    тогда дату берут из имени подпапки. session дополняется нулём до 2 цифр.
    """
    m = _TK_RE.match(name)
    if not m:
        return None
    local_id = int(m.group(1))
    session = f"{int(m.group(2)):02d}"
    date = normalize_mmyy(m.group(3))
    return (local_id, session, date)


def parse_tk_subfolder_date(folder_name: str) -> str | None:
    """Имя подпапки «… фото … MMYY» -> "YYYY-MM" (берётся последний 4-значный токен)."""
    tokens = re.findall(r"\d{4}", folder_name)
    if not tokens:
        return None
    return normalize_mmyy(tokens[-1])


_PW_RE = re.compile(r"^(\d+)-(\d+)\s*\((\d+)\)\.jpe?g$", re.IGNORECASE)


def parse_pw_filename(name: str) -> tuple[int, str, int] | None:
    """"{id}-{session} ({shot}).JPG" -> (local_id, session, shot) | None."""
    m = _PW_RE.match(name)
    if not m:
        return None
    return (int(m.group(1)), f"{int(m.group(2)):02d}", int(m.group(3)))


def parse_lab_stem(stem: str) -> tuple[int, int]:
    """Стем LAB-файла: "03"->(3,0); "03.1"->(3,1); "1"->(1,0).

    Сырой парс по имени. Ошибочную метку (1.jpg ≡ копия 10.jpg) чинит md5-дедуп — не здесь.
    """
    parts = stem.split(".")
    local_id = int(parts[0])
    variant = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    return (local_id, variant)


def parse_lab_date(subdir: str) -> str:
    """Имя подпапки LAB "DD.MM.YYYY" -> "YYYY-MM-DD".

    ValueError, если дата не существует в календаре или год вне 2023..2026.
    """
    dd, mm, yyyy = (int(p) for p in subdir.split("."))
    if not (_MIN_YEAR <= yyyy <= _MAX_YEAR):
        raise ValueError(f"год вне диапазона {_MIN_YEAR}..{_MAX_YEAR}: {subdir!r}")
    _date(yyyy, mm, dd)  # несуществующая дата (напр. 31 февраля) -> ValueError
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
