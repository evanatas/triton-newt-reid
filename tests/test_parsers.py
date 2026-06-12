"""Тесты чистых парсеров (без I/O, TDD)."""
import pytest

from triton_data.parsers import (
    normalize_mmyy,
    parse_tk_filename,
    parse_tk_subfolder_date,
    parse_pw_filename,
    parse_lab_stem,
    parse_lab_date,
)


# --- normalize_mmyy: MMYY -> "YYYY-MM" ---
def test_normalize_mmyy_standard():
    assert normalize_mmyy("0125") == "2025-01"
    assert normalize_mmyy("1224") == "2024-12"


def test_normalize_mmyy_five_digit_typo():
    # аномалия 74-02-01025: лишний 0 в позиции 2 -> Jan 2025
    assert normalize_mmyy("01025") == "2025-01"


def test_normalize_mmyy_rejects_bad():
    assert normalize_mmyy("1325") is None   # месяц 13
    assert normalize_mmyy("0122") is None   # год 2022 < 2023 (ловит опечатку «2002»)
    assert normalize_mmyy("xx25") is None   # не цифры
    assert normalize_mmyy("") is None
    assert normalize_mmyy("012") is None     # 3 цифры — мусор


# --- parse_tk_filename: "{id}-{session}-{MMYY}.JPG" ---
def test_parse_tk_filename_standard():
    assert parse_tk_filename("05-02-0125.JPG") == (5, "02", "2025-01")


def test_parse_tk_filename_session_no_leading_zero():
    assert parse_tk_filename("70-1-1224.JPG") == (70, "01", "2024-12")


def test_parse_tk_filename_five_digit_date():
    assert parse_tk_filename("74-02-01025.JPG") == (74, "02", "2025-01")


def test_parse_tk_filename_lowercase_ext():
    assert parse_tk_filename("12-03-0325.jpg") == (12, "03", "2025-03")


def test_parse_tk_filename_non_dated_returns_none():
    assert parse_tk_filename("IMG_0856.JPG") is None
    assert parse_tk_filename("Иллюстрация.jpg") is None


# --- parse_tk_subfolder_date: «… фото … MMYY» -> "YYYY-MM" ---
def test_parse_tk_subfolder_date_variants():
    assert parse_tk_subfolder_date("Дополнительные фото от 0125") == "2025-01"
    assert parse_tk_subfolder_date("Доп фото 0225") == "2025-02"
    assert parse_tk_subfolder_date("Дополниетльные фото от 0125") == "2025-01"  # опечатка (папка 66)
    assert parse_tk_subfolder_date("Фото от 0225") == "2025-02"


def test_parse_tk_subfolder_date_none():
    assert parse_tk_subfolder_date("случайная папка без даты") is None


# --- parse_pw_filename: "{id}-{session} ({shot}).JPG" ---
def test_parse_pw_filename():
    assert parse_pw_filename("15-02 (3).JPG") == (15, "02", 3)
    assert parse_pw_filename("01-01 (1).JPG") == (1, "01", 1)


def test_parse_pw_filename_none():
    assert parse_pw_filename("IMG_1.JPG") is None
    assert parse_pw_filename("05-02-0125.JPG") is None  # это TK-формат, не PW


# --- parse_lab_stem: "03"->(3,0); "03.1"->(3,1); "1"->(1,0) ---
def test_parse_lab_stem():
    assert parse_lab_stem("03") == (3, 0)
    assert parse_lab_stem("03.1") == (3, 1)
    assert parse_lab_stem("1") == (1, 0)
    assert parse_lab_stem("10") == (10, 0)


# --- parse_lab_date: "DD.MM.YYYY" -> "YYYY-MM-DD" ---
def test_parse_lab_date():
    assert parse_lab_date("05.08.2025") == "2025-08-05"
    assert parse_lab_date("30.12.2025") == "2025-12-30"


def test_parse_lab_date_rejects_invalid():
    with pytest.raises(ValueError):
        parse_lab_date("1.2.3")        # год 3 — вне диапазона лет
    with pytest.raises(ValueError):
        parse_lab_date("31.02.2025")   # 31 февраля не существует
