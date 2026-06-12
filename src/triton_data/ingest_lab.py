"""Ингест LAB (лабораторная когорта): подпапки-даты DD.MM.YYYY; имя файла = локальный id.

ОТДЕЛЬНАЯ целевая когорта (НЕ перепоимки TK). Каждая дата — отдельная сессия одной особи.
Ошибочную метку (1.jpg ≡ копия 10.jpg) чинит md5-дедуп — не здесь.
"""
from . import parsers
from .ingest_common import base_record, is_image


def ingest(spec, cfg) -> list[dict]:
    root = spec.abs_dir(cfg.workspace_root)
    records: list[dict] = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            iso = parsers.parse_lab_date(date_dir.name)
        except ValueError:
            continue  # подпапка не вида DD.MM.YYYY
        for entry in sorted(date_dir.iterdir()):
            if not is_image(entry, cfg.image_extensions):
                continue
            try:
                local_id, variant = parsers.parse_lab_stem(entry.stem)
            except ValueError:
                continue  # имя-картинка не вида локального id (напр. «пояснение.jpg») → пропустить (как TK/PW)
            rec = base_record(entry, spec, cfg)
            rec["local_id"] = local_id
            rec["date"], rec["date_source"] = iso, "subfolder"
            rec["session"], rec["shot"] = iso, variant
            records.append(rec)
    return records
