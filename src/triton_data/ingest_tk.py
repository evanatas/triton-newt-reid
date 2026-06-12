"""Ингест TK (Карелина): папка=особь; даты из имени файла или из имени подпапки «Доп фото»."""
from . import parsers
from .ingest_common import base_record, is_image


def ingest(spec, cfg) -> list[dict]:
    root = spec.abs_dir(cfg.workspace_root)
    records: list[dict] = []
    for id_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if id_dir.name in spec.skip_dirs:
            continue
        try:
            folder_id = int(id_dir.name)
        except ValueError:
            continue  # непарсимая папка-особь (не из skip) — пропускаем осознанно
        for entry in sorted(id_dir.iterdir()):
            if is_image(entry, cfg.image_extensions):
                records.append(_file_record(entry, folder_id, None, spec, cfg))
            elif entry.is_dir():
                subdate = parsers.parse_tk_subfolder_date(entry.name)
                for sub in sorted(entry.iterdir()):
                    if is_image(sub, cfg.image_extensions):
                        records.append(_file_record(sub, folder_id, subdate, spec, cfg))
    return records


def _file_record(path, folder_id, subdate, spec, cfg) -> dict:
    rec = base_record(path, spec, cfg)
    rec["local_id"] = folder_id
    parsed = parsers.parse_tk_filename(path.name)
    if parsed is not None:
        pid, session, date = parsed
        rec["session"] = session
        if date is not None:
            rec["date"], rec["date_source"] = date, "filename"
        elif subdate is not None:
            rec["date"], rec["date_source"] = subdate, "subfolder"
        else:
            rec["date_source"] = "unparsed"
        if pid != folder_id:
            rec["notes"] = f"id из имени ({pid}) ≠ папка ({folder_id})"
    else:
        # IMG_* и прочие без даты в имени — дату берём из имени подпапки
        if subdate is not None:
            rec["date"], rec["date_source"] = subdate, "subfolder"
    return rec
