"""Ингест PW (Ребристый): папка=особь, дат нет, имена «{id}-{session} ({shot}).JPG»."""
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
            continue
        for entry in sorted(id_dir.iterdir()):
            if not is_image(entry, cfg.image_extensions):
                continue
            rec = base_record(entry, spec, cfg)
            rec["local_id"] = folder_id
            parsed = parsers.parse_pw_filename(entry.name)
            if parsed is not None:
                pid, session, shot = parsed
                rec["session"], rec["shot"] = session, shot
                if pid != folder_id:
                    rec["notes"] = f"id из имени ({pid}) ≠ папка ({folder_id})"
            records.append(rec)
    return records
