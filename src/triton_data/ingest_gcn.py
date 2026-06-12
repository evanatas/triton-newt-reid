"""Ингест GCN (внешний датасет): metadata.csv + Raw_Data/{id}/. Только каталогизация.

GCN не target: в gallery/probe/KPI не входит (анти-утечка). RLE-маски НЕ декодируются в Блоке 1 —
лишь сохраняется построчный префикс HxW (он непостоянен!) и флаг пустой маски для Блока 2.
"""
import re
from pathlib import Path

import pandas as pd

from .imageio import file_md5, read_image_stats
from .ingest_common import rel_path

_RLE_PREFIX = re.compile(r"^(\d+)x(\d+):(.*)$", re.DOTALL)


def _parse_rle(value) -> tuple[int | None, int | None, bool]:
    """Префикс «HxW:тело» → (h, w, mask_empty). Непарсимое/пустое → (None, None, True)."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return None, None, True
    m = _RLE_PREFIX.match(str(value).strip())
    if not m:
        return None, None, True
    return int(m.group(1)), int(m.group(2)), m.group(3).strip() == ""


def ingest(spec, cfg) -> list[dict]:
    root = spec.abs_dir(cfg.workspace_root)
    csv_path = root / spec.extra["metadata_csv"]
    raw_subdir = spec.extra.get("raw_subdir", "Raw_Data")
    df = pd.read_csv(csv_path, dtype=str)

    records: list[dict] = []
    for _, row in df.iterrows():
        identity, file_name = row["identity"], row["file_name"]
        abs_path = root / raw_subdir / str(identity) / str(file_name)
        rle_h, rle_w, mask_empty = _parse_rle(row.get("segmentation_mask_rle"))
        if abs_path.exists():
            st = read_image_stats(abs_path)
            md5, width, height, orient, note = (
                file_md5(abs_path), st.width, st.height, st.orientation, "")
            rel = rel_path(abs_path, cfg.workspace_root)
        else:
            md5, width, height, orient = f"MISSING:{spec.dir}/{raw_subdir}/{identity}/{file_name}", 0, 0, None
            note = "файл отсутствует на диске"
            rel = f"{spec.dir}/{raw_subdir}/{identity}/{file_name}"
        records.append({
            "cohort": "GCN", "species": spec.species, "role": spec.role,
            "local_id": int(identity), "rel_path": rel, "md5": md5,
            "width": width, "height": height, "orientation": orient,
            "date": None, "date_source": "none", "session": None, "shot": None,
            "rle_h": rle_h, "rle_w": rle_w, "mask_empty": mask_empty,
            "bbox": row.get("bbox"), "recapture_id": row.get("recapture_id"),
            "survey": row.get("survey"), "notes": note,
        })
    return records
