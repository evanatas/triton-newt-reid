"""Сборка манифеста: ingest → глобальные id → дедуп → сплиты → 2 CSV.

Выход:
  data/manifest.csv           — target (TK+LAB+PW), с осями сплита;
  data/manifest_external.csv  — GCN (external), со своими полями (RLE/bbox), без сплитов.

Порядок строк детерминирован (сортировка) → манифест воспроизводим побайтово.
"""
from pathlib import Path

import pandas as pd

from . import ingest_gcn, ingest_lab, ingest_pw, ingest_tk
from .config import load_config
from .dedup import deduplicate
from .splits import assign_splits

_INGEST = {
    "TK": ingest_tk.ingest,
    "PW": ingest_pw.ingest,
    "LAB": ingest_lab.ingest,
    "GCN": ingest_gcn.ingest,
}

_TARGET_COLS = [
    "cohort", "species", "role", "kpi_scope", "individual_id", "local_id", "rel_path", "md5",
    "width", "height", "orientation", "date", "date_source", "session", "shot",
    "dup_group", "dup_keep", "split_role", "split_fold", "split_scheme", "is_new_open", "notes",
]
_EXTERNAL_COLS = [
    "cohort", "species", "role", "kpi_scope", "individual_id", "local_id", "rel_path", "md5",
    "width", "height", "orientation", "recapture_id", "survey", "bbox",
    "rle_h", "rle_w", "mask_empty", "dup_group", "dup_keep", "notes",
]
_SORT_KEYS = ["cohort", "individual_id", "rel_path"]


def build_manifest(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать (target_df, external_df) из всех когорт конфигурации."""
    records: list[dict] = []
    for spec in cfg.datasets:
        recs = _INGEST[spec.id_prefix](spec, cfg)
        for r in recs:
            r["individual_id"] = f"{spec.id_prefix}-{int(r['local_id']):03d}"
        records.extend(recs)

    records = deduplicate(records)                      # кросс-когортный md5-скан
    df = pd.DataFrame(records)
    if cfg.merge_individuals:
        df["individual_id"] = df["individual_id"].replace(cfg.merge_individuals)
    df = assign_splits(df, seed=cfg.seed)

    target = _project(df[df["role"] == "target"], _TARGET_COLS)
    external = _project(df[df["role"] == "external"], _EXTERNAL_COLS)
    return target, external


def _project(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols].sort_values(_SORT_KEYS).reset_index(drop=True)


def write_manifests(target: pd.DataFrame, external: pd.DataFrame, data_dir) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target.to_csv(data_dir / "manifest.csv", index=False, encoding="utf-8")
    external.to_csv(data_dir / "manifest_external.csv", index=False, encoding="utf-8")


def build_and_write(cfg=None, data_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать и записать оба манифеста. По умолчанию — реальный конфиг и REPO/data."""
    if cfg is None:
        cfg = load_config()
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"
    target, external = build_manifest(cfg)
    write_manifests(target, external, data_dir)
    return target, external
