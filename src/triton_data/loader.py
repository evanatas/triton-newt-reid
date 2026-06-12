"""Контракт-loader манифеста для downstream-блоков (2–6).

Брать строки ТОЛЬКО через select(...): это единственное место, где зашиты правильные
фильтры (target/external, dup_keep, role, fold, scope). Так нельзя случайно подмешать
дубли, GCN, open_new или LAB-temporal_aux в headline-KPI или вскрыть запечатанный test.

Стадии:
  train      — чем учить эмбеддер (target, dup_keep, fold=train = все галереи);
  gallery    — эталонная база для поиска (target, dup_keep, role=gallery, без open_new);
  dev        — пробы для A-B и тюнинга блоков 2–6 (role=probe, fold=dev, по scope);
  test       — ЗАПЕЧАТАННЫЕ пробы headline-KPI (role=probe, fold=test, по scope) — 1 раз в конце;
  open_dev   — open-set «новые» особи для настройки порога known/new (fold=dev);
  open_test  — open-set «новые» особи, запечатанные (fold=test);
  pretrain   — внешний GCN (только pretrain блоков 2/4, не в KPI).

scope (для dev/test): kpi_core (TK+PW — официальный 75/95) · temporal_aux (LAB) · all_target.
"""
from pathlib import Path

import pandas as pd

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"
_BOOL_COLS = ("dup_keep", "is_new_open", "mask_empty")


def read_manifests(data_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прочитать оба манифеста с диска, приведя bool-колонки к настоящему bool."""
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA
    target = pd.read_csv(data_dir / "manifest.csv")
    external = pd.read_csv(data_dir / "manifest_external.csv")
    for df in (target, external):
        for col in _BOOL_COLS:
            if col in df.columns and df[col].dtype != bool:
                low = df[col].astype(str).str.lower()
                mapped = low.map({"true": True, "false": False})
                # мусор (не true/false и не пустая ячейка) — ошибка, а не тихий False
                bad = mapped.isna() & df[col].notna()
                if bad.any():
                    raise ValueError(
                        f"неразбираемое bool-значение в колонке {col}: "
                        f"{sorted(low[bad].unique())}")
                df[col] = mapped.fillna(False).astype(bool)
    return target, external


def _scope_filter(df: pd.DataFrame, scope) -> pd.DataFrame:
    if scope in (None, "all_target"):
        return df
    return df[df["kpi_scope"] == scope]


def select(target: pd.DataFrame, external: pd.DataFrame, stage: str,
           scope: str = "kpi_core") -> pd.DataFrame:
    """Вернуть строки для конкретной стадии downstream с правильными фильтрами."""
    if stage == "pretrain":
        return external[external["dup_keep"] == True].copy()  # noqa: E712

    t = target[target["dup_keep"] == True]  # noqa: E712
    if stage == "train":
        return t[t["split_fold"] == "train"].copy()
    if stage == "gallery":
        g = t[(t["split_role"] == "gallery") & (~t["is_new_open"])]
        return _scope_filter(g, scope).copy()   # галерея уважает scope (kpi_core по умолчанию)

    if stage in ("dev", "test"):
        closed = _scope_filter(t[(t["split_role"] == "probe") & (~t["is_new_open"])], scope)
        return closed[closed["split_fold"] == stage].copy()
    if stage in ("open_dev", "open_test"):
        opens = t[t["is_new_open"]]
        return opens[opens["split_fold"] == stage.split("_")[1]].copy()

    raise ValueError(f"неизвестная стадия: {stage!r}")
