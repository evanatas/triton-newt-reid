"""crops_manifest: сборка/чтение data/crops_manifest.csv + контракт select_crops для Блока 4.

Ключ — md5 (FK к manifest.csv Блока 1; стабильный, НЕ абсолютный путь). Блок 4 берёт кропы
ТОЛЬКО через select_crops (как loader.select) — нельзя случайно взять fallback-кроп или вскрыть test.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_COLS = [
    "md5", "rel_path", "cohort", "individual_id", "split_role", "split_fold", "kpi_scope",
    "variant", "crop_status", "crop_path", "head_x", "head_y", "cloaca_x", "cloaca_y",
    "orientation_deg", "head_up_conf", "seg_conf", "canon_size", "mirrored", "pipeline_version",
    "belly_axis_start_x", "belly_axis_start_y", "belly_axis_end_x", "belly_axis_end_y",
    "unroll_method",
]
_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"


def _variant_row(row, variant, crop_status, crop_path, canon_size, pipeline_version,
                 h, c, orientation_deg, head_up_conf, seg_conf, bs, be, unroll_method):
    return {
        "md5": row["md5"], "rel_path": row.get("rel_path"), "cohort": row.get("cohort"),
        "individual_id": row.get("individual_id"), "split_role": row.get("split_role"),
        "split_fold": row.get("split_fold"), "kpi_scope": row.get("kpi_scope"),
        "variant": variant, "crop_status": crop_status, "crop_path": crop_path,
        "head_x": h[0], "head_y": h[1], "cloaca_x": c[0], "cloaca_y": c[1],
        "orientation_deg": orientation_deg, "head_up_conf": head_up_conf, "seg_conf": seg_conf,
        "canon_size": int(canon_size), "mirrored": False, "pipeline_version": pipeline_version,
        "belly_axis_start_x": bs[0], "belly_axis_start_y": bs[1],
        "belly_axis_end_x": be[0], "belly_axis_end_y": be[1], "unroll_method": unroll_method,
    }


def build_crops_manifest(rows_df, results, crops_dir, pipeline_version: str = "v0") -> pd.DataFrame:
    """rows_df (срез манифеста Блока 1) + results (list[CropResult], тот же порядок) → DataFrame.

    Ключ строки = (md5, variant): базовая строка (belly_oriented/belly_mask/full) + по строке на
    каждый распрямлённый вариант `unroll_{метод}` из CropResult.unroll_variants (Блок 3).
    """
    if len(rows_df) != len(results):                          # контракт: ровно один CropResult на строку
        raise ValueError(f"build_crops_manifest: len(rows_df)={len(rows_df)} != len(results)={len(results)}")
    recs = []
    for (_, row), res in zip(rows_df.iterrows(), results):
        h = res.head_xy or (np.nan, np.nan)
        c = res.cloaca_xy or (np.nan, np.nan)
        bs = res.belly_axis_start or (np.nan, np.nan)
        be = res.belly_axis_end or (np.nan, np.nan)
        recs.append(_variant_row(row, res.variant, res.crop_status, f"{crops_dir}/{row['md5']}.png",
                                 int(res.image.shape[0]), pipeline_version, h, c, res.orientation_deg,
                                 res.head_up_conf, res.seg_conf, bs, be, ""))
        for m, img_u in (res.unroll_variants or {}).items():
            recs.append(_variant_row(row, f"unroll_{m}", (res.unroll_status or {}).get(m, "ok"),
                                     f"{crops_dir}/{row['md5']}__{m}.png", int(img_u.shape[0]),
                                     pipeline_version, h, c, res.orientation_deg, res.head_up_conf,
                                     res.seg_conf, bs, be, m))
    df = pd.DataFrame(recs, columns=_COLS)
    return df.sort_values(["md5", "variant"]).reset_index(drop=True)


def write_crops_manifest(df, data_dir) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "crops_manifest.csv", index=False, encoding="utf-8")


def read_crops_manifest(data_dir=None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA
    df = pd.read_csv(data_dir / "crops_manifest.csv")
    for c in _COLS:                              # бэкфилл колонок старого (до-Блок-3) CSV → нет KeyError
        if c not in df.columns:
            df[c] = np.nan
    if "mirrored" in df.columns and df["mirrored"].dtype != bool:
        df["mirrored"] = (df["mirrored"].astype(str).str.lower()
                          .map({"true": True, "false": False}).fillna(False).astype(bool))
    return df


def select_crops(crops_df, rows_df, variant: str = "belly_oriented") -> pd.DataFrame:
    """Inner-join кропов к УЖЕ выбранным строкам Блока 1 (loader.select) по md5 + фильтр variant.

    Ключ (md5, variant): для конкретного variant — ровно 1 строка на md5 (иначе ValueError). Контракт
    Блоков 4/5. ВНИМАНИЕ: unroll-варианты покрывают НЕ все md5 (кадры без оси) → выборка может быть
    МЕНЬШЕ belly_oriented; вызывающий сам решает про fallback (Блок 4 обязан повторить bo-fallback A/B).
    """
    sel = crops_df if variant is None else crops_df[crops_df["variant"] == variant]
    if variant is not None and not sel["md5"].is_unique:
        raise ValueError(f"select_crops: дубли md5 у variant={variant!r} — нарушен ключ (md5, variant)")
    extra = [c for c in ("individual_id", "split_role", "split_fold", "kpi_scope")
             if c in rows_df.columns and c not in sel.columns]
    return sel.merge(rows_df[["md5"] + extra], on="md5", how="inner")


def select_crops_with_fallback(crops_df, rows_df, variant: str = "unroll_ribbon",
                               fallback: str = "belly_oriented") -> pd.DataFrame:
    """Для Блока 4: на каждый md5 из rows_df берём `variant`, иначе `fallback` (тот же fallback-протокол,
    что в A/B `cmd_ab`), а НЕ молчаливо меньше строк (как inner-join select_crops). Покрытие == len(rows_df).
    -> по строке на md5 + колонка `variant_used` (variant | fallback | None если нет ни того, ни другого)."""
    def _index_by_md5(v):                       # как select_crops: ключ (md5, variant) обязан быть уникален
        sl = crops_df[crops_df["variant"] == v]
        if not sl["md5"].is_unique:
            raise ValueError(f"select_crops_with_fallback: дубли md5 у variant={v!r} — нарушен ключ (md5, variant)")
        return dict(zip(sl["md5"], sl.index))
    prim = _index_by_md5(variant)
    fb = _index_by_md5(fallback)
    recs = []
    for md5 in rows_df["md5"]:
        if md5 in prim:
            rec = crops_df.loc[prim[md5]].to_dict(); rec["variant_used"] = variant
        elif md5 in fb:
            rec = crops_df.loc[fb[md5]].to_dict(); rec["variant_used"] = fallback
        else:
            rec = {"md5": md5, "variant_used": None}
        recs.append(rec)
    return pd.DataFrame(recs)


def merge_crops_manifest(existing_df, new_df) -> pd.DataFrame:
    """Аддитивно слить существующий и новый crops_manifest по ключу (md5, variant): при дубле побеждает
    НОВАЯ строка (свежий кроп). Колонки выравниваются по объединению. Существующие строки НЕ теряются."""
    cat = pd.concat([existing_df, new_df], ignore_index=True)
    cat = cat.drop_duplicates(["md5", "variant"], keep="last")     # last = new_df перекрывает existing
    return cat.sort_values(["md5", "variant"]).reset_index(drop=True)


def crops_parity(crops_df, rows_df, variant_a: str = "belly_oriented", variant_b: str = "unroll_ribbon"):
    """Паритет покрытия: множества md5 (среди rows_df), покрытые variant_a и variant_b. -> (only_a, only_b, both).
    Блок 4 обязан сравнивать baseline/challenger на ОДНОМ наборе md5 (иначе сравнение невоспроизводимо)."""
    want = set(rows_df["md5"])
    a = set(crops_df.loc[crops_df["variant"] == variant_a, "md5"]) & want
    b = set(crops_df.loc[crops_df["variant"] == variant_b, "md5"]) & want
    return a - b, b - a, a & b
