"""Бутстрап псевдо-меток: выбор маски животного (ЧИСТО) + обёртка ultralytics-SAM (модель, lazy).

Поток: фото → SAM 'everything' (много масок) → pick_animal (самая «тритоновая») → derive_pseudo_label.
pick_animal — чистый numpy/cv2 (тестируется на синтетике); ultralytics грузим лениво внутри SamMasker,
чтобы импорт модуля и юнит-тесты не требовали моделей.
"""
import cv2
import numpy as np
import pandas as pd


def pick_animal(masks, rgb, *, area_min: float = 0.015, area_max: float = 0.55,
                border_max: float = 0.25):
    """Из набора масок 'everything' выбрать самую похожую на тритона.

    Фильтры: площадь в [area_min, area_max] кадра; доля пикселей на рамке ≤ border_max
    (фон обычно липнет к краям). Скоринг прошедших: центральность × вытянутость(PCA) × текстура(Sobel).
    Возвращает bitmask uint8 (H×W) или None.
    """
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None]
    h, w = arr.shape[1:]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if (rgb.ndim == 3) else rgb
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    cx, cy = w / 2.0, h / 2.0
    diag = float(np.hypot(cx, cy))
    rng = np.random.RandomState(0)   # локальный seed=0: сабсэмплинг PCA-точек; на выбор маски влияет пренебрежимо
    best, best_score = None, -1.0    # (НЕ путать с проектным seed=42 — это лишь детерминизм внутреннего сабсэмплинга)
    for m in arr:
        mb = m > 0.5
        af = float(mb.mean())
        if af < area_min or af > area_max:
            continue
        ys, xs = np.where(mb)
        border = float(((xs <= 1) | (xs >= w - 2) | (ys <= 1) | (ys >= h - 2)).mean())
        if border > border_max:
            continue
        mxc, myc = float(xs.mean()), float(ys.mean())
        centr = max(1.0 - np.hypot(mxc - cx, myc - cy) / diag, 0.05)
        pts = np.stack([xs - mxc, ys - myc], 1).astype(float)
        if len(pts) > 4000:
            pts = pts[rng.choice(len(pts), 4000, replace=False)]
        ev = np.linalg.eigvalsh(np.cov(pts.T))
        elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        tex = float(grad[mb].mean()) + 1e-3
        score = centr * float(np.clip(elong, 1.0, 6.0)) * tex
        if score > best_score:
            best, best_score = mb.astype(np.uint8), score
    return best


class SamMasker:
    """Обёртка ultralytics-SAM (sam2_b) в режиме 'everything'. Модельная граница (lazy import)."""

    def __init__(self, weights: str = "FastSAM-s.pt", device: str = "mps", max_side: int = 1024):
        self.is_fast = "fastsam" in str(weights).lower()   # FastSAM — быстрый bulk; SAM2 — точечный reseg
        if self.is_fast:
            from ultralytics import FastSAM
            self.model = FastSAM(weights)
        else:
            from ultralytics import SAM
            self.model = SAM(weights)
        self.device = device
        self.max_side = max_side

    def animal_mask(self, rgb):
        """RGB (H×W×3 uint8) → (mask uint8 H×W | None, work_rgb). Маска — в координатах work_rgb (≤ max_side)."""
        h0, w0 = rgb.shape[:2]
        sc = self.max_side / max(h0, w0)
        work = (cv2.resize(rgb, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)
                if sc < 1 else rgb)
        kw = dict(retina_masks=True) if self.is_fast else {}
        try:
            res = self.model.predict(work, device=self.device, verbose=False, **kw)
        except Exception:
            self.device = "cpu"
            res = self.model.predict(work, device=self.device, verbose=False, **kw)
        md = res[0].masks
        if md is None or len(md.data) == 0:
            return None, work
        return pick_animal(md.data.cpu().numpy(), work), work

    def mask_from_point(self, rgb, point_norm):
        """Точечный промпт SAM по нормализованной точке (xn, yn) → (mask uint8 | None, work_rgb)."""
        h0, w0 = rgb.shape[:2]
        sc = self.max_side / max(h0, w0)
        work = (cv2.resize(rgb, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)
                if sc < 1 else rgb)
        wh, ww = work.shape[:2]
        pt = [[float(point_norm[0] * ww), float(point_norm[1] * wh)]]
        try:
            res = self.model.predict(work, points=pt, labels=[1], device=self.device, verbose=False)
        except Exception:
            self.device = "cpu"
            res = self.model.predict(work, points=pt, labels=[1], device=self.device, verbose=False)
        md = res[0].masks
        if md is None or len(md.data) == 0:
            return None, work
        arr = md.data.cpu().numpy()
        m = arr[int(arr.reshape(len(arr), -1).sum(1).argmax())]
        return (m > 0.5).astype(np.uint8), work


def derive_flags(head_conf, band_conf, notes="", *, head_conf_min: float = 0.2):
    """Флаги в очередь правки: неуверенная ориентация (узкая разность ширины концов) и
    подозрительная брюшная полоса (площадь вне ожидаемого диапазона)."""
    flags = []
    if head_conf < head_conf_min:
        flags.append("low_head_conf")
    if "band_suspect" in (notes or ""):
        flags.append("band_suspect")
    return tuple(flags)


def select_pilot(df, per_species: int = 40, cohorts=("TK", "PW"), folds=("train", "dev")):
    """~per_species/вид из заданных folds (по умолчанию train+dev, БЕЗ test-LOCK), без дублей,
    равномерно по md5. Для масштаба — folds=('train',): dev остаётся чистым для честного A/B."""
    keep = df["dup_keep"].astype(str).str.lower().isin(["true", "1", "1.0"])
    pool = df[keep & df["split_fold"].isin(list(folds)) & df["cohort"].isin(cohorts)]
    parts = []
    for c in cohorts:
        sub = pool[pool["cohort"] == c].sort_values("md5")
        if len(sub) == 0:
            continue
        idx = np.unique(np.linspace(0, len(sub) - 1, min(per_species, len(sub))).round().astype(int))
        parts.append(sub.iloc[idx])
    return pd.concat(parts) if parts else pool.iloc[:0]


def generate_pseudo_labels(rows, masker, labels_dir, overlays_dir, workspace_root,
                           pipeline_tag: str = "v2-sam2b-everything", head_conf_min: float = 0.2):
    """Кадр → SamMasker → derive_pseudo_label → LabelRecord(JSON) + overlay(PNG).

    rows — итерируемое словарей (md5, rel_path, cohort, individual_id, width, height).
    Маска None → status='redraw'+'no_mask'. Возвращает список (md5, status, flags)."""
    from pathlib import Path

    from triton_data.imageio import load_canonical

    from .labelio import LabelRecord, write_label
    from .pseudo_label import derive_pseudo_label
    from .viz import draw_overlay

    Path(overlays_dir).mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        common = dict(md5=row["md5"], rel_path=row["rel_path"], cohort=row["cohort"],
                      individual_id=str(row["individual_id"]),
                      img_w=int(row["width"]), img_h=int(row["height"]), pipeline=pipeline_tag)
        try:
            rgb = np.array(load_canonical(Path(workspace_root) / row["rel_path"]))
            mask, work = masker.animal_mask(rgb)
            h, w = work.shape[:2]
            if mask is None:
                rec = LabelRecord(belly_polygon=(), head_xy=None, cloaca_xy=None,
                                  source="pseudo", status="redraw", flags=("no_mask",), **common)
                ov = draw_overlay(work, None, None, None, None, redraw=True,
                                  label=f"{row['cohort']} {row['individual_id']} NO_MASK")
            else:
                pl = derive_pseudo_label(mask)
                poly_n = tuple((float(x) / w, float(y) / h) for x, y in pl.belly_polygon)
                rec = LabelRecord(
                    belly_polygon=poly_n,
                    head_xy=(pl.head_xy[0] / w, pl.head_xy[1] / h),
                    cloaca_xy=(pl.cloaca_xy[0] / w, pl.cloaca_xy[1] / h),
                    source="pseudo", status="auto",
                    flags=derive_flags(pl.head_conf, pl.band_conf, pl.notes, head_conf_min=head_conf_min),
                    head_conf=pl.head_conf, band_conf=pl.band_conf, **common)
                ov = draw_overlay(work, mask, pl.belly_polygon, pl.head_xy, pl.cloaca_xy,
                                  label=f"{row['cohort']} {row['individual_id']} hc={pl.head_conf:.2f}")
            write_label(rec, labels_dir)
            cv2.imwrite(str(Path(overlays_dir) / f"{row['md5']}.png"), ov)
            summary.append((row["md5"], rec.status, rec.flags))
        except Exception as exc:  # битый кадр не валит всю пачку — в очередь правки
            rec = LabelRecord(belly_polygon=(), head_xy=None, cloaca_xy=None,
                              source="pseudo", status="redraw", flags=("gen_error",), **common)
            write_label(rec, labels_dir)
            summary.append((row["md5"], "redraw", ("gen_error", str(exc)[:80])))
    return summary


def run_bootstrap(manifest_csv, workspace_root, labels_dir, overlays_dir, *,
                  per_species: int = 40, cohorts=("TK", "PW"), folds=("train", "dev"),
                  weights: str = "FastSAM-s.pt", device: str = "mps", reuse_labels_dir=None):
    """Оркестровка: манифест → select_pilot(folds) → переиспользовать уже исправленные метки
    (source=manual из reuse_labels_dir) → FastSAM-псевдо-метки для остальных → сводка."""
    import collections
    from pathlib import Path

    from .labelio import scan_labels, write_label

    df = pd.read_csv(manifest_csv)
    pool = select_pilot(df, per_species=per_species, cohorts=tuple(cohorts), folds=tuple(folds))
    reuse = {}
    if reuse_labels_dir and Path(reuse_labels_dir).exists():
        reuse = {r.md5: r for r in scan_labels(reuse_labels_dir) if r.source == "manual"}
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    to_gen, reused = [], 0
    for row in pool.to_dict("records"):
        if row["md5"] in reuse:
            write_label(reuse[row["md5"]], labels_dir)
            reused += 1
        else:
            to_gen.append(row)
    masker = SamMasker(weights, device=device)
    summary = generate_pseudo_labels(to_gen, masker, labels_dir, overlays_dir, workspace_root)
    status = collections.Counter(s for _, s, _ in summary)
    return {"n_pool": len(pool), "reused": reused, "generated": len(summary),
            "by_cohort": dict(pool["cohort"].value_counts()), "status": dict(status)}
