"""Экспорт corrected label-records в Ultralytics-датасет (seg ИЛИ pose). Сплит train/val ПО ОСОБЯМ.

seg — полигон брюшка (класс 0 'belly'); pose — bbox брюшка + 2 keypoints [head, cloaca].
Координаты в LabelRecord уже нормализованы → строки YOLO пишутся напрямую. Сплит по особи
(детерминированно, blake2b) — особь не попадает в train и val одновременно (как в Блоке 1).
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from .labelio import scan_labels


def bbox_from_polygon(poly):
    """Полигон (норм.) → (cx, cy, w, h) нормализованный bbox."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def seg_line(rec, cls: int = 0) -> str:
    coords = " ".join(f"{v:.6f}" for p in rec.belly_polygon for v in p)
    return f"{cls} {coords}"


def _bbox_covering(points, pad_frac: float = 0.05):
    """Минимальный bbox (cx, cy, w, h) по набору точек + относительный запас; кламп в [0, 1].
    Гарантирует, что все точки внутри bbox даже при смещённом авто-полигоне."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0, x1 = max(0.0, x0 - px), min(1.0, x1 + px)
    y0, y1 = max(0.0, y0 - py), min(1.0, y1 + py)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def pose_line(rec, cls: int = 0) -> str:
    hx, hy = rec.head_xy
    clx, cly = rec.cloaca_xy
    # bbox по ОБЪЕДИНЕНИЮ полигона и keypoints — смещённый авто-полигон не уводит bbox от верных точек.
    cx, cy, w, h = _bbox_covering(list(rec.belly_polygon) + [rec.head_xy, rec.cloaca_xy])
    return (f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
            f"{hx:.6f} {hy:.6f} 2 {clx:.6f} {cly:.6f} 2")


def assign_yolo_split(individual_id, val_frac: float = 0.2, seed: int = 42) -> str:
    """Фолд особи (функция ТОЛЬКО от id) → 'train'|'val'. Детерминированно."""
    raw = hashlib.blake2b(f"{seed}:{individual_id}".encode(), digest_size=8).digest()
    return "val" if int.from_bytes(raw, "big") / 2 ** 64 < val_frac else "train"


def _usable(rec) -> bool:
    """Годен для обучения, если проверен человеком (source=manual ИЛИ status=corrected),
    не помечен skip/redraw и есть полная геометрия."""
    return ((rec.status == "corrected" or rec.source == "manual")
            and rec.status not in ("skip", "redraw")
            and bool(rec.belly_polygon) and bool(rec.head_xy) and bool(rec.cloaca_xy))


def belly_consistent(rec, tol_frac: float = 0.03) -> bool:
    """Полигон пуза согласован с точками: середина головы↔клоаки внутри (или у края) полигона.
    Ловит кадры, где авто-маска села мимо (угол/фон), а точки человек поставил верно → такие в seg НЕ берём."""
    if not (rec.head_xy and rec.cloaca_xy and rec.belly_polygon and len(rec.belly_polygon) >= 3):
        return False
    mid = ((rec.head_xy[0] + rec.cloaca_xy[0]) / 2.0, (rec.head_xy[1] + rec.cloaca_xy[1]) / 2.0)
    poly = (np.asarray(rec.belly_polygon, np.float32) * 1000.0).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (mid[0] * 1000.0, mid[1] * 1000.0), True) >= -tol_frac * 1000.0


def build_yolo_dataset(records, image_root, out_dir, task: str = "seg",
                       val_frac: float = 0.2, seed: int = 42):
    """corrected-метки → структура Ultralytics (images симлинки + labels + data.yaml).

    task='seg'|'pose'. Возвращает {n, train_ids, val_ids}. Невычитанные (status≠corrected) пропускаются.
    """
    if task not in ("seg", "pose"):
        raise ValueError(f"task: {task}")
    out_dir, image_root = Path(out_dir), Path(image_root)
    for fold in ("train", "val"):
        (out_dir / "images" / fold).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / fold).mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, n = set(), set(), 0
    for rec in records:
        if not _usable(rec):
            continue
        if task == "seg" and not belly_consistent(rec):
            continue   # точки верны, но авто-полигон сел мимо → не учим seg на мусоре
        fold = assign_yolo_split(rec.individual_id, val_frac, seed)
        (val_ids if fold == "val" else train_ids).add(rec.individual_id)
        link = out_dir / "images" / fold / f"{rec.md5}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to((image_root / rec.rel_path).resolve())
        line = seg_line(rec) if task == "seg" else pose_line(rec)
        (out_dir / "labels" / fold / f"{rec.md5}.txt").write_text(line + "\n", encoding="utf-8")
        n += 1
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
            "names": {0: "belly"}}
    if task == "pose":
        data["kpt_shape"] = [2, 3]
        data["flip_idx"] = [0, 1]      # head/cloaca на средней линии → под h-flip не меняются местами
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"n": n, "train_ids": train_ids, "val_ids": val_ids}


def export_from_dir(labels_dir, image_root, out_root, val_frac: float = 0.2, seed: int = 42):
    """Прочитать corrected-метки из каталога и собрать ОБА датасета: out_root/seg и out_root/pose."""
    recs = scan_labels(labels_dir)
    out_root = Path(out_root)
    return {
        "seg": build_yolo_dataset(recs, image_root, out_root / "seg", "seg", val_frac, seed),
        "pose": build_yolo_dataset(recs, image_root, out_root / "pose", "pose", val_frac, seed),
    }
