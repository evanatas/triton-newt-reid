"""Тесты экспорта corrected-меток в Ultralytics-датасет (seg/pose): конвертеры + сборка + анти-утечка + согласованность полигона."""
import cv2
import numpy as np

from triton_crop.labelio import LabelRecord
from triton_crop.yolo_export import (assign_yolo_split, bbox_from_polygon, belly_consistent,
                                      build_yolo_dataset, pose_line, seg_line)


def _rec(md5="m1", ind="TK-1", **kw):
    base = dict(md5=md5, rel_path="x.jpg", cohort="TK", individual_id=ind, img_w=100, img_h=100,
                belly_polygon=((0.2, 0.4), (0.6, 0.4), (0.6, 0.8), (0.2, 0.8)),
                head_xy=(0.4, 0.45), cloaca_xy=(0.4, 0.75), status="corrected", source="manual")
    base.update(kw)
    return LabelRecord(**base)


def test_bbox_from_polygon():
    cx, cy, w, h = bbox_from_polygon(((0.2, 0.4), (0.6, 0.4), (0.6, 0.8), (0.2, 0.8)))
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.4, 0.6, 0.4, 0.4)


def test_seg_line_format():
    parts = seg_line(_rec()).split()
    assert parts[0] == "0" and len(parts) == 1 + 2 * 4   # класс + 4 вершины (x,y)


def test_pose_line_format():
    parts = pose_line(_rec()).split()
    assert parts[0] == "0" and len(parts) == 11          # класс + bbox(4) + 2*(x,y,v)
    assert parts[7] == "2" and parts[-1] == "2"          # видимость head и cloaca


def test_pose_bbox_covers_keypoints_when_polygon_offset():
    # Полигон сел мимо (в углу), но точки человек поставил верно → pose-bbox обязан покрыть обе keypoints.
    rec = _rec(belly_polygon=((0.02, 0.02), (0.10, 0.02), (0.10, 0.10), (0.02, 0.10)),
               head_xy=(0.60, 0.50), cloaca_xy=(0.60, 0.85))
    parts = pose_line(rec).split()
    cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    for kx, ky in (rec.head_xy, rec.cloaca_xy):
        assert x0 <= kx <= x1 and y0 <= ky <= y1   # keypoint внутри bbox несмотря на смещённый полигон


def test_assign_split_deterministic_per_individual():
    a = assign_yolo_split("TK-7", 0.2, 42)
    assert a in ("train", "val")
    assert assign_yolo_split("TK-7", 0.2, 42) == a       # та же особь → тот же фолд


def test_usable_accepts_manual_rejects_pseudo_and_skip():
    assert build_yolo_dataset  # модуль импортируется
    from triton_crop.yolo_export import _usable
    assert _usable(_rec(status="auto", source="manual"))         # пробел → правка точек
    assert not _usable(_rec(status="auto", source="pseudo"))      # не тронут
    assert not _usable(_rec(status="skip", source="manual"))      # помечен негодным


def test_belly_consistent_true_and_false():
    good = _rec(belly_polygon=((0.30, 0.30), (0.50, 0.30), (0.50, 0.80), (0.30, 0.80)),
                head_xy=(0.40, 0.30), cloaca_xy=(0.40, 0.80))
    bad = _rec(belly_polygon=((0.02, 0.02), (0.10, 0.02), (0.10, 0.10), (0.02, 0.10)),
               head_xy=(0.60, 0.50), cloaca_xy=(0.60, 0.85))
    assert belly_consistent(good) and not belly_consistent(bad)


def test_build_dataset_no_individual_leak(tmp_path):
    img_root = tmp_path / "imgs"; img_root.mkdir()
    recs = []
    for i in range(6):
        cv2.imwrite(str(img_root / f"{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
        recs.append(_rec(md5=f"m{i}", ind=f"TK-{i}", rel_path=f"{i}.jpg"))
    out = tmp_path / "ds"
    info = build_yolo_dataset(recs, img_root, out, task="seg", val_frac=0.5, seed=42)
    assert (out / "data.yaml").exists() and info["n"] == 6
    assert set(info["train_ids"]).isdisjoint(info["val_ids"])   # особь не в обоих фолдах


def test_build_dataset_skips_unreviewed(tmp_path):
    img_root = tmp_path / "imgs"; img_root.mkdir()
    cv2.imwrite(str(img_root / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    pseudo = _rec(md5="a", ind="TK-9", rel_path="a.jpg", status="auto", source="pseudo")
    info = build_yolo_dataset([pseudo], img_root, tmp_path / "ds", task="pose", val_frac=0.2)
    assert info["n"] == 0


def test_seg_skips_inconsistent_polygon_but_pose_keeps(tmp_path):
    img_root = tmp_path / "imgs"; img_root.mkdir()
    cv2.imwrite(str(img_root / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    bad = _rec(md5="a", ind="TK-1", rel_path="a.jpg",
               belly_polygon=((0.02, 0.02), (0.10, 0.02), (0.10, 0.10), (0.02, 0.10)),
               head_xy=(0.60, 0.50), cloaca_xy=(0.60, 0.85))   # точки верны, полигон в углу
    seg = build_yolo_dataset([bad], img_root, tmp_path / "s", task="seg")
    pose = build_yolo_dataset([bad], img_root, tmp_path / "p", task="pose")
    assert seg["n"] == 0 and pose["n"] == 1
