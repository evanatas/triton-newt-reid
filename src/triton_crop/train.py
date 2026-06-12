"""Обучение YOLO seg/pose на пилоте брюшка (Ultralytics, MPS). Модельная граница (lazy import).

Аугментации полевых условий съёмки: degrees=180 (тритон под любым углом), hsv_v (блики чашки), h-flip РАЗРЕШЁН
(flip_idx=[0,1] для pose — голова/клоака на средней линии). Веса → artifacts/runs/<name>/weights/best.pt.
"""
from pathlib import Path

_ART = Path(__file__).resolve().parents[2] / "artifacts"


def _train(base, data_yaml, *, name, epochs, imgsz, device, seed=42, **kw):
    from ultralytics import YOLO
    model = YOLO(base)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device,
                seed=seed, deterministic=True, patience=20,
                project=str(_ART / "runs"), name=name, exist_ok=True,
                degrees=180.0, hsv_v=0.4, scale=0.5, translate=0.1, fliplr=0.5, **kw)
    return model


def train_seg(data_yaml, base="yolov8n-seg.pt", epochs=100, imgsz=640, device="mps", seed=42):
    return _train(base, data_yaml, name="belly_seg", epochs=epochs, imgsz=imgsz, device=device, seed=seed)


def train_pose(data_yaml, base="yolov8n-pose.pt", epochs=120, imgsz=640, device="mps", seed=42):
    return _train(base, data_yaml, name="belly_pose", epochs=epochs, imgsz=imgsz, device=device, seed=seed)
