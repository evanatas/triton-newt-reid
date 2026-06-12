"""Обёртки обученных YOLO под интерфейс pipeline.canonical_belly_crop. Модель — lazy import.

parse_seg/parse_pose — ЧИСТЫЕ (тестируются): разбор результата YOLO без загрузки модели.
BellySegmenter/PoseEstimator — тонкие обёртки: BGR для ultralytics (не RGB!), imgsz=обучению=640.
pose_conf = МИНИМУМ уверенности keypoints (ориентация надёжна, только если ОБЕ точки уверенны).
"""
import numpy as np

from .masks import polygon_to_mask


def parse_seg(masks_xy, box_conf, h, w):
    """masks_xy — полигоны в координатах ОРИГИНАЛА (или None); box_conf — 1d уверенности боксов.
    -> (mask HxW bool | None, seg_conf). Берём детекцию с макс. уверенностью."""
    if masks_xy is None or box_conf is None or len(box_conf) == 0:
        return None, 0.0
    i = int(np.argmax(box_conf))
    poly = masks_xy[i] if i < len(masks_xy) else None
    if poly is None or len(poly) < 3:
        return None, float(box_conf[i])
    return polygon_to_mask(np.asarray(poly, float), h, w), float(box_conf[i])


def parse_pose(kpts, box_conf):
    """kpts — (N,2,3) [x,y,conf] (или None); box_conf — 1d. -> (head|None, cloaca|None, pose_conf).
    pose_conf = min(conf головы, conf клоаки) — обе точки должны быть уверенны для верной ориентации."""
    if kpts is None or box_conf is None or len(box_conf) == 0:
        return None, None, 0.0
    kp = np.asarray(kpts)[int(np.argmax(box_conf))]
    if kp.shape[0] < 2:
        return None, None, 0.0
    return ((float(kp[0, 0]), float(kp[0, 1])), (float(kp[1, 0]), float(kp[1, 1])),
            float(min(kp[0, 2], kp[1, 2])))


class BellySegmenter:
    def __init__(self, weights, device: str = "mps", imgsz: int = 640, conf: float = 0.10):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def predict(self, img_rgb):
        h, w = img_rgb.shape[:2]
        bgr = np.ascontiguousarray(np.asarray(img_rgb)[..., ::-1])   # ultralytics ждёт BGR
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return None, 0.0
        return parse_seg(r.masks.xy, r.boxes.conf.cpu().numpy(), h, w)


class PoseEstimator:
    def __init__(self, weights, device: str = "mps", imgsz: int = 640, conf: float = 0.10):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def predict(self, img_rgb):
        bgr = np.ascontiguousarray(np.asarray(img_rgb)[..., ::-1])
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None, None, 0.0
        return parse_pose(r.keypoints.data.cpu().numpy(), r.boxes.conf.cpu().numpy())
