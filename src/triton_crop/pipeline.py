"""★ Продукт Блока 2: canonical_belly_crop — единый детерминированный путь gallery=probe.

Сегментация пуза → ориентация по морда/клоака (поворот головой ВВЕРХ, БЕЗ зеркала) → тугой кроп
по маске → заливка фона → letterbox в канон (НЕ stretch) + unroll-прямоугольник для Блока 3.
Лестница fallback (никогда не падаем, всё помечено в crop_status): belly_oriented → belly_mask → full.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import (
    apply_affine_image,
    clip_axis_to_mask,
    head_up_angle_deg,
    mask_to_tight_bbox,
    rotate_points,
    rotation_matrix,
    unroll_rectangle,
)
from .masks import mask_area_frac
from .unroll import straighten_belly


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray            # canon×canon×3, фон залит
    variant: str                 # belly_oriented | belly_mask | full
    crop_status: str             # ok | empty_mask | low_seg_conf | tiny_mask | no_pose | low_pose_conf | clipped_after_rotation
    mask: np.ndarray | None      # маска пуза в координатах канона (для Блока 5/оверлея)
    head_xy: tuple | None
    cloaca_xy: tuple | None
    orientation_deg: float
    head_up_conf: float
    seg_conf: float
    unroll_rect: dict | None
    src_md5: str
    belly_axis_start: tuple | None = None    # опорные точки оси ВНУТРИ маски пуза (bounded, для Блока 3)
    belly_axis_end: tuple | None = None
    unroll_variants: dict | None = None      # {метод: распрямлённый кроп uint8 canon²} (Блок 3)
    unroll_status: dict | None = None        # {метод: статус straighten_belly}


def _centroid(mask):
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean()))


def _letterbox(img, mask, size, fill=(0, 0, 0)):
    """Resize с сохранением пропорций (НЕ stretch) + паддинг до квадрата. Маска — тем же преобразованием."""
    h, w = img.shape[:2]
    scale = size / max(h, w) if max(h, w) > 0 else 1.0
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    out = np.full((size, size, 3), fill, np.uint8)
    out_mask = np.zeros((size, size), bool)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    if mask is not None:
        out_mask[py:py + nh, px:px + nw] = cv2.resize(mask.astype(np.uint8), (nw, nh),
                                                      interpolation=cv2.INTER_NEAREST) > 0
    return out, out_mask, scale, (px, py)


def _apply_background(crop, mask, mode):
    if mode == "none":
        return crop
    out = crop.copy()
    if mode == "mean" and mask.any():
        out[~mask] = crop[mask].mean(axis=0).astype(np.uint8)
    else:
        out[~mask] = 0
    return out


def _full_fallback(img_rgb, status, seg_conf, cfg, src_md5):
    canon, _, _, _ = _letterbox(img_rgb, None, cfg.canon_size)
    return CropResult(canon, "full", status, None, None, None, 0.0, 0.0, float(seg_conf), None, src_md5)


def canonical_belly_crop(img_rgb, segmenter, pose, cfg, src_md5: str = "") -> CropResult:
    img_rgb = np.asarray(img_rgb)
    H, W = img_rgb.shape[:2]

    # 1) Сегментация пуза → fallback на полный кадр при пустой/мелкой/слабой маске
    mask, seg_conf = segmenter.predict(img_rgb)
    seg_conf = float(seg_conf or 0.0)
    if mask is None:
        return _full_fallback(img_rgb, "empty_mask", seg_conf, cfg, src_md5)
    if seg_conf < cfg.seg_conf_min:
        return _full_fallback(img_rgb, "low_seg_conf", seg_conf, cfg, src_md5)
    mask = np.asarray(mask) > 0
    if mask_area_frac(mask) < cfg.seg_area_frac_min:
        return _full_fallback(img_rgb, "tiny_mask", seg_conf, cfg, src_md5)

    # 2) Поза (морда/клоака) → ориентация; иначе belly_mask без поворота
    head, cloaca, pose_conf = pose.predict(img_rgb)
    pose_conf = float(pose_conf or 0.0)
    if head is not None and cloaca is not None and pose_conf >= cfg.pose_conf_min:
        angle = head_up_angle_deg(head, cloaca)
        area_before = int(mask.sum())
        M = rotation_matrix(angle, _centroid(mask))
        img_w = apply_affine_image(img_rgb, M, (H, W))
        mask_w = apply_affine_image((mask.astype(np.uint8) * 255), M, (H, W)) > 127
        head_w, cloaca_w = (tuple(p) for p in rotate_points([head, cloaca], M))
        variant, status, orientation_deg, head_up_conf = "belly_oriented", "ok", angle, pose_conf
        # Поворот пишется в холст (H, W) → концы наклонного брюшка у края могут срезаться (BORDER_CONSTANT).
        # Геометрию кропа НЕ меняем, лишь помечаем потерю площади маски, чтобы клиппинг не уходил молча.
        area_after = int(mask_w.sum())
        if area_before > 0 and area_after < (1.0 - cfg.clip_area_tol_frac) * area_before:
            status = "clipped_after_rotation"
    else:
        img_w, mask_w = img_rgb, mask
        head_w = cloaca_w = None
        variant = "belly_mask"
        status = "no_pose" if (head is None or cloaca is None) else "low_pose_conf"
        orientation_deg, head_up_conf = 0.0, 0.0

    # 3) Тугой кроп по маске + заливка фона + letterbox в канон
    x0, y0, x1, y1 = mask_to_tight_bbox(mask_w, cfg.margin_frac)
    crop = img_w[y0:y1, x0:x1].copy()
    mcrop = mask_w[y0:y1, x0:x1]
    crop = _apply_background(crop, mcrop, cfg.mask_background)
    canon, mask_canon, scale, (px, py) = _letterbox(crop, mcrop, cfg.canon_size)

    def _to_canon(p):
        return None if p is None else (float((p[0] - x0) * scale + px), float((p[1] - y0) * scale + py))

    head_c, cloaca_c = _to_canon(head_w), _to_canon(cloaca_w)
    unroll = (unroll_rectangle(head_c, cloaca_c, cfg.unroll_halfwidth_frac)
              if head_c is not None and cloaca_c is not None else None)
    belly_start = belly_end = None
    if head_c is not None and cloaca_c is not None and mask_canon is not None and mask_canon.any():
        belly_start, belly_end = clip_axis_to_mask(head_c, cloaca_c, mask_canon)

    # 4) Распрямление (Блок 3): только для belly_oriented с валидной осью; по каждому методу из cfg
    uv, us = {}, {}
    if variant == "belly_oriented" and belly_start is not None and belly_end is not None:
        for m in cfg.unroll_methods:
            img_u, st = straighten_belly(canon, mask_canon, belly_start, belly_end, cfg, m)
            us[m] = st
            if st == "ok":
                uv[m] = img_u

    return CropResult(canon, variant, status, mask_canon, head_c, cloaca_c,
                      float(orientation_deg), float(head_up_conf), seg_conf, unroll, src_md5,
                      belly_axis_start=belly_start, belly_axis_end=belly_end,
                      unroll_variants=(uv or None), unroll_status=(us or None))
