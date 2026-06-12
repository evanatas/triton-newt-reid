"""Конвертеры масок: COCO-RLE ↔ bitmask ↔ polygon (numpy/cv2/pycocotools). Без моделей.

КРИТИЧНО: COCO-RLE хранит размер как [H, W] (height, width). Префикс GCN-масок `2048x1536:`
тоже HxW (урок Блока 1) — здесь оси НЕ путаем.
"""
import cv2
import numpy as np
from pycocotools import mask as coco_mask


def rle_to_mask(counts, h: int, w: int) -> np.ndarray:
    """COCO-RLE counts + размеры (H, W) → bitmask uint8 (H×W)."""
    body = counts.encode("ascii") if isinstance(counts, str) else counts
    rle = {"size": [int(h), int(w)], "counts": body}
    return coco_mask.decode(rle).astype(np.uint8)


def mask_to_rle(mask) -> tuple[str, int, int]:
    """bitmask → (counts:str, h, w). pycocotools требует F-порядок и 3D-вход для одной маски."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    rle = coco_mask.encode(np.asfortranarray(m[:, :, None]))[0]
    counts = rle["counts"]
    counts = counts.decode("ascii") if isinstance(counts, bytes) else counts
    return counts, int(rle["size"][0]), int(rle["size"][1])


def mask_to_polygon(mask, eps_frac: float = 0.002, min_area_frac: float = 0.001):
    """Контур самой большой связной компоненты → полигон Nx2 (approxPolyDP). None если мелкая/пусто."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_frac * binary.size:
        return None
    eps = eps_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    return approx.reshape(-1, 2).astype(float)


def polygon_to_mask(poly, h: int, w: int) -> np.ndarray:
    """Полигон Nx2 → bitmask bool (H×W)."""
    m = np.zeros((int(h), int(w)), np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], 1)
    return m > 0


def mask_area_frac(mask) -> float:
    """Доля кадра, занятая маской."""
    arr = np.asarray(mask)
    return float((arr > 0).sum()) / float(arr.size)


def foreground_from_crop(rgb, bg_sum_thr: int = 30) -> np.ndarray:
    """Реконструировать маску пуза из кропа с занулённым в чёрный фоном.

    Пайплайн зануляет фон в чёрный → letterbox даёт чёрную рамку по краям. Прежний костыль `rgb.sum(2)>30`
    выкидывал тёмные пятна (сумма каналов <= bg_sum_thr) как «фон», теряя самые дискриминативные пятна.
    Здесь фон определяется как near-чёрное, СВЯЗАННОЕ С РАМКОЙ (flood-fill от 4 углов, 4-связность);
    foreground = всё остальное → внутренние тёмные пятна (не связаны с рамкой) ОСТАЮТСЯ в маске. -> (H,W) bool."""
    rgb = np.asarray(rgb)
    s = rgb.astype(np.int32).sum(2)
    darkish = (s <= int(bg_sum_thr)).astype(np.uint8)        # кандидаты фона (near-чёрные)
    h, w = darkish.shape
    ff = darkish.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[sy, sx] == 1:                                  # угол near-чёрный → это фон, заливаем
            cv2.floodFill(ff, ffmask, seedPoint=(int(sx), int(sy)), newVal=2, loDiff=0, upDiff=0)
    background = ff == 2                                      # связное с рамкой near-чёрное = фон
    return ~background
