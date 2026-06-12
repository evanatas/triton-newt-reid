"""Чистая 2D-геометрия канонического кропа (numpy/cv2). Без моделей → полностью тестируема.

Соглашение координат: x вправо, y ВНИЗ (как у изображения). «Вверх» = −Y.
Все повороты — БЕЗ отражения (det 2x2 = 1 > 0): зеркало меняет хиральность узора = другая особь.
"""
import math

import cv2
import numpy as np


def head_up_angle_deg(head_xy, cloaca_xy) -> float:
    """Угол (град) для cv2.getRotationMatrix2D, приводящий вектор клоака→голова к «вверх» (−Y).

    Голова уже над клоакой → 0. Детерминирован. Нормализован в [−180, 180) (вход 180° даёт −180°; это тот же поворот).
    """
    vx = float(head_xy[0]) - float(cloaca_xy[0])
    vy = float(head_xy[1]) - float(cloaca_xy[1])
    angle = math.degrees(math.atan2(vy, vx)) + 90.0
    return (angle + 180.0) % 360.0 - 180.0


def rotation_matrix(angle_deg, center_xy) -> np.ndarray:
    """2x3 аффинная матрица поворота вокруг center_xy. БЕЗ отражения (det 2x2 = 1)."""
    return cv2.getRotationMatrix2D((float(center_xy[0]), float(center_xy[1])), float(angle_deg), 1.0)


def rotate_points(pts, M) -> np.ndarray:
    """Применить аффинную 2x3 к точкам (Nx2) → Nx2."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ homog.T).T


def apply_affine_image(img, M, out_hw, fill=(0, 0, 0)) -> np.ndarray:
    """cv2.warpAffine с константной заливкой границ (детерминировано, INTER_LINEAR)."""
    h, w = out_hw
    return cv2.warpAffine(img, M, (int(w), int(h)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def mask_to_tight_bbox(mask, margin_frac):
    """(x0, y0, x1, y1) полуинтервал по маске; margin — доля стороны bbox; клампится к краям.
    Пустая маска → весь кадр (caller обрабатывает пустоту раньше)."""
    ys, xs = np.where(mask)
    height, width = mask.shape
    if len(xs) == 0:
        return (0, 0, width, height)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = int(round((x1 - x0) * margin_frac))
    my = int(round((y1 - y0) * margin_frac))
    return (max(0, x0 - mx), max(0, y0 - my), min(width, x1 + mx), min(height, y1 + my))


def unroll_rectangle(head_xy, cloaca_xy, halfwidth_frac) -> dict:
    """Ось-выровненный прямоугольник (angle=0) для Блока 3: центр между головой и клоакой,
    высота = длина оси, ширина = 2·halfwidth_frac·длина. Голова вверху."""
    hx, hy = float(head_xy[0]), float(head_xy[1])
    qx, qy = float(cloaca_xy[0]), float(cloaca_xy[1])
    length = math.hypot(qx - hx, qy - hy)
    return {"cx": (hx + qx) / 2.0, "cy": (hy + qy) / 2.0,
            "w": 2.0 * halfwidth_frac * length, "h": length, "angle": 0.0,
            "p_head": (hx, hy), "p_cloaca": (qx, qy)}


def clip_axis_to_mask(head_xy, cloaca_xy, mask, n_samples: int = 200):
    """Отрезок оси голова→клоака, ОБРЕЗАННЫЙ до маски брюшка (для распрямления в Блоке 3).

    Голова (морда) часто ВНЕ кропа пуза → её нельзя брать опорной точкой unroll. Сэмплируем линию
    голова→клоака, оставляем точки внутри маски и возвращаем первую (со стороны головы) и последнюю
    (со стороны клоаки) → bounded landmarks ВНУТРИ пуза. -> (start, end) или (None, None)."""
    head = np.asarray(head_xy, float)
    cloaca = np.asarray(cloaca_xy, float)
    ts = np.linspace(0.0, 1.0, int(n_samples))[:, None]
    pts = head[None, :] * (1.0 - ts) + cloaca[None, :] * ts      # (n, 2): x, y
    h, w = mask.shape[:2]
    xr = np.round(pts[:, 0]).astype(int)
    yr = np.round(pts[:, 1]).astype(int)
    on_frame = (xr >= 0) & (xr < w) & (yr >= 0) & (yr < h)   # точка реально в кадре (НЕ «прижата» clip к рамке)
    inside = np.zeros(len(pts), bool)                        # off-frame точки не считаем попавшими в маску
    if on_frame.any():
        inside[on_frame] = np.asarray(mask)[yr[on_frame], xr[on_frame]] > 0
    if not inside.any():
        return None, None
    idx = np.where(inside)[0]
    return ((float(pts[idx[0], 0]), float(pts[idx[0], 1])),
            (float(pts[idx[-1], 0]), float(pts[idx[-1], 1])))
