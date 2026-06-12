"""Оверлеи бутстрапа/кликера (ЧИСТО, cv2): маска тела + брюшная полоса + голова/клоака.

Возвращает BGR uint8 (готово для cv2.imwrite/imshow). Все аргументы геометрии опциональны
(None → не рисуем) — кадры с отсутствующей маской тоже визуализируются.
"""
import cv2
import numpy as np

_YEL = np.array([0, 255, 255])     # маска тела (BGR)
_MAG = (255, 0, 255)               # брюшная полоса
_GRN = (0, 255, 0)                 # голова
_RED = (0, 0, 255)                 # клоака


def draw_overlay(rgb, mask, polygon_px, head_px, cloaca_px, *, redraw=False, label=""):
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    if mask is not None:
        sel = np.asarray(mask) > 0
        tint = bgr.copy()
        tint[sel] = (0.5 * tint[sel] + 0.5 * _YEL).astype(np.uint8)
        out = cv2.addWeighted(bgr, 0.45, tint, 0.55, 0)
    if polygon_px is not None and len(polygon_px) >= 3:
        cv2.polylines(out, [np.asarray(polygon_px, np.int32).reshape(-1, 1, 2)], True, _MAG, 2)
    if head_px is not None:
        cv2.circle(out, (int(head_px[0]), int(head_px[1])), 7, _GRN, -1)
    if cloaca_px is not None:
        cv2.circle(out, (int(cloaca_px[0]), int(cloaca_px[1])), 7, _RED, -1)
    if redraw:
        cv2.putText(out, "REDRAW", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED, 2)
    if label:
        cv2.putText(out, label, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ───────────────────────── Блок 5 (пятна / матчинг созвездия) ─────────────────────────

_SPOT = (0, 165, 255)              # пятно (BGR, оранжевый)
_PAL = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (128, 0, 255), (0, 128, 255)]   # цвета пар совпавших пятен (циклично)


def draw_spot_overlay(rgb, spots, *, label=""):
    """Центроиды пятен (Spot) на кропе: кружки + номера. -> BGR uint8 (QA детектора)."""
    out = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR).copy()
    for i, s in enumerate(spots or []):
        cv2.circle(out, (int(round(s.x)), int(round(s.y))), 5, _SPOT, 2)
        cv2.putText(out, str(i), (int(round(s.x)) + 5, int(round(s.y)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _SPOT, 1)
    if label:
        cv2.putText(out, label, (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def draw_match_overlay(rgb_probe, spots_probe, rgb_gallery, spots_gallery, matched_pairs,
                       score=None, *, label=""):
    """SIDE-BY-SIDE двух кропов + ЛИНИИ между совпавшими пятнами (matched_pairs[(i_probe, j_gallery)]) + %.
    Запрос заказчика №1 (интерпретируемость). Совпавшие — цветом по индексу пары. -> BGR uint8."""
    p = cv2.cvtColor(np.ascontiguousarray(rgb_probe), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(np.ascontiguousarray(rgb_gallery), cv2.COLOR_RGB2BGR)
    h = max(p.shape[0], g.shape[0])

    def _pad(im):
        out = np.zeros((h, im.shape[1], 3), np.uint8)
        out[: im.shape[0], :, :] = im
        return out

    p2, g2 = _pad(p), _pad(g)
    wp = p2.shape[1]
    canvas = np.hstack([p2, g2])
    for s in (spots_probe or []):
        cv2.circle(canvas, (int(round(s.x)), int(round(s.y))), 5, _SPOT, 1)
    for s in (spots_gallery or []):
        cv2.circle(canvas, (int(round(s.x)) + wp, int(round(s.y))), 5, _SPOT, 1)
    for k, (ia, ib) in enumerate(matched_pairs or []):
        col = _PAL[k % len(_PAL)]
        x1, y1 = int(round(spots_probe[ia].x)), int(round(spots_probe[ia].y))
        x2, y2 = int(round(spots_gallery[ib].x)) + wp, int(round(spots_gallery[ib].y))
        cv2.line(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.circle(canvas, (x1, y1), 6, col, -1)
        cv2.circle(canvas, (x2, y2), 6, col, -1)
    if score is not None:
        cv2.putText(canvas, f"match {score * 100:.0f}%", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if label:
        cv2.putText(canvas, label, (6, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas
