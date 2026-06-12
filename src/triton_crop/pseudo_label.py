"""Псевдо-метки брюшка из маски ВСЕГО тела (ЧИСТО: numpy/scipy). Без моделей.

Выводим точки голова/клоака + полигон брюшной полосы (пектораль→клоака), исключая
голову/горло/хвост/лапы. Грубый bootstrap — потом правится вручную и заменяется YOLO-pose.
Ось — из PCA; 180°-неоднозначность снимает арбитр «голова шире хвоста» (плечевой пояс).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PseudoLabel:
    belly_polygon: np.ndarray  # Nx2 px
    head_xy: tuple
    cloaca_xy: tuple
    head_conf: float           # нормированная разность ширины концов (уверенность арбитра)
    band_conf: float
    notes: str


def _largest_cc(mask):
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl, float), lbl, range(1, n + 1))
    return lbl == (1 + int(np.argmax(sizes)))


def derive_pseudo_label(animal_mask, cfg=None, head_frac: float = 0.18,
                        cloaca_frac: float = 0.66) -> PseudoLabel:
    """Маска тела → (belly_polygon, head_xy, cloaca_xy). Доли head_frac/cloaca_frac — вдоль оси от головы."""
    mask = ndimage.binary_fill_holes(_largest_cc(np.asarray(animal_mask) > 0))
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    centroid = pts.mean(0)
    rel = pts - centroid

    # главная ось (PCA) + перпендикуляр
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    perp = np.array([-axis[1], axis[0]])
    t = rel @ axis           # координата вдоль оси
    d = rel @ perp           # координата поперёк
    tmin, tmax = float(t.min()), float(t.max())
    span = tmax - tmin

    # профиль ширины по 40 срезам; концы сравниваем → широкий конец = голова (плечевой пояс)
    nb = 40
    edges = np.linspace(tmin, tmax, nb + 1)
    width = np.zeros(nb)
    for i in range(nb):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if sel.any():
            dd = d[sel]
            width[i] = float(dd.max() - dd.min())
    k = max(1, nb // 5)
    w_low, w_high = float(width[:k].mean()), float(width[-k:].mean())
    head_at_tmax = w_high >= w_low
    head_conf = abs(w_high - w_low) / (max(w_high, w_low) + 1e-6)
    head_t, tail_t = (tmax, tmin) if head_at_tmax else (tmin, tmax)

    def axis_point(frac):
        return centroid + (head_t + (tail_t - head_t) * frac) * axis

    head_xy = axis_point(0.0)
    cloaca_xy = axis_point(cloaca_frac)

    # брюшная полоса: вдоль оси [head_frac..cloaca_frac]; ширину урезаем до центральных 50%
    # (перцентили 25/75) → выкидываем боковые выступы ЛАП.
    rail_lo, rail_hi = [], []
    for frac in np.linspace(head_frac, cloaca_frac, 24):
        tt = head_t + (tail_t - head_t) * frac
        sel = np.abs(t - tt) <= max(span / nb, 1.0)
        if not sel.any():
            continue
        dlo, dhi = np.percentile(d[sel], [25, 75])
        rail_lo.append(centroid + tt * axis + dlo * perp)
        rail_hi.append(centroid + tt * axis + dhi * perp)
    belly_polygon = np.array(rail_lo + rail_hi[::-1], dtype=float)

    band_area = 0.5 * abs(_shoelace(belly_polygon)) if len(belly_polygon) >= 3 else 0.0
    ratio = band_area / float(mask.sum() + 1e-6)
    band_conf = float(np.clip(1.0 - abs(ratio - 0.35) / 0.35, 0.0, 1.0))
    notes = "" if 0.15 <= ratio <= 0.55 else "band_suspect"

    return PseudoLabel(
        belly_polygon=belly_polygon,
        head_xy=(float(head_xy[0]), float(head_xy[1])),
        cloaca_xy=(float(cloaca_xy[0]), float(cloaca_xy[1])),
        head_conf=float(head_conf), band_conf=band_conf, notes=notes)


def _shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
