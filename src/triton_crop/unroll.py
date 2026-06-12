"""Блок 3 — распрямление брюшка (unroll): чистая геометрия (cv2 + numpy), без моделей.

Вход: канонический кроп `belly_oriented` (uint8 HxWx3, голова вверху, фон чёрный) + маска пуза
в каноне + bounded опорные точки оси (внутри маски). Выход: распрямлённый кроп ТОГО ЖЕ размера
(letterbox-канвас не меняется — единый препроцессинг), фон чёрный — вход для эмбеддера (Блок 4) и матчинга
пятен (Блок 5).

Методы:
  debend — сдвиг строк по центроиду тела (убирает изгиб, минимально искажает узор) — безопасный дефолт;
  wnorm  — debend + нормировка ширины срезов (бьёт по растяжению «поел/переварил»; рискованный — узор
           искажается горизонтально, принимается в эмбеддер только под гейтом сохранения узора C10c);
  ribbon — арк-ленс развёртка по кривой (добавляется в шаге 4; «лента» для Блока 5);
  off    — тождество (вариант существует и сравним в A/B).
Все функции детерминированы (одинаковый вход → побайтово одинаковый выход; без рандома). Идентичность
gallery=probe обеспечивается этим И кропом одной сборкой cv2/numpy (на разных платформах cv2.remap/
np.polyfit возможны ±1 LSB — gallery и probe кропать одной сборкой; golden-hash — задел Блока 4).
ВАЖНО: детерминизм НЕ доказывает re-ID-безопасность распрямления — это решает только A/B-гейт (Rosa).
"""
import cv2
import numpy as np

_OK_METHODS = ("debend", "wnorm", "ribbon", "off")


def _runs(idx):
    """Список непрерывных отрезков (runs) из отсортированных индексов столбцов строки."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def belly_centerline(mask, axis_start, axis_end, poly_deg=3):
    """Гладкая центральная линия пуза по центроиду НАИБОЛЬШЕЙ 1D-компоненты каждой строки.

    y-диапазон ОГРАНИЧЕН осью пуза [axis_start.y .. axis_end.y] (если задана) ∩ маска — чтобы не тянуть
    линию по голове/хвосту/лапам вне оси (ось НЕ декоративна). Дыры интерполируем, линию сглаживаем
    полиномом степени poly_deg. Если значимая доля строк МУЛЬТИ-ЛОБНА (C/U-петля: ≥2 сопоставимых
    куска) → None (центроид строки не следует за осью). -> (y_all, xc_smooth, width_profile) или None.
    """
    mask = np.asarray(mask) > 0
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    if axis_start is not None and axis_end is not None:
        ay0, ay1 = int(min(axis_start[1], axis_end[1])), int(max(axis_start[1], axis_end[1])) + 1
        y0, y1 = max(y0, ay0), min(y1, ay1)        # ограничить диапазон осью пуза (не тянуть линию по голове/хвосту)
    if y1 - y0 < 2:
        return None
    y_all = np.arange(y0, y1)
    xc = np.full(y_all.shape, np.nan, float)
    ww = np.zeros(y_all.shape, float)
    multilobe = 0
    for i, y in enumerate(y_all):
        runs = _runs(np.where(mask[y])[0])
        if not runs:
            continue
        runs.sort(key=len, reverse=True)
        main = runs[0]
        xc[i] = float(main.mean())
        ww[i] = float(main.max() - main.min() + 1)
        if len(runs) >= 2 and len(runs[1]) >= 0.5 * len(main):
            multilobe += 1
    valid = ~np.isnan(xc)
    nvalid = int(valid.sum())
    if nvalid < int(poly_deg) + 2:
        return None
    if multilobe > 0.15 * nvalid:                  # C/U-петля → линия по маске недостоверна
        return None
    xc = np.interp(y_all, y_all[valid], xc[valid])
    ww = np.interp(y_all, y_all[valid], ww[valid])
    deg = min(int(poly_deg), max(1, nvalid - 1))
    yc0 = y_all - float(y_all.mean())   # центрируем y для обусловленности фита (RankWarning), значения xc_smooth те же
    xc_smooth = np.polyval(np.polyfit(yc0, xc, deg), yc0)
    return y_all, xc_smooth, ww


def centerline_curvature(xc_smooth, y_all) -> float:
    """Интегральная кривизна центральной линии: средний |x''| по дуге. Прямая → ~0
    (инвариант C10b, проверяемый ТЕСТАМИ (TDD), а не runtime-гард — в проде не enforce-ится)."""
    xc = np.asarray(xc_smooth, float)
    y = np.asarray(y_all, float)
    if xc.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.gradient(np.gradient(xc, y), y))))


def straighten_belly(canon_image, mask_canon, axis_start, axis_end, cfg, method="debend"):
    """Распрямить пузо. -> (uint8 HxWx3 ТОГО ЖЕ размера, status).

    status ∈ ok | no_mask | no_axis | axis_too_short | degenerate_centerline | bad_method.
    При любом не-ok возвращает вход без изменений (вариант просто не пишется как unrolled).
    """
    img = np.ascontiguousarray(np.asarray(canon_image), np.uint8)
    if method not in _OK_METHODS:
        return img, "bad_method"
    if method == "off":
        return img.copy(), "ok"
    if mask_canon is None or not (np.asarray(mask_canon) > 0).any():
        return img, "no_mask"
    if axis_start is None or axis_end is None:
        return img, "no_axis"
    mask = np.asarray(mask_canon) > 0
    cl = belly_centerline(mask, axis_start, axis_end, cfg.unroll_poly_deg)
    if cl is None:
        return img, "degenerate_centerline"
    y_all, xc_smooth, ww = cl
    if int(y_all[-1] - y_all[0] + 1) < int(cfg.unroll_min_rows):
        return img, "axis_too_short"
    if method == "ribbon":
        return _ribbon(img, mask, y_all, xc_smooth, ww, cfg)
    return _debend(img, mask, y_all, xc_smooth, ww, cfg, width_norm=(method == "wnorm"))


def _debend(img, mask, y_all, xc_smooth, ww, cfg, width_norm=False):
    """Сдвиг каждой строки тела так, чтобы центральная линия стала вертикалью W/2 (cv2.remap).

    Строка едет целиком → вертикальные координаты пятен инвариантны, площади сохранены.
    width_norm=True (метод wnorm): дополнительно масштабирует ширину среза к медианной.
    """
    H, W = mask.shape
    x_target = W / 2.0
    maxs = cfg.unroll_max_shift_frac * W
    xc_clamped = np.clip(np.asarray(xc_smooth, float), x_target - maxs, x_target + maxs)
    median_w = float(np.median(ww)) if ww.size else 1.0
    xs = np.arange(W, dtype=np.float32)
    map_x = np.tile(xs, (H, 1))
    map_y = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    for i, y in enumerate(y_all):
        scale = (ww[i] / median_w) if (width_norm and ww[i] > 1 and median_w > 0) else 1.0
        # dst(x,y) = src(xc + (x - x_target)*scale) → центрирует ось на W/2, опц. нормирует ширину
        map_x[int(y)] = (xc_clamped[i] + (xs - x_target) * scale).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mout = cv2.remap((mask.astype(np.uint8) * 255), map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
    out[~mout] = 0
    return np.ascontiguousarray(out, np.uint8), "ok"


# Порог «синего» пятна — СИНХРОНИЗИРОВАН с цветом синтетического пятна (20,20,230) в pattern_preserved ниже.
_BLUE_B_MIN, _BLUE_RG_MAX, _MIN_SPOT_AREA = 150, 110, 3


def _measure_spot_offsets(rgb, w_center):
    """Центроиды контрастных (синих) пятен и их смещение по x от вертикали w_center. -> [(off, y)] по y."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    blue = ((b > _BLUE_B_MIN) & (r < _BLUE_RG_MAX) & (g < _BLUE_RG_MAX)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(blue, 8)
    out = [(float(cents[i][0]) - w_center, float(cents[i][1]))
           for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= _MIN_SPOT_AREA]
    return sorted(out, key=lambda p: p[1])


def pattern_preserved(method, cfg) -> bool:
    """ХАРД-гейт сохранения узора как ИЗМЕРЕНИЕ (а не хардкод): на синтетике ПЕРЕМЕННОЙ ширины (узкое
    тело + резкий бугор «поел») с пятнами известного смещения от оси проверяем, что метод НЕ разносит
    центроиды пятен. debend/ribbon → True (смещение от оси сохранено); wnorm → False (масштабирует
    ширину срезов → пятна в широкой части едут). Используется для измеренной pattern-safety на решении."""
    H, W = 200, 180
    cx, y0, y1 = 80, 30, 176

    def hw(y):
        return 10.0 + 18.0 * np.exp(-((y - 105.0) / 14.0) ** 2)

    mask = np.zeros((H, W), np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    rows = list(range(y0, y1))
    band = np.array([(int(round(cx - hw(y))), y) for y in rows]
                    + [(int(round(cx + hw(y))), y) for y in reversed(rows)], np.int32)
    cv2.fillPoly(mask, [band], 255)
    cv2.fillPoly(rgb, [band], (170, 130, 80))
    spots = [(105, 18), (60, 6), (150, -5), (90, 10)]              # (y, смещение от оси); бугор-спот off=18
    for ys, off in spots:
        cv2.circle(rgb, (int(round(cx + off)), ys), 3, (20, 20, 230), -1)
    img_u, st = straighten_belly(rgb, mask > 0, (cx, y0 + 4), (cx, y1 - 6), cfg, method)
    if st != "ok":
        return False
    after = _measure_spot_offsets(img_u, W / 2.0)
    if len(after) != len(spots):
        return False
    tol = cfg.unroll_pattern_tol_frac * W
    for (ys, off), (off_a, ya) in zip(sorted(spots, key=lambda p: p[0]), after):
        if abs(off_a - off) > tol or abs(ya - ys) > tol + 2:
            return False
    return True


def pattern_safe_methods(cfg, methods=("debend", "ribbon", "wnorm")) -> dict:
    """{метод: pattern_preserved(метод, cfg)} — измеренная pattern-safety для правила адопции (cli ab)."""
    return {m: pattern_preserved(m, cfg) for m in methods}


def _ribbon(img, mask, y_all, xc_smooth, ww, cfg):
    """Арк-ленс развёртка: выходные строки расставлены РАВНОМЕРНО ПО ДЛИНЕ ДУГИ центральной кривой,
    в каждой — перпендикуляр (нормаль к оси) натурального масштаба. Равномерная плотность вдоль тела
    → без потери площади и без перекрытия/дыр (в отличие от построчного семплинга). «Выпрямленная
    лента» для матчинга созвездия пятен (Блок 5). -> (uint8, "ok").
    Нюансы реализации: (1) сдвиг НЕ клампится через unroll_max_shift_frac (опора на mask-reapply, в отличие
    от _debend); (2) нижняя строка y1 НЕ входит в развёртку — out_rows=arange(y0, y1), полуоткрытый интервал."""
    H, W = mask.shape
    y_all = np.asarray(y_all, float)
    xc = np.asarray(xc_smooth, float)
    y0, y1 = int(y_all[0]), int(y_all[-1])
    dxc = np.gradient(xc, y_all)                              # касательная dx/dy
    seg = np.sqrt(1.0 + dxc * dxc)                            # |d(точка)/dy|
    s = np.concatenate([[0.0], np.cumsum((seg[:-1] + seg[1:]) / 2.0 * np.diff(y_all))])
    total = float(s[-1]) if s[-1] > 0 else 1.0
    out_rows = np.arange(y0, y1)
    target_s = (out_rows - y0) / max(1, (y1 - 1 - y0)) * total
    yc = np.interp(target_s, s, y_all)                        # центр выходной строки (равномерно по дуге)
    xcen = np.interp(yc, y_all, xc)
    dyc = np.interp(yc, y_all, dxc)
    cols = np.arange(W, dtype=np.float32) - W / 2.0
    map_x = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    map_y = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    for i, v in enumerate(out_rows):
        tx, ty = float(dyc[i]), 1.0
        nrm = float(np.hypot(tx, ty))
        nx, ny = ty / nrm, -tx / nrm                         # единичная нормаль ⟂ касательной
        map_x[int(v)] = (xcen[i] + cols * nx).astype(np.float32)
        map_y[int(v)] = (yc[i] + cols * ny).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mout = cv2.remap((mask.astype(np.uint8) * 255), map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
    out[~mout] = 0
    return np.ascontiguousarray(out, np.uint8), "ok"
