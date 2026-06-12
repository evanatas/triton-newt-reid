"""Тесты Блока 3 (распрямление): belly_centerline, centerline_curvature, straighten_belly.

Чистая геометрия на синтетике (изогнутое пузо + вшитые пятна), без моделей. Покрывает гейты
C10a (детерминизм), C10b (кривизна падает), C10c (узор сохранён), контракт выхода, fallback-статусы.
"""
import cv2
import numpy as np

from triton_crop.config import CropConfig
from triton_crop.unroll import belly_centerline, centerline_curvature, straighten_belly


def _spot_centroids(rgb):
    """Центроиды синих пятен на кадре (по цвету) -> list[(x, y)] сорт. по y."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    blue = ((b > 150) & (r < 110) & (g < 110)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(blue, 8)
    out = [(float(cents[i][0]), float(cents[i][1])) for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= 4]
    return sorted(out, key=lambda p: p[1])


def _tiny_mask(rows):
    """Заполненный прямоугольник высотой `rows` строк (для fallback-тестов)."""
    m = np.zeros((rows + 20, 60), bool)
    m[5:5 + rows, 20:40] = True
    return m


# --- belly_centerline следует кривой оси ---
def test_centerline_follows_curve(synthetic_curved_newt):
    s = synthetic_curved_newt
    y_all, xc, ww = belly_centerline(s.mask, s.axis_start, s.axis_end, poly_deg=3)
    err = max(abs(xc[i] - s.axis_x(y_all[i])) for i in range(len(y_all)))
    assert err < 2.5
    assert (ww > 0).all()


# --- C10b: де-бенд снижает кривизну центральной линии ---
def test_debend_reduces_curvature(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    y0, xc0, _ = belly_centerline(s.mask, s.axis_start, s.axis_end, cfg.unroll_poly_deg)
    k_before = centerline_curvature(xc0, y0)
    img_u, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "debend")
    assert st == "ok"
    mask_u = img_u.any(axis=2)
    y1, xc1, _ = belly_centerline(mask_u, s.axis_start, s.axis_end, cfg.unroll_poly_deg)
    assert centerline_curvature(xc1, y1) < 0.3 * k_before


# --- голова остаётся вверху (вертикаль не инвертирована) ---
def test_debend_keeps_head_up(synthetic_curved_newt):
    s = synthetic_curved_newt
    img_u, _ = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, CropConfig(), "debend")
    rows = np.where(img_u.any(axis=2).any(axis=1))[0]
    assert rows.min() < rows.max()
    # верх тела пришёл с верха (узкая зона головы по y сохранила порядок)
    assert rows.min() <= s.axis_start[1] + 5


# --- C10c: узор сохранён (центроиды пятен едут с телом, не разрушаются) ---
def test_debend_preserves_pattern(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    img_u, _ = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "debend")
    after = _spot_centroids(img_u)
    assert len(after) == len(s.spot_centroids)          # пятна не слиты/не потеряны
    W = s.rgb.shape[1]
    tol = cfg.unroll_pattern_tol_frac * W + 2
    for (xs, ys), (xa, ya) in zip(s.spot_centroids, after):
        assert abs(ya - ys) <= tol                       # вертикаль сохранена
        off_before = xs - s.axis_x(ys)                   # смещение от оси тела
        off_after = xa - W / 2.0                          # ось теперь по центру кадра
        assert abs(off_after - off_before) <= tol         # смещение от тела сохранено


# --- C10a: детерминизм (gallery=probe) ---
def test_debend_deterministic(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    a, sa = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "debend")
    b, sb = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "debend")
    assert sa == sb == "ok" and np.array_equal(a, b)


# --- контракт выхода: тот же размер/тип, фон чёрный ---
def test_debend_output_contract(synthetic_curved_newt):
    s = synthetic_curved_newt
    img_u, _ = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, CropConfig(), "debend")
    assert img_u.shape == s.rgb.shape and img_u.dtype == np.uint8
    assert (img_u[:10, :10] == 0).all()                  # угол-фон чёрный


# --- wnorm выпрямляет (статус ok), узор остаётся целым по числу пятен ---
def test_wnorm_runs_ok(synthetic_curved_newt):
    s = synthetic_curved_newt
    img_u, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, CropConfig(), "wnorm")
    assert st == "ok"
    assert len(_spot_centroids(img_u)) == len(s.spot_centroids)


# --- off = тождество ---
def test_off_is_identity(synthetic_curved_newt):
    s = synthetic_curved_newt
    img_u, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, CropConfig(), "off")
    assert st == "ok" and np.array_equal(img_u, s.rgb)


# --- fallback-статусы (без исключений) ---
def test_fallback_no_mask(synthetic_curved_newt):
    s = synthetic_curved_newt
    img_u, st = straighten_belly(s.rgb, None, s.axis_start, s.axis_end, CropConfig(), "debend")
    assert st == "no_mask" and np.array_equal(img_u, s.rgb)


def test_fallback_no_axis(synthetic_curved_newt):
    s = synthetic_curved_newt
    _, st = straighten_belly(s.rgb, s.mask, None, s.axis_end, CropConfig(), "debend")
    assert st == "no_axis"


def test_fallback_axis_too_short():
    cfg = CropConfig()
    m = _tiny_mask(rows=10)                               # 10 строк < unroll_min_rows(24)
    img = np.zeros((*m.shape, 3), np.uint8)
    img[m] = (170, 130, 80)
    _, st = straighten_belly(img, m, (30, 6), (30, 14), cfg, "debend")
    assert st == "axis_too_short"


def test_fallback_degenerate_centerline():
    cfg = CropConfig()
    m = _tiny_mask(rows=2)                                # < poly_deg+2 валидных строк
    img = np.zeros((*m.shape, 3), np.uint8)
    img[m] = (170, 130, 80)
    _, st = straighten_belly(img, m, (30, 6), (30, 7), cfg, "debend")
    assert st == "degenerate_centerline"


def test_bad_method(synthetic_curved_newt):
    s = synthetic_curved_newt
    _, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, CropConfig(), "nope")
    assert st == "bad_method"


# --- ribbon (метод для Блока 5): выпрямляет, узор цел, детерминирован ---
def test_ribbon_runs_and_reduces_curvature(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    y0, xc0, _ = belly_centerline(s.mask, s.axis_start, s.axis_end, cfg.unroll_poly_deg)
    k_before = centerline_curvature(xc0, y0)
    img_u, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "ribbon")
    assert st == "ok"
    y1, xc1, _ = belly_centerline(img_u.any(axis=2), s.axis_start, s.axis_end, cfg.unroll_poly_deg)
    assert centerline_curvature(xc1, y1) < 0.3 * k_before
    assert len(_spot_centroids(img_u)) == len(s.spot_centroids)


def test_ribbon_deterministic(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    a, _ = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "ribbon")
    b, _ = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "ribbon")
    assert np.array_equal(a, b)


# --- узор-тест ИМЕННО для ribbon (принятый метод) на ИЗОГНУТОЙ оси + переменной ширине ---
def _curved_var_band(H=200, W=180, cx=85, A=40.0, R=80.0, ymid=100, y0=30, y1=176):
    def ax(y):
        return cx + A * ((y - ymid) / R) ** 2

    def axd(y):
        return 2 * A * (y - ymid) / (R * R)

    def hw(y):
        return 11.0 + 9.0 * np.sin(np.pi * (y - y0) / (y1 - y0))

    mask = np.zeros((H, W), np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    rows = list(range(y0, y1))
    band = np.array([(int(round(ax(y) - hw(y))), y) for y in rows]
                    + [(int(round(ax(y) + hw(y))), y) for y in reversed(rows)], np.int32)
    cv2.fillPoly(mask, [band], 255)
    cv2.fillPoly(rgb, [band], (170, 130, 80))
    spots = []
    for ys, d in [(60, -6), (95, 5), (130, -4), (160, 6)]:    # d = перпендикулярное смещение от оси
        tx, ty = axd(ys), 1.0
        nrm = (tx * tx + ty * ty) ** 0.5
        nx, ny = ty / nrm, -tx / nrm
        cv2.circle(rgb, (int(round(ax(ys) + d * nx)), int(round(ys + d * ny))), 3, (20, 20, 230), -1)
        spots.append((d, ys))
    return mask > 0, rgb, (int(round(ax(y0 + 4))), y0 + 4), (int(round(ax(y1 - 6))), y1 - 6), spots


def test_ribbon_preserves_spot_offset_on_curved_axis():
    mask, rgb, a0, a1, spots = _curved_var_band()
    cfg, W = CropConfig(), 180
    img_u, st = straighten_belly(rgb, mask, a0, a1, cfg, "ribbon")
    assert st == "ok"
    after = _spot_centroids(img_u)
    assert len(after) == len(spots)                          # узор не разрушен/не слит
    tol = cfg.unroll_pattern_tol_frac * W + 3
    for (d, ys), (xa, ya) in zip(sorted(spots, key=lambda p: p[1]), after):
        assert abs((xa - W / 2.0) - d) <= tol                # перпендикулярное смещение от оси сохранено


# --- геометрия на трудных формах ---
def test_debend_handles_single_s_curve():
    from triton_crop.unroll import belly_centerline, centerline_curvature
    H, W, ymid, y0, y1 = 220, 150, 110, 30, 196

    def ax(y):
        return 75 + 34 * ((y - ymid) / 80.0) ** 3            # кубическая S (один перегиб)

    mask = np.zeros((H, W), np.uint8)
    rows = list(range(y0, y1))
    band = np.array([(int(round(ax(y) - 14)), y) for y in rows]
                    + [(int(round(ax(y) + 14)), y) for y in reversed(rows)], np.int32)
    cv2.fillPoly(mask, [band], 255)
    mask = mask > 0
    img = np.zeros((H, W, 3), np.uint8)
    img[mask] = (170, 130, 80)
    a0, a1 = (int(ax(y0 + 4)), y0 + 4), (int(ax(y1 - 6)), y1 - 6)
    y, xc0, _ = belly_centerline(mask, a0, a1, 3)
    k0 = centerline_curvature(xc0, y)
    img_u, st = straighten_belly(img, mask, a0, a1, CropConfig(), "debend")
    assert st == "ok"
    y2, xc1, _ = belly_centerline(img_u.any(axis=2), a0, a1, 3)
    assert centerline_curvature(xc1, y2) < 0.5 * k0          # S распрямлена


def test_centerline_survives_holes_in_mask():
    from triton_crop.unroll import belly_centerline
    m = np.zeros((150, 80), bool)
    m[20:130, 30:50] = True
    m[60:75, 30:50] = False                                  # дыра в маске (выкус)
    assert belly_centerline(m, (40, 25), (40, 125), 3) is not None


def test_centerline_mask_touching_top_edge():
    from triton_crop.unroll import belly_centerline
    m = np.zeros((120, 80), bool)
    m[0:100, 30:50] = True                                   # маска у верхнего края (partial crop)
    assert belly_centerline(m, (40, 4), (40, 95), 3) is not None


def test_straighten_degenerate_when_axis_outside_mask_yrange():
    m = np.zeros((150, 80), bool)
    m[20:60, 30:50] = True                                   # маска по y: 20..59
    img = np.zeros((150, 80, 3), np.uint8)
    _, st = straighten_belly(img, m, (40, 100), (40, 120), CropConfig(), "debend")
    assert st in ("degenerate_centerline", "axis_too_short")  # ось вне y-диапазона маски


def _u_shape():
    m = np.zeros((140, 100), bool)
    m[20:120, 25:37] = True      # левый рукав
    m[20:120, 63:75] = True      # правый рукав
    m[108:120, 25:75] = True     # перемычка снизу (U-петля)
    return m


# --- multi-lobe-гард: C/U-петля → центральная линия недостоверна ---
def test_belly_centerline_rejects_multilobe():
    from triton_crop.unroll import belly_centerline
    assert belly_centerline(_u_shape(), (31, 25), (69, 115), poly_deg=3) is None


def test_straighten_degenerate_on_multilobe():
    m = _u_shape()
    img = np.zeros((*m.shape, 3), np.uint8)
    img[m] = (170, 130, 80)
    _, st = straighten_belly(img, m, (31, 25), (69, 115), CropConfig(), "debend")
    assert st == "degenerate_centerline"


# --- ось НЕ декоративна: y-диапазон ограничен осью пуза ---
def test_centerline_bounded_by_axis():
    from triton_crop.unroll import belly_centerline
    m = np.zeros((200, 80), bool)
    m[20:180, 30:50] = True                                    # маска по y: 20..179
    y_all, xc, ww = belly_centerline(m, (40, 60), (40, 120), poly_deg=3)
    assert y_all.min() >= 60 and y_all.max() <= 120            # диапазон ограничен осью


# --- ribbon arc-length: без существенной потери площади ---
def test_ribbon_preserves_area(synthetic_curved_newt):
    s, cfg = synthetic_curved_newt, CropConfig()
    before = int(s.mask.sum())
    img_u, st = straighten_belly(s.rgb, s.mask, s.axis_start, s.axis_end, cfg, "ribbon")
    after = int(img_u.any(axis=2).sum())
    assert st == "ok" and after >= 0.85 * before


# --- pattern-safety ИЗМЕРЯЕТСЯ (а не хардкод): wnorm ловится на переменной ширине ---
def test_pattern_preserved_classifies_methods():
    from triton_crop.unroll import pattern_preserved, pattern_safe_methods
    cfg = CropConfig()
    assert pattern_preserved("debend", cfg) is True
    assert pattern_preserved("ribbon", cfg) is True
    assert pattern_preserved("wnorm", cfg) is False        # нормировка ширины разносит узор — по делу
    assert pattern_safe_methods(cfg) == {"debend": True, "ribbon": True, "wnorm": False}
