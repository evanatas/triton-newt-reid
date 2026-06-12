"""Тесты детектора пятен (Блок 5, шаг 5.3) — гейт S1.

Детектор подключаемый (deviation/darkness/log/dog), работает ВНУТРИ маски пуза. На synthetic_curved_newt
известны 4 вшитых центроида (spot_centroids) → проверяем точность ±2px. Чистый CV, герметично.
"""
import cv2
import numpy as np
import pytest

from triton_crop.config import SpotConfig


def _match_centroids(spots, truth, tol=2.5):
    """Каждому истинному центроиду нашлось детектированное пятно в пределах tol (px)."""
    got = np.array([[s.x, s.y] for s in spots], float)
    ok = 0
    for (tx, ty) in truth:
        if len(got) and np.min(np.hypot(got[:, 0] - tx, got[:, 1] - ty)) <= tol:
            ok += 1
    return ok


def test_detect_deviation_four_on_synthetic(synthetic_curved_newt):
    from triton_crop.spots import detect_spots
    d = synthetic_curved_newt
    spots = detect_spots(d.rgb, d.mask, SpotConfig())                    # default = deviation
    assert len(spots) == 4                                              # ровно 4 пятна
    assert _match_centroids(spots, d.spot_centroids) == 4               # все 4 центроида ±2px


def test_detect_empty_mask_returns_empty(synthetic_curved_newt):
    from triton_crop.spots import detect_spots
    d = synthetic_curved_newt
    spots = detect_spots(d.rgb, np.zeros_like(d.mask), SpotConfig())
    assert spots == []


def test_detect_deterministic(synthetic_curved_newt):
    from triton_crop.spots import detect_spots
    d = synthetic_curved_newt
    a = detect_spots(d.rgb, d.mask, SpotConfig())
    b = detect_spots(d.rgb, d.mask, SpotConfig())
    assert [(s.x, s.y, s.area) for s in a] == [(s.x, s.y, s.area) for s in b]


def test_detect_top_n_caps_overdetection():
    from dataclasses import replace
    from triton_crop.spots import detect_spots
    rgb = np.full((120, 120, 3), (170, 130, 80), np.uint8)              # тело
    mask = np.ones((120, 120), bool)
    rng = np.random.RandomState(0)
    for _ in range(40):                                                 # 40 синих точек (over-detection)
        x, y = rng.randint(5, 115), rng.randint(5, 115)
        cv2.circle(rgb, (x, y), 2, (20, 20, 230), -1)
    spots = detect_spots(rgb, mask, replace(SpotConfig(), spot_top_n=5))
    assert len(spots) == 5                                              # top-N по площади обрезает
    areas = [s.area for s in spots]
    assert areas == sorted(areas, reverse=True)                        # отсортировано по площади


def test_detect_darkness_on_dark_spots():
    from dataclasses import replace
    from triton_crop.spots import detect_spots
    rgb = np.full((100, 100, 3), 200, np.uint8)                        # светлое тело
    mask = np.ones((100, 100), bool)
    for (x, y) in [(30, 30), (70, 40), (50, 75)]:
        cv2.circle(rgb, (x, y), 5, (30, 30, 30), -1)                   # тёмные пятна
    spots = detect_spots(rgb, mask, replace(SpotConfig(), detect_method="darkness"))
    assert _match_centroids(spots, [(30, 30), (70, 40), (50, 75)]) == 3


def test_detect_log_finds_spots_on_synthetic(synthetic_curved_newt):
    from dataclasses import replace
    from triton_crop.spots import detect_spots
    d = synthetic_curved_newt
    spots = detect_spots(d.rgb, d.mask, replace(SpotConfig(), detect_method="log"))
    assert 3 <= len(spots) <= 6                                        # blob-детектор приблизителен
    assert _match_centroids(spots, d.spot_centroids) >= 3


def test_detect_mask_erosion_drops_border_artifacts():
    # Итерация 5c (визуальный QA): тёмные пятна у ГРАНИЦЫ маски = краевые артефакты (letterbox/обрез),
    # не пятна узора. Эрозия маски их убирает, центральные настоящие пятна остаются.
    from dataclasses import replace
    from triton_crop.spots import detect_spots
    rgb = np.full((80, 80, 3), (170, 130, 80), np.uint8)
    mask = np.zeros((80, 80), bool); mask[10:70, 10:70] = True
    cv2.circle(rgb, (40, 40), 5, (20, 20, 20), -1)        # настоящее пятно в центре
    cv2.circle(rgb, (12, 12), 3, (20, 20, 20), -1)        # артефакт у границы маски
    spots = detect_spots(rgb, mask, replace(SpotConfig(), detect_method="darkness", mask_erode_px=6))
    assert any(abs(s.x - 40) < 3 and abs(s.y - 40) < 3 for s in spots)    # центр остался
    assert not any(s.x < 16 and s.y < 16 for s in spots)                 # краевой ушёл после эрозии


def test_detect_illum_norm_does_not_break_synthetic(synthetic_curved_newt):
    # CLAHE-нормализация (illum_norm) для выравнивания бледных↔ярких реальных кадров — не должна ломать
    # чистую детекцию на синтетике (равномерное тело → CLAHE не плодит ложных пятен).
    from dataclasses import replace
    from triton_crop.spots import detect_spots
    d = synthetic_curved_newt
    spots = detect_spots(d.rgb, d.mask, replace(SpotConfig(), illum_norm=True))
    assert 3 <= len(spots) <= 6


def test_select_prefers_high_contrast_small_over_large_shadow():
    # Регресс: большая слабоконтрастная тень (area велика, score низок) НЕ должна вытеснять
    # маленькое высококонтрастное пятно. salience = area×contrast, не только площадь.
    from dataclasses import replace
    from triton_crop.spots import Spot, _select
    shadow = Spot(x=5, y=5, area=200.0, score=0.2)         # большая тень, низкий контраст → salience 40
    spot = Spot(x=9, y=9, area=20.0, score=5.0)            # маленькое яркое пятно → salience 100
    cfg = replace(SpotConfig(), spot_top_n=1, select_by="salience")
    assert _select([shadow, spot], cfg) == [spot]
    # select_by="area" сохраняет старое поведение (по площади)
    cfg_area = replace(SpotConfig(), spot_top_n=1, select_by="area")
    assert _select([shadow, spot], cfg_area) == [shadow]


def test_spots_to_array_shape(synthetic_curved_newt):
    from triton_crop.spots import detect_spots, spots_to_array
    d = synthetic_curved_newt
    arr = spots_to_array(detect_spots(d.rgb, d.mask, SpotConfig()))
    assert arr.shape == (4, 2)
    assert spots_to_array([]).shape == (0, 2)
