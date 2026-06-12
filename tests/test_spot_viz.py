"""Тесты оверлеев пятен/матчей (Блок 5, шаг 5.8). draw_spot_overlay (QA детектора) +
draw_match_overlay (side-by-side + линии между совпавшими пятнами + % — запрос заказчика №1).
"""
import numpy as np

from triton_crop.spots import Spot


def test_draw_spot_overlay_shape():
    from triton_crop.viz import draw_spot_overlay
    rgb = np.full((80, 60, 3), 120, np.uint8)
    out = draw_spot_overlay(rgb, [Spot(10, 20, 30, 1.0), Spot(40, 50, 25, 0.9)], label="bo")
    assert out.shape == (80, 60, 3) and out.dtype == np.uint8
    assert draw_spot_overlay(rgb, []).shape == (80, 60, 3)        # пустой список — не падает


def test_draw_match_overlay_side_by_side_and_lines():
    from triton_crop.viz import draw_match_overlay
    rp = np.full((80, 60, 3), 100, np.uint8)
    rg = np.full((80, 60, 3), 150, np.uint8)
    sp = [Spot(10, 20, 30, 1.0), Spot(40, 60, 25, 0.9)]
    sg = [Spot(12, 22, 30, 1.0), Spot(38, 58, 25, 0.9)]
    no_lines = draw_match_overlay(rp, sp, rg, sg, [], score=None)
    with_lines = draw_match_overlay(rp, sp, rg, sg, [(0, 0), (1, 1)], score=0.5, label="K7")
    assert with_lines.shape[1] == 120                            # side-by-side: 60+60
    assert with_lines.shape[0] == 80
    assert not np.array_equal(no_lines, with_lines)              # линии меняют картинку
