"""Тест bounded belly-axis (clip_axis_to_mask): ось обрезается до маски пуза (для Блока 3)."""
import numpy as np

from triton_crop.geometry import clip_axis_to_mask


def test_clip_axis_to_mask_bounds_to_belly():
    mask = np.zeros((100, 100), bool); mask[30:70, 40:60] = True    # полоса пуза y∈[30,70)
    s, e = clip_axis_to_mask((50, 0), (50, 90), mask)               # голова сверху вне, клоака снизу вне
    assert 29 <= s[1] <= 31 and 68 <= e[1] <= 70                    # ось обрезана к маске
    assert abs(s[0] - 50) < 2 and abs(e[0] - 50) < 2


def test_clip_axis_none_when_axis_misses_mask():
    mask = np.zeros((50, 50), bool); mask[0:5, 0:5] = True
    assert clip_axis_to_mask((40, 40), (45, 45), mask) == (None, None)


def test_clip_axis_ignores_offframe_points_at_border_mask():
    # маска касается ЛЕВОЙ рамки; голова — ВНЕ кадра слева. Раньше clip x<0→0 ложно считал «внутри»
    # и возвращал ось с отрицательным x. Теперь off-frame точки не считаются попавшими в маску.
    mask = np.zeros((20, 20), bool); mask[:, 0:4] = True            # левая полоса касается рамки x=0
    s, e = clip_axis_to_mask((-15, 10), (2, 10), mask)             # голова далеко вне кадра слева, клоака внутри
    assert s is not None and e is not None
    # старый баг: clip x<0→0 + mask на рамке → start≈(-15,10) (голова вне кадра). Теперь start у границы маски.
    assert s[0] > -1.0 and 0 <= e[0] < 4                            # НЕ далёкая off-frame точка
    assert 0 <= s[1] < 20 and 0 <= e[1] < 20
