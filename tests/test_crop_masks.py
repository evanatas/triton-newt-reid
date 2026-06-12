"""Тесты конвертеров масок (RLE/polygon/bitmask, TDD).
Особое внимание — НЕ перепутать HxW (урок GCN-масок Блока 1)."""
import numpy as np

from triton_crop.masks import (
    mask_area_frac,
    mask_to_polygon,
    mask_to_rle,
    polygon_to_mask,
    rle_to_mask,
)


def test_rle_roundtrip_preserves_mask_and_hxw():
    m = np.zeros((6, 4), np.uint8)  # h=6, w=4 (АСИММЕТРИЧНО — ловит свап осей)
    m[1:4, 1:3] = 1
    counts, h, w = mask_to_rle(m)
    assert (h, w) == (6, 4)
    back = rle_to_mask(counts, h, w)
    assert back.shape == (6, 4)
    assert np.array_equal(back > 0, m > 0)


def test_rle_decode_hxw_not_swapped():
    m = np.zeros((8, 3), np.uint8)  # h=8, w=3
    m[:, 0] = 1                     # левый столбец
    counts, h, w = mask_to_rle(m)
    back = rle_to_mask(counts, h, w)
    assert back.shape == (8, 3)
    assert (back[:, 0] > 0).all() and not (back[:, 2] > 0).any()


def test_mask_to_polygon_takes_largest_cc():
    m = np.zeros((100, 100), bool)
    m[10:14, 10:14] = True   # маленькая клякса
    m[40:80, 40:80] = True   # большая
    poly = mask_to_polygon(m.astype(np.uint8))
    assert poly is not None
    assert 35 < poly[:, 0].mean() < 85 and 35 < poly[:, 1].mean() < 85  # вокруг большой


def test_polygon_to_mask_roundtrip():
    poly = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], float)
    m = polygon_to_mask(poly, 80, 80)
    assert m[40, 40] and not m[5, 5]
    assert mask_to_polygon(m.astype(np.uint8)) is not None


def test_mask_area_frac():
    m = np.zeros((10, 10), bool)
    m[:5, :] = True
    assert abs(mask_area_frac(m) - 0.5) < 1e-9


def test_mask_to_polygon_none_on_tiny():
    m = np.zeros((100, 100), bool)
    m[0, 0] = True
    assert mask_to_polygon(m.astype(np.uint8), min_area_frac=0.01) is None
