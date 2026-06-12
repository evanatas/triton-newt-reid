"""Тест оверлея правки (чисто, cv2): форма/тип сохраняются + цветные маркеры головы/клоаки."""
import numpy as np

from triton_crop.viz import draw_overlay


def test_overlay_shape_and_markers():
    H, W = 120, 100
    rgb = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8); mask[40:80, 30:70] = 1
    poly = np.array([[30, 40], [70, 40], [70, 80], [30, 80]], float)
    out = draw_overlay(rgb, mask, poly, (50, 45), (50, 75))
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    # BGR: голова зелёная (канал G), клоака красная (канал R)
    assert out[45, 50, 1] > 200 and out[45, 50, 0] < 90 and out[45, 50, 2] < 90
    assert out[75, 50, 2] > 200 and out[75, 50, 0] < 90 and out[75, 50, 1] < 90


def test_overlay_handles_none_mask_and_points():
    out = draw_overlay(np.zeros((50, 50, 3), np.uint8), None, None, None, None)
    assert out.shape == (50, 50, 3) and out.dtype == np.uint8
