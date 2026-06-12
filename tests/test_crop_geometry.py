"""Тесты чистой геометрии кропа (TDD)."""
import numpy as np

from triton_crop.geometry import (
    apply_affine_image,
    head_up_angle_deg,
    mask_to_tight_bbox,
    rotate_points,
    rotation_matrix,
    unroll_rectangle,
)


def test_head_up_angle_already_up():
    assert abs(head_up_angle_deg((65, 20), (65, 162))) < 1e-6  # голова выше клоаки → 0°


def test_head_up_angle_cases():
    assert abs(head_up_angle_deg((200, 100), (100, 100)) - 90) < 1e-6    # голова справа → 90
    assert abs(head_up_angle_deg((100, 100), (200, 100)) - (-90)) < 1e-6  # голова слева → −90
    assert abs(abs(head_up_angle_deg((100, 200), (100, 100))) - 180) < 1e-6  # голова снизу → ±180


def test_rotation_no_mirror_det_positive():
    M = rotation_matrix(37.0, (10, 10))
    assert np.linalg.det(M[:, :2]) > 0  # поворот, НЕ отражение (хиральность = идентичность)


def test_rotate_brings_head_above_cloaca():
    head, cloaca = (200, 100), (100, 100)  # голова справа
    M = rotation_matrix(head_up_angle_deg(head, cloaca), cloaca)
    (hx, hy), (cx, cy) = rotate_points([head, cloaca], M)
    assert abs(hx - cx) < 1e-3  # голова ровно над клоакой по x
    assert hy < cy              # и выше (меньший y)


def test_rotate_points_invertible():
    pts = np.array([[10, 20], [30, 40.0]])
    out = rotate_points(pts, rotation_matrix(25.0, (0, 0)))
    back = rotate_points(out, rotation_matrix(-25.0, (0, 0)))
    assert np.allclose(back, pts, atol=1e-6)


def test_mask_to_tight_bbox_margin_and_clamp():
    m = np.zeros((100, 80), bool)
    m[40:60, 30:50] = True
    assert mask_to_tight_bbox(m, 0.0) == (30, 40, 50, 60)
    assert mask_to_tight_bbox(m, 5.0) == (0, 0, 80, 100)  # огромный margin → клампится к краям


def test_unroll_rectangle_axis_vertical_head_top():
    r = unroll_rectangle((65, 20), (65, 160), 0.55)
    assert r["angle"] == 0.0
    assert abs(r["h"] - 140) < 1e-6
    assert abs(r["w"] - 2 * 0.55 * 140) < 1e-6
    assert r["p_head"][1] < r["p_cloaca"][1]


def test_apply_affine_image_rotates_180():
    img = np.zeros((50, 50, 3), np.uint8)
    img[0:10, :] = 255  # белая полоса сверху
    out = apply_affine_image(img, rotation_matrix(180.0, (25, 25)), (50, 50))
    assert out[40:50, :].mean() > out[0:10, :].mean()  # полоса ушла вниз
