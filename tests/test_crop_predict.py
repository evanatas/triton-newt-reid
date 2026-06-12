"""Тесты разбора результата YOLO (чисто, без загрузки моделей) + smoke-импорт обёрток."""
import numpy as np

from triton_crop.predict import BellySegmenter, PoseEstimator, parse_pose, parse_seg


def test_parse_seg_picks_best_and_builds_mask():
    sq = [[10, 10], [30, 10], [30, 30], [10, 30]]
    m, c = parse_seg([sq, [[0, 0], [1, 0], [1, 1]]], np.array([0.9, 0.2]), 40, 40)
    assert abs(c - 0.9) < 1e-9 and m is not None and m.sum() > 0     # выбрал детекцию с conf 0.9


def test_parse_seg_empty_or_degenerate():
    assert parse_seg(None, None, 10, 10) == (None, 0.0)
    assert parse_seg([], np.array([]), 10, 10) == (None, 0.0)
    m, c = parse_seg([[[0, 0], [1, 0]]], np.array([0.5]), 10, 10)     # полигон <3 точек → None
    assert m is None and abs(c - 0.5) < 1e-9


def test_parse_pose_returns_min_keypoint_conf():
    kp = np.array([[[5, 5, 0.9], [5, 9, 0.2]]])      # голова conf 0.9, клоака 0.2 → min 0.2
    head, cloaca, c = parse_pose(kp, np.array([0.8]))
    assert head == (5.0, 5.0) and cloaca == (5.0, 9.0) and abs(c - 0.2) < 1e-9


def test_parse_pose_empty():
    assert parse_pose(None, None) == (None, None, 0.0)


def test_predict_classes_importable():
    assert callable(BellySegmenter) and callable(PoseEstimator)
