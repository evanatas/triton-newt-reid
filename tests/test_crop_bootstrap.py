"""Тесты бутстрапа: выбор маски животного, флаги правки, отбор пилота (анти-утечка test). Без моделей."""
import cv2
import numpy as np
import pandas as pd

from triton_crop.sam_bootstrap import derive_flags, pick_animal, select_pilot


def _blank(h=200, w=160):
    return np.zeros((h, w), np.uint8)


def test_picks_central_elongated_over_distractors():
    H, W = 200, 160
    A = _blank(H, W)
    cv2.ellipse(A, (80, 100), (18, 70), 0, 0, 360, 1, -1)   # центральный вытянутый = животное
    B = np.ones((H, W), np.uint8)                           # весь кадр = фон (area>max, border)
    C = _blank(H, W); cv2.circle(C, (18, 18), 4, 1, -1)      # крошечный угловой
    D = _blank(H, W); cv2.circle(D, (80, 100), 22, 1, -1)    # центральный, но круглый (не вытянут)
    rgb = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.where(A > 0)
    rgb[ys, xs] = np.random.RandomState(0).randint(0, 255, (len(xs), 3))  # текстура на животном
    masks = np.stack([B, C, D, A]).astype(np.uint8)
    chosen = pick_animal(masks, rgb)
    assert chosen is not None
    assert np.array_equal(chosen > 0, A > 0)


def test_returns_none_when_no_plausible_animal():
    H, W = 200, 160
    B = np.ones((H, W), np.uint8)
    C = _blank(H, W); cv2.circle(C, (10, 10), 3, 1, -1)
    rgb = np.zeros((H, W, 3), np.uint8)
    assert pick_animal(np.stack([B, C]).astype(np.uint8), rgb) is None


def test_accepts_single_synthetic_newt(synthetic_newt):
    chosen = pick_animal(synthetic_newt.mask.astype(np.uint8)[None], synthetic_newt.rgb)
    assert chosen is not None and np.array_equal(chosen > 0, synthetic_newt.mask)


def test_derive_flags_low_head_conf():
    assert derive_flags(0.1, 0.9, "") == ("low_head_conf",)


def test_derive_flags_band_suspect():
    assert derive_flags(0.5, 0.4, "band_suspect") == ("band_suspect",)


def test_derive_flags_both_and_clean():
    both = derive_flags(0.1, 0.4, "band_suspect")
    assert "low_head_conf" in both and "band_suspect" in both
    assert derive_flags(0.6, 0.9, "") == ()


def test_select_pilot_excludes_test_and_dups():
    df = pd.DataFrame({
        "md5": [f"{i:02d}" for i in range(6)],
        "cohort": ["TK"] * 6,
        "dup_keep": [True] * 5 + [False],
        "split_fold": ["train", "dev", "test", "train", "dev", "train"],
        "individual_id": ["TK-1"] * 6,
        "width": [10] * 6, "height": [10] * 6, "rel_path": ["x"] * 6,
    })
    out = select_pilot(df, per_species=10, cohorts=("TK",))
    assert set(out["split_fold"]) <= {"train", "dev"}   # test-LOCK не попал
    assert bool(out["dup_keep"].all())                  # дубли отброшены


def test_select_pilot_folds_train_only_excludes_dev():
    df = pd.DataFrame({
        "md5": [f"{i:02d}" for i in range(4)],
        "cohort": ["TK"] * 4, "dup_keep": [True] * 4,
        "split_fold": ["train", "dev", "train", "dev"],
        "individual_id": ["TK-1"] * 4, "width": [10] * 4, "height": [10] * 4, "rel_path": ["x"] * 4,
    })
    out = select_pilot(df, per_species=10, cohorts=("TK",), folds=("train",))
    assert set(out["split_fold"]) == {"train"}          # dev исключён → чист для A/B
