"""Тесты продуктового пайплайна canonical_belly_crop (со стабами seg/pose). TDD.
Проверяем: ok-путь, fallback-лестницу, детерминизм, заливку фона."""
import numpy as np

from triton_crop.config import CropConfig
from triton_crop.pipeline import canonical_belly_crop


class _Seg:
    def __init__(self, mask, conf):
        self.mask, self.conf = mask, conf

    def predict(self, img):
        return self.mask, self.conf


class _Pose:
    def __init__(self, head, cloaca, conf):
        self.head, self.cloaca, self.conf = head, cloaca, conf

    def predict(self, img):
        return self.head, self.cloaca, self.conf


def _img_mask():
    H, W = 200, 160
    img = np.full((H, W, 3), 50, np.uint8)        # серый фон
    mask = np.zeros((H, W), bool)
    mask[40:160, 60:100] = True
    img[mask] = (150, 120, 80)                    # «брюшко»
    return img, mask


def test_belly_oriented_ok():
    img, mask = _img_mask()
    r = canonical_belly_crop(img, _Seg(mask, 0.9), _Pose((80, 40), (80, 150), 0.9),
                             CropConfig(canon_size=128))
    assert r.variant == "belly_oriented" and r.crop_status == "ok"
    assert r.image.shape == (128, 128, 3)
    assert r.unroll_rect is not None and r.head_xy is not None


def test_empty_mask_full_fallback():
    img, _ = _img_mask()
    r = canonical_belly_crop(img, _Seg(None, 0.0), _Pose((1, 1), (1, 2), 0.9),
                             CropConfig(canon_size=128))
    assert r.variant == "full" and r.crop_status == "empty_mask"
    assert r.image.shape == (128, 128, 3) and r.mask is None


def test_low_seg_conf_full_fallback():
    img, mask = _img_mask()
    r = canonical_belly_crop(img, _Seg(mask, 0.1), _Pose((1, 1), (1, 2), 0.9),
                             CropConfig(canon_size=128))
    assert r.crop_status == "low_seg_conf" and r.variant == "full"


def test_no_pose_belly_mask_no_rotation():
    img, mask = _img_mask()
    r = canonical_belly_crop(img, _Seg(mask, 0.9), _Pose(None, None, 0.0),
                             CropConfig(canon_size=128))
    assert r.variant == "belly_mask" and r.crop_status == "no_pose"
    assert r.orientation_deg == 0.0


def test_determinism_same_input_same_output():
    img, mask = _img_mask()
    seg, pose = _Seg(mask, 0.9), _Pose((80, 40), (80, 150), 0.9)
    r1 = canonical_belly_crop(img, seg, pose, CropConfig(canon_size=96))
    r2 = canonical_belly_crop(img, seg, pose, CropConfig(canon_size=96))
    assert np.array_equal(r1.image, r2.image) and r1.orientation_deg == r2.orientation_deg


def test_background_black_fills_more_zeros_than_none():
    img, mask = _img_mask()
    seg, pose = _Seg(mask, 0.9), _Pose((80, 40), (80, 150), 0.9)
    rb = canonical_belly_crop(img, seg, pose, CropConfig(canon_size=128, mask_background="black"))
    rn = canonical_belly_crop(img, seg, pose, CropConfig(canon_size=128, mask_background="none"))
    assert (rb.image > 0).sum() < (rn.image > 0).sum()   # чёрная заливка фона убирает серый


def test_belly_oriented_produces_unrolled():
    img, mask = _img_mask()
    r = canonical_belly_crop(img, _Seg(mask, 0.9), _Pose((80, 40), (80, 150), 0.9),
                             CropConfig(canon_size=128))
    assert r.variant == "belly_oriented"
    assert r.unroll_variants is not None and "debend" in r.unroll_variants
    assert r.unroll_status["debend"] == "ok"
    assert r.unroll_variants["debend"].shape == (128, 128, 3)   # тот же канон (единый контракт выхода)


def test_fallback_variant_has_no_unrolled():
    img, _ = _img_mask()
    r = canonical_belly_crop(img, _Seg(None, 0.0), _Pose((1, 1), (1, 2), 0.9),
                             CropConfig(canon_size=128))
    assert r.variant == "full" and r.unroll_variants is None


def test_clipped_after_rotation_flagged():
    # Горизонтальное вытянутое брюшко, центроид у верхнего края → поворот на 90° в холст (H,W)
    # уводит дальний конец за рамку (BORDER_CONSTANT) → потеря площади маски помечается.
    H, W = 200, 160
    img = np.full((H, W, 3), 50, np.uint8)
    mask = np.zeros((H, W), bool)
    mask[5:15, 25:140] = True              # длинная горизонтальная полоса у верхнего края
    img[mask] = (150, 120, 80)
    r = canonical_belly_crop(img, _Seg(mask, 0.9), _Pose((140, 10), (25, 10), 0.9),
                             CropConfig(canon_size=128))
    assert r.variant == "belly_oriented"
    assert r.crop_status == "clipped_after_rotation"
