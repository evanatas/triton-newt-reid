"""Тесты embed_augment (Блок 4): корректные для объекта аугментации, детерминируемые, БЕЗ зеркала.

Ключевое требование заказчика: НЕ зеркалить (зеркальный тритон = ДРУГАЯ особь). Проверяем (1) в пайплайне
нет flip-трансформов; (2) асимметрия лево/право не переворачивается; (3) узор (пятна) переживает геометрию;
(4) детерминизм seed=42. albumentations импортируется внутри модуля (lazy).
"""
import numpy as np


def test_augment_pipeline_has_no_mirror():
    from triton_crop.embed_augment import build_train_augment
    names = {type(t).__name__ for t in build_train_augment().transforms}
    assert names.isdisjoint({"HorizontalFlip", "VerticalFlip", "Flip", "Transpose", "RandomRotate90"})


def test_augment_deterministic_and_changes():
    from triton_crop.embed_augment import augment_image
    img = np.random.default_rng(0).integers(0, 255, (48, 48, 3), np.uint8)
    a = augment_image(img, seed=42)
    b = augment_image(img, seed=42)
    c = augment_image(img, seed=7)
    assert np.array_equal(a, b)                       # один seed → идентичный результат
    assert a.shape == img.shape and a.dtype == np.uint8
    assert not np.array_equal(a, img)                 # реально аугментирует
    assert not np.array_equal(a, c)                   # другой seed → другой результат


def test_augment_does_not_mirror_left_right():
    # яркое пятно строго в ЛЕВОЙ половине → после умеренной геометрии центроид остаётся слева (не зеркалит)
    from triton_crop.embed_augment import augment_image
    H, W = 64, 64
    img = np.zeros((H, W, 3), np.uint8)
    img[24:34, 8:18] = (240, 30, 30)
    for seed in range(8):
        out = augment_image(img, seed=seed, p_color=0.0, p_elastic=0.0, p_noise=0.0,
                            p_affine=1.0, rotate=8, translate=0.04, scale=(0.95, 1.05))
        xs = np.where(out[..., 0] > 150)[1]
        if xs.size:
            assert xs.mean() < W * 0.5 + 4            # центроид не перепрыгнул на правую половину


def test_augment_preserves_pattern(synthetic_curved_newt):
    # синие пятна (узор) переживают умеренную аугментацию — не стираются геометрией
    from triton_crop.embed_augment import augment_image
    rgb = synthetic_curved_newt.rgb
    before = int((rgb[..., 2] > 150).sum())
    out = augment_image(rgb, seed=3, p_color=0.0, p_elastic=0.0, p_noise=0.0,
                        p_affine=1.0, scale=(0.95, 1.05), rotate=5, translate=0.02)
    after = int((out[..., 2] > 150).sum())
    assert after >= before * 0.6 and before > 0       # узор сохранён (а не уничтожен)
