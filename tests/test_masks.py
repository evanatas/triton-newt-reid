"""Тесты реконструкции foreground из кропа (Блок 5).

Кроп пайплайна зануляет фон в чёрный (letterbox даёт чёрную рамку). Прежний костыль `rgb.sum(2)>30`
выкидывал тёмные пятна с суммой каналов <=30 как «фон». foreground_from_crop восстанавливает маску
flood-fill'ом фона ОТ РАМКИ → внутренние тёмные пятна (не связаны с рамкой) остаются в маске.
"""
import numpy as np


def test_foreground_keeps_interior_dark_spot():
    from triton_crop.masks import foreground_from_crop
    rgb = np.zeros((40, 40, 3), np.uint8)          # чёрный фон (letterbox-рамка)
    rgb[8:32, 8:32] = (180, 150, 40)               # тело пуза (яркое)
    rgb[18:22, 18:22] = (5, 5, 5)                  # ТЁМНОЕ пятно ВНУТРИ тела (sum=15 <= 30)
    fg = foreground_from_crop(rgb, bg_sum_thr=30)
    assert fg[20, 20]                              # внутреннее тёмное пятно ОСТАЁТСЯ в маске
    assert fg[20, 9]                               # тело — в маске
    assert not fg[1, 1]                            # угловой фон — НЕ в маске
    assert fg.dtype == bool and fg.shape == (40, 40)


def test_foreground_all_body_when_no_dark_border():
    # если рамка не чёрная (нет фона) → всё foreground (безопасно, не теряем пятна)
    from triton_crop.masks import foreground_from_crop
    rgb = np.full((20, 20, 3), 100, np.uint8)      # всё яркое
    fg = foreground_from_crop(rgb, bg_sum_thr=30)
    assert fg.all()
