"""Тесты эвристики псевдо-меток (голова/клоака + брюшная полоса). TDD на synthetic_newt."""
from triton_crop.masks import polygon_to_mask
from triton_crop.pseudo_label import derive_pseudo_label


def test_head_above_cloaca(synthetic_newt):
    pl = derive_pseudo_label(synthetic_newt.mask)
    assert pl.head_xy[1] < pl.cloaca_xy[1]   # голова выше клоаки
    assert pl.head_xy[1] < 80                # голова в верхней части (GT y=20)
    assert pl.cloaca_xy[1] > 120             # клоака в нижней части (GT ~162)


def test_belly_polygon_excludes_legs_and_extremes(synthetic_newt):
    pl = derive_pseudo_label(synthetic_newt.mask)
    band = polygon_to_mask(pl.belly_polygon, *synthetic_newt.mask.shape)
    assert band[110, 65]                          # центр торса — внутри
    assert not band[20, 65] and not band[203, 65]  # кончик головы и хвоста — снаружи
    assert not band[100, 43] and not band[100, 87]  # боковые «лапы» — снаружи


def test_head_arbiter_resolves_180(synthetic_newt):
    # перевёрнутая маска: широкая «голова» теперь снизу → арбитр (шире=голова) находит её там
    flipped = synthetic_newt.mask[::-1, :].copy()
    pl = derive_pseudo_label(flipped)
    assert pl.head_xy[1] > pl.cloaca_xy[1]
