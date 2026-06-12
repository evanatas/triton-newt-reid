"""Тесты матчера созвездий (Блок 5, шаг 5.5) — гейты S2/S3/S4.

Своя особь (две сессии) → высокий score; чужая → низкий; зеркало штрафуется (другая особь);
инвариантность к повороту/масштабу/сдвигу; симметрия + детерминизм; устойчивость к лишним пятнам
(RANSAC: outliers). NN — слабый baseline (только same>diff). Чистая геометрия.
"""
from dataclasses import replace

import numpy as np

from triton_crop.config import SpotConfig


def _sim(theta_deg, scale, t, pts):
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (pts @ R.T) * scale + np.asarray(t, float)


def test_match_same_high_diff_low(synthetic_two_sessions):   # SpotConfig() → дефолт guided (не ransac)
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    s_same, pairs = match_constellations(d.pts_a, d.pts_b, SpotConfig())
    s_diff, _ = match_constellations(d.pts_a, d.pts_other, SpotConfig())
    assert s_same >= 0.8 and len(pairs) >= 6
    assert s_diff < s_same and s_diff <= 0.5


def test_match_rotation_scale_translation_invariant(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    b2 = _sim(123.0, 0.4, [-30.0, 15.0], d.pts_a)         # сильный поворот+масштаб+сдвиг
    s, _ = match_constellations(d.pts_a, b2, SpotConfig())
    assert s >= 0.8                                       # score инвариантен к similarity


def test_match_mirror_penalized(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    s_same, _ = match_constellations(d.pts_a, d.pts_b, SpotConfig())
    s_mir, _ = match_constellations(d.pts_a, d.pts_a_mirror, SpotConfig())
    assert s_mir < s_same                                 # зеркало = другая особь → ниже


def test_match_symmetric_and_deterministic(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    s1, _ = match_constellations(d.pts_a, d.pts_b, SpotConfig())
    s2, _ = match_constellations(d.pts_b, d.pts_a, SpotConfig())
    s1b, _ = match_constellations(d.pts_a, d.pts_b, SpotConfig())
    assert abs(s1 - s2) <= 0.15                           # симметрия (равные размеры)
    assert s1 == s1b                                      # детерминизм (seed)


def test_match_robust_to_extra_points(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    rng = np.random.RandomState(1)
    noisy = np.vstack([d.pts_b, rng.uniform(-3.0, 3.0, size=(5, 2))])     # +5 шумовых пятен
    s_noisy, _ = match_constellations(d.pts_a, noisy, SpotConfig())
    s_diff, _ = match_constellations(d.pts_a, d.pts_other, SpotConfig())
    assert s_noisy >= 0.5                                 # match выживает (max-норма штрафует, но не рушит)
    assert s_noisy > s_diff                               # своя+шум всё равно выше чужой


def test_score_not_degenerate_small_vs_large():
    # Дефект Блока 5 (визуальный QA): маленькое чужое созвездие (3 пятна) НЕ должно давать ~1.0 при матче
    # к большой пробе (20). score_norm=max + min_inliers это чинит.
    from triton_crop.constellation import match_constellations
    rng = np.random.RandomState(7)
    big = rng.uniform(-3.0, 3.0, size=(20, 2))            # большая чужая проба
    small = rng.uniform(-3.0, 3.0, size=(3, 2))           # крошечное чужое созвездие
    s, _ = match_constellations(big, small, SpotConfig())
    assert s < 0.5                                        # НЕ спурьёзная единица


def test_min_spots_guard(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    s, pairs = match_constellations(d.pts_a[:2], d.pts_b, SpotConfig())    # < min_spots_for_match=3
    assert s == 0.0 and pairs == []


def test_min_inliers_gate_returns_empty_pairs():
    # гейт min_inliers: score=0 ⇒ пары пустые (оверлей демо не должен рисовать «совпавшие» пятна при 0%)
    from triton_crop.constellation import match_constellations
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    cfg = replace(SpotConfig(), min_spots_for_match=3, min_inliers=10)     # совпадёт ≤4 пар < 10 → гейт
    s, pairs = match_constellations(pts, pts.copy(), cfg)
    assert s == 0.0 and pairs == []


def test_guided_beats_random_ransac_on_hard_case():
    # 5 истинных соответствий среди 20 (15 дистракторов с каждой стороны). Случайный 2-точечный RANSAC
    # при низких итерациях недосэмплирует верную гипотезу; guided (дескриптор-кандидаты) — находит.
    from triton_crop.constellation import match_constellations
    from triton_crop.config import SpotConfig
    rng = np.random.RandomState(0)
    sig = rng.uniform(-1, 1, size=(5, 2)) * np.array([1.0, 2.0])      # «подпись» из 5 пятен (кластер у центра)
    A = np.vstack([sig, rng.uniform(-4, 4, size=(15, 2))])           # +15 дистракторов
    B = np.vstack([_sim(31.0, 1.4, [7.0, -3.0], sig), rng.uniform(-4, 4, size=(15, 2))])
    B = B[rng.permutation(len(B))]
    cfg_g = replace(SpotConfig(), match_method="guided")
    cfg_r = replace(SpotConfig(), match_method="ransac", ransac_iters=120)
    s_g, pairs_g = match_constellations(A, B, cfg_g)
    s_r, _ = match_constellations(A, B, cfg_r)
    assert len(pairs_g) >= 5                          # guided ловит все 5 истинных соответствий
    assert s_g > s_r                                  # guided строго лучше random@120 (контроль iteration-starvation)


def test_guided_same_high_diff_low_mirror(synthetic_two_sessions):
    # дефолт-матчер = guided: своя особь высокий score, чужая ниже, зеркало штрафуется, детерминизм
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    cfg = SpotConfig()                                # дефолт = guided
    s_same, pairs = match_constellations(d.pts_a, d.pts_b, cfg)
    s_diff, _ = match_constellations(d.pts_a, d.pts_other, cfg)
    s_mir, _ = match_constellations(d.pts_a, d.pts_a_mirror, cfg)
    assert s_same >= 0.8 and len(pairs) >= 6
    assert s_diff < s_same and s_mir < s_same
    assert match_constellations(d.pts_a, d.pts_b, cfg)[0] == s_same   # детерминизм


def test_ransac_method_still_available(synthetic_two_sessions):
    # явный match_method="ransac" покрывает случайный путь (сохранён для sensitivity-сравнения)
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    cfg = replace(SpotConfig(), match_method="ransac")
    s_same, _ = match_constellations(d.pts_a, d.pts_b, cfg)
    s_diff, _ = match_constellations(d.pts_a, d.pts_other, cfg)
    assert s_same >= 0.8 and s_diff < s_same


def test_nn_baseline_same_higher_than_diff(synthetic_two_sessions):
    from triton_crop.constellation import match_constellations
    d = synthetic_two_sessions
    cfg = replace(SpotConfig(), match_method="nn")
    s_same, _ = match_constellations(d.pts_a, d.pts_b, cfg)
    s_diff, _ = match_constellations(d.pts_a, d.pts_other, cfg)
    assert s_same > s_diff


def test_build_sim_matrix_diagonal_dominant(synthetic_two_sessions):
    from triton_crop.constellation import build_match_sim_matrix
    d = synthetic_two_sessions
    # probe = [a], gallery = [b(своя), other(чужая)] → своя должна получить выше
    sim = build_match_sim_matrix([d.pts_a], [d.pts_b, d.pts_other], SpotConfig())
    assert sim.shape == (1, 2)
    assert sim[0, 0] > sim[0, 1]
