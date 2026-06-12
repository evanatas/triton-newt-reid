"""Тесты геометрии созвездия (Блок 5, шаг 5.4) — гейт S2 (similarity, БЕЗ зеркала).

estimate_similarity_no_reflection восстанавливает поворот+масштаб; на зеркале даёт БОЛЬШОЙ остаток
(proper similarity не умеет отражать → зеркальный тритон не сматчится); normalize_constellation
инвариантна к сдвигу/масштабу. Чистая геометрия на synthetic_two_sessions.
"""
import numpy as np


def test_estimate_recovers_similarity(synthetic_two_sessions):
    from triton_crop.constellation import apply_similarity, estimate_similarity_no_reflection
    d = synthetic_two_sessions
    M = estimate_similarity_no_reflection(d.pts_a, d.pts_b)
    assert M is not None
    assert np.allclose(apply_similarity(M, d.pts_a), d.pts_b, atol=1e-6)   # точно восстановил


def test_estimate_mirror_high_residual(synthetic_two_sessions):
    from triton_crop.constellation import apply_similarity, estimate_similarity_no_reflection
    d = synthetic_two_sessions
    M = estimate_similarity_no_reflection(d.pts_a, d.pts_a_mirror)         # proper similarity ≠ отражение
    resid = np.linalg.norm(apply_similarity(M, d.pts_a) - d.pts_a_mirror, axis=1).mean()
    scale = np.sqrt(((d.pts_a - d.pts_a.mean(0)) ** 2).sum(1).mean())
    assert resid > 0.3 * scale                                            # зеркало НЕ воспроизводится


def test_estimate_degenerate_returns_none():
    from triton_crop.constellation import estimate_similarity_no_reflection
    pts = np.zeros((3, 2))
    assert estimate_similarity_no_reflection(pts, pts) is None            # совпадающие точки
    assert estimate_similarity_no_reflection(pts[:1], pts[:1]) is None    # < 2 точек


def test_normalize_translation_scale_invariant(synthetic_two_sessions):
    from triton_crop.constellation import normalize_constellation
    d = synthetic_two_sessions
    na = normalize_constellation(d.pts_a)
    assert np.allclose(na.mean(0), 0.0, atol=1e-9)                        # центрирован
    assert abs(np.sqrt((na ** 2).sum(1).mean()) - 1.0) < 1e-6            # RMS-радиус = 1
    nb = normalize_constellation(d.pts_a * 3.0 + np.array([10.0, -7.0]))  # сдвиг+масштаб
    assert np.allclose(np.sort(np.linalg.norm(na, axis=1)),
                       np.sort(np.linalg.norm(nb, axis=1)), atol=1e-6)    # инвариант


def test_local_descriptor_similarity_invariant_not_mirror(synthetic_two_sessions):
    # дескриптор пятна (по knn соседям) инвариантен к similarity (масштаб+поворот), но НЕ к зеркалу
    from triton_crop.constellation import _local_descriptor
    d = synthetic_two_sessions
    da = _local_descriptor(d.pts_a, knn=5)
    db = _local_descriptor(d.pts_b, knn=5)          # pts_b = similarity(pts_a), порядок точек тот же
    dm = _local_descriptor(d.pts_a_mirror, knn=5)   # зеркало (хиральность нарушена)
    assert da.shape == (d.n, 3 * 5)
    assert np.allclose(da, db, atol=1e-6)           # similarity-инвариант: дескриптор той же точки совпадает
    assert not np.allclose(da, dm, atol=1e-3)       # зеркало меняет знак угловой части → отличается


def test_best_hypothesis_matches_reference():
    # 2-точечная similarity в _best_hypothesis (инлайн через z=(d2-d1)/(s2-s1)) эквивалентна
    # референсу estimate_similarity_no_reflection (для 2 точек least-squares = точное решение)
    from triton_crop.constellation import estimate_similarity_no_reflection
    rng = np.random.RandomState(42)
    for _ in range(20):
        src = rng.uniform(-10.0, 10.0, (2, 2))
        dst = rng.uniform(-10.0, 10.0, (2, 2))
        M = estimate_similarity_no_reflection(src, dst)            # [[a,-b,tx],[b,a,ty]]
        assert M is not None
        s1, s2 = src[0], src[1]
        d1, d2 = dst[0], dst[1]
        cs = (s2[0] - s1[0]) + 1j * (s2[1] - s1[1])                # формулы _best_hypothesis
        cd = (d2[0] - d1[0]) + 1j * (d2[1] - d1[1])
        z = cd / cs
        a, b = z.real, z.imag
        tx = d1[0] - (a * s1[0] - b * s1[1])
        ty = d1[1] - (b * s1[0] + a * s1[1])
        assert np.allclose([a, b, tx, ty], [M[0, 0], M[1, 0], M[0, 2], M[1, 2]], atol=1e-9)


def test_candidate_correspondences_link_true_matches(synthetic_two_sessions):
    # кандидаты по дескриптору должны предлагать истинные пары i->i (pts_b = similarity(pts_a), тот же порядок)
    from triton_crop.constellation import _candidate_correspondences
    from triton_crop.config import SpotConfig
    d = synthetic_two_sessions
    cands = _candidate_correspondences(d.pts_a, d.pts_b, SpotConfig())
    pairs = set(cands)
    hit = sum((i, i) in pairs for i in range(d.n))
    assert hit >= d.n - 1                            # почти все истинные пары среди кандидатов
    assert all(0 <= i < d.n and 0 <= j < d.n for i, j in cands)
