"""Тесты статистики A/B: Wilson CI, парный McNemar, парный bootstrap."""
from triton_crop.stats import mcnemar, paired_bootstrap_recall_diff, wilson_ci


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi and 0.0 <= lo < hi <= 1.0


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_clamps_invalid_hits_no_nan():
    import math
    lo, hi = wilson_ci(5, 3)                  # hits>n → клип к n (p=1), без NaN под корнем
    assert not (math.isnan(lo) or math.isnan(hi))
    assert 0.0 <= lo <= hi <= 1.0 and hi == 1.0


def test_mcnemar_detects_crop_better():
    raw = [0] * 10
    crop = [1] * 8 + [0] * 2          # crop попал там, где raw нет
    b, c, p = mcnemar(raw, crop)
    assert b == 0 and c == 8 and p < 0.05


def test_mcnemar_large_n_normal_approx_no_overflow():
    import math
    raw = [1] * 700 + [0] * 500          # b=700 (raw попал, crop нет)
    crop = [0] * 700 + [1] * 500         # c=500 (crop попал, raw нет); n=1200>1000 → нормальное приближение
    b, c, p = mcnemar(raw, crop)
    assert b == 700 and c == 500
    assert not math.isnan(p) and 0.0 <= p <= 1.0 and p < 0.05   # значимо, без OverflowError


def test_mcnemar_symmetric_not_significant():
    raw = [1, 1, 0, 0]
    crop = [0, 0, 1, 1]              # симметрично → не значимо
    _, _, p = mcnemar(raw, crop)
    assert p > 0.05


def test_paired_bootstrap_positive_ci():
    raw = [0] * 20
    crop = [1] * 10 + [0] * 10        # diff = +0.5
    d, lo, hi = paired_bootstrap_recall_diff(raw, crop, n_boot=500)
    assert abs(d - 0.5) < 1e-9 and lo > 0.0 and hi >= d
