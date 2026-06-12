"""Статистика для A/B (ЧИСТО, numpy): Wilson CI доли, парный McNemar, парный bootstrap по probe.

Нужно, чтобы заявление «кроп ≥ raw» держалось не на агрегатах, а на ПАРНОМ сравнении одних и тех же
проб (raw_hit vs crop_hit по каждому probe) с доверительными интервалами.
"""
from math import comb

import numpy as np


def wilson_ci(hits: int, n: int, z: float = 1.96):
    """Доверительный интервал Wilson для доли hits/n (по умолчанию 95%)."""
    if n == 0:
        return (0.0, 0.0)
    hits = max(0, min(int(hits), n))            # защита: hits>n / hits<0 → клип (иначе NaN под корнем)
    p = hits / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(raw_hits, crop_hits):
    """Парный McNemar по ОДНИМ И ТЕМ ЖЕ probe. -> (b, c, p_value).

    b = raw попал, crop нет; c = crop попал, raw нет. p — точный двусторонний биномиальный (p=0.5).
    """
    raw = np.asarray(raw_hits, bool)
    crop = np.asarray(crop_hits, bool)
    b = int((raw & ~crop).sum())
    c = int((~raw & crop).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n > 1000:                                   # большой n: точный 2**n/comb переполняется → нормальное приближение
        import math
        z = (abs(b - c) - 1.0) / math.sqrt(n)      # McNemar с поправкой непрерывности
        return b, c, min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))   # двусторонний p
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def paired_bootstrap_recall_diff(raw_hits, crop_hits, n_boot: int = 2000, seed: int = 42):
    """Бутстрап разницы recall (crop - raw) по ПАРАМ probe. -> (diff, lo95, hi95)."""
    raw = np.asarray(raw_hits, float)
    crop = np.asarray(crop_hits, float)
    n = len(raw)
    diff = float(crop.mean() - raw.mean()) if n else 0.0
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    diffs = [crop[idx].mean() - raw[idx].mean()
             for idx in (rng.randint(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diff, float(lo), float(hi)
