"""Тесты S-гейтов Блока 5 (шаг 5.9): run_spot_gates.

S6 — детекция только на belly_oriented/unroll_debend (НЕ ribbon, хард); S7 — анти-утечка запечатанного
test (хард, как C9); S1 — покрытие детектора (мягкий warning); S5 — A/B записан и ПЕРЕСЧИТЫВАЕТСЯ по
артефакту (хард, как C8 Блока 3 — защита от рассинхрона/ручной правки decision).
"""
import pandas as pd
import pytest

from triton_crop.validate import ValidationError, run_spot_gates


def _spots(variant="belly_oriented", n=5):
    return pd.DataFrame({"md5": [f"m{i}" for i in range(4)], "detect_variant": [variant] * 4,
                         "n_spots": [n, n, n, n]})


def test_s6_rejects_ribbon_surface():
    with pytest.raises(ValidationError):
        run_spot_gates(_spots("unroll_ribbon"))


def test_s6_passes_allowed_surface():
    assert isinstance(run_spot_gates(_spots("belly_oriented")), list)
    assert isinstance(run_spot_gates(_spots("unroll_debend")), list)


def test_s7_rejects_test_leak():
    spots = _spots("belly_oriented")
    man = pd.DataFrame({"md5": ["m1"], "split_fold": ["test"], "kpi_scope": ["kpi_core"]})
    with pytest.raises(ValidationError):
        run_spot_gates(spots, manifest_df=man)
    assert isinstance(run_spot_gates(spots, manifest_df=man, unsealed=True), list)  # финальный режим: S7 не бьёт


def test_s1_coverage_warning():
    spots = _spots("belly_oriented", n=1)                 # < min_spots=3 у всех
    w = run_spot_gates(spots, min_spots=3, coverage_min=0.5)
    assert any("S1" in x for x in w)                      # мягкий warning, не падение


def _cmp(r1m=0.5):
    return {"embedder": {"overall": {"recall@1": 0.2, "recall@5": 0.4}},
            "matcher": {"overall": {"recall@1": r1m, "recall@5": 0.6}},
            "stats_matcher_vs_embedder@1": {"mcnemar_p": 0.01, "mcnemar_c": 10, "mcnemar_b": 2},
            "stats_matcher_vs_embedder@5": {"mcnemar_p": 0.2, "mcnemar_c": 5, "mcnemar_b": 3}}


def test_s5_ab_recompute_ok():
    ab = {"matcher_adopt_decision": "matcher", "compare": _cmp(),
          "matcher_adopt_rationale": {"primary_k": 1}, "recall_ks": [1, 5]}
    assert isinstance(run_spot_gates(_spots(), ab_metrics=ab), list)   # decision == пересчёт → ок


def test_s5_ab_recompute_mismatch_raises():
    ab = {"matcher_adopt_decision": "embedder", "compare": _cmp(),    # записано НЕ то, что даёт пересчёт
          "matcher_adopt_rationale": {"primary_k": 1}, "recall_ks": [1, 5]}
    with pytest.raises(ValidationError):
        run_spot_gates(_spots(), ab_metrics=ab)
