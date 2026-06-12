"""Тесты sealed_test.build_sealed_report (финальные числа ВКР) — чистая логика на синтетике (без моделей/IO)."""
import math

import numpy as np
import pytest

from triton_crop.sealed_test import SEALED_STAGES, assert_unsealed, build_sealed_report


def test_assert_unsealed_single_source_gate():
    # единый гейт C9: запечатанные test/open_test требуют явного unseal; остальное проходит
    assert set(SEALED_STAGES) == {"test", "open_test"}
    assert_unsealed(["dev", "gallery"], False)       # не запечатанные стадии → ок (без unseal)
    assert_unsealed(["test", "open_test"], True)     # явное вскрытие → ок
    for stages in (["test"], ["dev", "open_test"], ("open_test",)):
        with pytest.raises(ValueError, match="C9|запечат|unseal"):
            assert_unsealed(stages, False)


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _setup():
    # 2 особи в галерее: A≈e0, B≈e1 (по 2 фото); тест: проба A→e0, проба B→e1; open: новая особь ≈ e3 (далеко).
    gallery_emb = _unit([[1, 0, 0, 0], [0.97, 0.24, 0, 0], [0, 1, 0, 0], [0, 0.97, 0.24, 0]])
    gallery_ids = np.array(["A", "A", "B", "B"])
    test_emb = _unit([[0.99, 0.10, 0, 0], [0.05, 0.99, 0, 0]])
    test_ids = np.array(["A", "B"])
    test_cohorts = np.array(["TK", "PW"])
    open_emb = _unit([[0, 0, 0, 1]])           # ортогональна плоскости галереи → низкий max_sim (новая особь)
    return gallery_emb, gallery_ids, test_emb, test_ids, test_cohorts, open_emb


def test_recall_perfect_when_separable():
    g, gid, te, ti, tc, oe = _setup()
    rec = build_sealed_report(te, ti, tc, oe, g, gid, ks=(1, 5))["recall"]
    assert rec["overall"]["recall@1"] == 1.0
    assert rec["overall"]["recall@5"] == 1.0
    assert "TK" in rec["per_cohort"] and "PW" in rec["per_cohort"]      # temporal (TK) + PW срезы
    assert rec["per_cohort"]["TK"]["recall@1"] == 1.0
    assert rec["coverage"]["n_true_id_in_gallery"] == 2
    assert rec["coverage"]["n_excluded_no_true_id"] == 0


def test_openset_auroc_and_policy():
    g, gid, te, ti, tc, oe = _setup()
    o = build_sealed_report(te, ti, tc, oe, g, gid, ks=(1, 5), openset_threshold=0.5)["openset"]
    assert o["n_known"] == 2 and o["n_new"] == 1
    assert o["auroc"] == 1.0                       # known (высокий max_sim) vs new (низкий) → идеальное разделение
    assert o["known_kept_rate"] == 1.0             # все known выше порога 0.5
    assert o["new_falsely_known_rate"] == 0.0      # новая особь ниже порога → не помечена known


def test_openset_empty_open_is_nan():
    g, gid, te, ti, tc, _ = _setup()
    o = build_sealed_report(te, ti, tc, None, g, gid, ks=(1, 5))["openset"]
    assert o["n_new"] == 0
    assert math.isnan(o["auroc"])


def test_coverage_excludes_probe_without_gallery_identity():
    g, gid, te, ti, tc, oe = _setup()
    ti2 = np.array(["A", "Z"])                      # Z нет в галерее → исключается из closed-set reid-recall
    rec = build_sealed_report(te, ti2, tc, oe, g, gid, ks=(1, 5))["recall"]
    assert rec["coverage"]["n_true_id_in_gallery"] == 1
    assert rec["coverage"]["n_excluded_no_true_id"] == 1
    assert rec["closedset_reid_recall"][1] == 1.0  # единственная покрытая проба (A) опознана
