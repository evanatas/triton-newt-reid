"""Тесты гейтов Блока 2 (run_crop_gates): позитив + негатив на каждый. TDD."""
import pandas as pd
import pytest

from triton_crop.validate import ValidationError, run_crop_gates, write_crop_eda


def _crops():
    return pd.DataFrame([
        {"md5": "a", "variant": "belly_oriented", "crop_status": "ok", "canon_size": 384,
         "mirrored": False, "pipeline_version": "v1", "kpi_scope": "kpi_core", "split_role": "gallery"},
        {"md5": "b", "variant": "belly_oriented", "crop_status": "ok", "canon_size": 384,
         "mirrored": False, "pipeline_version": "v1", "kpi_scope": "kpi_core", "split_role": "probe"},
    ])


def _manifest():
    return pd.DataFrame([
        {"md5": "a", "dup_keep": True, "kpi_scope": "kpi_core", "split_role": "gallery"},
        {"md5": "b", "dup_keep": True, "kpi_scope": "kpi_core", "split_role": "probe"},
    ])


def test_gates_pass_on_clean():
    assert isinstance(run_crop_gates(_crops(), _manifest()), list)


def test_c1_bites_on_missing_coverage():
    man = _manifest()
    man.loc[len(man)] = {"md5": "c", "dup_keep": True, "kpi_scope": "kpi_core", "split_role": "probe"}
    with pytest.raises(ValidationError):
        run_crop_gates(_crops(), man)   # кропа для 'c' нет


def test_c2_bites_on_orphan_crop():
    c = _crops()
    c.loc[len(c)] = {"md5": "X", "variant": "belly_oriented", "crop_status": "ok", "canon_size": 384,
                     "mirrored": False, "pipeline_version": "v1", "kpi_scope": "kpi_core", "split_role": "gallery"}
    with pytest.raises(ValidationError):
        run_crop_gates(c, _manifest())  # 'X' нет в манифесте


def test_c3_bites_on_mirror():
    c = _crops()
    c.loc[0, "mirrored"] = True
    with pytest.raises(ValidationError):
        run_crop_gates(c, _manifest())


def test_c4_bites_on_mixed_canon_size():
    c = _crops()
    c.loc[0, "canon_size"] = 256
    with pytest.raises(ValidationError):
        run_crop_gates(c, _manifest())


def test_c7_bites_on_fallback_budget():
    c = _crops()
    c.loc[0, "crop_status"] = "full_fallback"
    c.loc[1, "crop_status"] = "empty_mask"
    with pytest.raises(ValidationError):
        run_crop_gates(c, _manifest(), fallback_budget=0.1)


def test_c9_bites_on_test_leak():
    man = _manifest()
    man["split_fold"] = ["train", "dev"]
    man.loc[len(man)] = {"md5": "t", "dup_keep": True, "kpi_scope": "kpi_core",
                         "split_role": "probe", "split_fold": "test"}
    c = _crops()
    c.loc[len(c)] = {"md5": "t", "variant": "belly_oriented", "crop_status": "ok", "canon_size": 384,
                     "mirrored": False, "pipeline_version": "v1", "kpi_scope": "kpi_core", "split_role": "probe"}
    with pytest.raises(ValidationError):
        run_crop_gates(c, man)   # кроп для запечатанного test 't' — утечка
    assert isinstance(run_crop_gates(c, man, unsealed=True), list)   # финальный режим: test законен → C9 не бьёт


def _crops_with_unroll():
    c = _crops()
    for md5 in ("a", "b"):
        c.loc[len(c)] = {"md5": md5, "variant": "unroll_debend", "crop_status": "ok", "canon_size": 384,
                         "mirrored": False, "pipeline_version": "v1", "kpi_scope": "kpi_core",
                         "split_role": "gallery"}
    return c


def test_c4b_bites_on_unroll_canon_mismatch():
    c = _crops_with_unroll()
    c.loc[c.variant == "unroll_debend", "canon_size"] = 256        # unroll-канон ≠ belly_oriented
    with pytest.raises(ValidationError):
        run_crop_gates(c, _manifest(), ab_metrics={"unroll_debend": {}, "unroll_adopt_decision": "x"})


def test_c7_excludes_unroll_rows():
    c = _crops_with_unroll()
    c.loc[c.variant == "belly_oriented", "crop_status"] = ["full_fallback", "empty_mask"]
    with pytest.raises(ValidationError):                            # 2/2 belly_oriented fallback > бюджет;
        run_crop_gates(c, _manifest(), fallback_budget=0.1,         # ok-строки unroll НЕ разбавляют
                       ab_metrics={"unroll_debend": {}, "unroll_adopt_decision": "x"})


def test_c8_bites_on_missing_ab_variant():
    with pytest.raises(ValidationError):
        run_crop_gates(_crops_with_unroll(), _manifest(),
                       ab_metrics={"raw": {}, "unroll_adopt_decision": "x"})   # нет unroll_debend


def test_c8_hard_fails_without_ab_metrics():
    with pytest.raises(ValidationError):                            # unroll есть, но A/B нет → блокер (Rosa)
        run_crop_gates(_crops_with_unroll(), _manifest())


def test_c8_bites_on_bad_decision():
    with pytest.raises(ValidationError):                            # decision не из belly_oriented ∪ unroll_*
        run_crop_gates(_crops_with_unroll(), _manifest(),
                       ab_metrics={"unroll_debend": {}, "unroll_adopt_decision": "unroll_ghost"})


def test_c8_passes_with_valid_decision():
    out = run_crop_gates(_crops_with_unroll(), _manifest(),
                         ab_metrics={"unroll_debend": {}, "unroll_adopt_decision": "belly_oriented"})
    assert isinstance(out, list)


def _ab_metrics_full(decision):
    return {
        "belly_oriented": {"overall": {"recall@1": 0.5, "recall@5": 0.8}},
        "unroll_debend": {"overall": {"recall@1": 0.55, "recall@5": 0.86}},
        "stats_unroll_debend_vs_bo@1": {"mcnemar_p": 1.0},
        "stats_unroll_debend_vs_bo@5": {"mcnemar_p": 0.01},
        "unroll_adopt_decision": decision,
        "unroll_adopt_rationale": {"alpha": 0.05, "primary_k": 5, "allow_k1_significance": False,
                                   "baseline": "belly_oriented",
                                   "considered": [{"variant": "unroll_debend", "pattern_safe": True,
                                                   "eligible": True}]},
    }


def test_c8_recompute_passes_consistent():
    out = run_crop_gates(_crops_with_unroll(), _manifest(), ab_metrics=_ab_metrics_full("unroll_debend"))
    assert isinstance(out, list)


def test_c8_recompute_catches_tampered_decision():
    with pytest.raises(ValidationError):                  # decision вручную на belly_oriented, хотя debend eligible
        run_crop_gates(_crops_with_unroll(), _manifest(), ab_metrics=_ab_metrics_full("belly_oriented"))


def test_write_crop_eda(tmp_path):
    write_crop_eda(_crops(), tmp_path)
    assert (tmp_path / "crop_eda.md").exists()
