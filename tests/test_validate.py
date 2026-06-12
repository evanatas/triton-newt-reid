"""Тесты гейтов: позитив (чистый манифест) + НЕГАТИВ на каждый гейт (доказать, что падает)."""
import pandas as pd
import pytest

from triton_data.manifest import build_manifest
from triton_data.validate import ValidationError, _anomaly_log, run_gates, write_eda


def _m(ws):
    return build_manifest(ws)


def test_gates_pass_on_clean_manifest(synthetic_workspace):
    target, external = _m(synthetic_workspace)
    warnings = run_gates(target, external, synthetic_workspace)
    assert isinstance(warnings, list)  # без исключений (предупреждения о разрешении допустимы)


# ─────────── НЕГАТИВНЫЕ: каждый гейт должен «укусить» ───────────

def test_g1_bites_on_gallery_probe_md5_leak(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    probe_md5 = t[t.split_role == "probe"]["md5"].iloc[0]
    t.loc[t[t.split_role == "gallery"].index[0], "md5"] = probe_md5
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g2_bites_when_probe_individual_has_no_gallery(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    ind = t[(t.split_role == "probe") & (~t.is_new_open)]["individual_id"].iloc[0]
    gal = (t.individual_id == ind) & (t.split_role == "gallery")
    t.loc[gal, "individual_id"] = ind.split("-")[0] + "-999"  # увели галерею под другой id
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g3_bites_when_external_has_nonexternal(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    e.loc[e.index[0], "role"] = "target"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g4_bites_on_two_survivors(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[(t.dup_group >= 0) & (~t.dup_keep)].index[0], "dup_keep"] = True
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g5_bites_on_keep_without_scheme(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[t.dup_keep & t.split_scheme.notna()].index[0], "split_scheme"] = None
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g6_bites_on_probe_in_train(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[t.split_role == "probe"].index[0], "split_fold"] = "train"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g7_bites_on_absolute_path(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t.index[0], "rel_path"] = "/etc/passwd"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g8_bites_on_prefix_cohort_mismatch(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    ind = t[t.cohort == "TK"]["individual_id"].iloc[0]
    t.loc[t.individual_id == ind, "individual_id"] = "ZZ-001"  # префикс ≠ когорта
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g9_bites_on_invalid_month(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[t.date.notna()].index[0], "date"] = "2025-13"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g9_bites_on_impossible_calendar_date(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[(t.cohort == "LAB") & t.date.notna()].index[0], "date"] = "2025-02-31"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g11_bites_on_duplicate_md5_among_keep(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    gal = t[t.split_role == "gallery"].index[:2]
    t.loc[gal[1], "md5"] = t.loc[gal[0], "md5"]
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


def test_g12_bites_on_wrong_kpi_scope(synthetic_workspace):
    t, e = _m(synthetic_workspace)
    t.loc[t[t.cohort == "TK"].index[0], "kpi_scope"] = "external"
    with pytest.raises(ValidationError):
        run_gates(t, e, synthetic_workspace)


# ─────────── EDA ───────────

def test_anomaly_log_excludes_empty_and_nan_notes():
    target = pd.DataFrame([
        {"rel_path": "a/01.jpg", "cohort": "LAB", "notes": float("nan")},
        {"rel_path": "a/02.jpg", "cohort": "LAB", "notes": ""},
        {"rel_path": "a/1.jpg", "cohort": "LAB", "notes": "md5-дубль группы 14; выживший 10.jpg"},
    ])
    external = pd.DataFrame(columns=["rel_path", "cohort", "notes", "mask_empty"])
    out = _anomaly_log(target, external)
    assert "10.jpg" in out
    assert "01.jpg" not in out and "02.jpg" not in out


def test_eda_writes_report(synthetic_workspace, tmp_path):
    target, external = _m(synthetic_workspace)
    write_eda(target, external, synthetic_workspace, tmp_path / "reports")
    assert (tmp_path / "reports" / "eda.md").exists()
    assert (tmp_path / "reports" / "eda.md").read_text(encoding="utf-8").strip()
