"""Тесты контракт-loader'а для downstream (Блок 2+) — TDD-контракт.

Цель: downstream берёт строки ТОЛЬКО через явные стадии, без шанса случайно подмешать
dup_keep=False / external / open_new / LAB в headline KPI / запечатанный test.
"""
import pandas as pd
import pytest

from triton_data.loader import read_manifests, select
from triton_data.manifest import build_manifest, write_manifests


def test_train_is_target_keep_train_fold(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    tr = select(t, e, "train")
    assert (tr["split_fold"] == "train").all()
    assert (tr["role"] == "target").all()
    assert tr["dup_keep"].all()


def test_gallery_excludes_open_new(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    g = select(t, e, "gallery")
    assert (g["split_role"] == "gallery").all()
    assert (~g["is_new_open"]).all()


def test_gallery_respects_scope(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    g_core = select(t, e, "gallery", scope="kpi_core")
    assert g_core["cohort"].isin(["TK", "PW"]).all()        # LAB-галерея вне kpi_core
    g_all = select(t, e, "gallery", scope="all_target")
    assert set(g_all["cohort"]) >= set(g_core["cohort"])    # all_target ⊇ kpi_core


def test_test_is_locked_kpi_core_probes(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    te = select(t, e, "test", scope="kpi_core")
    assert (te["split_fold"] == "test").all()
    assert (te["split_role"] == "probe").all()
    assert (~te["is_new_open"]).all()
    assert te["cohort"].isin(["TK", "PW"]).all()  # LAB (temporal_aux) НЕ в headline


def test_dev_and_test_md5_disjoint(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    dev = select(t, e, "dev", scope="all_target")
    test = select(t, e, "test", scope="all_target")
    assert set(dev["md5"]).isdisjoint(set(test["md5"]))


def test_pretrain_is_external_only(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    pr = select(t, e, "pretrain")
    assert (pr["role"] == "external").all()
    assert len(pr) >= 1


def test_open_stages_are_new_individuals(synthetic_workspace):
    t, e = build_manifest(synthetic_workspace)
    od = select(t, e, "open_dev")
    ot = select(t, e, "open_test")
    assert od["is_new_open"].all() if len(od) else True
    assert ot["is_new_open"].all() if len(ot) else True


def test_read_manifests_roundtrip_bool(synthetic_workspace, tmp_path):
    t, e = build_manifest(synthetic_workspace)
    write_manifests(t, e, tmp_path)
    t2, e2 = read_manifests(tmp_path)
    assert t2["dup_keep"].dtype == bool
    assert t2["is_new_open"].dtype == bool
    assert e2["mask_empty"].dtype == bool
    # стадии работают и на прочитанном с диска манифесте
    assert (select(t2, e2, "test", scope="kpi_core")["split_fold"] == "test").all()


def test_read_manifests_rejects_garbage_bool(synthetic_workspace, tmp_path):
    # мусор в bool-колонке не должен молча превращаться в False
    t, e = build_manifest(synthetic_workspace)
    write_manifests(t, e, tmp_path)
    raw = pd.read_csv(tmp_path / "manifest.csv", dtype=str)
    raw.loc[0, "dup_keep"] = "maybe"
    raw.to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError):
        read_manifests(tmp_path)
