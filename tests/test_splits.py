"""Тесты сплиттера: 3-частный split_fold (train/dev/test-LOCK) + kpi_scope + анти-утечка.

Модель (аналогия экзамена):
  gallery-фото → train (учим эмбеддер + эталоны базы);
  probe-фото   → dev (тюнинг/A-B блоков 2–6) ИЛИ test (ЗАПЕЧАТАНО, headline KPI один раз в конце).
Probe НИКОГДА не в train. headline-KPI = kpi_scope=kpi_core (TK+PW) & split_fold=test & probe.
"""
import pandas as pd

from triton_data.splits import assign_splits


def _row(**kw):
    base = dict(cohort="TK", role="target", dup_keep=True, individual_id="TK-1",
                date=None, session=None, md5="m", rel_path="p")
    base.update(kw)
    return base


def _df(rows):
    return pd.DataFrame(rows)


def _ten_temporal():
    rows = []
    for k in range(1, 11):
        rows += [
            _row(individual_id=f"TK-{k}", date="2024-12", md5=f"a{k}", rel_path=f"a{k}"),
            _row(individual_id=f"TK-{k}", date="2025-01", md5=f"b{k}", rel_path=f"b{k}"),
        ]
    return _df(rows)


def _many_probes():
    rows = [_row(individual_id="TK-1", date="2024-12", md5="g0", rel_path="g0")]
    for i in range(12):
        rows.append(_row(individual_id="TK-1", date="2025-01", md5=f"p{i}", rel_path=f"p{i}"))
    return _df(rows)


def test_temporal_gallery_train_probes_dev_or_test():
    out = assign_splits(_df([
        _row(individual_id="TK-1", date="2024-12", md5="a", rel_path="a"),
        _row(individual_id="TK-1", date="2025-01", md5="b", rel_path="b"),
        _row(individual_id="TK-1", date="2025-02", md5="c", rel_path="c"),
    ]), seed=42)
    g = out[out.split_role == "gallery"]
    p = out[out.split_role == "probe"]
    assert set(g["split_fold"]) == {"train"}
    assert set(p["split_fold"]) <= {"dev", "test"}
    assert set(g["date"]) == {"2024-12"}


def test_probe_never_train_gallery_always_train():
    out = assign_splits(_ten_temporal(), seed=42, open_new_frac=0.0)
    assert out[out.split_role == "probe"]["split_fold"].isin(["dev", "test"]).all()
    assert (out[out.split_role == "gallery"]["split_fold"] == "train").all()


def test_dev_and_test_both_present_and_disjoint():
    out = assign_splits(_many_probes(), seed=42, test_frac=0.5, open_new_frac=0.0)
    folds = set(out[out.split_role == "probe"]["split_fold"])
    assert folds == {"dev", "test"}  # на 12 пробах при 0.5 оба бакета непусты


def test_fold_deterministic_under_shuffle():
    df = _many_probes()
    a = assign_splits(df.copy(), seed=42).set_index("rel_path")["split_fold"].sort_index()
    shuf = df.sample(frac=1, random_state=3).reset_index(drop=True)
    b = assign_splits(shuf, seed=42).set_index("rel_path")["split_fold"].sort_index()
    assert a.equals(b)


def test_kpi_scope_derivation():
    out = assign_splits(_df([
        _row(individual_id="TK-1", cohort="TK", md5="a", rel_path="a"),
        _row(individual_id="TK-1", cohort="TK", md5="a2", rel_path="a2"),
        _row(individual_id="PW-1", cohort="PW", md5="p", rel_path="p1"),
        _row(individual_id="PW-1", cohort="PW", md5="p2", rel_path="p2"),
        _row(individual_id="LAB-1", cohort="LAB", date="2025-08-05", md5="l", rel_path="l"),
        _row(individual_id="LAB-1", cohort="LAB", date="2025-09-23", md5="l2", rel_path="l2"),
        _row(individual_id="GCN-1", cohort="GCN", role="external", md5="g", rel_path="g"),
    ]), seed=42, open_new_frac=0.0)
    sc = out.set_index("rel_path")["kpi_scope"]
    assert sc["a"] == "kpi_core" and sc["p1"] == "kpi_core"
    assert sc["l"] == "temporal_aux"
    assert sc["g"] == "external"


def test_external_has_no_role_or_fold():
    out = assign_splits(_df([_row(individual_id="GCN-1", cohort="GCN", role="external",
                                  md5="g", rel_path="g")]), seed=42)
    r = out.iloc[0]
    assert pd.isna(r["split_role"]) and pd.isna(r["split_fold"])
    assert r["split_scheme"] == "external"


def test_open_new_probes_dev_or_test_never_train():
    rows = [_row(individual_id=f"PW-{k}", cohort="PW", md5=f"m{k}_{i}", rel_path=f"p{k}_{i}",
                 session="01")
            for k in range(1, 7) for i in range(3)]
    out = assign_splits(_df(rows), seed=42, open_new_frac=0.5, test_frac=0.5)
    new = out[out.is_new_open]
    assert (new["split_role"] == "probe").all()
    assert new["split_fold"].isin(["dev", "test"]).all()
    assert (new["split_fold"] == "train").sum() == 0
    gal_ids = set(out[out.split_role == "gallery"]["individual_id"])
    assert set(new["individual_id"]).isdisjoint(gal_ids)


def test_no_gallery_probe_md5_leak():
    out = assign_splits(_ten_temporal(), seed=42)
    g = set(out[out.split_role == "gallery"]["md5"])
    p = set(out[out.split_role == "probe"]["md5"])
    assert g.isdisjoint(p)


def test_closed_probe_individual_in_gallery():
    out = assign_splits(_ten_temporal(), seed=42, open_new_frac=0.0)
    cp = set(out[(out.split_role == "probe") & (~out.is_new_open)]["individual_id"])
    gi = set(out[out.split_role == "gallery"]["individual_id"])
    assert cp <= gi


def test_gallery_only_single_photo_is_train():
    out = assign_splits(_df([_row(individual_id="PW-9", cohort="PW", md5="z", rel_path="z")]),
                        seed=42, open_new_frac=0.0)
    assert set(out["split_scheme"]) == {"gallery_only"}
    assert set(out["split_role"]) == {"gallery"}
    assert set(out["split_fold"]) == {"train"}


def test_random_scheme_one_probe():
    rows = [_row(individual_id="PW-1", cohort="PW", md5=f"m{i}", rel_path=f"p{i}", session="01")
            for i in range(4)]
    out = assign_splits(_df(rows), seed=42, open_new_frac=0.0)
    assert set(out["split_scheme"]) == {"random"}
    assert (out.split_role == "probe").sum() == 1
    assert (out.split_role == "gallery").sum() == 3


def test_random_probe_session_tiebreak_deterministic():
    # сессии равной частоты: выбор probe не должен зависеть от порядка строк
    rows = [_row(individual_id="PW-1", cohort="PW", md5=f"m{i}", rel_path=f"p{i}", session=s)
            for i, s in enumerate(["01", "01", "02", "02"])]
    df = _df(rows)
    a = assign_splits(df.copy(), seed=42, open_new_frac=0.0)
    b = assign_splits(df.iloc[::-1].reset_index(drop=True), seed=42, open_new_frac=0.0)
    pa = sorted(a[a.split_role == "probe"]["rel_path"])
    pb = sorted(b[b.split_role == "probe"]["rel_path"])
    assert pa == pb
