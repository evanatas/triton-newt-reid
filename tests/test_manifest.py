"""Тесты сборки манифеста (2 файла, TDD)."""
import pandas.testing as pdt

from triton_data.manifest import build_manifest, write_manifests


def test_counts_and_namespaces(synthetic_workspace):
    target, external = build_manifest(synthetic_workspace)
    assert len(target) == 14          # TK 6 + LAB 5 + PW 3
    assert len(external) == 2          # GCN
    assert set(target["role"]) == {"target"}
    assert set(external["role"]) == {"external"}
    assert (external["cohort"] == "GCN").all()
    assert target["individual_id"].str.match(r"^(TK|LAB|PW)-\d{3}$").all()
    assert (~target["rel_path"].str.startswith("/")).all()


def test_dedup_marks_two_non_keep(synthetic_workspace):
    target, _ = build_manifest(synthetic_workspace)
    # LAB: 03.1≡03 и 1≡10 -> ровно 2 не-выживших
    assert (~target["dup_keep"]).sum() == 2
    keep = target[target["dup_keep"]]
    assert keep["split_scheme"].notna().all()


def test_no_gallery_probe_md5_leak(synthetic_workspace):
    target, _ = build_manifest(synthetic_workspace)
    g = set(target[target.split_role == "gallery"]["md5"])
    p = set(target[target.split_role == "probe"]["md5"])
    assert g.isdisjoint(p)


def test_external_has_gcn_fields(synthetic_workspace):
    _, external = build_manifest(synthetic_workspace)
    for col in ("rle_h", "rle_w", "mask_empty", "bbox", "recapture_id", "survey"):
        assert col in external.columns
    # split-колонок во внешнем манифесте нет
    assert "split_role" not in external.columns


def test_deterministic_bytes(synthetic_workspace, tmp_path):
    t1, e1 = build_manifest(synthetic_workspace)
    t2, e2 = build_manifest(synthetic_workspace)
    pdt.assert_frame_equal(t1, t2)
    pdt.assert_frame_equal(e1, e2)
    write_manifests(t1, e1, tmp_path / "a")
    write_manifests(t2, e2, tmp_path / "b")
    assert (tmp_path / "a" / "manifest.csv").read_bytes() == (tmp_path / "b" / "manifest.csv").read_bytes()
    assert (tmp_path / "a" / "manifest_external.csv").read_bytes() == (tmp_path / "b" / "manifest_external.csv").read_bytes()
