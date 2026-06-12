"""Тесты crops_manifest: build/write/read + контракт select_crops (анти-утечка). TDD."""
import numpy as np
import pandas as pd
import pytest

from triton_crop.crops_manifest import (
    build_crops_manifest,
    crops_parity,
    read_crops_manifest,
    select_crops,
    select_crops_with_fallback,
    write_crops_manifest,
)
from triton_crop.pipeline import CropResult


def _res(variant, status, head=None, cloaca=None):
    return CropResult(np.zeros((8, 8, 3), np.uint8), variant, status, None, head, cloaca,
                      12.0, 0.8, 0.9, None, "")


def _res_unroll(methods=("debend", "ribbon")):
    uv = {m: np.zeros((8, 8, 3), np.uint8) for m in methods}
    us = {m: "ok" for m in methods}
    return CropResult(np.zeros((8, 8, 3), np.uint8), "belly_oriented", "ok", None, (4, 2), (4, 6),
                      12.0, 0.8, 0.9, None, "", belly_axis_start=(4, 2), belly_axis_end=(4, 6),
                      unroll_variants=uv, unroll_status=us)


def test_build_schema_and_sort():
    rows = pd.DataFrame([
        {"md5": "zzz", "rel_path": "p1", "cohort": "TK", "individual_id": "TK-1",
         "split_role": "gallery", "split_fold": "train", "kpi_scope": "kpi_core"},
        {"md5": "aaa", "rel_path": "p2", "cohort": "PW", "individual_id": "PW-1",
         "split_role": "probe", "split_fold": "dev", "kpi_scope": "kpi_core"},
    ])
    df = build_crops_manifest(rows, [_res("belly_oriented", "ok", (4, 2), (4, 6)),
                                     _res("full", "empty_mask")], "crops_belly", pipeline_version="v1")
    assert list(df["md5"]) == ["aaa", "zzz"]            # отсортировано по md5
    for c in ("variant", "crop_status", "crop_path", "orientation_deg", "canon_size",
              "mirrored", "pipeline_version"):
        assert c in df.columns
    assert (df["mirrored"] == False).all()             # noqa: E712 — поворот без зеркала
    assert df.loc[df.md5 == "aaa", "crop_path"].iloc[0].endswith("aaa.png")


def test_write_read_roundtrip(tmp_path):
    rows = pd.DataFrame([{"md5": "aaa", "rel_path": "p", "cohort": "TK", "individual_id": "TK-1",
                          "split_role": "gallery", "split_fold": "train", "kpi_scope": "kpi_core"}])
    df = build_crops_manifest(rows, [_res("belly_oriented", "ok", (1, 1), (1, 2))], "crops_belly", "v1")
    write_crops_manifest(df, tmp_path)
    df2 = read_crops_manifest(tmp_path)
    assert list(df2["md5"]) == ["aaa"] and df2["mirrored"].dtype == bool


def test_select_crops_join_and_variant_filter():
    crops = pd.DataFrame([{"md5": "a", "variant": "belly_oriented"},
                          {"md5": "b", "variant": "full"},
                          {"md5": "c", "variant": "belly_oriented"}])
    rows = pd.DataFrame([{"md5": "a", "individual_id": "TK-1"}, {"md5": "c", "individual_id": "TK-2"}])
    out = select_crops(crops, rows, variant="belly_oriented")
    assert set(out["md5"]) == {"a", "c"} and "individual_id" in out.columns


def test_select_crops_excludes_unselected_rows():
    crops = pd.DataFrame([{"md5": "a", "variant": "belly_oriented"},
                          {"md5": "t", "variant": "belly_oriented"}])  # 't' = test-строка, не в выборке
    out = select_crops(crops, pd.DataFrame([{"md5": "a", "individual_id": "X"}]), variant="belly_oriented")
    assert set(out["md5"]) == {"a"}


def test_build_emits_unroll_rows():
    rows = pd.DataFrame([{"md5": "aaa", "rel_path": "p", "cohort": "TK", "individual_id": "TK-1",
                          "split_role": "gallery", "split_fold": "train", "kpi_scope": "kpi_core"}])
    df = build_crops_manifest(rows, [_res_unroll(("debend", "ribbon"))], "crops_belly", "v1")
    assert set(df["variant"]) == {"belly_oriented", "unroll_debend", "unroll_ribbon"}
    assert (df["md5"] == "aaa").all()                  # ключ (md5, variant) → несколько строк на md5
    deb = df[df.variant == "unroll_debend"].iloc[0]
    assert deb["crop_path"].endswith("aaa__debend.png") and deb["unroll_method"] == "debend"
    assert df[df.variant == "belly_oriented"].iloc[0]["crop_path"].endswith("aaa.png")


def test_select_crops_unroll_variant():
    crops = pd.DataFrame([{"md5": "a", "variant": "belly_oriented"},
                          {"md5": "a", "variant": "unroll_debend"},
                          {"md5": "b", "variant": "unroll_debend"}])
    out = select_crops(crops, pd.DataFrame([{"md5": "a", "individual_id": "TK-1"}]),
                       variant="unroll_debend")
    assert set(out["md5"]) == {"a"} and (out["variant"] == "unroll_debend").all()


def test_select_crops_rejects_dup_md5():
    crops = pd.DataFrame([{"md5": "a", "variant": "unroll_debend"},
                          {"md5": "a", "variant": "unroll_debend"}])   # нарушен ключ (md5, variant)
    with pytest.raises(ValueError):
        select_crops(crops, pd.DataFrame([{"md5": "a"}]), variant="unroll_debend")


def test_build_crops_manifest_len_mismatch():
    rows = pd.DataFrame([{"md5": "a"}, {"md5": "b"}])
    with pytest.raises(ValueError):
        build_crops_manifest(rows, [_res("belly_oriented", "ok")], "crops_belly")   # 2 строки, 1 результат


def test_select_with_fallback_no_silent_drop():
    crops = pd.DataFrame([{"md5": "a", "variant": "belly_oriented", "crop_path": "a.png"},
                          {"md5": "a", "variant": "unroll_ribbon", "crop_path": "a__r.png"},
                          {"md5": "b", "variant": "belly_oriented", "crop_path": "b.png"}])  # b без ribbon
    out = select_crops_with_fallback(crops, pd.DataFrame([{"md5": "a"}, {"md5": "b"}]),
                                     "unroll_ribbon", "belly_oriented")
    assert len(out) == 2                                       # покрытие == rows (нет тихого выпадения)
    used = dict(zip(out["md5"], out["variant_used"]))
    assert used["a"] == "unroll_ribbon" and used["b"] == "belly_oriented"


def test_select_with_fallback_rejects_dup_md5():
    # как select_crops: дубль (md5, variant) → ValueError (а не молчаливый выбор последней строки)
    crops = pd.DataFrame([{"md5": "a", "variant": "unroll_ribbon", "crop_path": "a.png"},
                          {"md5": "a", "variant": "unroll_ribbon", "crop_path": "a2.png"}])  # дубль
    with pytest.raises(ValueError):
        select_crops_with_fallback(crops, pd.DataFrame([{"md5": "a"}]), "unroll_ribbon", "belly_oriented")


def test_merge_crops_manifest_additive():
    import pandas as pd
    from triton_crop.crops_manifest import merge_crops_manifest
    existing = pd.DataFrame({"md5": ["a", "a", "b"], "variant": ["belly_oriented", "unroll_ribbon",
                             "belly_oriented"], "crop_path": ["a.png", "a_r.png", "b.png"]})
    new = pd.DataFrame({"md5": ["b", "c"], "variant": ["belly_oriented", "belly_oriented"],
                        "crop_path": ["b_NEW.png", "c.png"]})
    merged = merge_crops_manifest(existing, new)
    # старые TK/PW-строки сохранены; новая особь c добавлена; (md5,variant) уникальны; new побеждает дубль
    assert ("a", "unroll_ribbon") in set(zip(merged["md5"], merged["variant"]))
    assert "c" in set(merged["md5"])
    assert merged.duplicated(["md5", "variant"]).sum() == 0
    assert merged.loc[(merged["md5"] == "b") & (merged["variant"] == "belly_oriented"),
                      "crop_path"].iloc[0] == "b_NEW.png"     # свежий кроп перекрывает старый


def test_crops_parity_reports_gap():
    crops = pd.DataFrame([{"md5": "a", "variant": "belly_oriented"},
                          {"md5": "a", "variant": "unroll_ribbon"},
                          {"md5": "b", "variant": "belly_oriented"}])  # b без ribbon
    only_bo, only_rib, both = crops_parity(crops, pd.DataFrame([{"md5": "a"}, {"md5": "b"}]),
                                           "belly_oriented", "unroll_ribbon")
    assert only_bo == {"b"} and only_rib == set() and both == {"a"}
