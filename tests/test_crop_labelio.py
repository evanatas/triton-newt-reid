"""Тесты схемы label-record (контракт бутстрап↔кликер↔тренер): round-trip JSON + мутации."""
from triton_crop.labelio import LabelRecord, read_label, scan_labels, write_label


def _rec(**kw):
    base = dict(
        md5="abc123", rel_path="TK/1/01.jpg", cohort="TK", individual_id="TK-001",
        img_w=3024, img_h=4032,
        belly_polygon=((0.4, 0.3), (0.6, 0.3), (0.6, 0.7), (0.4, 0.7)),
        head_xy=(0.5, 0.2), cloaca_xy=(0.5, 0.66),
        source="pseudo", status="auto", flags=(),
        head_conf=0.3, band_conf=0.9, pipeline="v2",
    )
    base.update(kw)
    return LabelRecord(**base)


def test_dict_roundtrip():
    r = _rec()
    assert LabelRecord.from_dict(r.to_dict()) == r


def test_json_roundtrip(tmp_path):
    r = _rec()
    p = write_label(r, tmp_path)
    assert p.exists() and p.suffix == ".json"
    assert read_label(p) == r


def test_flip_swaps_head_and_cloaca_and_marks_manual():
    r = _rec()
    f = r.flip()
    assert f.head_xy == r.cloaca_xy and f.cloaca_xy == r.head_xy
    assert f.source == "manual" and f != r


def test_set_points_and_mark():
    r = _rec()
    r2 = r.set_head((0.1, 0.1)).set_cloaca((0.2, 0.2)).mark("corrected", "redraw")
    assert r2.head_xy == (0.1, 0.1) and r2.cloaca_xy == (0.2, 0.2)
    assert r2.status == "corrected" and "redraw" in r2.flags
    assert r2.source == "manual"


def test_scan_labels_sorted_by_md5(tmp_path):
    write_label(_rec(md5="bbb"), tmp_path)
    write_label(_rec(md5="aaa"), tmp_path)
    assert [x.md5 for x in scan_labels(tmp_path)] == ["aaa", "bbb"]
