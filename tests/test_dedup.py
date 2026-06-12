"""Тесты md5-дедупа и детерминированного выбора «выжившего» (TDD)."""
from triton_data.dedup import deduplicate


def _rec(cohort, rel_path, md5, **extra):
    r = {"cohort": cohort, "rel_path": rel_path, "md5": md5, "notes": ""}
    r.update(extra)
    return r


def test_unique_records_kept_group_minus1():
    out = deduplicate([_rec("TK", "TK/1/a.JPG", "aaa"), _rec("TK", "TK/2/b.JPG", "bbb")])
    assert all(r["dup_keep"] for r in out)
    assert all(r["dup_group"] == -1 for r in out)


def test_pair_exactly_one_survivor_same_group():
    out = deduplicate([_rec("TK", "TK/1/x.JPG", "same"), _rec("TK", "TK/1/y.JPG", "same")])
    assert len([r for r in out if r["dup_keep"]]) == 1
    assert {r["dup_group"] for r in out} == {0}


def test_named_beats_img_dump():
    out = deduplicate([
        _rec("TK", "tk_cohort/5/Доп фото 0125/IMG_0001.JPG", "z"),
        _rec("TK", "tk_cohort/5/05-02-0125.JPG", "z"),
    ])
    survivor = next(r for r in out if r["dup_keep"])
    assert survivor["rel_path"].endswith("05-02-0125.JPG")


def test_lab_variant_loses_to_plain():
    out = deduplicate([
        _rec("LAB", "LAB/05.08.2025/03.1.jpg", "z"),
        _rec("LAB", "LAB/05.08.2025/03.jpg", "z"),
    ])
    survivor = next(r for r in out if r["dup_keep"])
    assert survivor["rel_path"].endswith("/03.jpg")


def test_lab_bare_single_digit_loses_to_two_digit():
    # 1.jpg ≡ 10.jpg (ошибочная метка): выжить должен 10.jpg (особь 10)
    out = deduplicate([
        _rec("LAB", "LAB/25.10.2025/1.jpg", "z", local_id=1),
        _rec("LAB", "LAB/25.10.2025/10.jpg", "z", local_id=10),
    ])
    survivor = next(r for r in out if r["dup_keep"])
    dropped = next(r for r in out if not r["dup_keep"])
    assert survivor["rel_path"].endswith("/10.jpg")
    assert survivor["local_id"] == 10
    assert dropped["rel_path"].endswith("/1.jpg")
    assert dropped["notes"]  # помечен дублем/аномалией


def test_group_indices_deterministic_sorted_by_md5():
    out = deduplicate([
        _rec("TK", "TK/1/a.JPG", "mmm"), _rec("TK", "TK/1/b.JPG", "mmm"),
        _rec("TK", "TK/2/c.JPG", "aaa"), _rec("TK", "TK/2/d.JPG", "aaa"),
    ])
    grp = {r["md5"]: r["dup_group"] for r in out}
    assert grp["aaa"] == 0 and grp["mmm"] == 1  # нумерация групп по сортировке md5


def test_preserves_input_order():
    recs = [_rec("TK", "TK/1/a.JPG", "x"), _rec("TK", "TK/2/b.JPG", "y")]
    out = deduplicate(recs)
    assert [r["rel_path"] for r in out] == ["TK/1/a.JPG", "TK/2/b.JPG"]
