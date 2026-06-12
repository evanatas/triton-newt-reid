"""Тесты демо-движка (финализация): чистая логика ранжира / калибровки / known-new (без моделей/IO)."""
import numpy as np
import pytest

from triton_crop.demo_backend import (
    GalleryIndex, calibrate_confidence, known_new_verdict, rank_individuals,
)


def test_calibrate_confidence_monotonic_and_bounds():
    assert calibrate_confidence(0.10) == 0.0        # ниже lo → 0
    assert calibrate_confidence(0.60) == 99.0       # выше hi → потолок 99, не 100
    assert 0.0 < calibrate_confidence(0.35) < 100.0
    assert calibrate_confidence(0.45) > calibrate_confidence(0.30)   # монотонно возрастает
    with pytest.raises(ValueError):
        calibrate_confidence(0.3, lo=0.5, hi=0.5)   # hi <= lo


def _gallery():
    embs = np.array([[1, 0, 0, 0], [0.96, 0.28, 0, 0], [0, 1, 0, 0], [0, 0.96, 0.28, 0]], float)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return GalleryIndex(emb=embs, ids=np.array(["A", "A", "B", "B"]),
                        md5=np.array(["a1", "a2", "b1", "b2"]), cohort=np.array(["TK"] * 4),
                        crop_paths=["a1.png", "a2.png", "b1.png", "b2.png"], embed_variant="belly_oriented")


def test_rank_individuals_identity_level_and_topk():
    g = _gallery()
    r = rank_individuals(np.array([1.0, 0, 0, 0]), g, topk=2)    # запрос ближе к особи A
    assert [x["individual_id"] for x in r] == ["A", "B"]         # A первой
    assert r[0]["n_photos"] == 2 and r[0]["best_md5"] == "a1"    # агрегация фото → особь; лучшее фото
    assert 0.0 <= r[0]["confidence"] <= 100.0
    assert r[0]["sim"] > r[1]["sim"]                             # своя особь выше чужой


def test_calibration_range_genuine_vs_impostor():
    # Калибровка шкалы: lo берётся из косинусов РАЗНЫХ особей (impostor), hi — из ОДНОЙ особи (genuine);
    # своя особь → ~100 %, чужая → низко (без сатурации всех к 100 %).
    from triton_crop.demo_backend import calibrate_confidence, calibration_range
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8)                           # 3 особи × 4 фото: внутри тесно, между — далеко
    embs, ids = [], []
    for k in range(3):
        for _ in range(4):
            v = centers[k] + 0.05 * rng.standard_normal(8)
            embs.append(v / np.linalg.norm(v)); ids.append(f"ID{k}")
    g = GalleryIndex(emb=np.array(embs), ids=np.array(ids), md5=np.array([f"m{i}" for i in range(12)]),
                     cohort=np.array(["TK"] * 12), crop_paths=[f"m{i}.png" for i in range(12)],
                     embed_variant="belly_oriented")
    lo, hi = calibration_range(g)
    assert lo < hi                                   # impostor-уровень ниже genuine-уровня
    assert hi > 0.7 and lo < 0.5                     # тесные genuine-кластеры, далёкие impostor
    assert calibrate_confidence(hi + 0.05, lo, hi) == 99.0    # уверенное своё → потолок 99 %
    assert calibrate_confidence(lo - 0.05, lo, hi) == 0.0     # типичный чужой → 0 %


def test_calibration_range_small_gallery_fallback():
    # мало genuine-пар (по 2 фото на особь) → fallback на все пары, диапазон валиден
    from triton_crop.demo_backend import calibration_range
    lo, hi = calibration_range(_gallery())
    assert hi > lo and -1.0 <= lo <= 1.0 and -1.0 <= hi <= 1.0


def test_load_heldout_roundtrip(tmp_path):
    from triton_crop.demo_backend import load_heldout
    d = tmp_path / "heldout"; d.mkdir()
    np.save(d / "belly_oriented.npy", np.eye(3, dtype=np.float32))
    np.save(d / "md5.npy", np.array(["m0", "m1", "m2"]))
    np.save(d / "individual_id.npy", np.array(["TK-1", "TK-2", "NEW-9"]))
    np.save(d / "cohort.npy", np.array(["TK", "TK", "PW"]))
    np.save(d / "is_new.npy", np.array([False, False, True]))
    items = load_heldout(d, tmp_path / "crops", "belly_oriented")
    assert len(items) == 3
    assert items[0]["individual_id"] == "TK-1" and items[0]["is_new"] is False
    assert items[2]["is_new"] is True                       # open_test = новая особь
    assert items[0]["crop_path"].endswith("m0.png")
    assert items[0]["emb"].shape == (3,)


def test_load_headline_parses_sealed_numbers(tmp_path):
    # витрина демо/НИР читает финальные sealed-числа ПРОГРАММНО из artifacts/ab_test_headline.json
    import json

    from triton_crop.demo_backend import load_headline
    h = {"headline": {
        "recall": {
            "overall": {"n": 112, "recall@1": 0.25, "recall@5": 0.44642857142857145},
            "per_cohort": {"PW": {"n": 11, "recall@1": 0.818, "recall@5": 1.0},
                           "TK": {"n": 101, "recall@1": 0.188, "recall@5": 0.386}},
            "pipeline_recall": {"1": 0.2478, "5": 0.4425},
            "closedset_reid_recall": {"1": 0.2617, "5": 0.4673}},
        "openset": {"auroc": 0.44642857142857145},
        "n_gallery": 238}}
    p = tmp_path / "ab_test_headline.json"
    p.write_text(json.dumps(h), encoding="utf-8")
    d = load_headline(p)
    assert d is not None
    assert d["overall@1"] == 0.25 and round(d["overall@5"], 3) == 0.446
    assert d["PW@1"] == 0.818 and d["PW@5"] == 1.0
    assert d["TK@1"] == 0.188 and d["TK@5"] == 0.386
    assert d["PW_n"] == 11 and d["TK_n"] == 101      # per-cohort n — тоже из артефакта, не «на глаз»
    assert round(d["auroc"], 3) == 0.446
    assert d["n"] == 112 and d["n_gallery"] == 238
    assert d["pipeline@1"] == 0.2478


def test_load_headline_missing_returns_none(tmp_path):
    from triton_crop.demo_backend import load_headline
    assert load_headline(tmp_path / "нет.json") is None


def test_load_headline_malformed_returns_none(tmp_path):
    # битый/неполный JSON → None (демо просто покажет только dev, без падения)
    from triton_crop.demo_backend import load_headline
    p = tmp_path / "h.json"
    p.write_text('{"foo": 1}', encoding="utf-8")           # нет ключа headline
    assert load_headline(p) is None
    p.write_text('{"headline": {"recall": {}}}', encoding="utf-8")   # неполная структура
    assert load_headline(p) is None
    p.write_text("не json вовсе", encoding="utf-8")
    assert load_headline(p) is None


def test_filter_scope_by_cohort():
    from triton_crop.demo_backend import filter_scope
    g = GalleryIndex(emb=np.eye(4), ids=np.array(["A", "B", "C", "D"]), md5=np.array(["a", "b", "c", "d"]),
                     cohort=np.array(["TK", "PW", "TK", "PW"]),
                     crop_paths=["a.png", "b.png", "c.png", "d.png"], embed_variant="belly_oriented")
    assert filter_scope(g, "Вся база") is g                      # без среза
    pw = filter_scope(g, "PW")
    assert list(pw.ids) == ["B", "D"] and pw.crop_paths == ["b.png", "d.png"]   # crop_paths согласованы
    assert pw.emb.shape == (2, 4) and list(pw.cohort) == ["PW", "PW"]


def test_known_new_verdict_thresholds():
    assert known_new_verdict([{"confidence": 92.0}, {"confidence": 55.0}])["verdict"] == "known"
    assert known_new_verdict([{"confidence": 58.0}, {"confidence": 54.0}])["verdict"] == "new_candidate"  # top1<70
    assert known_new_verdict([{"confidence": 85.0}, {"confidence": 83.0}])["verdict"] == "new_candidate"  # margin мал
    assert known_new_verdict([])["verdict"] == "new_candidate"   # пустой
