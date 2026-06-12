"""Тесты ингесторов 4 когорт (TDD)."""
from triton_data import ingest_tk, ingest_pw, ingest_lab, ingest_gcn


def _spec(cfg, name):
    return next(d for d in cfg.datasets if d.name == name)


def _basename(rec):
    return rec["rel_path"].split("/")[-1]


def test_ingest_tk(synthetic_workspace):
    cfg = synthetic_workspace
    recs = ingest_tk.ingest(_spec(cfg, "karelinii"), cfg)
    assert len(recs) == 6  # 99 (skip) полностью пропущена
    assert all("99 (skip)" not in r["rel_path"] for r in recs)
    by = {_basename(r): r for r in recs}
    assert "Иллюстрация.jpg" not in by

    r = by["01-01-1224.JPG"]
    assert r["local_id"] == 1 and r["session"] == "01"
    assert r["date"] == "2024-12" and r["date_source"] == "filename"

    r = by["IMG_1000.JPG"]  # дата из имени подпапки
    assert r["local_id"] == 1 and r["date"] == "2025-02" and r["date_source"] == "subfolder"

    r = by["IMG_2000.JPG"]  # typo-подпапка папки 66
    assert r["local_id"] == 66 and r["date"] == "2025-01" and r["date_source"] == "subfolder"

    assert all(not r["rel_path"].startswith("/") for r in recs)
    assert all(r["md5"] and r["width"] > 0 and r["height"] > 0 for r in recs)
    assert all(r["cohort"] == "TK" and r["role"] == "target" for r in recs)


def test_ingest_lab(synthetic_workspace):
    cfg = synthetic_workspace
    recs = ingest_lab.ingest(_spec(cfg, "lab"), cfg)
    assert len(recs) == 5
    assert all("пояснение" not in _basename(r) for r in recs)  # непарсимое имя пропущено, ingest не упал
    by = {_basename(r) + "@" + r["session"]: r for r in recs}

    r = by["03.jpg@2025-08-05"]
    assert r["local_id"] == 3 and r["date"] == "2025-08-05" and r["shot"] == 0 and r["cohort"] == "LAB"
    assert by["03.1.jpg@2025-08-05"]["shot"] == 1
    assert by["1.jpg@2025-10-25"]["local_id"] == 1  # сырой парс; дедуп починит позже


def test_ingest_pw(synthetic_workspace):
    cfg = synthetic_workspace
    recs = ingest_pw.ingest(_spec(cfg, "pleurodeles"), cfg)
    assert len(recs) == 3
    r = next(r for r in recs if _basename(r) == "01-01 (1).JPG")
    assert r["local_id"] == 1 and r["session"] == "01" and r["shot"] == 1
    assert r["date"] is None and r["date_source"] == "none"
    assert all(r["cohort"] == "PW" and r["role"] == "target" for r in recs)


def test_ingest_gcn(synthetic_workspace):
    cfg = synthetic_workspace
    recs = ingest_gcn.ingest(_spec(cfg, "gcnid"), cfg)
    assert len(recs) == 2
    by = {r["local_id"]: r for r in recs}
    assert set(by) == {1, 10}

    r = by[1]
    assert r["cohort"] == "GCN" and r["role"] == "external"
    assert r["rel_path"].endswith("Raw_Data/1/IMG_2530.JPEG")
    assert r["rle_h"] == 2048 and r["rle_w"] == 1536 and r["mask_empty"] is False
    assert r["recapture_id"] == "1" and r["survey"] == "2"

    r10 = by[10]
    assert r10["rle_h"] == 1536 and r10["rle_w"] == 2048 and r10["mask_empty"] is True
