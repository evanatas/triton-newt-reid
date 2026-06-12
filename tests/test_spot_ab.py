"""Тесты A/B-обёртки матчера (Блок 5, шаги 5.6–5.7) — гейты S4/S5.

Матчер выдаёт ПРОИЗВОЛЬНУЮ sim-матрицу (доля совпавших пятен), а не эмбеддинги → нужна обёртка под
ab_harness. Проверяем: identity-ранжир из готовой sim; формат ab-словаря == эмбеддерскому; честный
McNemar матчер↔эмбеддер; Rosa-адопция (значимость+non-regression, primary_k=1); grid выбирает лучший
recall@1. Герметично (mock_embedder + синтетические sim).
"""
import numpy as np


def _gp(n):
    gids = [f"K{i}" for i in range(n) for _ in range(2)]
    pids = [f"K{i}" for i in range(n)]
    return gids, pids, ["TK"] * n


def test_identity_hits_from_sim_separable():
    from triton_crop.spot_ab import identity_hits_from_sim
    gids, pids, _ = _gp(10)
    ga = np.array(gids)
    sim = np.full((10, len(gids)), 0.1)
    for i, pid in enumerate(pids):
        sim[i, ga == pid] = 1.0                          # истинная особь — максимум
    h = identity_hits_from_sim(sim, pids, gids, ks=(1, 5))
    assert h[1].mean() == 1.0 and h[5].mean() == 1.0     # recall@1 = 1 на разделимом


def test_summarize_sim_format_matches_embedder(mock_embedder):
    from triton_crop.spot_ab import summarize_sim
    gids, pids, coh = _gp(8)
    g = mock_embedder(gids, sep=1.5, seed=1); p = mock_embedder(pids, sep=1.5, seed=2)
    sim = np.asarray(p) @ np.asarray(g).T
    s = summarize_sim(sim, pids, gids, coh, ks=(1, 5))
    assert "overall" in s and "recall@1" in s["overall"] and "recall@5" in s["overall"]
    assert "TK" in s                                     # per-cohort, как у эмбеддера


def test_recall_split_sim_pipeline_vs_closedset():
    from triton_crop.spot_ab import recall_split_sim
    gids = ["A", "A", "B", "B", "C", "C"]
    pids = ["A", "B", "X"]                               # X — особи нет в галерее
    ga = np.array(gids)
    sim = np.full((3, 6), 0.1)
    sim[0, ga == "A"] = 1.0; sim[1, ga == "B"] = 1.0
    rs = recall_split_sim(sim, pids, gids, ks=(1,), n_official_probe=4)
    assert rs["coverage"]["n_excluded_no_true_id"] == 1
    assert rs["closedset_reid_recall"][1] == 1.0         # A,B попали → 2/2
    assert rs["pipeline_recall"][1] == 0.5               # 2 попадания / 4 офиц. probe


def test_compare_matcher_beats_weak_embedder_and_adopt(mock_embedder):
    from triton_crop.spot_ab import adopt_matcher, compare_matcher_vs_embedder
    n = 15
    gids, pids, coh = _gp(n)
    ga = np.array(gids)
    g = mock_embedder(gids, sep=0.12, noise=0.3, seed=1)     # слабый эмбеддер
    p = mock_embedder(pids, sep=0.12, noise=0.3, seed=2)
    sim = np.full((n, len(gids)), 0.1)
    for i, pid in enumerate(pids):
        sim[i, ga == pid] = 1.0                              # матчер знает истинную особь
    cmp = compare_matcher_vs_embedder(sim, p, g, pids, gids, coh, ks=(1, 5))
    assert cmp["matcher"]["overall"]["recall@1"] == 1.0
    dec = adopt_matcher(cmp, primary_k=1)
    assert dec["decision"] == "matcher" and dec["significant"] and dec["non_regression"]


def test_adopt_baseline_when_no_gain(mock_embedder):
    from triton_crop.spot_ab import adopt_matcher, compare_matcher_vs_embedder
    n = 12
    gids, pids, coh = _gp(n)
    g = mock_embedder(gids, sep=1.5, seed=1); p = mock_embedder(pids, sep=1.5, seed=2)
    sim = np.asarray(p) @ np.asarray(g).T                   # матчер == эмбеддер (та же sim) → нет прироста
    cmp = compare_matcher_vs_embedder(sim, p, g, pids, gids, coh, ks=(1, 5))
    dec = adopt_matcher(cmp, primary_k=1)
    assert dec["decision"] == "embedder" and not dec["significant"]


def test_detailed_rows_schema():
    # per-probe аудит — true_id, top1 матчера, ранги матчера/эмбеддера, hit@k, n пятен
    from triton_crop.spot_ab import detailed_rows
    sim_m = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])         # 2 probe × 3 gallery
    emb_p = np.eye(2, 4); emb_g = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float)
    rows = detailed_rows(sim_m, emb_p, emb_g, probe_ids=["X", "Y"], gallery_ids=["X", "Y", "Z"],
                         probe_md5=["m1", "m2"], cohorts=["tk", "tk"], n_spots_probe=[8, 7])
    r0 = rows[0]
    for col in ("probe_md5", "true_id", "cohort", "n_spots_probe", "top1_id", "top1_score",
                "embedder_rank", "matcher_rank", "hit@1", "hit@5", "matched_pairs_count"):
        assert col in r0
    assert "true_id_rank" not in r0                  # дубль matcher_rank убран
    assert r0["true_id"] == "X" and r0["top1_id"] == "X" and r0["hit@1"] is True
    assert r0["matcher_rank"] == 1 and r0["embedder_rank"] == 1


def test_grid_ab_picks_best_recall1():
    from triton_crop.spot_ab import grid_ab
    gids, pids, coh = _gp(12)
    ga = np.array(gids)
    good = np.full((12, len(gids)), 0.1)
    for i, pid in enumerate(pids):
        good[i, ga == pid] = 1.0
    bad = np.random.RandomState(0).random((12, len(gids))) * 0.2     # шум
    res = grid_ab({"good": good, "bad": bad}, pids, gids, coh, ks=(1, 5))
    assert res["best"] == "good"
    assert res["table"]["good"]["recall@1"] >= res["table"]["bad"]["recall@1"]
