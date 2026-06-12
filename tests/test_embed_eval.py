"""Тесты embed_eval (Блок 4): open-set оценка known/new (заказчик прямо назвал задачу open metric learning).

Гейт: на синтетике (mock_embedder) известные особи получают высокий score, новые — низкий; AUROC
разделяет (known≫new); порог по Юдену делит; политика known/new работает. open_set_auroc требует оба
класса (негатив). Калибровка порога — на open_dev (open_test НЕ вскрывать — это дисциплина cli, Этап C).
"""
import numpy as np


def test_known_new_separation_and_auroc(mock_embedder):
    from triton_crop.embed_eval import (
        apply_policy, known_new_scores, open_set_auroc, tune_threshold,
    )
    gal_ids = [f"K{i}" for i in range(8) for _ in range(3)]          # 8 известных особей × 3 кадра
    gallery = mock_embedder(gal_ids, sep=1.0, noise=0.05, seed=1)
    probe_ids = [f"K{i}" for i in range(8)] + [f"NEW{i}" for i in range(8)]   # 8 known + 8 new
    probes = mock_embedder(probe_ids, sep=1.0, noise=0.05, seed=2)
    is_new = np.array([False] * 8 + [True] * 8)

    sc = known_new_scores(probes, gallery, gal_ids)
    assert sc["max_sim"][:8].mean() > sc["max_sim"][8:].mean() + 0.1   # known ближе к галерее, чем new
    assert open_set_auroc(sc["max_sim"], is_new) > 0.9                 # known≫new разделяется
    thr = tune_threshold(sc["max_sim"], is_new)
    known_pred = apply_policy(sc["max_sim"], sc["margin"], thr, margin_min=0.0)
    assert known_pred[:8].mean() >= 0.75                              # большинство known → known
    assert (~known_pred[8:]).mean() >= 0.75                           # большинство new → new


def test_known_new_scores_predict_id_and_margin(mock_embedder):
    from triton_crop.embed_eval import known_new_scores
    gal_ids = [f"K{i}" for i in range(6) for _ in range(2)]
    gallery = mock_embedder(gal_ids, sep=1.5, noise=0.02, seed=1)
    probe_ids = [f"K{i}" for i in range(6)]
    probes = mock_embedder(probe_ids, sep=1.5, noise=0.02, seed=9)
    sc = known_new_scores(probes, gallery, gal_ids)
    assert (sc["pred_id"] == np.array(probe_ids)).mean() >= 0.8        # разнесённые кластеры → верный id
    assert (sc["margin"] >= -1e-9).all()                              # margin = top1 − лучший другой ≥ 0


def test_open_set_auroc_needs_both_classes():
    from triton_crop.embed_eval import open_set_auroc
    assert np.isnan(open_set_auroc([0.9, 0.8, 0.7], [False, False, False]))   # только known → nan
    assert np.isnan(open_set_auroc([0.9, 0.8], [True, True]))                 # только new → nan


def test_tune_threshold_finite_on_degenerate():
    # вырожденный вход НЕ должен молча давать inf-порог (apply_policy(score>=inf) → «все new»):
    # один класс или неразделяющий score → конечный порог (sklearn добавляет +inf).
    from triton_crop.embed_eval import apply_policy, tune_threshold
    assert np.isfinite(tune_threshold([0.9, 0.8, 0.7], [False, False, False]))   # все known
    assert np.isfinite(tune_threshold([0.5, 0.4], [True, True]))                 # все new
    scores = np.array([0.2, 0.3, 0.8, 0.9])
    is_new = np.array([False, False, True, True])           # инверсия (new имеет высокий score)
    thr = tune_threshold(scores, is_new)
    assert np.isfinite(thr)
    known = apply_policy(scores, np.ones_like(scores), thr)
    assert known.any()                                       # порог не выродился в «отвергнуть всех»


def test_tune_threshold_between_clusters():
    from triton_crop.embed_eval import apply_policy, tune_threshold
    scores = np.array([0.95, 0.92, 0.90, 0.30, 0.25, 0.20])
    is_new = np.array([False, False, False, True, True, True])
    thr = tune_threshold(scores, is_new)
    assert 0.30 < thr <= 0.90                                         # порог между кластерами known/new
    known = apply_policy(scores, np.ones_like(scores), thr)
    assert known.tolist() == [True, True, True, False, False, False]
