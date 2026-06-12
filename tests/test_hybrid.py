"""Тесты гибрида Блока 6 (шаг 6.1): эмбеддер top-K shortlist → матчер переранжирует внутри.

Ключевые свойства: (1) если матчер уверенно знает истинную особь среди top-K эмбеддера — recall@1 растёт;
(2) если матчер бесполезен (нули) — гибрид == эмбеддер (НЕ вредит); (3) матчер не вытаскивает особь
из-за пределов top-K эмбеддера (shortlist-гейт). Герметично на синтетических sim-матрицах.
"""
import numpy as np


def _sims_true_rank3():
    """Сконструировать: эмбеддер ставит истинную особь на ранг 3 (внутри top-K), матчер — на 1-е место."""
    # 5 особей (по 1 фото), 4 пробы; истинная особь probe i = особь i
    gids = np.array(["A", "B", "C", "D", "E"])
    pids = np.array(["A", "B", "C", "D"])
    P, G = 4, 5
    embed = np.full((P, G), 0.1)
    matcher = np.zeros((P, G))
    for i in range(P):
        # эмбеддер: две чужие особи выше истинной, истинная — 3-я по величине
        embed[i, i] = 0.5                       # истинная
        embed[i, (i + 1) % G] = 0.7             # чужая выше
        embed[i, (i + 2) % G] = 0.6             # чужая выше
        matcher[i, i] = 1.0                     # матчер уверенно знает истинную
    return embed, matcher, gids, pids


def test_fuse_rerank_improves_recall1():
    from triton_crop.hybrid import embedder_hits, fuse_hits
    embed, matcher, gids, pids = _sims_true_rank3()
    base = embedder_hits(embed, gids, pids, ks=(1, 5))
    hyb = fuse_hits(embed, matcher, gids, pids, k=5, method="rerank", ks=(1, 5))
    assert base[1].mean() == 0.0               # эмбеддер: истинная на ранге 3 → recall@1=0
    assert hyb[1].mean() == 1.0                # гибрид: матчер поднял истинную на 1-е → recall@1=1


def test_fuse_no_harm_when_matcher_zero():
    from triton_crop.hybrid import embedder_hits, fuse_hits
    embed, _, gids, pids = _sims_true_rank3()
    zero = np.zeros_like(embed)
    base = embedder_hits(embed, gids, pids, ks=(1, 5))
    hyb = fuse_hits(embed, zero, gids, pids, k=5, method="rerank", ks=(1, 5))
    assert (base[1] == hyb[1]).all() and (base[5] == hyb[5]).all()   # нули матчера → как эмбеддер


def test_fuse_cannot_rescue_outside_topk():
    from triton_crop.hybrid import fuse_hits
    # истинная особь — ранг 4 у эмбеддера, но top-K=2 → матчер не достаёт её
    gids = np.array(["A", "B", "C", "D"])
    pids = np.array(["A"])
    embed = np.array([[0.2, 0.9, 0.8, 0.7]])    # истинная A — последняя (ранг 4)
    matcher = np.array([[1.0, 0.0, 0.0, 0.0]])  # матчер любит A
    hyb = fuse_hits(embed, matcher, gids, pids, k=2, method="rerank", ks=(1,))
    assert hyb[1].mean() == 0.0                 # A вне top-2 эмбеддера → не спасти


def test_compare_hybrid_vs_embedder_mcnemar():
    from triton_crop.hybrid import compare_hybrid_vs_embedder
    embed, matcher, gids, pids = _sims_true_rank3()
    cmp = compare_hybrid_vs_embedder(embed, matcher, gids, pids, ["TK"] * 4, k=5,
                                     method="rerank", ks=(1, 5))
    assert cmp["hybrid"]["overall"]["recall@1"] == 1.0
    assert cmp["embedder"]["overall"]["recall@1"] == 0.0
    assert "stats_hybrid_vs_embedder@1" in cmp


def test_provenance_mismatch_raises():
    # --reuse-sims не должен молча использовать stale sim при смене конфига
    import pytest

    from triton_crop.hybrid import check_sims_provenance
    stored = {"scope": "kpi_core", "surface": "belly_oriented", "method": "darkness",
              "matcher": "guided", "ransac_iters": 500, "embed_variant": "unroll_ribbon"}
    check_sims_provenance(stored, dict(stored))                 # совпадает → не падает
    with pytest.raises(ValueError):
        check_sims_provenance(stored, dict(stored, matcher="ransac"))   # рассинхрон → ValueError


def test_fuse_grid_picks_best():
    from triton_crop.hybrid import fuse_grid
    embed, matcher, gids, pids = _sims_true_rank3()
    res = fuse_grid(embed, matcher, gids, pids, ["TK"] * 4,
                    methods=("rerank",), ks_topk=(5,), alphas=(0.5,), recall_ks=(1, 5))
    assert res["best"]["recall@1"] == 1.0
    assert "table" in res
