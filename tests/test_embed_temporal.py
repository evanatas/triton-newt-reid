"""TDD embed_temporal: temporal-срез recall (probe-сессия ≠ gallery-сессия) + парный McNemar конфигов.

Герметично (mock_embedder + inline сессии). Эмбеддинги mock зависят ТОЛЬКО от особи (не от сессии) →
проверяем ПРОВОДКУ (маска temporal, recall, McNemar), а не реальный прирост.
"""
import numpy as np

from triton_crop.embed_temporal import temporal_probe_mask, temporal_recall, build_ablation_report


def test_temporal_probe_mask_flags_recaptures():
    # X: gallery s1, probe s2 → temporal (перепоимка). Y: gallery s1, probe s1 → НЕ temporal.
    p_ids = np.array(["X", "Y"]); p_sess = np.array(["s2", "s1"])
    g_ids = np.array(["X", "Y"]); g_sess = np.array(["s1", "s1"])
    m = temporal_probe_mask(p_ids, p_sess, g_ids, g_sess)
    assert list(m) == [True, False]


def _toy_oof(mock_embedder, variant="unroll_ribbon"):
    # 4 особи; на каждую gallery(s1) + probe(s2) = temporal; cohort TK; эмбеддинги разделимы
    ids = np.array([f"TK-{i}" for i in range(4)] * 2)
    role = np.array(["gallery"] * 4 + ["probe"] * 4)
    sess = np.array(["s1"] * 4 + ["s2"] * 4)
    coh = np.array(["TK"] * 8)
    emb = mock_embedder(ids, sep=1.5, seed=1)
    return {variant: emb, "role": role, "individual_id": ids, "session": sess, "cohort": coh}


def test_temporal_recall_separable_is_one(mock_embedder):
    oof = _toy_oof(mock_embedder)
    rep, hits = temporal_recall(oof, variant="unroll_ribbon", cohort_filter="TK", ks=(1, 5))
    assert rep["n"] == 4                                  # 4 temporal-пробы TK
    assert rep["recall@1"] == 1.0                         # разделимо → попадает
    assert set(hits) == {1, 5} and len(hits[1]) == 4


def test_build_ablation_report_has_mcnemar(mock_embedder):
    c0 = _toy_oof(mock_embedder)                          # «zero-shot»
    c2 = _toy_oof(mock_embedder)                          # «session-aware» (идентичные mock-эмбеддинги)
    rep = build_ablation_report({"C0": c0, "C2": c2}, variant="unroll_ribbon",
                                cohort_filter="TK", ks=(1, 5), primary_k=1)
    assert rep["per_config"]["C0"]["recall@1"] == 1.0
    assert "C2_vs_C0" in rep["mcnemar"]
    mc = rep["mcnemar"]["C2_vs_C0"]
    assert {"b", "c", "p"} <= set(mc) and 0.0 <= mc["p"] <= 1.0   # парный McNemar посчитан
