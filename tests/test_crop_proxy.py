"""Тесты метрик recall proxy-эмбеддера (чистая часть, без модели). TDD."""
import numpy as np

from triton_crop.proxy_embed import recall_metrics


def test_recall_metrics_known_ranks():
    gallery_ids = np.array(["A", "B", "C", "A"])
    probe_ids = np.array(["A", "B", "C"])
    sim = np.array([
        [0.9, 0.1, 0.2, 0.3],    # A → top1 gallery0=A ✓
        [0.1, 0.5, 0.8, 0.2],    # B → top1 gallery2=C ✗, gallery1=B на ранге 2
        [0.4, 0.3, 0.05, 0.35],  # C → ранги A,A,B,C → C только на ранге 4
    ])
    assert recall_metrics(sim, gallery_ids, probe_ids, ks=(1,))["recall@1"] == 1 / 3
    assert recall_metrics(sim, gallery_ids, probe_ids, ks=(2,))["recall@2"] == 2 / 3
    assert recall_metrics(sim, gallery_ids, probe_ids, ks=(4,))["recall@4"] == 1.0


def test_recall_metrics_reports_n_probe():
    sim = np.eye(3)
    m = recall_metrics(sim, np.array(["X", "Y", "Z"]), np.array(["X", "Y", "Z"]), ks=(1,))
    assert m["n_probe"] == 3 and m["recall@1"] == 1.0
