"""Тесты embed_verify : read-only аудит артефактов эмбеддера.

Ловит классы ошибок, рассинхрон metrics.json ↔ .npy, фото-порядок
md5 между Mega/MiewID, утечку запечатанных (test/open_test) md5 в OOF, несовпадение размерности
чекпойнта. Герметично на mock_embedder (без torch/моделей).
"""
import numpy as np


def _oof(mock_embedder, sep=1.5, seed=1):
    ids = [f"K{i}" for i in range(8)] * 2
    role = np.array(["gallery"] * 8 + ["probe"] * 8)
    bo = mock_embedder(ids, sep=sep, noise=0.05, seed=seed)
    rib = mock_embedder(ids, sep=sep, noise=0.05, seed=seed + 10)
    return {"belly_oriented": bo, "unroll_ribbon": rib, "md5": np.array([f"m{i}" for i in range(16)]),
            "role": role, "individual_id": np.array(ids), "cohort": np.array(["TK"] * 16)}


def test_recompute_oof_recall_has_photo_and_identity(mock_embedder):
    from triton_crop.embed_verify import recompute_oof_recall
    rc = recompute_oof_recall(_oof(mock_embedder), variants=("belly_oriented", "unroll_ribbon"), ks=(1, 5))
    assert "belly_oriented" in rc and "unroll_ribbon" in rc
    assert set(rc["belly_oriented"]) == {"photo", "identity"}
    assert 1 in rc["belly_oriented"]["photo"] and 5 in rc["belly_oriented"]["photo"]


def test_verify_passes_on_consistent_artifacts(mock_embedder):
    from triton_crop.embed_verify import recompute_oof_recall, verify_embed_artifacts
    oof = _oof(mock_embedder)
    rc = recompute_oof_recall(oof)
    expected = {v: {k: rc[v]["photo"][k] for k in (1, 5)} for v in ("belly_oriented", "unroll_ribbon")}
    rep = verify_embed_artifacts(oof, expected_recall=expected, forbidden_md5=set(),
                                 ckpt_embed_dim=oof["belly_oriented"].shape[1], compare_md5=oof["md5"])
    assert rep["ok"] is True
    assert all(c["ok"] for c in rep["checks"])
    names = {c["name"] for c in rep["checks"]}
    assert {"shapes_consistent", "metrics_json_matches_npy", "no_sealed_md5_in_oof"} <= names


def test_verify_catches_metrics_desync_major1(mock_embedder):
    # Регресс: «авторитетный» JSON не соответствует текущим .npy → проверка ОБЯЗАНА упасть.
    from triton_crop.embed_verify import verify_embed_artifacts
    oof = _oof(mock_embedder)
    wrong = {"belly_oriented": {1: 0.999, 5: 0.999}}        # заведомо не из этих .npy
    rep = verify_embed_artifacts(oof, expected_recall=wrong)
    assert rep["ok"] is False
    mm = next(c for c in rep["checks"] if c["name"] == "metrics_json_matches_npy")
    assert mm["ok"] is False


def test_verify_catches_sealed_md5_leak(mock_embedder):
    from triton_crop.embed_verify import verify_embed_artifacts
    oof = _oof(mock_embedder)
    rep = verify_embed_artifacts(oof, forbidden_md5={"m3", "m9"})   # эти md5 есть в OOF
    assert rep["ok"] is False
    sealed = next(c for c in rep["checks"] if c["name"] == "no_sealed_md5_in_oof")
    assert sealed["ok"] is False


def test_verify_catches_md5_order_mismatch(mock_embedder):
    from triton_crop.embed_verify import verify_embed_artifacts
    oof = _oof(mock_embedder)
    other = oof["md5"].copy()[::-1]                                 # другой порядок
    rep = verify_embed_artifacts(oof, compare_md5=other)
    assert rep["ok"] is False
