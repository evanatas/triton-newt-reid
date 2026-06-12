"""Тесты embed_ab (Блок 4): строгий A/B на дообученном эмбеддере (склейка поверх ab_harness).

Контракты: эмбеддинги в A/B — L2-norm; richer ab_detailed (rank/score/margin) согласован с per_probe_hits
(probe и gallery — одним путём). Честное правило адопции (как Блок 3): finetuned принимается ТОЛЬКО при non-regression
+ значимом McNemar (c>b), иначе baseline=zero_shot. Универсально для пары моделей (zero↔ft, Mega↔MiewID).
Гоняем на mock_embedder (без torch) — логика A/B герметична.
"""
import numpy as np
import pytest


def test_adopt_finetune_rejects_inconsistent_ks(mock_embedder):
    # дефект: .get(...,0.0) делал non-regression вакуумно проходимой по неизмеренному k.
    # Теперь рассогласование ks (нет recall@k в данных) и primary_k∉ks → явная ошибка, не тихий пропуск.
    from triton_crop.embed_ab import adopt_finetune, compare_models
    gids = [f"K{i}" for i in range(6) for _ in range(2)]
    pids = [f"K{i}" for i in range(6)]
    g = mock_embedder(gids, sep=1.0, seed=1)
    p = mock_embedder(pids, sep=1.0, seed=2)
    cmp = compare_models(g, p, g, p, gids, pids, ["TK"] * 6, ks=(1,))      # измерено только @1
    with pytest.raises(ValueError):
        adopt_finetune(cmp, primary_k=5, ks=(1, 5))                       # recall@5 отсутствует в данных
    with pytest.raises(ValueError):
        adopt_finetune(cmp, primary_k=3, ks=(1,))                        # primary_k не входит в ks


def test_load_oof_npy_roundtrip(tmp_path):
    # Этап C: загрузка OOF-выгрузки с Colab (artifacts/embed/oof/*.npy) → dict для to_ab_inputs/run_ab_analysis
    from triton_crop.embed_ab import load_oof_npy
    for v in ("belly_oriented", "unroll_ribbon"):
        np.save(tmp_path / f"{v}.npy", np.random.default_rng(0).random((5, 8)).astype(np.float32))
    np.save(tmp_path / "md5.npy", np.array([f"m{i}" for i in range(5)]))
    np.save(tmp_path / "role.npy", np.array(["gallery"] * 3 + ["probe"] * 2))
    np.save(tmp_path / "individual_id.npy", np.array([f"K{i}" for i in range(5)]))
    np.save(tmp_path / "cohort.npy", np.array(["TK"] * 5))
    oof = load_oof_npy(tmp_path)
    assert oof["belly_oriented"].shape == (5, 8)
    assert set(oof) >= {"belly_oriented", "unroll_ribbon", "md5", "role", "individual_id", "cohort"}
    assert list(oof["role"]) == ["gallery", "gallery", "gallery", "probe", "probe"]


def test_build_finetuned_ab_structure(mock_embedder):
    # Этап C: один словарь даёт ОБА сравнения — zero-shot↔finetuned и belly_oriented↔ribbon на finetuned.
    from triton_crop.embed_ab import build_finetuned_ab
    n = 12
    ids = [f"K{i}" for i in range(n)] * 2
    role = ["gallery"] * n + ["probe"] * n
    ft_bo = mock_embedder(ids, sep=2.0, seed=1)         # finetuned — сильнее
    ft_rib = mock_embedder(ids, sep=2.2, seed=1)
    zero_bo = mock_embedder(ids, sep=0.5, seed=2)       # zero-shot — слабее
    zero_rib = mock_embedder(ids, sep=0.5, seed=3)
    oof = {"belly_oriented": ft_bo, "unroll_ribbon": ft_rib, "role": np.array(role),
           "individual_id": np.array(ids), "cohort": np.array(["TK"] * (2 * n))}
    ab = build_finetuned_ab(oof, zero_bo, zero_ribbon=zero_rib)
    assert {"raw", "belly_oriented", "unroll_ribbon"} <= set(ab)   # raw=zero-shot bo, bo=ft bo, ribbon=ft
    assert "unroll_adopt_decision" in ab                          # bo↔ribbon на finetuned
    assert "adopt_finetuned_bo" in ab and "adopt_finetuned_ribbon" in ab   # zero↔finetuned
    assert "finetuned_vs_zeroshot_bo" in ab
    # finetuned (sep 2.0) не хуже zero-shot (sep 0.5) по recall@1
    assert ab["belly_oriented"]["overall"]["recall@1"] >= ab["raw"]["overall"]["recall@1"]


def test_ab_detailed_matches_hits_and_l2(mock_embedder):
    from triton_crop.ab_harness import per_probe_hits
    from triton_crop.embed_ab import ab_detailed
    gids = [f"K{i}" for i in range(6) for _ in range(2)]
    g = mock_embedder(gids, sep=0.8, noise=0.1, seed=1)
    pids = [f"K{i}" for i in range(6)]
    p = mock_embedder(pids, sep=0.8, noise=0.1, seed=2)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-5)        # эмбеддинги L2-norm
    det = ab_detailed(p, pids, g, gids, ks=(1, 5))
    h = per_probe_hits(p, pids, g, gids, ks=(1, 5))
    assert (det["hit@1"].to_numpy() == h[1]).all()                      # один путь → согласовано
    assert (det["hit@5"].to_numpy() == h[5]).all()
    assert set(det.columns) >= {"individual_id", "pred_id", "score_top1", "margin_top1_top2", "true_rank"}
    assert (det["margin_top1_top2"] >= -1e-9).all()


def test_finetune_adopted_only_if_significant(mock_embedder):
    from triton_crop.embed_ab import adopt_finetune, compare_models
    n = 20
    gids = [f"K{i}" for i in range(n) for _ in range(3)]
    pids = [f"K{i}" for i in range(n)]
    coh = ["TK"] * n
    # zero-shot почти сливает особей; finetuned их разносит
    g_zero = mock_embedder(gids, dim=32, sep=0.15, noise=0.3, seed=1)
    p_zero = mock_embedder(pids, dim=32, sep=0.15, noise=0.3, seed=2)
    g_ft = mock_embedder(gids, dim=32, sep=3.0, noise=0.03, seed=1)
    p_ft = mock_embedder(pids, dim=32, sep=3.0, noise=0.03, seed=2)

    # сценарий A: finetuned ЗНАЧИМО лучше → принят
    cmp_a = compare_models(g_zero, p_zero, g_ft, p_ft, gids, pids, coh)
    dec_a = adopt_finetune(cmp_a)
    assert dec_a["decision"] == "finetuned"
    assert dec_a["significant"] and dec_a["non_regression"]
    assert cmp_a["finetuned"]["overall"]["recall@1"] > cmp_a["zero_shot"]["overall"]["recall@1"]

    # сценарий B: «finetuned» == zero (идентичны) → НЕ принят, baseline
    cmp_b = compare_models(g_ft, p_ft, g_ft, p_ft, gids, pids, coh)
    dec_b = adopt_finetune(cmp_b)
    assert dec_b["decision"] == "zero_shot"
    assert not dec_b["significant"]


def test_compare_models_is_symmetric_labels(mock_embedder):
    # та же механика годится для Mega↔MiewID (просто другие подписи)
    from triton_crop.embed_ab import compare_models
    gids = [f"K{i}" for i in range(8) for _ in range(2)]
    pids = [f"K{i}" for i in range(8)]
    g = mock_embedder(gids, sep=1.5, noise=0.05, seed=1)
    p = mock_embedder(pids, sep=1.5, noise=0.05, seed=2)
    cmp = compare_models(g, p, g, p, gids, pids, ["TK"] * 8,
                         label_a="megadescriptor", label_b="miewid")
    assert "megadescriptor" in cmp and "miewid" in cmp
    assert "stats_miewid_vs_megadescriptor@5" in cmp
    # идентичные эмбеддеры → нулевая разница
    assert cmp["stats_miewid_vs_megadescriptor@5"]["bootstrap_diff"] == 0.0


# ───────────── identity-level CMC вместо photo-level ─────────────

def test_identity_hits_equals_photo_when_one_photo_per_id(mock_embedder):
    # когда у каждой особи ровно 1 фото в галерее — агрегация ничего не меняет: identity==photo
    from triton_crop.ab_harness import per_probe_hits, per_probe_identity_hits
    gids = [f"K{i}" for i in range(10)]            # по 1 фото на особь
    pids = [f"K{i}" for i in range(10)]
    g = mock_embedder(gids, sep=1.2, noise=0.05, seed=1)
    p = mock_embedder(pids, sep=1.2, noise=0.05, seed=2)
    photo = per_probe_hits(p, pids, g, gids, ks=(1, 5))
    ident = per_probe_identity_hits(p, pids, g, gids, ks=(1, 5), aggregation="max")
    assert (photo[1] == ident[1]).all() and (photo[5] == ident[5]).all()


def test_identity_hits_differ_when_wrong_id_has_many_photos():
    # КОНТРАКТ ТЗ: top-K = список ОСОБЕЙ. Конструируем: чужая особь W имеет 2 фото на ранге 1-2,
    # истинная особь T — на ранге 3. Photo-level @2 промахивается, identity-level @2 попадает.
    import numpy as np
    from triton_crop.ab_harness import per_probe_hits, per_probe_identity_hits
    # 3 особи в галерее: W (2 фото), T (1 фото), X (1 фото). Probe близок к W, T второй по близости.
    gallery = np.array([[1.0, 0.0],     # W фото1 (ближайшее)
                        [0.98, 0.0],    # W фото2 (2-е)
                        [0.6, 0.8],     # T фото (3-е по косинусу с пробой)
                        [-1.0, 0.0]])   # X
    gids = np.array(["W", "W", "T", "X"])
    probe = np.array([[1.0, 0.0]])      # ближе всего к W, потом T
    pids = np.array(["T"])              # истинная особь — T
    photo = per_probe_hits(probe, pids, gallery, gids, ks=(1, 2))
    ident = per_probe_identity_hits(probe, pids, gallery, gids, ks=(1, 2), aggregation="max")
    assert photo[2][0] == False        # top-2 ФОТО = {W, W} → T не попал
    assert ident[2][0] == True         # top-2 ОСОБЕЙ = {W, T} → T попал
    assert ident[1][0] == False        # top-1 особь = W (не T)


def test_ab_detailed_has_identity_fields(mock_embedder):
    from triton_crop.embed_ab import ab_detailed
    gids = [f"K{i}" for i in range(6) for _ in range(2)]   # по 2 фото на особь
    pids = [f"K{i}" for i in range(6)]
    g = mock_embedder(gids, sep=0.8, noise=0.1, seed=1)
    p = mock_embedder(pids, sep=0.8, noise=0.1, seed=2)
    det = ab_detailed(p, pids, g, gids, ks=(1, 5))
    need = {"true_identity_rank", "top1_identity", "n_gallery_photos_true_id",
            "max_true_id_score", "best_wrong_id_score", "margin_identity_top1_top2", "hit_id@1", "hit_id@5"}
    assert need <= set(det.columns)
    # identity-rank согласован с per_probe_identity_hits (один путь)
    from triton_crop.ab_harness import per_probe_identity_hits
    ih = per_probe_identity_hits(p, pids, g, gids, ks=(1, 5))
    assert (det["hit_id@1"].to_numpy() == ih[1]).all()
    assert (det["hit_id@5"].to_numpy() == ih[5]).all()
    # n фото истинной особи в галерее = 2 (если особь есть в галерее)
    assert (det["n_gallery_photos_true_id"] == 2).all()


# ───────────── разнести pipeline_recall vs closedset_reid_recall + coverage ─────────────

def test_recall_split_pipeline_vs_closedset():
    # closedset_reid_recall — только probe, чья особь ЕСТЬ в OOF-галерее; pipeline_recall —
    # все ОФИЦИАЛЬНЫЕ probe (нет-кропа / нет-в-галерее = промах). + coverage.
    from triton_crop.embed_ab import recall_split
    gemb = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    gids = np.array(["A", "B", "C"])
    pemb = np.array([[1., 0.], [0., 1.], [0.5, 0.5]])
    pids = np.array(["A", "B", "X"])               # X — особи нет в галерее
    role = np.array(["gallery"] * 3 + ["probe"] * 3)
    oof = {"belly_oriented": np.vstack([gemb, pemb]), "role": role,
           "individual_id": np.concatenate([gids, pids]), "cohort": np.array(["TK"] * 6)}
    rs = recall_split(oof, "belly_oriented", ks=(1,), n_official_probe=4, n_official_gallery=5)
    assert rs["coverage"]["n_excluded_no_true_id"] == 1            # X
    assert rs["coverage"]["n_true_id_in_gallery"] == 2
    assert rs["coverage"]["n_official_probe"] == 4 and rs["coverage"]["n_probe_oof"] == 3
    assert rs["closedset_reid_recall"][1] == 1.0                   # A,B попали → 2/2
    assert rs["pipeline_recall"][1] == 0.5                         # 2 попадания / 4 офиц. probe
