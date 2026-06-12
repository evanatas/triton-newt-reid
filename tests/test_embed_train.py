"""Тесты embed_train (Блок 4, Этап B): cross-fit out-of-fold ПО ОСОБЯМ (анти-утечка train→gallery).

ЧИСТАЯ часть (assign_oof_folds, to_ab_inputs) — без torch, на synthetic_embed_dataset/mock_embedder.
МОДЕЛЬНАЯ часть (train_one_fold/cross_fit_embed/extract/checkpoint) — на КРОШЕЧНОМ CPU-backbone
(инъекция build_embedder_fn + image_loader), без timm/MegaDescriptor: проверяем ОРКЕСТРАЦИЮ —
главное анти-утечку (ни одна особь не эмбеддится моделью, обучавшейся на ней), выравнивание
под run_ab_analysis, детерминизм seed=42, save/load round-trip. torch/torchvision — внутри функций.
"""
import numpy as np
import pandas as pd


# ── крошечный backbone + детерминированный загрузчик кадров (вместо timm + реальных PNG) ──
def _tiny_factory(name="x", device="cpu", num_classes=0):
    import torch
    import torchvision.transforms as T

    class TinyBackbone(torch.nn.Module):
        num_features = 16

        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8, 16)

        def forward(self, x):
            return self.fc(x.flatten(1))

    return {"model": TinyBackbone().to(device),
            "transform": T.Compose([T.Resize((8, 8)), T.ToTensor()]), "device": device}


def _fake_loader(path):
    import hashlib
    s = int(hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.RandomState(s).randint(0, 255, (10, 12, 3), dtype=np.uint8)


def _tiny_cfg(folds=3):
    from dataclasses import replace
    from triton_crop.config import EmbedConfig
    # augment=False: тесты проверяют ОРКЕСТРАЦИЮ (анти-утечка/выравнивание/детерминизм), аугментация
    # покрыта отдельно (test_embed_augment); на Colab cfg.augment=True из embed.yaml.
    return replace(EmbedConfig(), cross_fit_folds=folds, batch_p=2, batch_k=2,
                   freeze_backbone_stages=0, augment=False)


# ───────────────────────── ЧИСТАЯ часть ─────────────────────────

def test_assign_oof_folds_individual_in_one_fold(synthetic_embed_dataset):
    from triton_crop.embed_train import assign_oof_folds
    d = synthetic_embed_dataset
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df,
                           variants=("belly_oriented", "unroll_ribbon"), n_folds=3, seed=42)
    assert set(oof["role"]) == {"gallery", "probe"}
    assert {"path_belly_oriented", "path_unroll_ribbon", "cross_fit_fold"} <= set(oof.columns)
    # ИНВАРИАНТ анти-утечки: все кадры особи (любой role) — в ОДНОМ фолде
    per_ind = oof.groupby("individual_id")["cross_fit_fold"].nunique()
    assert (per_ind == 1).all()
    # детерминизм seed=42
    oof2 = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=3, seed=42)
    assert oof["cross_fit_fold"].tolist() == oof2["cross_fit_fold"].tolist()


def test_to_ab_inputs_feeds_run_ab_analysis(mock_embedder):
    from triton_crop.ab_harness import run_ab_analysis
    from triton_crop.embed_train import to_ab_inputs
    ids = [f"K{i}" for i in range(6)] * 2
    role = ["gallery"] * 6 + ["probe"] * 6
    bo = mock_embedder(ids, sep=1.2, seed=1)
    rib = mock_embedder(ids, sep=1.2, seed=1)
    raw = mock_embedder(ids, sep=0.3, seed=2)
    oof = {"belly_oriented": bo, "unroll_ribbon": rib, "role": np.array(role),
           "individual_id": np.array(ids), "cohort": np.array(["TK"] * 12)}
    kw, extra = to_ab_inputs(oof, raw, embed_variant="unroll_ribbon")
    ab = run_ab_analysis(**kw, extra_variants=extra)
    assert "unroll_ribbon" in ab and "stats_unroll_ribbon_vs_bo@5" in ab
    assert ab["belly_oriented"]["overall"]["n"] == 6                  # 6 probe-кадров


def test_assign_oof_folds_carries_session():
    import pandas as pd
    from triton_crop.embed_train import assign_oof_folds
    g = pd.DataFrame({"md5": ["a", "b"], "individual_id": ["X", "Y"], "cohort": ["TK", "TK"],
                      "session": ["s1", "s1"]})
    p = pd.DataFrame({"md5": ["c", "d"], "individual_id": ["X", "Y"], "cohort": ["TK", "TK"],
                      "session": ["s2", "s2"]})
    crops = pd.DataFrame({
        "md5": ["a", "b", "c", "d"], "variant": ["belly_oriented"] * 4,
        "crop_path": [f"c/{m}.png" for m in "abcd"], "individual_id": ["X", "Y", "X", "Y"],
        "cohort": ["TK"] * 4})
    oof = assign_oof_folds(g, p, crops, variants=("belly_oriented",), n_folds=2, seed=42)
    assert "session" in oof.columns
    assert set(oof["session"]) == {"s1", "s2"}            # сессии протянуты из обоих источников


def test_assign_oof_folds_session_default_when_missing(synthetic_embed_dataset):
    # rows без колонки session (как synthetic_embed_dataset) → session заполняется (не NaN), без падения
    from triton_crop.embed_train import assign_oof_folds
    d = synthetic_embed_dataset
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=3, seed=42)
    assert "session" in oof.columns
    assert oof["session"].notna().all()                  # дефолт проставлен (fallback = md5)


def test_select_train_frames_gallery_only_vs_all_sessions():
    import pandas as pd
    from triton_crop.embed_train import select_train_frames
    # X: gallery s1 + probe s2 (мультисессия); X в fold 1 → для fold 0 X — обучающая особь.
    oof = pd.DataFrame({
        "md5": ["x1", "x2", "y1", "y2"],
        "individual_id": ["X", "X", "Y", "Y"],
        "role": ["gallery", "probe", "gallery", "probe"],
        "session": ["s1", "s2", "s1", "s2"],
        "cross_fit_fold": [1, 1, 0, 0],
    })
    g = select_train_frames(oof, 0, train_all_sessions=False)        # дефолт: только gallery
    assert set(g["role"]) == {"gallery"}
    assert set(g["individual_id"]) == {"X"}                           # особи ∉ fold 0
    assert int(g.groupby("individual_id").session.nunique().max()) == 1   # одна сессия
    a = select_train_frames(oof, 0, train_all_sessions=True)         # все сессии train-особей
    assert set(a["role"]) == {"gallery", "probe"}
    assert int(a[a.individual_id == "X"].session.nunique()) == 2      # X стала мультисессией → cross-session пары


# ───────────────────────── МОДЕЛЬНАЯ часть (tiny CPU) ─────────────────────────

def test_cross_fit_no_leak_39(synthetic_embed_dataset):
    from triton_crop.embed_train import assign_oof_folds, cross_fit_embed
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    r = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=2)
    assert r["belly_oriented"].shape[0] == len(oof)
    for i, ind in enumerate(r["individual_id"]):
        assert r["embedded_by"][i] is not None
        assert ind not in r["embedded_by"][i]        # анти-утечка: особь не в train-наборе своей fold-модели


def test_cross_fit_deterministic(synthetic_embed_dataset):
    from triton_crop.embed_train import assign_oof_folds, cross_fit_embed
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    r1 = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=3)
    r2 = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=3)
    for v in ("belly_oriented", "unroll_ribbon"):
        assert np.allclose(r1[v], r2[v], atol=1e-5)


def test_cross_fit_to_ab_pipeline(synthetic_embed_dataset, mock_embedder):
    from triton_crop.ab_harness import run_ab_analysis
    from triton_crop.embed_train import assign_oof_folds, cross_fit_embed, to_ab_inputs
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    r = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=2)
    # raw отдельно (не дообучается); в проде raw и finetuned — оба MegaDescriptor (одна размерность),
    # здесь tiny backbone даёт 16-D, поэтому raw тоже 16-D
    raw = mock_embedder(r["individual_id"], dim=r["belly_oriented"].shape[1], sep=0.5, seed=9)
    kw, extra = to_ab_inputs(r, raw, embed_variant="unroll_ribbon")
    ab = run_ab_analysis(**kw, extra_variants=extra)
    assert "unroll_ribbon" in ab
    assert ab["raw"]["overall"]["n"] == int((r["role"] == "probe").sum())


def test_cross_fit_resume_from_checkpoint(tmp_path, synthetic_embed_dataset):
    # сброс рантайма Colab ~90 мин: повторный запуск с тем же ckpt_dir не переобучает готовые фолды (load),
    # эмбеддинги идентичны, анти-утечка сохраняется.
    from triton_crop.embed_train import assign_oof_folds, cross_fit_embed
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    r1 = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader,
                         ckpt_dir=tmp_path, max_steps=2)
    r2 = cross_fit_embed(oof, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader,
                         ckpt_dir=tmp_path, max_steps=2)               # resume: грузит fold*.pt
    for v in ("belly_oriented", "unroll_ribbon"):
        assert np.allclose(r1[v], r2[v], atol=1e-5)
    for i, ind in enumerate(r2["individual_id"]):
        assert ind not in r2["embedded_by"][i]                        # анти-утечка держится и после resume


def test_train_one_fold_infers_embed_dim_without_num_features(synthetic_embed_dataset):
    # MiewID (transformers AutoModel) НЕ имеет .num_features → embed_dim определяется прогоном кадра
    from triton_crop.embed_train import assign_oof_folds, train_one_fold

    def _factory_no_numfeat(name="x", device="cpu", num_classes=0):
        import torch
        import torchvision.transforms as T

        class BBNoNF(torch.nn.Module):                       # без атрибута num_features (как MiewIdNet)
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3 * 8 * 8, 16)
            def forward(self, x):
                return self.fc(x.flatten(1))

        return {"model": BBNoNF().to(device),
                "transform": T.Compose([T.Resize((8, 8)), T.ToTensor()]), "device": device}

    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    state = train_one_fold(oof, 0, cfg, build_embedder_fn=_factory_no_numfeat,
                           image_loader=_fake_loader, max_steps=2)
    assert state["embed_dim"] == 16                          # определён прогоном, не из num_features


def test_checkpoint_roundtrip(tmp_path, synthetic_embed_dataset):
    from triton_crop.embed_train import (
        assign_oof_folds, extract_embeddings, load_checkpoint, save_checkpoint, train_one_fold,
    )
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    state = train_one_fold(oof, 0, cfg, build_embedder_fn=_tiny_factory,
                           image_loader=_fake_loader, max_steps=2)
    held = oof[oof["cross_fit_fold"] == 0]
    e1 = extract_embeddings(state["embedder"], held, "path_belly_oriented", _fake_loader)
    p = tmp_path / "fold0.pt"
    save_checkpoint(state, p, cfg)
    st2 = load_checkpoint(p, cfg, build_embedder_fn=_tiny_factory)
    e2 = extract_embeddings(st2["embedder"], held, "path_belly_oriented", _fake_loader)
    assert np.allclose(e1, e2, atol=1e-5)                             # save→load даёт те же эмбеддинги


def test_checkpoint_stores_and_validates_model_meta(tmp_path, synthetic_embed_dataset):
    # Регресс: чекпойнт без model-metadata — повторяемая ловушка после путаницы Mega/MiewID.
    from dataclasses import replace

    import pytest

    from triton_crop.embed_train import (
        assign_oof_folds, load_checkpoint, save_checkpoint, train_one_fold,
    )
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    state = train_one_fold(oof, 0, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=2)
    p = tmp_path / "fold0.pt"
    save_checkpoint(state, p, cfg)
    st = load_checkpoint(p, cfg, build_embedder_fn=_tiny_factory)
    assert st["model_meta"]["base_model"] == cfg.base_model           # метаданные модели сохранены
    assert st["model_meta"]["embed_dim"] == 16
    # загрузка тем же чекпойнтом, но НЕ ТЕМ base_model (Mega-конфиг ↔ MiewID-чекпойнт) → явная ошибка
    cfg_wrong = replace(cfg, base_model="hf-hub:OTHER/model")
    with pytest.raises(ValueError):
        load_checkpoint(p, cfg_wrong, build_embedder_fn=_tiny_factory)


def test_train_one_fold_session_aware_smoke(synthetic_embed_dataset):
    # session-aware ветка обучается на крошечном backbone без падения и даёт embed_dim
    from dataclasses import replace
    from triton_crop.config import EmbedConfig
    from triton_crop.embed_train import assign_oof_folds, train_one_fold
    d = synthetic_embed_dataset
    cfg = replace(EmbedConfig(), cross_fit_folds=3, batch_p=2, batch_k=2,
                  freeze_backbone_stages=0, augment=False, session_aware_sampling=True)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    state = train_one_fold(oof, 0, cfg, build_embedder_fn=_tiny_factory,
                           image_loader=_fake_loader, max_steps=2)
    assert state["embed_dim"] == 16                      # обучение прошло (session-aware сэмплер сработал)


def test_train_one_fold_emits_train_log(synthetic_embed_dataset):
    # Требование: без логов сходимости нельзя отличить «потолок» от «недообучили».
    from triton_crop.embed_train import assign_oof_folds, train_one_fold
    d = synthetic_embed_dataset
    cfg = _tiny_cfg(folds=3)
    oof = assign_oof_folds(d.gallery_rows, d.probe_rows, d.crops_df, n_folds=cfg.cross_fit_folds)
    state = train_one_fold(oof, 0, cfg, build_embedder_fn=_tiny_factory, image_loader=_fake_loader, max_steps=3)
    log = state["train_log"]
    assert isinstance(log, list) and len(log) >= 1
    assert {"epoch", "loss_mean", "lr_head", "n_steps"} <= set(log[0])
    assert log[0]["loss_mean"] >= 0.0 and log[0]["n_steps"] >= 1


def test_train_one_fold_session_aware_selects_session_sampler(monkeypatch):
    # Проводка флага: при session_aware_sampling=True train_one_fold РЕАЛЬНО конструирует
    # SessionAwarePKSampler и передаёт ему колонку session. RED при регрессе «всегда PKSampler».
    # Патчим модуль-источник embed_dataset (train_one_fold делает локальный from .embed_dataset import ...).
    import pandas as pd
    from dataclasses import replace
    import triton_crop.embed_dataset as ED
    from triton_crop.config import EmbedConfig
    from triton_crop.embed_train import assign_oof_folds, train_one_fold

    inds = ["X", "Y", "Z", "W", "U", "V"]                # >=2 сессий на особь (иначе SA вырождается в PK)
    g = pd.DataFrame({"md5": [f"{i}{s}" for i in inds for s in ("1", "2")],
                      "individual_id": [i for i in inds for _ in range(2)],
                      "cohort": ["TK"] * (2 * len(inds)),
                      "session": ["s1", "s2"] * len(inds)})
    p = pd.DataFrame({"md5": [f"{i}p" for i in inds], "individual_id": inds,
                      "cohort": ["TK"] * len(inds), "session": ["s2"] * len(inds)})
    all_md5 = list(g["md5"]) + list(p["md5"])
    crops = pd.DataFrame({"md5": all_md5, "variant": ["belly_oriented"] * len(all_md5),
                          "crop_path": [f"c/{m}.png" for m in all_md5],
                          "individual_id": list(g["individual_id"]) + list(p["individual_id"]),
                          "cohort": ["TK"] * len(all_md5)})
    oof = assign_oof_folds(g, p, crops, variants=("belly_oriented",), n_folds=3, seed=42)
    cfg = replace(EmbedConfig(), cross_fit_folds=3, batch_p=2, batch_k=2,
                  freeze_backbone_stages=0, augment=False, session_aware_sampling=True,
                  embed_variant="belly_oriented")        # обучаем по единственному собранному варианту

    seen = []
    RealSA, RealPK = ED.SessionAwarePKSampler, ED.PKSampler
    monkeypatch.setattr(ED, "SessionAwarePKSampler",
                        lambda *a, **k: seen.append(("SA", a)) or RealSA(*a, **k))
    monkeypatch.setattr(ED, "PKSampler",
                        lambda *a, **k: seen.append(("PK", a)) or RealPK(*a, **k))
    train_one_fold(oof, 0, cfg, build_embedder_fn=_tiny_factory,
                   image_loader=_fake_loader, max_steps=2)
    assert seen and seen[0][0] == "SA"                   # выбран session-aware (RED если всегда PKSampler)
    assert seen[0][1][1] is not None                     # 2-й позиционный арг = sessions, реально передан
