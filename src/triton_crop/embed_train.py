"""Cross-fit дообучение эмбеддера re-ID (Блок 4, Этап B) — оркестрация out-of-fold ПО ОСОБЯМ (анти-утечка по особям).

Граница как в проекте:
  • ЧИСТЫЕ (numpy/pandas, в шапке — тестируются без torch на synthetic_embed_dataset/mock_embedder):
      assign_oof_folds — плоский OOF-набор union(gallery+probe), общий fold по особям (анти-утечка через
                         probe — главный источник train-on-gallery сдвига 2.0); to_ab_inputs — выровненные
                         массивы под ab_harness.run_ab_analysis.
  • МОДЕЛЬНЫЕ (torch/timm импортируются ВНУТРИ функций; backbone_factory + image_loader инжектируются →
    локальный CPU-тест на крошечном backbone; тяжёлый прогон — на Colab GPU, Этап B):
      train_one_fold / extract_embeddings / cross_fit_embed / save_checkpoint / load_checkpoint.

Принцип честности: для фолда f эмбеддер учится ТОЛЬКО на gallery-кадрах особей ∉ f; затем им
эмбеддятся ВСЕ кадры (gallery+probe) особей ∈ f. Так каждый кадр эмбеддится моделью, не видевшей его
особь. raw НЕ дообучается (zero-shot baseline) — подаётся в to_ab_inputs отдельно.
"""
import numpy as np
import pandas as pd

from .embed_dataset import build_records, kfold_by_individual


def assign_oof_folds(gallery_rows, probe_rows, crops_df, variants=("belly_oriented", "unroll_ribbon"),
                     fallback: str = "belly_oriented", n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Плоский OOF-набор: по строке на кадр union(gallery+probe), с путями всех `variants` и общим
    cross_fit_fold ПО ОСОБЯМ. Фолд назначается на ОБЪЕДИНЕНИИ особей → gallery- и probe-кадры одной
    особи получают ОДИН фолд (инвариант анти-утечки). -> DataFrame[md5, individual_id, cohort, role,
    path_<variant>, used_<variant>, cross_fit_fold]."""
    frames = []
    for rows, role in ((gallery_rows, "gallery"), (probe_rows, "probe")):
        base = rows[["md5", "individual_id", "cohort"]].drop_duplicates("md5").copy()
        base["role"] = role
        if "session" in rows.columns:                    # C2: сессия для cross-session сэмплинга
            smap = rows.drop_duplicates("md5").set_index("md5")["session"]
            base["session"] = base["md5"].map(smap).fillna(base["md5"])
        else:
            base["session"] = base["md5"]                # нет дат (старые rows) → каждый кадр = своя сессия
        for v in variants:
            rec = build_records(crops_df, rows, variant=v, fallback=fallback)
            base = base.merge(
                rec[["md5", "crop_path", "variant_used"]].rename(
                    columns={"crop_path": f"path_{v}", "variant_used": f"used_{v}"}),
                on="md5", how="left")
        frames.append(base)
    flat = pd.concat(frames, ignore_index=True)
    # кадры без кропа (raw-only fallback — модель не нашла пузо) эмбеддить кропом нельзя → исключаем
    flat = flat.dropna(subset=[f"path_{v}" for v in variants]).reset_index(drop=True)
    ids = flat["individual_id"].to_numpy()
    fold_of = np.full(len(ids), -1, int)
    for f, (_, va) in enumerate(kfold_by_individual(ids, n_folds=n_folds, seed=seed)):
        fold_of[va] = f
    flat["cross_fit_fold"] = fold_of
    return flat


def to_ab_inputs(oof, raw_emb, embed_variant: str = "unroll_ribbon",
                 baseline_variant: str = "belly_oriented", seg_conf: float = 1.0):
    """OOF-эмбеддинги (dict вариант→(N,D) + role/individual_id/cohort) + raw_emb → (kwargs, extra_variants)
    для run_ab_analysis. seg_conf=1.0 на все кадры (кропы уже выбраны fallback'ом — порог raw-fallback не
    нужен). raw подаётся отдельно: он zero-shot baseline и в дообучении не участвует."""
    role = np.asarray(oof["role"])
    g, p = role == "gallery", role == "probe"
    bo = np.asarray(oof[baseline_variant], float)
    ids, coh = np.asarray(oof["individual_id"]), np.asarray(oof["cohort"])
    raw = np.asarray(raw_emb, float)
    conf = np.full(len(role), float(seg_conf))
    kw = dict(g_raw_e=raw[g], g_bo_e=bo[g], g_conf=conf[g], g_ids=ids[g],
              p_raw_e=raw[p], p_bo_e=bo[p], p_conf=conf[p], p_ids=ids[p], p_coh=coh[p])
    extra = None
    if embed_variant in oof and embed_variant != baseline_variant:
        var = np.asarray(oof[embed_variant], float)
        extra = {embed_variant: (var[g], var[p])}
    return kw, extra


def select_train_frames(oof, fold, train_all_sessions: bool = False):
    """Кадры для обучения фолда f: особи ∉ fold. По умолчанию — только role=gallery (поведение Блока 4).
    train_all_sessions=True → все сессии (gallery+probe) train-особей → cross-session позитивы для
    session-aware сэмплера (анти-утечка сохраняется: фолд по особям, оценочный фолд исключён)."""
    sel = oof["cross_fit_fold"] != fold
    if not train_all_sessions:
        sel = sel & (oof["role"] == "gallery")
    return oof[sel].reset_index(drop=True)


# ───────────────────────── МОДЕЛЬНЫЕ (lazy torch; инъекция backbone/loader) ─────────────────────────

def _default_build_embedder(name, device):
    from .embed_model import build_embedder
    return build_embedder(name, device, num_classes=0)


def train_one_fold(oof, fold: int, cfg, build_embedder_fn=None, image_loader=None, device: str = "cpu",
                   train_variant=None, augment: bool = False, epochs=None, max_steps=None) -> dict:
    """Дообучить ArcFace на gallery-кадрах особей ∉ fold (probe в обучение НЕ идёт). epochs (по умолчанию
    cfg.epochs) с линейным warmup и cosine-annealing; max_steps ограничивает шаги (CPU-тест). Детерминизм:
    torch.manual_seed(cfg.seed+fold). -> state {embedder(backbone), head, label_encoder, fold, embed_dim,
    train_individuals}."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from .embed_dataset import LabelEncoder, PKSampler, SessionAwarePKSampler
    from .embed_model import EmbedNet, build_param_groups, freeze_backbone_stages
    torch.manual_seed(int(cfg.seed) + int(fold))
    bef = build_embedder_fn or _default_build_embedder
    path_col = f"path_{train_variant or cfg.embed_variant}"
    train_df = select_train_frames(oof, fold, getattr(cfg, "train_all_sessions", False))
    le = LabelEncoder(train_df["individual_id"])
    emb = bef(cfg.base_model, device)
    backbone, transform = emb["model"], emb["transform"]
    embed_dim = getattr(backbone, "num_features", None)
    if embed_dim is None:                                    # MiewID (AutoModel) — определяем dim прогоном кадра
        with torch.no_grad():
            x0 = transform(Image.fromarray(image_loader(train_df.iloc[0][path_col]))).unsqueeze(0).to(device)
            embed_dim = int(backbone(x0).shape[1])
    net = EmbedNet(backbone, len(le), embed_dim=embed_dim,
                   margin=cfg.arcface_margin, scale=cfg.arcface_scale).to(device)
    freeze_backbone_stages(backbone, cfg.freeze_backbone_stages)
    opt = torch.optim.SGD(build_param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
                          momentum=cfg.momentum)
    n_epochs = 1 if max_steps is not None else int(epochs if epochs is not None else cfg.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs))
    base_lrs = [g["lr"] for g in opt.param_groups]
    net.train()
    step = 0
    train_log = []                                                       # логи сходимости (потолок vs недообучили)
    for ep in range(n_epochs):
        warm = bool(cfg.warmup_epochs) and ep < cfg.warmup_epochs        # линейный LR-warmup (анти-оверфит головы)
        if warm:
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * float(ep + 1) / float(cfg.warmup_epochs)
        ep_losses = []
        if getattr(cfg, "session_aware_sampling", False):
            sampler = SessionAwarePKSampler(train_df["individual_id"].to_numpy(),
                                            train_df["session"].to_numpy(),
                                            cfg.batch_p, cfg.batch_k, seed=int(cfg.seed) + ep)
        else:
            sampler = PKSampler(train_df["individual_id"].to_numpy(), cfg.batch_p, cfg.batch_k,
                                seed=int(cfg.seed) + ep)
        for idx in sampler:
            if max_steps is not None and step >= max_steps:
                break
            rows = train_df.iloc[idx]
            imgs = [image_loader(pth) for pth in rows[path_col]]
            if augment:
                from .embed_augment import augment_image
                imgs = [augment_image(im, seed=int(cfg.seed) + step + j) for j, im in enumerate(imgs)]
            x = torch.stack([transform(Image.fromarray(im)) for im in imgs]).to(device)
            y = torch.as_tensor(le.transform(rows["individual_id"]), device=device)
            loss = F.cross_entropy(net(x, y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
            step += 1
        lrs = [g["lr"] for g in opt.param_groups]                        # LR этой эпохи (до sched.step)
        train_log.append({"epoch": int(ep), "loss_mean": float(np.mean(ep_losses)) if ep_losses else 0.0,
                          "lr_head": float(max(lrs)), "lr_backbone": float(min(lrs)),
                          "n_steps": len(ep_losses), "warmup": bool(warm)})
        if not warm:
            sched.step()                                                 # cosine после warmup (T=epochs — урок 2.0)
        if max_steps is not None and step >= max_steps:
            break
    net.eval()
    return {"embedder": {"model": net.backbone, "transform": transform, "device": device},
            "head": net.head, "label_encoder": le, "fold": int(fold), "embed_dim": net.embed_dim,
            "epochs": int(n_epochs), "train_log": train_log,
            "train_individuals": set(train_df["individual_id"].tolist())}


def extract_embeddings(embedder, records, path_col: str, image_loader, batch_size: int = 16) -> np.ndarray:
    """records[path_col] → L2-norm эмбеддинги (N,D) в порядке строк (eval, no_grad, без аугментаций).
    Переиспользует embed_model.embed_images (probe и gallery — одним путём). image_loader(path) → HWC uint8."""
    from .embed_model import embed_images
    imgs = [image_loader(p) for p in records[path_col]]
    return embed_images(embedder, imgs, batch_size=batch_size)


def cross_fit_embed(oof, cfg, build_embedder_fn=None, image_loader=None, device: str = "cpu",
                    variants=("belly_oriented", "unroll_ribbon"), ckpt_dir=None, max_steps=None,
                    epochs=None, resume: bool = True) -> dict:
    """ГЛАВНАЯ оркестрация: для каждого фолда — обучить (train_one_fold) и эмбеддить held-out (особи фолда)
    его моделью. Раскладывает эмбеддинги в глобальные массивы по позиции строки oof. -> {<variant>: (N,D),
    md5, role, individual_id, cohort, embedded_by[i]=train-особи модели, эмбеддившей кадр i (для проверки анти-утечки)}.

    resume + ckpt_dir: если fold{f}.pt уже есть — грузим (не переобучаем) → переживает сброс Colab ~90 мин
    на уровне фолда (фолд на малом датасете << 90 мин). train_individuals восстанавливаются из oof.
    """
    from pathlib import Path
    md5 = oof["md5"].to_numpy()
    pos = {m: i for i, m in enumerate(md5)}
    per, embedded_by = {}, [None] * len(oof)
    train_logs = {}                                                      # логи сходимости по фолдам
    for f in range(cfg.cross_fit_folds):
        ckpt = Path(ckpt_dir) / f"fold{f}.pt" if ckpt_dir is not None else None
        train_inds = set(oof[(oof["role"] == "gallery") & (oof["cross_fit_fold"] != f)]["individual_id"])
        if ckpt is not None and resume and ckpt.exists():
            state = load_checkpoint(ckpt, cfg, build_embedder_fn, device)   # resume: готовый фолд не переобучаем
            state["train_individuals"] = train_inds
            print(f"[cross-fit] fold {f + 1}/{cfg.cross_fit_folds}: загружен из чекпойнта (resume)", flush=True)
        else:
            print(f"[cross-fit] fold {f + 1}/{cfg.cross_fit_folds}: обучение "
                  f"({int(epochs or cfg.epochs)} эпох · {len(train_inds)} особей)…", flush=True)
            state = train_one_fold(oof, f, cfg, build_embedder_fn, image_loader, device,
                                   augment=cfg.augment, epochs=epochs, max_steps=max_steps)
            if ckpt is not None:
                save_checkpoint(state, ckpt, cfg)
            print(f"[cross-fit] fold {f + 1}/{cfg.cross_fit_folds}: ✓ обучен", flush=True)
        train_logs[f] = state.get("train_log", [])
        held = oof[oof["cross_fit_fold"] == f]
        hp = [pos[m] for m in held["md5"]]
        for v in variants:
            e = extract_embeddings(state["embedder"], held, f"path_{v}", image_loader)
            if v not in per:
                per[v] = np.zeros((len(oof), e.shape[1]), np.float32)
            per[v][hp] = e
        for i in hp:
            embedded_by[i] = state["train_individuals"]
        del state                                            # освобождаем VRAM фолда перед следующим (T4 тесна)
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    if ckpt_dir is not None:                                  # сохранить логи сходимости рядом с чекпойнтами
        import json
        from pathlib import Path
        (Path(ckpt_dir) / "train_log.json").write_text(
            json.dumps(train_logs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**per, "md5": md5, "role": oof["role"].to_numpy(),
            "individual_id": oof["individual_id"].to_numpy(), "cohort": oof["cohort"].to_numpy(),
            "embedded_by": embedded_by, "train_log": train_logs}


def build_checkpoint_meta(state, cfg) -> dict:
    """Метаданные модели для чекпойнта (воспроизводимость + защита от путаницы Mega/MiewID).
    ЧИСТАЯ (без torch) → тестируема и переиспользуема при миграции существующих чекпойнтов."""
    n_classes = len(state["label_encoder"].classes_) if "label_encoder" in state else len(state.get("classes", []))
    return {"base_model": cfg.base_model, "image_size": int(cfg.image_size),
            "miewid_image_size": int(cfg.miewid_image_size), "miewid_revision": cfg.miewid_revision,
            "freeze_backbone_stages": int(cfg.freeze_backbone_stages),
            "epochs": int(state.get("epochs", cfg.epochs)), "seed": int(cfg.seed),
            "arcface_margin": float(cfg.arcface_margin), "arcface_scale": float(cfg.arcface_scale),
            "batch_p": int(cfg.batch_p), "batch_k": int(cfg.batch_k),
            "embed_dim": int(state["embed_dim"]), "n_classes": int(n_classes)}


def save_checkpoint(state, path, cfg) -> None:
    """Сохранить фолд-чекпойнт (backbone+head+classes+fold+embed_dim + model_meta + train_log). На Colab — в Drive."""
    from pathlib import Path

    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": state["embedder"]["model"].state_dict(), "head": state["head"].state_dict(),
                "classes": [str(c) for c in state["label_encoder"].classes_], "fold": int(state["fold"]),
                "embed_dim": int(state["embed_dim"]),
                "model_meta": build_checkpoint_meta(state, cfg),         # воспроизводимость
                "train_log": state.get("train_log", [])}, path)          # сходимость


def load_checkpoint(path, cfg, build_embedder_fn=None, device: str = "cpu", strict_meta: bool = True) -> dict:
    """Восстановить фолд-чекпойнт → state с embedder(backbone) для extract_embeddings (без переобучения).
    strict_meta: если в чекпойнте записан base_model и он НЕ совпадает с cfg.base_model — явная ошибка
    (загрузка MiewID-чекпойнта под Mega-конфиг раньше падала непонятно/тихо)."""
    import numpy as np
    import torch

    from .embed_model import EmbedNet
    bef = build_embedder_fn or _default_build_embedder
    # weights_only=True (защита от pickle-исполнения: чекпойнты качаются извне без sha256). safe_globals —
    # классы из старых чекпойнтов (label_encoder.classes_ = numpy.str_); новые чекпойнты чисто-типные.
    # Int64/Float64DType — запас под numpy-ЧИСЛОВЫЕ скаляры в meta сторонних/Colab-чекпойнтов
    # (иначе громкий UnpicklingError); собственные save_checkpoint кастуют всё в нативные типы.
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType,
                                           np.dtypes.Int64DType, np.dtypes.Float64DType]):
        ck = torch.load(path, map_location=device, weights_only=True)
    meta = ck.get("model_meta", {})
    if strict_meta and meta.get("base_model") and meta["base_model"] != cfg.base_model:
        raise ValueError(
            f"checkpoint model_meta.base_model={meta['base_model']!r} != cfg.base_model={cfg.base_model!r}: "
            f"смените конфиг под чекпойнт (ловушка Mega/MiewID.")
    emb = bef(cfg.base_model, device)
    net = EmbedNet(emb["model"], len(ck["classes"]), embed_dim=ck["embed_dim"],
                   margin=cfg.arcface_margin, scale=cfg.arcface_scale).to(device)
    net.backbone.load_state_dict(ck["backbone"])
    net.head.load_state_dict(ck["head"])
    net.eval()
    return {"embedder": {"model": net.backbone, "transform": emb["transform"], "device": device},
            "head": net.head, "classes": ck["classes"], "fold": ck["fold"], "embed_dim": ck["embed_dim"],
            "model_meta": meta, "train_log": ck.get("train_log", [])}
