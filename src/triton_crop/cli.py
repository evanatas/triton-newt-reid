"""CLI triton_crop — единые точки входа конвейера (Блоки 2–6 + sealed-test).

Карта команд (метки: [локально] · [Colab GPU] · [SEALED, --unseal] · [аудит, read-only] · [демо — app/demo.py]):

  ── Блок 2 (кроп брюшка) ──
  bootstrap          [локально] псевдо-метки (FastSAM 'everything' → pick_animal → pseudo_label) + overlays
  correct            [локально] мини-кликер ручной правки (cv2); --reseg — ре-сегментация по клику
  export             [локально] corrected-метки → Ultralytics seg+pose датасеты (сплит по особям)
  train-seg/-pose    [локально] обучение YOLO seg/pose (Ultralytics, MPS)
  crop               [локально/SEALED] canonical_belly_crop → crops_belly/ + crops_manifest (test/open_test — только --unseal)
  ab                 [локально] A/B Rosa (raw↔кроп↔unroll) + unroll_adopt_decision
  validate-crops     [аудит] гейты C1–C9 + EDA; --unsealed — финальный режим (манифест законно содержит test)
  ── Блок 4 (эмбеддер re-ID) ──
  embed-build        [локально] embed-records + label-map + cross-fit фолды по особям (stage∈{train,gallery})
  embed-train        [Colab GPU] finetune ArcFace (Этап B)
  embed-ab           [локально] строгий A/B finetuned: zero-shot↔finetuned, bo↔ribbon (Этап C)
  embed-eval-openset [локально] open-set AUROC + порог Юдена на open_dev (open_test НЕ вскрывать)
  embed-test         [SEALED, --unseal] ФИНАЛ ВКР: вскрыть test+open_test → ab_test_headline.json (1 раз, потом не тюнить)
  embed-verify       [аудит] read-only сверка метрики↔npy, md5-порядок, нет sealed в OOF (exit 1 при рассинхроне)
  embed-pack         [локально] zip-упаковка для Colab (артефакты+кропы+код+sha256)
  ── Блок 5 (матчинг пятен) ──
  detect-spots       [локально] детекция пятен → spots-манифест (поверхность belly_oriented/unroll_debend, НЕ ribbon)
  spot-validate      [аудит] S-гейты (поверхность≠ribbon, анти-утечка test, A/B-решение); --unsealed — финальный режим
  spot-ab            [локально] A/B матчер созвездия vs эмбеддер (grid + McNemar + Rosa + sensitivity-sweep)
  match              [локально] один probe → top-K + оверлеи совпавших пятен
  ── Блок 6 (гибрид) ──
  hybrid             [локально] эмбеддер top-K → матчер переранжирует (fusion A/B vs эмбеддер; --reuse-sims с провенансом)

Гейт C9 (sealed_test.assert_unsealed): test/open_test запечатаны — вскрытие только с --unseal (необратимо; после — не тюнить).
Не путать: --unseal = ДЕЙСТВИЕ (разрешить обработку запечатанных test/open_test — crop, embed-test, detect-spots, match) ·
--unsealed = СОСТОЯНИЕ аудита (validate-crops, spot-validate: манифест/spots ЗАКОННО содержат test после вскрытия — гейты C9/S7 не применять).
"""
import argparse
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "data"
_ART = _REPO / "artifacts"
_REPORTS = _REPO / "reports"


def _workspace():
    p = _REPO / "configs" / "paths.yaml"
    if not p.exists():       # paths.yaml в .gitignore → на свежем clone его нет; подсказываем шаблон
        raise SystemExit("configs/paths.yaml не найден — скопируйте configs/paths.example.yaml "
                         "в configs/paths.yaml и подставьте свои пути (см. шапку шаблона).")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return raw["workspace_root"]


def cmd_bootstrap(a):
    from .sam_bootstrap import run_bootstrap
    print("bootstrap:", run_bootstrap(
        _DATA / "manifest.csv", _workspace(), a.labels, a.overlays,
        per_species=a.per_species, cohorts=tuple(a.cohorts.split(",")),
        folds=tuple(a.folds.split(",")), weights=a.weights, device=a.device,
        reuse_labels_dir=a.reuse_labels))


def cmd_correct(a):
    from .corrector import run_corrector
    masker = None
    if a.reseg:
        from .sam_bootstrap import SamMasker
        masker = SamMasker(a.weights, device=a.device)
    run_corrector(a.labels, _workspace(), masker=masker, pending_only=a.pending)


def cmd_export(a):
    from .yolo_export import export_from_dir
    print("export:", export_from_dir(a.labels, _workspace(), a.out, val_frac=a.val_frac))


def cmd_train_seg(a):
    from .train import train_seg
    train_seg(a.data or (Path(a.out) / "seg" / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, device=a.device)


def cmd_train_pose(a):
    from .train import train_pose
    train_pose(a.data or (Path(a.out) / "pose" / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, device=a.device)


def cmd_validate_crops(a):
    import json

    import pandas as pd

    from .crops_manifest import read_crops_manifest
    from .validate import run_crop_gates, write_crop_eda
    from .config import load_crop_config
    crops = read_crops_manifest(_DATA)
    target = pd.read_csv(_DATA / "manifest.csv")
    ab_path = _ART / "ab_metrics.json"
    ab_metrics = json.loads(ab_path.read_text(encoding="utf-8")) if ab_path.exists() else None
    unsealed = getattr(a, "unsealed", False)        # финальный режим: манифест законно содержит test (после sealed-test)
    for w in run_crop_gates(crops, target, cfg=load_crop_config(), ab_metrics=ab_metrics, unsealed=unsealed):
        print("  ⚠", w)
    write_crop_eda(crops, _REPORTS)
    print(f"✓ Гейты Блока 2/3 пройдены{' (финальный режим --unsealed)' if unsealed else ''}. "
          f"EDA: {_REPORTS / 'crop_eda.md'}")


def cmd_spot_validate(a):
    """Блок 5: S-гейты матчинга по spots-манифестам + ab_spots_metrics (production gate).
    S6 поверхность≠ribbon · S7 анти-утечка test · S1 покрытие (warning) · S5 A/B-решение пересчитывается.
    ValidationError → ненулевой выход (гейт ВКР)."""
    import glob
    import json

    import pandas as pd

    from .validate import run_spot_gates
    files = sorted(glob.glob(str(_DATA / "spots_*.csv")))
    if not files:
        print("spot-validate: нет data/spots_*.csv (сначала detect-spots)"); return
    spots_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    manifest = pd.read_csv(_DATA / "manifest.csv")
    ab_path = _ART / "ab_spots_metrics.json"
    ab_metrics = json.loads(ab_path.read_text(encoding="utf-8")) if ab_path.exists() else None
    unsealed = getattr(a, "unsealed", False)
    for w in run_spot_gates(spots_df, manifest_df=manifest, ab_metrics=ab_metrics, unsealed=unsealed):
        print("  ⚠", w)
    print(f"✓ S-гейты Блока 5 пройдены ({len(files)} spots-файлов, {len(spots_df)} строк"
          f"{', финальный режим --unsealed' if unsealed else ''}).")


def cmd_crop(a):
    # Гейт C9 (единый, sealed_test.assert_unsealed): test/open_test ЗАПЕЧАТАНЫ — кропать только при явном
    # --unseal. Срабатывает ДО загрузки моделей.
    from .sealed_test import assert_unsealed
    assert_unsealed(a.stages.split(","), getattr(a, "unseal", False))

    import cv2
    import numpy as np
    import pandas as pd

    from triton_data.imageio import load_canonical
    from triton_data.loader import read_manifests, select

    from .config import load_crop_config
    from .crops_manifest import build_crops_manifest, write_crops_manifest
    from .pipeline import canonical_belly_crop
    from .predict import BellySegmenter, PoseEstimator

    from dataclasses import replace
    cfg = load_crop_config()
    if a.seg_conf_min is not None:
        cfg = replace(cfg, seg_conf_min=a.seg_conf_min)
    if a.pose_conf_min is not None:
        cfg = replace(cfg, pose_conf_min=a.pose_conf_min)
    seg = BellySegmenter(a.seg_weights, device=a.device, imgsz=a.imgsz, conf=a.conf)
    pose = PoseEstimator(a.pose_weights, device=a.device, imgsz=a.imgsz, conf=a.conf)
    target, external = read_manifests(_DATA)
    ws = Path(_workspace())
    crops_dir = _REPO / "crops_belly"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for stage in a.stages.split(","):
        rows = select(target, external, stage, scope=a.scope)
        results = []
        for _, row in rows.iterrows():
            rgb = np.array(load_canonical(ws / row["rel_path"]))
            res = canonical_belly_crop(rgb, seg, pose, cfg, src_md5=row["md5"])
            cv2.imwrite(str(crops_dir / f"{row['md5']}.png"), cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
            for m, img_u in (res.unroll_variants or {}).items():        # Блок 3: распрямлённые варианты
                cv2.imwrite(str(crops_dir / f"{row['md5']}__{m}.png"),
                            cv2.cvtColor(img_u, cv2.COLOR_RGB2BGR))
            results.append(res)
        frames.append(build_crops_manifest(rows, results, "crops_belly", pipeline_version=a.pipeline))
        print(f"  {stage}: {len(results)} кропов")
    full = (pd.concat(frames).drop_duplicates(["md5", "variant"])    # ключ (md5, variant)
            .sort_values(["md5", "variant"]).reset_index(drop=True))
    if getattr(a, "append", False):                                  # аддитивно: не стереть TK/PW
        from .crops_manifest import merge_crops_manifest, read_crops_manifest
        full = merge_crops_manifest(read_crops_manifest(_DATA), full)
    write_crops_manifest(full, _DATA)
    print("crop variants:", dict(full["variant"].value_counts()),
          "| статусы:", dict(full["crop_status"].value_counts()))


def cmd_ab(a):
    import csv
    import json

    import cv2
    import numpy as np

    from triton_data.imageio import load_canonical
    from triton_data.loader import read_manifests, select

    from .ab_harness import gate_passed, run_ab_analysis, unroll_adopt_explain
    from .crops_manifest import read_crops_manifest
    from .proxy_embed import build_proxy, embed

    target, external = read_manifests(_DATA)
    gallery = select(target, external, "gallery", scope=a.gallery_scope)   # C3: галерея уважает scope
    probe = select(target, external, "dev", scope=a.scope)
    crops = read_crops_manifest(_DATA)
    ws = Path(_workspace())

    def _paths(variant):
        sub = crops[crops["variant"] == variant]
        return dict(zip(sub["md5"], sub["crop_path"]))

    bo = crops[crops["variant"] == "belly_oriented"]
    bo_path, bo_conf = dict(zip(bo["md5"], bo["crop_path"])), dict(zip(bo["md5"], bo["seg_conf"]))
    unroll_vs = sorted(v for v in crops["variant"].unique() if str(v).startswith("unroll_"))  # Блок 3
    var_paths = {v: _paths(v) for v in unroll_vs}

    def _raw(rel):
        im = np.array(load_canonical(ws / rel))
        s = 512.0 / max(im.shape[:2])
        return cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s))) if s < 1 else im

    def _crop(path_map, md5):
        p = path_map.get(md5)
        if not p:
            return None
        im = cv2.imread(str(_REPO / p))
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    proxy = build_proxy(a.model, device=a.device)

    def collect(rows):
        recs = list(rows[["md5", "individual_id", "cohort", "rel_path"]].itertuples(index=False))
        raw_e = embed(proxy, [_raw(r.rel_path) for r in recs])
        conf = [float(bo_conf.get(r.md5, 0.0)) if _crop(bo_path, r.md5) is not None else 0.0 for r in recs]

        def var_emb(path_map, fallback_e):           # нет кропа варианта → fallback на fallback_e
            imgs = [_crop(path_map, r.md5) for r in recs]
            ve = fallback_e.copy()
            idx = [i for i, c in enumerate(imgs) if c is not None]
            if idx:
                ce = embed(proxy, [imgs[i] for i in idx])
                for j, i in enumerate(idx):
                    ve[i] = ce[j]
            return ve

        bo_e = var_emb(bo_path, raw_e)               # belly_oriented: fallback на raw (нет кропа пуза)
        # unroll: fallback на belly_oriented (нет распрямления = ориентированный кроп) → паритет галерей bo↔unroll
        var_es = {v: var_emb(var_paths[v], bo_e) for v in unroll_vs}
        return recs, raw_e, bo_e, np.asarray(conf, float), var_es

    print(f"A/B: gallery ({a.gallery_scope}) · probe-dev ({a.scope}) · unroll {unroll_vs} · {a.model}")
    g_recs, g_raw_e, g_bo_e, g_conf, g_var = collect(gallery)
    p_recs, p_raw_e, p_bo_e, p_conf, p_var = collect(probe)
    g_ids = [r.individual_id for r in g_recs]
    p_ids = [r.individual_id for r in p_recs]
    p_coh = [r.cohort for r in p_recs]
    extra = {v: (g_var[v], p_var[v]) for v in unroll_vs}
    ab = run_ab_analysis(g_raw_e, g_bo_e, g_conf, g_ids, p_raw_e, p_bo_e, p_conf, p_ids, p_coh,
                         thr=a.seg_conf_min, extra_variants=extra)
    pp = ab.pop("_perprobe")
    from .config import load_crop_config
    from .unroll import pattern_safe_methods
    cfg = load_crop_config()
    pattern_ok = pattern_safe_methods(cfg)                       # ИЗМЕРЕННАЯ pattern-safety, строго @5
    explain = unroll_adopt_explain(ab, pattern_ok=pattern_ok)
    decision = explain["decision"]
    ab["unroll_adopt_decision"] = decision
    ab["unroll_adopt_rationale"] = explain
    ab["pattern_safety_protocol"] = {"tol_frac": cfg.unroll_pattern_tol_frac, "methods": pattern_ok,
                                     "canon_size": cfg.canon_size,
                                     "scenario": "synthetic переменной ширины + пятна; смещение от оси ≤ tol"}
    ab["protocol"] = {"gallery_scope": a.gallery_scope, "probe_scope": a.scope,
                      "n_gallery": len(g_ids), "n_probe": len(p_ids), "model": a.model,
                      "headline_seg_conf_min": a.seg_conf_min, "unroll_variants": unroll_vs,
                      "note": "pilot: галерея БЕЗ cross-fit (train-on-gallery сдвиг ещё возможен)"}

    with open(_ART / "ab_detailed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["probe_md5", "individual_id", "cohort", "true_id_in_gallery",
                "raw_hit@1", "crop_hit@1", "raw_hit@5", "crop_hit@5"]
        for v in unroll_vs:
            cols += [f"{v}_hit@1", f"{v}_hit@5"]
        w.writerow(cols)
        for i, r in enumerate(p_recs):
            rowv = [r.md5, r.individual_id, r.cohort, int(pp["in_gallery"][i]),
                    int(pp["raw"][1][i]), int(pp["crop"][1][i]), int(pp["raw"][5][i]), int(pp["crop"][5][i])]
            for v in unroll_vs:
                rowv += [int(pp[v][1][i]), int(pp[v][5][i])]
            w.writerow(rowv)
    (_ART / "ab_metrics.json").write_text(json.dumps(ab, ensure_ascii=False, indent=2), encoding="utf-8")

    o_r, o_c = ab["raw"]["overall"], ab["belly_oriented"]["overall"]
    print(f"[FULL n={o_r['n']}]  raw @1={o_r['recall@1']:.3f} @5={o_r['recall@5']:.3f}  |  "
          f"belly_oriented @1={o_c['recall@1']:.3f} @5={o_c['recall@5']:.3f}")
    for v in unroll_vs:
        ov = ab[v]["overall"]
        s1, s5 = ab.get(f"stats_{v}_vs_bo@1", {}), ab.get(f"stats_{v}_vs_bo@5", {})
        print(f"  {v}: @1={ov['recall@1']:.3f} @5={ov['recall@5']:.3f}  vs bo → "
              f"@1[c={s1.get('mcnemar_c')} p={s1.get('mcnemar_p')}] @5[c={s5.get('mcnemar_c')} p={s5.get('mcnemar_p')}]")
    s = ab["stats_closed@1"]
    print(f"McNemar(bo vs raw) b={s['mcnemar_b']} c={s['mcnemar_c']} p={s['mcnemar_p']}  Δ={s['bootstrap_diff']} CI{s['bootstrap_ci95']}")
    print("ГЕЙТ Блока 2 (bo≥raw):", "✅" if gate_passed(ab) else "❌")
    print("РЕШЕНИЕ Блока 3 (вход эмбеддера):", decision, "—", explain["rule"])


# ───────────────────────── Блок 4 (эмбеддер re-ID) ─────────────────────────

def cmd_embed_build(a):
    """Собрать embed-records + label-map + cross-fit фолды ПО ОСОБЯМ (чисто, детерминированно). Локально.

    Производит artifacts/embed/{embed_records.csv (+cross_fit_fold), label_map.json, cross_fit_folds.json}.
    Повторный запуск при том же seed → идентичные фолды/мапа (verification Этапа A).
    """
    import json

    import numpy as np

    from triton_data.loader import read_manifests, select

    from .config import load_embed_config
    from .crops_manifest import read_crops_manifest
    from .embed_dataset import LabelEncoder, build_records, kfold_by_individual
    from .sealed_test import assert_unsealed
    assert_unsealed({a.stage}, False)         # гейт C9: build обучающих артефактов из test/open_test ЗАПРЕЩЁН
    cfg = load_embed_config()
    target, external = read_manifests(_DATA)
    rows = select(target, external, a.stage, scope=a.scope)
    crops = read_crops_manifest(_DATA)
    rec = build_records(crops, rows, variant=cfg.embed_variant, fallback=cfg.fallback_variant)
    ids = rec["individual_id"].to_numpy()
    folds = kfold_by_individual(ids, n_folds=cfg.cross_fit_folds, seed=cfg.seed)
    fold_of = np.full(len(ids), -1, int)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f
    rec = rec.assign(cross_fit_fold=fold_of)
    out = _ART / "embed"
    out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out / "embed_records.csv", index=False, encoding="utf-8")
    le = LabelEncoder(ids)
    (out / "label_map.json").write_text(
        json.dumps({"classes": le.classes_, "n_classes": len(le)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    meta = [{"fold": f, "n_train": int(len(tr)), "n_val": int(len(va)),
             "val_individuals": sorted(set(ids[va].tolist()))} for f, (tr, va) in enumerate(folds)]
    (out / "cross_fit_folds.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # OOF-набор для cross-fit (gallery train + dev probe, ОБЩИЙ fold ПО ОСОБЯМ — анти-утечка через probe)
    from .embed_train import assign_oof_folds
    dev_rows = select(target, external, "dev", scope=a.scope)
    oof = assign_oof_folds(rows, dev_rows, crops, variants=(cfg.fallback_variant, cfg.embed_variant),
                           fallback=cfg.fallback_variant, n_folds=cfg.cross_fit_folds, seed=cfg.seed)
    oof.to_csv(out / "oof_records.csv", index=False, encoding="utf-8")
    print(f"embed-build: {len(rec)} записей · {len(le)} особей · {cfg.cross_fit_folds} фолдов "
          f"· variant={cfg.embed_variant} (fallback {cfg.fallback_variant}) → {out}")
    print("  variant_used:", {k: int(v) for k, v in rec["variant_used"].value_counts().items()})
    print(f"  oof_records: {len(oof)} кадров (gallery+dev) · роли "
          f"{ {k: int(v) for k, v in oof['role'].value_counts().items()} } · общий fold по особям")


def cmd_embed_train(a):
    """Дообучение эмбеддера (ArcFace finetune) — тяжёлый GPU-шаг, Этап B (Colab). Здесь печатает рецепт."""
    from .config import load_embed_config
    cfg = load_embed_config()
    print("embed-train: дообучение эмбеддера — Этап B (Colab GPU, см. notebooks/block4_colab.ipynb).")
    print(f"  модель={cfg.base_model} · ArcFace(m={cfg.arcface_margin}, s={cfg.arcface_scale}) · "
          f"{cfg.optimizer} mom={cfg.momentum} · cosine · epochs={cfg.epochs} (warmup {cfg.warmup_epochs})")
    print(f"  P×K={cfg.batch_p}×{cfg.batch_k} · freeze_stages={cfg.freeze_backbone_stages} · "
          f"augment={cfg.augment} · cross-fit={cfg.cross_fit_folds} · чекпойнты→{cfg.ckpt_dir}")


def _to_jsonable(o):
    """numpy-типы → JSON (для сериализации ab-словаря)."""
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def cmd_embed_ab(a):
    """Этап C: строгий A/B на дообученном эмбеддере. Берёт OOF finetuned-эмбеддинги с Colab
    (artifacts/embed/oof/*.npy) + считает zero-shot baseline на ТЕХ ЖЕ кропах (proxy MegaDescriptor) →
    build_finetuned_ab: zero-shot↔finetuned + belly_oriented↔ribbon на finetuned + честные решения.
    Пишет artifacts/ab_embed_metrics.json + ab_embed_detailed.csv (rank/score/margin)."""
    import json

    import cv2
    import numpy as np
    import pandas as pd

    from .config import load_embed_config
    from .embed_ab import ab_detailed, build_finetuned_ab, load_oof_npy
    from .proxy_embed import build_proxy, embed
    cfg = load_embed_config()
    oof = load_oof_npy(Path(a.oof_dir), variants=("belly_oriented", cfg.embed_variant))
    rec = pd.read_csv(_ART / "embed" / "oof_records.csv").drop_duplicates("md5").set_index("md5")
    proxy = build_proxy(a.model, device=a.device)

    def _zeroshot(col):                                   # zero-shot эмбеддинги по тем же кропам (порядок oof)
        imgs = []
        for m in oof["md5"]:
            im = cv2.imread(str(_REPO / rec.loc[m, col]))
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return embed(proxy, imgs)

    print(f"embed-ab: zero-shot baseline ({a.model}) на {len(oof['md5'])} кропах…")
    zero_bo = _zeroshot("path_belly_oriented")
    zero_rib = _zeroshot(f"path_{cfg.embed_variant}")
    ab = build_finetuned_ab(oof, zero_bo, zero_ribbon=zero_rib, embed_variant=cfg.embed_variant,
                            thr=a.seg_conf_min)
    role = np.asarray(oof["role"]); g = role == "gallery"; p = role == "probe"
    det = ab_detailed(oof[cfg.embed_variant][p], oof["individual_id"][p],
                      oof[cfg.embed_variant][g], oof["individual_id"][g], probe_md5=oof["md5"][p])
    det.to_csv(_ART / "ab_embed_detailed.csv", index=False, encoding="utf-8")
    (_ART / "ab_embed_metrics.json").write_text(
        json.dumps(ab, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")

    r, b, v = ab["raw"]["overall"], ab["belly_oriented"]["overall"], ab[cfg.embed_variant]["overall"]
    print(f"  zero-shot  bo     @1={r['recall@1']:.3f} @5={r['recall@5']:.3f}")
    print(f"  finetuned  bo     @1={b['recall@1']:.3f} @5={b['recall@5']:.3f}")
    print(f"  finetuned  {cfg.embed_variant} @1={v['recall@1']:.3f} @5={v['recall@5']:.3f}")
    print("  РЕШЕНИЕ finetune≻zero-shot (bo):", ab["adopt_finetuned_bo"]["decision"],
          "| ribbon≻bo на finetuned:", ab["unroll_adopt_decision"])
    print("  →", _ART / "ab_embed_metrics.json", "+ ab_embed_detailed.csv")


def cmd_embed_eval_openset(a):
    """Этап C: open-set known/new на дообученном эмбеддере. Грузит fold-чекпойнт (скачанный с Colab),
    эмбеддит галерею known + пробы (dev known + open_dev new), считает AUROC + порог Юдена на open_dev.
    Требует: (1) кропы open_dev (`cli crop --stages open_dev`); (2) чекпойнт artifacts/embed/ckpt/foldN.pt.
    open_test НЕ вскрывать."""
    import json

    import cv2
    import numpy as np

    from triton_data.loader import read_manifests, select

    from .config import load_embed_config
    from .crops_manifest import read_crops_manifest, select_crops_with_fallback
    from .embed_eval import apply_policy, known_new_scores, open_set_auroc, tune_threshold
    from .embed_train import extract_embeddings, load_checkpoint
    cfg = load_embed_config()
    state = load_checkpoint(Path(a.ckpt), cfg, device=a.device)
    target, external = read_manifests(_DATA)
    crops = read_crops_manifest(_DATA)

    def _loader(p):
        im = cv2.imread(str(_REPO / p))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def _emb(rows):
        sel = select_crops_with_fallback(crops, rows, variant=cfg.embed_variant, fallback=cfg.fallback_variant)
        sel = sel[sel["variant_used"].notna()].reset_index(drop=True)
        return extract_embeddings(state["embedder"], sel, "crop_path", _loader), sel["individual_id"].to_numpy()

    g_emb, g_ids = _emb(select(target, external, "gallery", scope=a.scope))
    d_emb, _ = _emb(select(target, external, "dev", scope=a.scope))           # known (есть в галерее)
    o_emb, _ = _emb(select(target, external, "open_dev"))                     # new (особей нет в галерее)
    probe = np.vstack([d_emb, o_emb])
    is_new = np.array([False] * len(d_emb) + [True] * len(o_emb))
    sc = known_new_scores(probe, g_emb, g_ids)
    thr = tune_threshold(sc["max_sim"], is_new)
    known = apply_policy(sc["max_sim"], sc["margin"], thr, margin_min=cfg.open_set_margin_min)
    out = {"auroc": float(open_set_auroc(sc["max_sim"], is_new)), "threshold": float(thr),
           "n_known": int((~is_new).sum()), "n_new": int(is_new.sum()),
           "known_kept_rate": float(known[~is_new].mean()) if (~is_new).any() else None,
           "new_falsely_known_rate": float(known[is_new].mean()) if is_new.any() else None,
           "ckpt": str(a.ckpt), "embed_variant": cfg.embed_variant, "margin_min": cfg.open_set_margin_min}
    (_ART / "openset_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"open-set: AUROC={out['auroc']:.3f} · порог={out['threshold']:.3f} · known={out['n_known']} "
          f"new={out['n_new']} · known-удержано={out['known_kept_rate']} "
          f"new-ложно-known={out['new_falsely_known_rate']}")
    print("  →", _ART / "openset_metrics.json", "(open_test НЕ вскрыт)")


def cmd_embed_test(a):
    """ФИНАЛ ВКР — sealed-test (НЕОБРАТИМО): вскрыть запечатанные test+open_test и посчитать headline-числа
    заявленной лучшей системы (zero-shot MegaDescriptor + unroll_ribbon). Identity-level top-1/5 (overall +
    per-cohort TK(temporal)/PW + coverage) против галереи + open-set (known=test, new=open_test; AUROC +
    политика при пороге, калиброванном на dev+open_dev — НЕ на test). Требует --unseal (гейт C9).
    Пишет artifacts/ab_test_headline.json + artifacts/embed/heldout/*.npy (belly_oriented zero-shot для демо).
    После вскрытия настройки НЕ тюнить — числа финальные."""
    from .sealed_test import SEALED_STAGES, assert_unsealed
    assert_unsealed(SEALED_STAGES, getattr(a, "unseal", False))   # гейт C9 ДО любых тяжёлых импортов/моделей
    # Guard повторного вскрытия: финальные числа уже зафиксированы — не затирать без явного --force-rerun-sealed.
    # Маркер вскрытия — И headline-json, И heldout-npy (живые npy при удалённом json тоже блокируют перепрогон).
    headline_path = _ART / "ab_test_headline.json"
    heldout_dir = _ART / "embed" / "heldout"
    opened = headline_path.exists() or (heldout_dir.exists() and any(heldout_dir.glob("*.npy")))
    if opened and not getattr(a, "force_rerun_sealed", False):
        raise ValueError(f"{headline_path.name} и/или artifacts/embed/heldout/*.npy уже существуют — "
                         f"sealed-test вскрыт, числа ФИНАЛЬНЫЕ. "
                         f"Повторный прогон затрёт финал (запрещено тюнить настройки на test). "
                         f"Если действительно нужно пересчитать — явный флаг --force-rerun-sealed.")
    import json

    import numpy as np

    from triton_data.loader import read_manifests, select

    from .config import load_embed_config
    from .crops_manifest import read_crops_manifest, select_crops_with_fallback
    from .embed_eval import known_new_scores, tune_threshold
    from .proxy_embed import build_proxy, embed
    from .sealed_test import build_sealed_report

    cfg = load_embed_config()
    head_v, base_v = cfg.embed_variant, cfg.fallback_variant          # unroll_ribbon (headline) + belly_oriented
    target, external = read_manifests(_DATA)
    crops = read_crops_manifest(_DATA)
    proxy = build_proxy(a.model, device=a.device)

    def _emb(rows, variant):                                          # zero-shot эмбеддинги kropов (fallback→bo)
        sel = select_crops_with_fallback(crops, rows, variant=variant, fallback=base_v)
        sel = sel[sel["variant_used"].notna()].reset_index(drop=True)
        imgs = [_load_crop_rgb(p) for p in sel["crop_path"]]
        keep = [i for i, im in enumerate(imgs) if im is not None]
        sel = sel.iloc[keep].reset_index(drop=True)
        e = embed(proxy, [imgs[i] for i in keep]) if keep else np.zeros((0, 1536), np.float32)
        return e, sel["individual_id"].to_numpy(), sel["cohort"].to_numpy(), sel["md5"].to_numpy()

    g_rows = select(target, external, "gallery", scope=a.scope)
    t_rows = select(target, external, "test", scope=a.scope)          # 113 — ВСКРЫВАЕМ
    o_rows = select(target, external, "open_test")     # 21 новых — ВСКРЫВАЕМ; scope для open-стадий игнорируется loader-ом (берутся все is_new_open fold=test)
    d_rows = select(target, external, "dev", scope=a.scope)           # для калибровки порога (не sealed)
    od_rows = select(target, external, "open_dev")

    report = {"protocol": {"system": "zero-shot MegaDescriptor + unroll_ribbon (заявленная лучшая система)",
                           "model": a.model, "scope": a.scope, "headline_variant": head_v,
                           "secondary_variant": base_v, "sealed_unsealed": True,
                           "note": "ВСКРЫТЫ test+open_test (необратимо). Числа финальные; настройки НЕ тюнить. "
                                   "Порог open-set калиброван на dev+open_dev, НЕ на test."},
              "by_variant": {}}
    heldout = None
    for v in dict.fromkeys((head_v, base_v)):                         # ribbon (headline) + bo (secondary), без дублей
        g_e, g_id, _, _ = _emb(g_rows, v)
        t_e, t_id, t_coh, t_md5 = _emb(t_rows, v)
        o_e, o_id, o_coh, o_md5 = _emb(o_rows, v)
        thr = None                                                   # калибровка порога на dev+open_dev (НЕ на test)
        d_e, *_ = _emb(d_rows, v)
        od_e, *_ = _emb(od_rows, v)
        if len(od_e) and len(d_e):
            sc = known_new_scores(np.vstack([d_e, od_e]), g_e, g_id)
            is_new_dev = np.array([False] * len(d_e) + [True] * len(od_e))
            thr = float(tune_threshold(sc["max_sim"], is_new_dev))
        rep = build_sealed_report(t_e, t_id, t_coh, o_e if len(o_e) else None, g_e, g_id,
                                  ks=(1, 5), n_official_test=len(t_rows), openset_threshold=thr,
                                  margin_min=cfg.open_set_margin_min)
        rep["n_gallery"], rep["n_test"], rep["n_open_test"] = int(len(g_id)), int(len(t_id)), int(len(o_id))
        report["by_variant"][v] = rep
        if v == base_v:                                              # held-out для демо = belly_oriented (zero-shot)
            heldout = {"emb": np.vstack([t_e, o_e]) if len(o_e) else t_e,
                       "md5": np.concatenate([t_md5, o_md5]), "individual_id": np.concatenate([t_id, o_id]),
                       "cohort": np.concatenate([t_coh, o_coh]),
                       "is_new": np.array([False] * len(t_md5) + [True] * len(o_md5))}
    report["headline"] = report["by_variant"][head_v]

    (_ART / "ab_test_headline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")
    ho = _ART / "embed" / "heldout"; ho.mkdir(parents=True, exist_ok=True)
    np.save(ho / f"{base_v}.npy", np.asarray(heldout["emb"], np.float32))
    for col in ("md5", "individual_id", "cohort", "is_new"):
        np.save(ho / f"{col}.npy", heldout[col])

    h = report["headline"]["recall"]; ops = report["headline"]["openset"]
    print(f"SEALED-TEST (zero-shot Mega + {head_v}) — ФИНАЛ ВКР:")
    print(f"  overall  @1={h['overall']['recall@1']:.3f} @5={h['overall']['recall@5']:.3f}  "
          f"(n={h['overall']['n']}, gallery={report['headline']['n_gallery']})")
    for c, blk in h["per_cohort"].items():
        print(f"  {c:4s}    @1={blk['recall@1']:.3f} @5={blk['recall@5']:.3f}  (n={blk['n']})"
              + ("   ← temporal" if c == "TK" else ""))
    print(f"  open-set AUROC={ops['auroc']} · known={ops['n_known']} new={ops['n_new']} · "
          f"порог(dev)={ops['threshold']} known-удержано={ops['known_kept_rate']} "
          f"new-ложно-known={ops['new_falsely_known_rate']}")
    print("  →", _ART / "ab_test_headline.json", "+ artifacts/embed/heldout/*.npy")
    print("  ⚠ test/open_test ВСКРЫТЫ. Настройки НЕ тюнить — числа финальные.")


def _load_crop_rgb(rel):
    """Кроп (PNG) → RGB uint8. None если файла нет."""
    import cv2
    im = cv2.imread(str(_REPO / rel))
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _spots_for_stage(cfg, stage, scope, crops, target, external):
    """Детектировать пятна на кропах stage (variant cfg.detect_variant, fallback). Маска = непустые
    (не чёрные) пиксели кропа. -> DataFrame[md5, individual_id, cohort, split_fold, n_spots, spots(list[Spot])]."""
    import pandas as pd

    from triton_data.loader import select

    from .crops_manifest import select_crops_with_fallback
    from .masks import foreground_from_crop
    from .spots import detect_spots
    rows = select(target, external, stage) if stage in ("open_dev", "open_test") \
        else select(target, external, stage, scope=scope)
    sel = select_crops_with_fallback(crops, rows, variant=cfg.detect_variant, fallback=cfg.fallback_variant)
    sel = sel[sel["variant_used"].notna()].reset_index(drop=True)
    recs = []
    for _, r in sel.iterrows():
        rgb = _load_crop_rgb(r["crop_path"])
        if rgb is None:
            continue
        # foreground через flood-fill фона (сохраняет внутренние тёмные пятна), НЕ rgb.sum(2)>30
        spots = detect_spots(rgb, foreground_from_crop(rgb, cfg.bg_sum_thr), cfg)
        recs.append({"md5": r["md5"], "individual_id": r.get("individual_id"), "cohort": r.get("cohort"),
                     "split_fold": r.get("split_fold"), "n_spots": len(spots), "spots": spots,
                     "crop_path": r["crop_path"],
                     # метаколонки для S-гейтов и аудита
                     "detect_variant": cfg.detect_variant, "detect_method": cfg.detect_method,
                     "illum_norm": bool(cfg.illum_norm)})
    return pd.DataFrame(recs)


def cmd_detect_spots(a):
    """Блок 5: детектировать пятна на кропах stage(ов) → spots-манифест (+ опц. QA-оверлеи).
    Поверхность belly_oriented/unroll_debend (НЕ ribbon). Маска = непустые пиксели кропа."""
    import json
    from dataclasses import replace

    from triton_data.loader import read_manifests

    from .config import load_spot_config
    from .crops_manifest import read_crops_manifest
    from .sealed_test import assert_unsealed
    assert_unsealed(a.stages.split(","), getattr(a, "unseal", False))   # гейт C9: не детектить пятна на test
    cfg = replace(load_spot_config(), detect_variant=a.variant, detect_method=a.method,
                  illum_norm=a.illum_norm)
    target, external = read_manifests(_DATA)
    crops = read_crops_manifest(_DATA)
    frames = []
    for stage in a.stages.split(","):
        df = _spots_for_stage(cfg, stage, a.scope, crops, target, external)
        df["stage"] = stage
        frames.append(df)
    import pandas as pd
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = full.copy()
    if len(out):
        out["spots"] = out["spots"].apply(
            lambda ss: json.dumps([[round(s.x, 2), round(s.y, 2), round(s.area, 1), round(s.score, 3)] for s in ss]))
        out = out.drop(columns=["crop_path"], errors="ignore")
    dst = _DATA / f"spots_{cfg.detect_variant}_{cfg.detect_method}.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    med = int(full["n_spots"].median()) if len(full) else 0
    cov = float((full["n_spots"] >= cfg.min_spots_for_match).mean()) if len(full) else 0.0
    print(f"detect-spots: {len(full)} кропов · variant={cfg.detect_variant} method={cfg.detect_method} · "
          f"медиана пятен {med} · покрытие(>={cfg.min_spots_for_match}) {cov:.0%} → {dst}")


def cmd_spot_ab(a):
    """Блок 5: строгий A/B матчера созвездия vs эмбеддер (baseline). Сетка детектор×матчер×поверхность
    (grid_ab) → лучший по recall@1 на dev → McNemar vs эмбеддер (overall + TK temporal + PW) + Rosa-решение.
    Эмбеддер-baseline берётся из готовых OOF (artifacts/embed/oof, Mega) выравниванием по md5. test НЕ трогаем."""
    import json
    from dataclasses import replace

    import numpy as np

    from triton_data.loader import read_manifests, select

    from .config import load_spot_config
    from .constellation import build_match_sim_matrix, match_constellations
    from .crops_manifest import read_crops_manifest
    from .embed_ab import load_oof_npy
    from .spot_ab import (
        adopt_matcher, compare_matcher_vs_embedder, detailed_rows, grid_ab, recall_split_sim, summarize_sim,
    )
    base = load_spot_config()
    target, external = read_manifests(_DATA)
    crops = read_crops_manifest(_DATA)
    surfaces = a.surfaces.split(",")
    methods = a.methods.split(",")
    matchers = a.matchers.split(",")

    # эмбеддер-baseline (Mega OOF, вариант shortlist) → sim + выравнивание по md5
    oof = load_oof_npy(Path(a.oof_dir), variants=("belly_oriented", a.embed_variant))
    oof_md5 = {m: i for i, m in enumerate(oof["md5"])}
    emb = np.asarray(oof[a.embed_variant], float)
    role = np.asarray(oof["role"])

    # пятна по поверхностям×методам (один раз на (surface,method))
    spot_dfs = {}
    for surf in surfaces:
        for meth in methods:
            cfg = replace(base, detect_variant=surf, detect_method=meth, ransac_iters=a.ransac_iters,
                          illum_norm=a.illum_norm)
            g = _spots_for_stage(cfg, "gallery", a.scope, crops, target, external)
            p = _spots_for_stage(cfg, "dev", a.scope, crops, target, external)
            spot_dfs[(surf, meth)] = (cfg, g, p)

    # общий набор probe/gallery: пятна И присутствие в OOF (одни md5 для матчера и эмбеддера)
    any_g = next(iter(spot_dfs.values()))[1]; any_p = next(iter(spot_dfs.values()))[2]
    g_md5 = [m for m in any_g["md5"] if m in oof_md5 and role[oof_md5[m]] == "gallery"]
    p_md5 = [m for m in any_p["md5"] if m in oof_md5 and role[oof_md5[m]] == "probe"]
    gset, pset = set(g_md5), set(p_md5)

    def _aligned(df, keep):
        d = df[df["md5"].isin(keep)].drop_duplicates("md5").set_index("md5")
        return d

    # ids/cohort — одни для всех комбо (общий md5-порядок g_md5/p_md5)
    gi0, pi0 = _aligned(any_g, gset), _aligned(any_p, pset)
    g_ids = gi0.loc[g_md5, "individual_id"].to_numpy()
    p_ids = pi0.loc[p_md5, "individual_id"].to_numpy()
    p_coh = pi0.loc[p_md5, "cohort"].to_numpy()

    sims = {}
    for (surf, meth), (cfg, g, p) in spot_dfs.items():
        gi, pi = _aligned(g, gset), _aligned(p, pset)
        g_spots = [gi.loc[m, "spots"] for m in g_md5]
        p_spots = [pi.loc[m, "spots"] for m in p_md5]
        for matcher in matchers:
            mcfg = replace(cfg, match_method=matcher)
            name = f"{surf}|{meth}|{matcher}"
            print(f"  матч {name}: {len(p_md5)}×{len(g_md5)} (ransac_iters={mcfg.ransac_iters})…", flush=True)
            sims[name] = build_match_sim_matrix(p_spots, g_spots, mcfg)

    grid = grid_ab(sims, p_ids, g_ids, p_coh, ks=tuple(base.recall_ks), primary_k=1)
    best_name = grid["best"]
    best_sim = sims[best_name]

    # спот-листы лучшего combo (для matched_counts и sensitivity sweep)
    b_surf, b_meth, _bm = best_name.split("|")
    b_cfg, _bg, _bp = spot_dfs[(b_surf, b_meth)]
    _bgi, _bpi = _aligned(_bg, gset), _aligned(_bp, pset)
    bg_spots = [_bgi.loc[m, "spots"] for m in g_md5]
    bp_spots = [_bpi.loc[m, "spots"] for m in p_md5]

    # эмбеддер-baseline на ТЕХ ЖЕ probe/gallery
    emb_g = emb[[oof_md5[m] for m in g_md5]]
    emb_p = emb[[oof_md5[m] for m in p_md5]]
    cmp = compare_matcher_vs_embedder(best_sim, emb_p, emb_g, p_ids, g_ids, p_coh, ks=tuple(base.recall_ks))
    cmp.pop("_perprobe", None)
    dec = adopt_matcher(cmp, primary_k=1, ks=tuple(base.recall_ks))
    # denominator pipeline_recall = ОФИЦИАЛЬНЫЙ dev (а не выровненный 128), список исключённых md5
    dev_official = select(target, external, "dev", scope=a.scope)
    n_official = int(len(dev_official))
    excluded = sorted(set(dev_official["md5"].astype(str)) - set(map(str, p_md5)))
    split = recall_split_sim(best_sim, p_ids, g_ids, ks=tuple(base.recall_ks), n_official_probe=n_official)
    split["excluded_md5"] = excluded
    split["excluded_cause"] = "нет пятен / нет в OOF — исключены из выровненного matcher↔embedder набора"
    per_cohort = summarize_sim(best_sim, p_ids, g_ids, p_coh, ks=tuple(base.recall_ks))

    # per-probe detailed CSV (аудит: ранги матчера/эмбеддера, hit@k, n пятен)
    import pandas as pd
    n_spots_probe = [int(pi0.loc[m, "n_spots"]) for m in p_md5]
    matched_counts = [len(match_constellations(bp_spots[i], bg_spots[int(np.argmax(best_sim[i]))], b_cfg)[1])
                      for i in range(len(p_md5))]     # совпавших пар probe ↔ его top-1 gallery-фото
    rows = detailed_rows(best_sim, emb_p, emb_g, p_ids, g_ids, list(p_md5), list(p_coh),
                         n_spots_probe=n_spots_probe, matched_counts=matched_counts, ks=tuple(base.recall_ks))
    pd.DataFrame(rows).to_csv(_ART / "ab_spots_detailed.csv", index=False, encoding="utf-8")

    out = {"grid": grid, "best": best_name, "compare": cmp, "matcher_adopt_decision": dec["decision"],
           "matcher_adopt_rationale": dec, "per_cohort": per_cohort, "recall_split": split,
           "embed_variant": a.embed_variant, "recall_ks": list(base.recall_ks),
           "n_probe": len(p_md5), "n_gallery": len(g_md5), "n_official_dev": n_official}
    (_ART / "ab_spots_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")
    print(f"spot-ab: лучший={best_name} | matcher @1={cmp['matcher']['overall']['recall@1']:.3f} "
          f"vs embedder @1={cmp['embedder']['overall']['recall@1']:.3f} | "
          f"McNemar p={cmp[f'stats_matcher_vs_embedder@1']['mcnemar_p']} | решение={dec['decision']}")
    print("  →", _ART / "ab_spots_metrics.json", "+ ab_spots_detailed.csv")

    # --- sensitivity по числу итераций RANSAC + guided (контроль iteration-starvation: результат НЕ «голодный по итерациям») ---
    if getattr(a, "sweep_iters", ""):
        import time
        sweep = {"best_surface_method": f"{b_surf}|{b_meth}", "runtime_s": {}, "recall": {}}
        modes = [("guided", None)] + [("ransac", int(x)) for x in a.sweep_iters.split(",") if x.strip()]
        for mm, it in modes:
            key = f"{mm}@{it}" if it else mm
            scfg = replace(b_cfg, match_method=mm, ransac_iters=(it or b_cfg.ransac_iters))
            t0 = time.perf_counter()
            ssim = best_sim if (mm == "guided" and _bm == "guided") else build_match_sim_matrix(bp_spots, bg_spots, scfg)
            rt = round(time.perf_counter() - t0, 1)
            sc = summarize_sim(ssim, p_ids, g_ids, p_coh, ks=tuple(base.recall_ks))
            sweep["runtime_s"][key] = rt
            sweep["recall"][key] = {c: {f"recall@{k}": sc[c][f"recall@{k}"] for k in base.recall_ks} for c in sc}
            print(f"  sweep {key}: overall @1={sc['overall']['recall@1']:.3f} @5={sc['overall']['recall@5']:.3f} ({rt}s)", flush=True)
        (_ART / "ab_spots_sensitivity.json").write_text(
            json.dumps(sweep, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")
        print("  →", _ART / "ab_spots_sensitivity.json")


def cmd_match(a):
    """Блок 5: один probe (md5) → top-K особей по матчеру + side-by-side оверлеи совпавших пятен."""
    import numpy as np

    from triton_data.loader import read_manifests

    from .config import load_spot_config
    from .constellation import match_constellations
    from .crops_manifest import read_crops_manifest
    from dataclasses import replace

    from .spot_ab import identity_hits_from_sim  # noqa: F401  (контракт ранжира)
    from .sealed_test import assert_unsealed
    from .viz import draw_match_overlay
    assert_unsealed({a.gallery_stage}, getattr(a, "unseal", False))   # гейт C9: галерея матча не из test
    # Оверлеи под ЯВНУЮ конфигурацию (не молчаливый дефолт) — чтобы доказательная база = canonical config
    cfg = replace(load_spot_config(), detect_variant=a.surface, detect_method=a.method,
                  match_method=a.matcher, illum_norm=a.illum_norm)
    target, external = read_manifests(_DATA)
    crops = read_crops_manifest(_DATA)
    g = _spots_for_stage(cfg, a.gallery_stage, a.scope, crops, target, external)
    p = _spots_for_stage(cfg, "dev", a.scope, crops, target, external)
    prow = p[p["md5"] == a.probe]
    if not len(prow):
        print(f"match: probe {a.probe} не найден среди dev-кропов"); return
    prow = prow.iloc[0]
    scored = []
    for _, gr in g.iterrows():
        s, pairs = match_constellations(prow["spots"], gr["spots"], cfg)
        scored.append((s, gr["individual_id"], gr["md5"], gr["crop_path"], pairs))
    scored.sort(key=lambda t: -t[0])
    import cv2
    outdir = _REPORTS / "figs" / "match"
    outdir.mkdir(parents=True, exist_ok=True)
    rgb_p = _load_crop_rgb(prow["crop_path"])
    tag = f"{a.surface}_{a.method}_{a.matcher}"                  # конфиг в имя файла (доказательная база)
    for rank, (s, gid, gmd5, gpath, pairs) in enumerate(scored[: a.topk], 1):
        ov = draw_match_overlay(rgb_p, prow["spots"], _load_crop_rgb(gpath), g[g["md5"] == gmd5].iloc[0]["spots"],
                                pairs, score=s, label=f"#{rank} {gid} [{tag}]")
        cv2.imwrite(str(outdir / f"{a.probe[:10]}__{tag}__top{rank}_{gid}.png"), ov)
    print(f"match: probe {a.probe[:10]} → top{a.topk}:",
          [(round(s, 3), gid) for s, gid, *_ in scored[: a.topk]], "→", outdir)


def cmd_hybrid(a):
    """Блок 6: гибрид эмбеддер top-K shortlist → матчер созвездия переранжирует внутри. Строит эмбеддер-sim
    (OOF Mega, выровнено по md5) + матчер-sim (лучшая конфигурация), сохраняет npz (для дешёвого
    переподбора fusion), сетка fusion (method×K×alpha) → recall@1/@5 vs эмбеддер + McNemar. dev, test НЕ трогаем."""
    import json
    from dataclasses import replace

    import numpy as np

    from triton_data.loader import read_manifests

    from .config import load_spot_config
    from .constellation import build_match_sim_matrix
    from .crops_manifest import read_crops_manifest
    from .embed_ab import load_oof_npy
    from .hybrid import compare_hybrid_vs_embedder, fuse_grid
    npz = _ART / "hybrid_sims.npz"
    if a.reuse_sims and npz.exists():
        z = np.load(npz, allow_pickle=True)
        embed_sim, matcher_sim = z["embed_sim"], z["matcher_sim"]
        g_ids, p_ids, p_coh = z["g_ids"], z["p_ids"], z["p_coh"]
        if "provenance" in z.files:                    # не использовать молча stale sim
            import hashlib
            from .hybrid import check_sims_provenance
            _cm = _DATA / "crops_manifest.csv"
            sha_cm = hashlib.sha256(_cm.read_bytes()).hexdigest() if _cm.exists() else ""
            want = {"scope": a.scope, "surface": a.surface, "method": a.method, "matcher": a.matcher,
                    "ransac_iters": int(a.ransac_iters), "embed_variant": a.embed_variant,
                    "sha_crops_manifest": sha_cm}     # сверяем И хэш crops_manifest (ловит изменение кропов)
            check_sims_provenance(json.loads(str(z["provenance"])), want)
        print("hybrid: загружены сохранённые sim из", npz)
    else:
        base = replace(load_spot_config(), detect_variant=a.surface, detect_method=a.method,
                       match_method=a.matcher, ransac_iters=a.ransac_iters, illum_norm=a.illum_norm)
        target, external = read_manifests(_DATA)
        crops = read_crops_manifest(_DATA)
        oof = load_oof_npy(Path(a.oof_dir), variants=("belly_oriented", a.embed_variant))
        oof_md5 = {m: i for i, m in enumerate(oof["md5"])}
        role = np.asarray(oof["role"]); emb = np.asarray(oof[a.embed_variant], float)
        g = _spots_for_stage(base, "gallery", a.scope, crops, target, external)
        p = _spots_for_stage(base, "dev", a.scope, crops, target, external)
        g_md5 = [m for m in g["md5"] if m in oof_md5 and role[oof_md5[m]] == "gallery"]
        p_md5 = [m for m in p["md5"] if m in oof_md5 and role[oof_md5[m]] == "probe"]
        gi = g.drop_duplicates("md5").set_index("md5"); pi = p.drop_duplicates("md5").set_index("md5")
        g_ids = gi.loc[g_md5, "individual_id"].to_numpy(); p_ids = pi.loc[p_md5, "individual_id"].to_numpy()
        p_coh = pi.loc[p_md5, "cohort"].to_numpy()
        g_spots = [gi.loc[m, "spots"] for m in g_md5]; p_spots = [pi.loc[m, "spots"] for m in p_md5]
        print(f"hybrid: матчер-sim {len(p_md5)}×{len(g_md5)} ({a.surface}/{a.method}/{a.matcher})…", flush=True)
        matcher_sim = build_match_sim_matrix(p_spots, g_spots, base)
        embed_sim = emb[[oof_md5[m] for m in p_md5]] @ emb[[oof_md5[m] for m in g_md5]].T
        import hashlib
        _cm = _DATA / "crops_manifest.csv"
        sha_cm = hashlib.sha256(_cm.read_bytes()).hexdigest() if _cm.exists() else ""   # полный хэш (без [:16])
        prov = {"scope": a.scope, "surface": a.surface, "method": a.method, "matcher": a.matcher,
                "ransac_iters": int(a.ransac_iters), "embed_variant": a.embed_variant, "sha_crops_manifest": sha_cm}
        np.savez(npz, embed_sim=embed_sim, matcher_sim=matcher_sim, g_ids=g_ids, p_ids=p_ids, p_coh=p_coh,
                 g_md5=np.array(g_md5), p_md5=np.array(p_md5),
                 provenance=json.dumps(prov, ensure_ascii=False))     # провенанс для сверки
        print("hybrid: sim+провенанс сохранены в", npz)

    grid = fuse_grid(embed_sim, matcher_sim, g_ids, p_ids, p_coh)
    best = grid["best"]
    cmp = compare_hybrid_vs_embedder(embed_sim, matcher_sim, g_ids, p_ids, p_coh,
                                     k=best["k"], method=best["method"], alpha=best["alpha"] or 0.5)
    cmp.pop("_perprobe", None)
    out = {"grid": grid, "compare": cmp, "best_fusion": best, "embedder": grid["embedder"],
           "n_probe": int(len(p_ids)), "n_gallery": int(len(g_ids)),
           "selection_note": ("best_fusion выбран на том же dev (best-on-dev exploratory); прирост, если есть, "
                              "НЕ независимый. Значимость — по McNemar (stats_hybrid_vs_embedder). "
                              "Headline-KPI — только на запечатанном test 1 раз в конце.")}
    (_ART / "ab_hybrid_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")
    e = grid["embedder"]; h = cmp["hybrid"]["overall"]
    print(f"hybrid: эмбеддер @1={e['recall@1']:.3f} @5={e['recall@5']:.3f} → "
          f"гибрид({best['method']},k={best['k']},a={best['alpha']}) @1={h['recall@1']:.3f} @5={h['recall@5']:.3f} | "
          f"McNemar@1 p={cmp['stats_hybrid_vs_embedder@1']['mcnemar_p']} c={cmp['stats_hybrid_vs_embedder@1']['mcnemar_c']} b={cmp['stats_hybrid_vs_embedder@1']['mcnemar_b']}")
    print("  →", _ART / "ab_hybrid_metrics.json")


def cmd_embed_verify(a):
    """Read-only аудит артефактов эмбеддера. Сверяет «авторитетный» JSON-метрики
    с пересчётом по oof/*.npy (ловит рассинхрон), фото-порядок md5 Mega↔MiewID, отсутствие
    запечатанных (test/open_test) md5 в OOF. Пишет verify_report.json; exit 1 при рассинхроне (гейт ВКР)."""
    import json
    import sys

    import numpy as np

    from .config import load_embed_config
    from .embed_ab import load_oof_npy
    from .embed_verify import verify_embed_artifacts
    cfg = load_embed_config()
    abj = Path(a.metrics)
    if not abj.exists() and not getattr(a, "allow_missing_metrics", False):   # fail-fast: иначе гейт «метрики↔npy» пройдёт вхолостую
        raise SystemExit(f"embed-verify: нет файла метрик {abj} — сверка metrics↔npy не выполнится. "
                         f"Укажи --metrics или явно --allow-missing-metrics (пропустить эту проверку).")
    variants = ("belly_oriented", cfg.embed_variant)
    oof = load_oof_npy(Path(a.oof_dir), variants=variants)

    expected = None
    if abj.exists():
        m = json.loads(abj.read_text(encoding="utf-8"))
        expected = {}
        for v in variants:
            blk = m.get(v, {}).get("overall") if isinstance(m.get(v), dict) else None
            if blk and "recall@1" in blk:
                expected[v] = {1: blk.get("recall@1"), 5: blk.get("recall@5")}

    compare = None
    other = Path(a.compare_oof)
    if (other / "md5.npy").exists():
        compare = np.load(other / "md5.npy", allow_pickle=True)

    # forbidden (sealed md5) — анти-утечка ОБЯЗАТЕЛЬНА: если загрузить не удалось, гейт не должен «проходить» вхолостую
    from triton_data.loader import read_manifests, select
    forbidden = set()
    target, external = read_manifests(_DATA)
    for stage in ("test", "open_test"):
        forbidden |= set(select(target, external, stage)["md5"].tolist())

    rep = verify_embed_artifacts(oof, expected_recall=expected, forbidden_md5=forbidden,
                                 compare_md5=compare, variants=variants)
    (_ART / "embed" / "verify_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=_to_jsonable), encoding="utf-8")
    for c in rep["checks"]:
        print(f"  [{'OK  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print("embed-verify:", "ВСЁ ОК" if rep["ok"] else "⚠️ ЕСТЬ РАССИНХРОН",
          "→", _ART / "embed" / "verify_report.json")
    if not rep["ok"]:
        sys.exit(1)


def cmd_embed_pack(a):
    """Упаковать самодостаточный архив для Colab (Этап B): артефакты + кропы + код + sha256-манифест.

    ЧИСТАЯ (zipfile/hashlib, без torch). Источник списка PNG — path_*/crop_path в --records (oof_records).
    Внутри zip: корень=имя_без_zip, дерево как в репо (configs/ data/ artifacts/embed/ src/ crops_belly/),
    + artifacts/embed/pack_manifest.json (sha256 каждого PNG/csv) для верификации после unzip на Colab.
    """
    import hashlib
    import json
    import zipfile

    import pandas as pd
    rec = pd.read_csv(a.records)
    png = set()
    for col in rec.columns:
        if col.startswith("path_") or col == "crop_path":
            png |= {str(p) for p in rec[col].dropna()}
    if a.include_fallback_png and "md5" in rec.columns:
        png |= {f"crops_belly/{m}.png" for m in rec["md5"]}
    png = sorted(p for p in png if (_REPO / p).exists())
    static = [f for f in ("configs/embed.yaml", "configs/paths.yaml", "data/crops_manifest.csv",
                          "artifacts/embed/oof_records.csv", "artifacts/embed/embed_records.csv",
                          "artifacts/embed/label_map.json", "artifacts/embed/cross_fit_folds.json",
                          "pyproject.toml") if (_REPO / f).exists()]
    src = sorted(str(p.relative_to(_REPO)) for p in (_REPO / "src").rglob("*.py")
                 if "__pycache__" not in str(p))

    def _sha(rel):
        return hashlib.sha256((_REPO / rel).read_bytes()).hexdigest()

    manifest = {"n_png": len(png), "n_records": int(len(rec)),
                "csv": {f: _sha(f) for f in static if f.endswith(".csv")},
                "png": {p: _sha(p) for p in png}}
    name = a.name or "triton_block4_pack.zip"
    root = name[:-4] if name.endswith(".zip") else name
    out_dir = Path(a.out)
    zip_path = out_dir / name
    files = static + src + png
    if a.dry_run:
        nbytes = sum((_REPO / f).stat().st_size for f in files)
        print(f"[dry-run] {len(files)} файлов ({len(png)} PNG, {len(src)} .py, {len(rec)} записей) · "
              f"~{nbytes // (1024 * 1024)} МБ → {zip_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(_REPO / f, f"{root}/{f}")
        z.writestr(f"{root}/artifacts/embed/pack_manifest.json",
                   json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"embed-pack: {zip_path} · {len(files)} файлов · {len(png)} PNG · {len(rec)} записей")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m triton_crop.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="псевдо-метки + overlays")
    b.add_argument("--labels", default=str(_ART / "labels_pilot"))
    b.add_argument("--overlays", default=str(_ART / "overlays_pilot"))
    b.add_argument("--per-species", type=int, default=40, dest="per_species")
    b.add_argument("--cohorts", default="TK,PW")
    b.add_argument("--folds", default="train,dev")
    b.add_argument("--reuse-labels", default=None, dest="reuse_labels")
    b.add_argument("--weights", default="FastSAM-s.pt")
    b.add_argument("--device", default="mps")
    b.set_defaults(func=cmd_bootstrap)

    c = sub.add_parser("correct", help="мини-кликер правки")
    c.add_argument("--labels", default=str(_ART / "labels_pilot"))
    c.add_argument("--reseg", action="store_true")
    c.add_argument("--pending", action="store_true", help="только нетронутые кадры (source=pseudo)")
    c.add_argument("--weights", default="sam2_b.pt")
    c.add_argument("--device", default="mps")
    c.set_defaults(func=cmd_correct)

    e = sub.add_parser("export", help="метки → YOLO seg+pose")
    e.add_argument("--labels", default=str(_ART / "labels_pilot"))
    e.add_argument("--out", default=str(_ART / "yolo"))
    e.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    e.set_defaults(func=cmd_export)

    for name, fn in (("train-seg", cmd_train_seg), ("train-pose", cmd_train_pose)):
        t = sub.add_parser(name, help="обучение YOLO (Ultralytics, MPS)")
        t.add_argument("--data", default=None)
        t.add_argument("--out", default=str(_ART / "yolo"))
        t.add_argument("--epochs", type=int, default=100)
        t.add_argument("--imgsz", type=int, default=640)   # 640 — проверенный (на 768 seg не сходится на MPS)
        t.add_argument("--device", default="mps")
        t.set_defaults(func=fn)

    vc = sub.add_parser("validate-crops", help="гейты + EDA")
    vc.add_argument("--unsealed", action="store_true",
                    help="финальный режим: манифест ЗАКОННО содержит test/open_test (после sealed-test) — C9 не применять")
    vc.set_defaults(func=cmd_validate_crops)

    cr = sub.add_parser("crop", help="прогон canonical_belly_crop → crops_manifest")
    cr.add_argument("--stages", default="dev")
    cr.add_argument("--scope", default="kpi_core")
    cr.add_argument("--seg-weights", dest="seg_weights",
                    default=str(_ART / "runs" / "belly_seg" / "weights" / "best.pt"))
    cr.add_argument("--pose-weights", dest="pose_weights",
                    default=str(_ART / "runs" / "belly_pose" / "weights" / "best.pt"))
    cr.add_argument("--imgsz", type=int, default=640)        # = imgsz обучения (640 — проверенный, на 768 seg не сходился)
    cr.add_argument("--conf", type=float, default=0.10)
    cr.add_argument("--seg-conf-min", type=float, default=None, dest="seg_conf_min")
    cr.add_argument("--pose-conf-min", type=float, default=None, dest="pose_conf_min")
    cr.add_argument("--device", default="mps")
    cr.add_argument("--pipeline", default="pilot-v1")
    cr.add_argument("--unseal", action="store_true",
                    help="разрешить кроп ЗАПЕЧАТАННЫХ test/open_test (гейт C9; необратимо, финальный замер ВКР)")
    cr.add_argument("--append", action="store_true",
                    help="аддитивно слить с существующим crops_manifest (не перезаписывать) — для кропа LAB")
    cr.set_defaults(func=cmd_crop)

    ab = sub.add_parser("ab", help="A/B Rosa (после кропа gallery+dev)")
    ab.add_argument("--scope", default="kpi_core")                          # scope проб (dev)
    ab.add_argument("--gallery-scope", default="kpi_core", dest="gallery_scope")
    ab.add_argument("--seg-conf-min", type=float, default=0.15, dest="seg_conf_min")
    ab.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-L-384")
    ab.add_argument("--device", default="mps")
    ab.set_defaults(func=cmd_ab)

    # --- Блок 4 (эмбеддер re-ID) ---
    eb = sub.add_parser("embed-build", help="Блок 4: embed-records + label-map + cross-fit фолды (локально)")
    eb.add_argument("--stage", default="train", choices=["train", "gallery"])  # только безопасные стадии (НЕ test)
    eb.add_argument("--scope", default="kpi_core")
    eb.set_defaults(func=cmd_embed_build)

    et = sub.add_parser("embed-train",
                        help="Блок 4: печатает рецепт finetune (обучение — Colab GPU, notebooks/block4_colab.ipynb)")
    et.set_defaults(func=cmd_embed_train)

    ea = sub.add_parser("embed-ab", help="Блок 4: строгий A/B finetuned (Этап C)")
    ea.add_argument("--scope", default="kpi_core")
    ea.add_argument("--gallery-scope", default="kpi_core", dest="gallery_scope")
    ea.add_argument("--oof-dir", default=str(_ART / "embed" / "oof"), dest="oof_dir",
                    help="OOF finetuned-эмбеддинги с Colab (*.npy)")
    ea.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-L-384")   # zero-shot baseline
    ea.add_argument("--seg-conf-min", type=float, default=0.15, dest="seg_conf_min")
    ea.add_argument("--device", default="mps")
    ea.set_defaults(func=cmd_embed_ab)

    eo = sub.add_parser("embed-eval-openset", help="Блок 4: open-set AUROC + порог на open_dev (Этап C)")
    eo.add_argument("--ckpt", default=str(_ART / "embed" / "ckpt" / "fold0.pt"),
                    help="дообученный fold-чекпойнт (скачать из Drive)")
    eo.add_argument("--scope", default="kpi_core")
    eo.add_argument("--device", default="mps")
    eo.set_defaults(func=cmd_embed_eval_openset)

    es = sub.add_parser("embed-test",
                        help="ФИНАЛ ВКР: sealed-test (вскрыть test+open_test, zero-shot Mega) → ab_test_headline.json")
    es.add_argument("--scope", default="kpi_core")
    es.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-L-384")
    es.add_argument("--device", default="mps")
    es.add_argument("--unseal", action="store_true",
                    help="необратимое вскрытие запечатанных test/open_test (гейт C9); после — настройки НЕ тюнить")
    es.add_argument("--force-rerun-sealed", action="store_true", dest="force_rerun_sealed",
                    help="перезаписать УЖЕ существующий ab_test_headline.json (финальные числа) — только осознанно")
    es.set_defaults(func=cmd_embed_test)

    ev = sub.add_parser("embed-verify", help="Блок 4: read-only аудит артефактов (метрики↔npy, md5-порядок, sealed)")
    ev.add_argument("--oof-dir", default=str(_ART / "embed" / "oof"), dest="oof_dir")
    ev.add_argument("--compare-oof", default=str(_ART / "embed" / "oof_miewid"), dest="compare_oof")
    ev.add_argument("--metrics", default=str(_ART / "ab_embed_metrics.json"))
    ev.add_argument("--allow-missing-metrics", action="store_true", dest="allow_missing_metrics",
                    help="не падать, если файла метрик нет (пропустить сверку metrics↔npy)")
    ev.set_defaults(func=cmd_embed_verify)

    # ── Блок 5: матчинг созвездия пятен ──
    sp = sub.add_parser("detect-spots", help="Блок 5: детекция пятен → spots-манифест")
    sp.add_argument("--stages", default="dev")
    sp.add_argument("--scope", default="kpi_core")
    sp.add_argument("--variant", default="belly_oriented", choices=["belly_oriented", "unroll_debend"])  # НЕ ribbon
    sp.add_argument("--method", default="deviation", choices=["deviation", "darkness", "log", "dog"])
    sp.add_argument("--illum-norm", action="store_true", dest="illum_norm")
    sp.add_argument("--unseal", action="store_true",
                    help="разрешить детекцию пятен на запечатанных test/open_test (гейт C9; обычно НЕ нужно)")
    sp.set_defaults(func=cmd_detect_spots)

    sv = sub.add_parser("spot-validate",
                        help="Блок 5: S-гейты матчинга (поверхность≠ribbon, анти-утечка test, A/B-решение)")
    sv.add_argument("--unsealed", action="store_true",
                    help="финальный режим: spots ЗАКОННО содержат test (после sealed-test) — S7 не применять")
    sv.set_defaults(func=cmd_spot_validate)

    sa = sub.add_parser("spot-ab", help="Блок 5: A/B матчер созвездия vs эмбеддер (grid + McNemar + Rosa)")
    sa.add_argument("--scope", default="kpi_core")
    sa.add_argument("--surfaces", default="belly_oriented,unroll_debend")
    sa.add_argument("--methods", default="deviation,darkness")
    sa.add_argument("--matchers", default="guided")              # канонический матчер (guided); ransac/nn — для sweep
    sa.add_argument("--ransac-iters", type=int, default=500, dest="ransac_iters")
    sa.add_argument("--sweep-iters", default="120,500,2000", dest="sweep_iters",
                    help="итерации RANSAC для sensitivity (контроль iteration-starvation); пусто = без sweep")
    sa.add_argument("--oof-dir", default=str(_ART / "embed" / "oof"), dest="oof_dir")
    sa.add_argument("--embed-variant", default="unroll_ribbon", dest="embed_variant")
    sa.add_argument("--illum-norm", action="store_true", dest="illum_norm")
    sa.set_defaults(func=cmd_spot_ab)

    sm = sub.add_parser("match", help="Блок 5: один probe → top-K + оверлеи совпавших пятен")
    sm.add_argument("--probe", required=True)
    sm.add_argument("--gallery-stage", default="gallery", dest="gallery_stage")
    sm.add_argument("--scope", default="kpi_core")
    sm.add_argument("--topk", type=int, default=5)
    sm.add_argument("--surface", default="belly_oriented", choices=["belly_oriented", "unroll_debend"])
    sm.add_argument("--method", default="deviation", choices=["deviation", "darkness", "log", "dog"])  # = канонический детектор (spot.yaml)
    sm.add_argument("--matcher", default="guided", choices=["guided", "ransac", "nn"])
    sm.add_argument("--illum-norm", action="store_true", dest="illum_norm")
    sm.add_argument("--unseal", action="store_true",
                    help="разрешить галерею матча из запечатанных test/open_test (гейт C9; обычно НЕ нужно)")
    sm.set_defaults(func=cmd_match)

    # ── Блок 6: гибрид эмбеддер + матчер ──
    hy = sub.add_parser("hybrid", help="Блок 6: эмбеддер top-K → матчер переранжирует (fusion A/B vs эмбеддер)")
    hy.add_argument("--scope", default="kpi_core")
    hy.add_argument("--surface", default="belly_oriented")
    # Дефолты = канонический конфиг (spot.yaml: deviation/guided/500). После их смены `--reuse-sims` без флагов
    # громко упадёт на check_sims_provenance против npz, считанного исторической конфигурацией, — это корректное
    # поведение (не молчаливый stale). Существующий npz записан с усечённым sha (до перехода на полный sha256)
    # и НЕ переиспользуется ни с какими флагами — перегенерировать без --reuse-sims.
    hy.add_argument("--method", default="deviation")          # детектор пятен (= канонический spot.yaml)
    hy.add_argument("--matcher", default="guided",
                    help="канонический guided; исторический прогон Блока 6 = --method darkness --matcher guided "
                         "--ransac-iters 120 (см. provenance artifacts/hybrid_sims.npz)")
    hy.add_argument("--ransac-iters", type=int, default=500, dest="ransac_iters")
    hy.add_argument("--illum-norm", action="store_true", dest="illum_norm")
    hy.add_argument("--oof-dir", default=str(_ART / "embed" / "oof"), dest="oof_dir")
    hy.add_argument("--embed-variant", default="unroll_ribbon", dest="embed_variant")
    hy.add_argument("--reuse-sims", action="store_true", dest="reuse_sims")
    hy.set_defaults(func=cmd_hybrid)

    ep = sub.add_parser("embed-pack", help="Блок 4: zip-упаковка для Colab (артефакты+кропы+код+sha256)")
    ep.add_argument("--records", default=str(_ART / "embed" / "oof_records.csv"))
    ep.add_argument("--out", default=str(_ART / "embed_pack"))
    ep.add_argument("--name", default=None)
    ep.add_argument("--include-fallback-png", action="store_true", dest="include_fallback_png")
    ep.add_argument("--dry-run", action="store_true", dest="dry_run")
    ep.set_defaults(func=cmd_embed_pack)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
