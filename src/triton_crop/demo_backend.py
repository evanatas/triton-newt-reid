"""Демо-движок (Streamlit MVP, финализация ВКР): браузерное опознание тритона по узору брюшка.

Концепция как у существующих систем фотоидентификации: загрузка фото → top-K похожих ОСОБЕЙ карточками
(миниатюра кропа + номер + «Уверенность %» + бейдж «Лучшее совпадение») → решение known/new.
ПЛЮС наш дифференциатор: side-by-side ОВЕРЛЕЙ совпавших пятен (матчинг созвездия центроидов;
embedding-only подход на DINOv2 структурно так не может).

Чистая логика (ранжир / калибровка шкалы / known-new) — numpy, ТЕСТИРУЕТСЯ. Модели (MegaDescriptor, YOLO)
и файловый IO — лениво (импорт внутри функций), чтобы импорт модуля и юнит-тесты не тянули веса.
Целевой запуск — ЛОКАЛЬНО, offline (ноутбук заказчика без интернета): `streamlit run app/demo.py`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GalleryIndex:
    """Индекс галереи известных особей (предвычисленные эмбеддинги + пути к кропам)."""
    emb: np.ndarray          # (G, D) L2-нормированные эмбеддинги
    ids: np.ndarray          # (G,) individual_id
    md5: np.ndarray          # (G,)
    cohort: np.ndarray       # (G,)
    crop_paths: list         # (G,) пути к кропам belly_oriented
    embed_variant: str


# ───────────────────────── ЧИСТАЯ логика (тестируется) ─────────────────────────

def calibrate_confidence(sim, lo: float = 0.15, hi: float = 0.55) -> float:
    """Косинус сходства → «уверенность» 0..99 % (монотонно, клип к [lo,hi]). Калибровка шкалы уверенности (монотонная, не вероятность):
    своя особь (высокий косинус) → ~90 %+, чужая/новая (низкий) → ~50-60 %. Это монотонная шкала, НЕ вероятность.
    Потолок 99 % — шкала никогда не показывает ровно 100 % (перекрытие genuine/impostor, потолок данных
    open-set AUROC≈0.45)."""
    if hi <= lo:
        raise ValueError("calibrate_confidence: требуется hi > lo")
    x = (float(sim) - lo) / (hi - lo)
    return float(min(np.clip(x, 0.0, 1.0) * 100.0, 99.0))


def calibration_range(gallery: "GalleryIndex", imp_pct: float = 95.0, gen_pct: float = 90.0):
    """Data-driven диапазон калибровки из РАЗДЕЛЁННЫХ распределений парных косинусов галереи:
    lo = перцентиль imp_pct косинусов РАЗНЫХ особей (impostor) — «уровень похожего чужого»,
    hi = перцентиль gen_pct косинусов ОДНОЙ особи (genuine)   — «уровень уверенного своего».
    Так своя особь → ~99 %, чужая/новая → ~40–65 %, БЕЗ сатурации всех к 100 %. (Раньше брались ВСЕ пары:
    потолок смешивал genuine+impostor и был занижен → несколько разных особей липли к 100 %.)
    Перекрытие genuine/impostor честно остаётся (потолок данных, open-set AUROC≈0.45). Fallback на все пары,
    если genuine/impostor мало. -> (lo, hi)."""
    E = np.asarray(gallery.emb, float)
    ids = np.asarray(gallery.ids)
    if len(E) < 3:
        return 0.15, 0.55
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    same = ids[iu[0]] == ids[iu[1]]
    gen, imp = S[iu][same], S[iu][~same]
    if len(gen) >= 5 and len(imp) >= 5:
        lo = float(np.percentile(imp, imp_pct)); hi = float(np.percentile(gen, gen_pct))
    else:                                          # мало genuine-пар → старый общий расчёт
        pair = S[iu]; lo = float(np.percentile(pair, 75.0)); hi = float(np.percentile(pair, 99.5))
    return (lo, hi) if hi > lo else (lo, lo + 1e-3)


def rank_individuals(query_emb, gallery: "GalleryIndex", topk: int = 7, aggregation: str = "max",
                     lo: float = 0.15, hi: float = 0.55) -> list:
    """query_emb (D,) → top-K ОСОБЕЙ по агрегированному косинусу (identity-level, как в A/B-оценке).
    Эмбеддинги L2-нормированы → скалярное произведение = косинус. (lo,hi) — диапазон калибровки уверенности
    (см. calibration_range для data-driven). -> list[dict] (отсортирован по убыванию)."""
    q = np.asarray(query_emb, float).ravel()
    sims = np.asarray(gallery.emb, float) @ q                  # (G,)
    rows = []
    for u in np.unique(gallery.ids):
        idx = np.where(gallery.ids == u)[0]
        s = sims[idx]
        agg = float(s.max()) if aggregation == "max" else float(s.mean())
        best = int(idx[int(np.argmax(s))])                     # лучшее фото особи (для миниатюры/оверлея)
        rows.append({"individual_id": str(u), "sim": agg, "confidence": round(calibrate_confidence(agg, lo, hi), 1),
                     "best_md5": str(gallery.md5[best]), "best_crop_path": gallery.crop_paths[best],
                     "cohort": str(gallery.cohort[best]), "n_photos": int(len(idx))})
    rows.sort(key=lambda r: -r["sim"])
    return rows[: int(topk)]


def filter_scope(gallery: "GalleryIndex", scope: str) -> "GalleryIndex":
    """Срез галереи по виду (cohort) или 'Вся база'/None → без среза. Чистая логика (тестируется);
    раньше жила в app/demo.py (нетестируемо). crop_paths режутся согласованно с эмбеддингами."""
    if scope in (None, "", "Вся база"):
        return gallery
    m = np.asarray(gallery.cohort) == scope
    return GalleryIndex(emb=gallery.emb[m], ids=gallery.ids[m], md5=gallery.md5[m],
                        cohort=gallery.cohort[m],
                        crop_paths=[p for p, k in zip(gallery.crop_paths, m) if k],
                        embed_variant=gallery.embed_variant)


def known_new_verdict(ranked: list, conf_thr: float = 70.0, margin_thr: float = 8.0) -> dict:
    """Решение known / кандидат-в-новую по top-1 уверенности И отрыву от top-2. -> dict."""
    if not ranked:
        return {"verdict": "new_candidate", "confidence": 0.0, "margin": 0.0}
    top1 = float(ranked[0]["confidence"])
    top2 = float(ranked[1]["confidence"]) if len(ranked) > 1 else 0.0
    margin = round(top1 - top2, 1)
    known = (top1 >= conf_thr) and (margin >= margin_thr)
    return {"verdict": "known" if known else "new_candidate", "confidence": top1, "margin": margin}


# ───────────────────────── IO / модели (лениво; маркер model) ─────────────────────────

def load_gallery(oof_dir, crops_dir, embed_variant: str = "belly_oriented") -> "GalleryIndex":
    """Галерея известных особей из OOF-выгрузки (role=='gallery', предвычисленные эмбеддинги). БЕЗ загрузки модели."""
    from .embed_ab import load_oof_npy
    oof = load_oof_npy(Path(oof_dir), variants=(embed_variant,))
    role = np.asarray(oof["role"]); md5 = np.asarray(oof["md5"])
    m = role == "gallery"
    md5g = md5[m]
    return GalleryIndex(emb=np.asarray(oof[embed_variant])[m], ids=np.asarray(oof["individual_id"])[m],
                        md5=md5g, cohort=np.asarray(oof["cohort"])[m],
                        crop_paths=[str(Path(crops_dir) / f"{x}.png") for x in md5g], embed_variant=embed_variant)


def sample_probes(oof_dir, crops_dir, embed_variant: str = "belly_oriented") -> list:
    """Готовые dev-пробы (role=='probe', предвычисленные эмбеддинги) — для НАДЁЖНОГО демо без загрузки модели."""
    from .embed_ab import load_oof_npy
    oof = load_oof_npy(Path(oof_dir), variants=(embed_variant,))
    role = np.asarray(oof["role"]); md5 = np.asarray(oof["md5"])
    m = role == "probe"
    emb = np.asarray(oof[embed_variant])[m]; md5p = md5[m]
    ids = np.asarray(oof["individual_id"])[m]; coh = np.asarray(oof["cohort"])[m]
    return [{"md5": str(md5p[i]), "individual_id": str(ids[i]), "cohort": str(coh[i]),
             "emb": emb[i], "crop_path": str(Path(crops_dir) / f"{md5p[i]}.png")} for i in range(len(md5p))]


def load_heldout(heldout_dir, crops_dir, embed_variant: str = "belly_oriented") -> list:
    """Held-out пробы sealed-test (нетронутые test+open_test): zero-shot эмбеддинги + ground-truth known/new.
    Формат как у `embed-test` (artifacts/embed/heldout/{variant,md5,individual_id,cohort,is_new}.npy).
    -> list[dict] как sample_probes + поле is_new (True = новая особь = open_test). БЕЗ загрузки модели."""
    d = Path(heldout_dir)
    emb = np.load(d / f"{embed_variant}.npy")
    # allow_pickle: строковые массивы сохранены как object-dtype (требуют pickle); файлы — СОБСТВЕННЫЕ
    # доверенные артефакты (artifacts/embed/heldout/, gitignored, пишутся нашим CLI embed-test).
    # Чужие .npy сюда не подкладывать.
    md5 = np.load(d / "md5.npy", allow_pickle=True)
    ids = np.load(d / "individual_id.npy", allow_pickle=True)
    coh = np.load(d / "cohort.npy", allow_pickle=True)
    is_new = np.load(d / "is_new.npy", allow_pickle=True)
    return [{"md5": str(md5[i]), "individual_id": str(ids[i]), "cohort": str(coh[i]),
             "is_new": bool(is_new[i]), "emb": emb[i],
             "crop_path": str(Path(crops_dir) / f"{md5[i]}.png")} for i in range(len(md5))]


def load_headline(path):
    """Финальные sealed-test числа ВКР из artifacts/ab_test_headline.json (блок `headline`).
    Единый источник правды для витрин (демо «О системе», README, текст НИР) — НЕ хардкодить «на глаз».
    -> dict (overall@1/5, PW@1/5, TK@1/5, PW_n/TK_n, pipeline@1/5, auroc, n, n_gallery; bo@1/bo@5/bo_PW@1 —
    sealed-числа варианта belly_oriented из by_variant, None при отсутствии блока) или None, если файла нет ИЛИ
    структура неполная/битая (тогда витрина просто покажет только dev — без падения демо)."""
    import json
    p = Path(path)
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        h = j["headline"]
        rec = h["recall"]; ov = rec["overall"]; pc = rec["per_cohort"]; pipe = rec["pipeline_recall"]
        out = {
            "overall@1": ov["recall@1"], "overall@5": ov["recall@5"], "n": ov["n"],
            "PW@1": pc["PW"]["recall@1"], "PW@5": pc["PW"]["recall@5"],
            "PW_n": pc["PW"]["n"], "TK_n": pc["TK"]["n"],
            "TK@1": pc["TK"]["recall@1"], "TK@5": pc["TK"]["recall@5"],
            "pipeline@1": pipe["1"], "pipeline@5": pipe["5"],
            "auroc": h["openset"]["auroc"], "n_gallery": h["n_gallery"],
        }
        # Вариант демо-конвейера (belly_oriented) — тоже из артефакта, не хардкодом в UI.
        bo = (j.get("by_variant") or {}).get("belly_oriented") or {}
        bo_rec = (bo.get("recall") or {})
        bo_ov = bo_rec.get("overall") or {}
        bo_pw = (bo_rec.get("per_cohort") or {}).get("PW") or {}
        out["bo@1"] = bo_ov.get("recall@1")
        out["bo@5"] = bo_ov.get("recall@5")
        out["bo_PW@1"] = bo_pw.get("recall@1")
        return out
    except (KeyError, ValueError, TypeError):                  # битый/неполный JSON → None (демо не падает)
        return None


def load_rgb(path):
    """PNG → RGB uint8 (или None)."""
    import cv2
    bgr = cv2.imread(str(path))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def embed_crop(rgb_crop, model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384", device: str = "mps", proxy=None):
    """Эмбеддинг одного кропа MegaDescriptor (лениво; путь «загрузить фото»). -> (D,) L2-normed."""
    from .proxy_embed import build_proxy, embed
    proxy = proxy or build_proxy(model_name, device)
    return embed(proxy, [np.asarray(rgb_crop)])[0]


def crop_from_raw(rgb, crop_cfg, seg_weights, pose_weights, device: str = "mps"):
    """Сырое фото → канонический кроп брюшка (belly_oriented) через YOLO seg+pose (лениво). -> rgb-кроп | None."""
    from .pipeline import canonical_belly_crop
    from .predict import BellySegmenter, PoseEstimator
    seg = BellySegmenter(str(seg_weights), device=device)
    pose = PoseEstimator(str(pose_weights), device=device)
    res = canonical_belly_crop(np.asarray(rgb), seg, pose, crop_cfg)
    return None if res is None or getattr(res, "image", None) is None else res.image


def spot_overlay(query_rgb, gallery_rgb, spot_cfg):
    """Оверлей совпавших пятен query↔gallery (наш дифференциатор; запрос заказчика №1).
    -> (overlay_BGR uint8, score 0..1, n_pairs). Детекция на belly_oriented (foreground flood-fill), матчер guided."""
    from .constellation import match_constellations
    from .masks import foreground_from_crop
    from .spots import detect_spots
    from .viz import draw_match_overlay
    qs = detect_spots(query_rgb, foreground_from_crop(query_rgb, spot_cfg.bg_sum_thr), spot_cfg)
    gs = detect_spots(gallery_rgb, foreground_from_crop(gallery_rgb, spot_cfg.bg_sum_thr), spot_cfg)
    score, pairs = match_constellations(qs, gs, spot_cfg)
    # подпись ASCII: cv2.putText не рисует кириллицу (иначе «???»); русский текст — в подписи Streamlit
    ov = draw_match_overlay(query_rgb, qs, gallery_rgb, gs, pairs, score=score,
                            label=f"matched spots: {len(pairs)}")
    return ov, float(score), len(pairs)
