"""A/B-гейт Rosa: помогает ли канонический кроп пуза re-ID vs raw-кадр. Модель снаружи, анализ — здесь.

ЧИСТАЯ часть (тестируется): per_probe_hits (парные попадания по probe), variant_embeddings
(переключение raw↔crop по порогу seg_conf с fallback на raw), summarize (recall@k + Wilson CI +
per-cohort), run_ab_analysis (полная сводка: full + paired-success-only + McNemar/bootstrap + sensitivity).
Честно: raw ВСЕГДА по полной raw-галерее; кадры без belly_oriented кропа → fallback на raw (не выкидываем).
"""
import numpy as np

from .stats import mcnemar, paired_bootstrap_recall_diff, wilson_ci


def per_probe_hits(probe_emb, probe_ids, gallery_emb, gallery_ids, ks=(1, 5)):
    """Для каждого probe — попал ли его individual_id в top-k галерейных ФОТО. -> {k: bool-массив}.

    PHOTO-level (технический proxy): ранжируются отдельные фото галереи. Если у одной чужой особи много
    фото, она может занять несколько мест top-k. Для продукта/ТЗ (top-K = список ОСОБЕЙ) используйте
    per_probe_identity_hits — на большой галерее результаты расходятся.
    """
    sim = np.asarray(probe_emb) @ np.asarray(gallery_emb).T
    ranked = np.asarray(gallery_ids)[np.argsort(-sim, axis=1, kind="stable")]
    pid = np.asarray(probe_ids)[:, None]
    return {k: (ranked[:, :k] == pid).any(axis=1) for k in ks}


def identity_scores(probe_emb, gallery_emb, gallery_ids, aggregation="max"):
    """Агрегировать фото галереи до ОСОБЕЙ: скор пробы к особи = max (или mean) по её фото.
    -> (agg[n_probe, n_identity], uniq_ids[n_identity]). uniq_ids отсортированы (np.unique)."""
    sim = np.asarray(probe_emb, float) @ np.asarray(gallery_emb, float).T
    gids = np.asarray(gallery_ids)
    uniq = np.unique(gids)
    agg = np.empty((sim.shape[0], len(uniq)), float)
    for j, u in enumerate(uniq):
        cols = sim[:, gids == u]
        agg[:, j] = cols.max(axis=1) if aggregation == "max" else cols.mean(axis=1)
    return agg, uniq


def per_probe_identity_hits(probe_emb, probe_ids, gallery_emb, gallery_ids, ks=(1, 5), aggregation="max"):
    """IDENTITY-level CMC (контракт ТЗ: top-K = список ОСОБЕЙ, не фото). Фото галереи агрегируются до
    особей (max/mean скор на особь), ранжируются УНИКАЛЬНЫЕ особи → попал ли true id в top-k ОСОБЕЙ.
    -> {k: bool-массив}. Основная метрика продукта; per_probe_hits (photo-level) — технический proxy."""
    agg, uniq = identity_scores(probe_emb, gallery_emb, gallery_ids, aggregation)
    ranked = uniq[np.argsort(-agg, axis=1, kind="stable")]   # stable: при равных score ранг детерминирован
    pid = np.asarray(probe_ids)[:, None]
    return {k: (ranked[:, :k] == pid).any(axis=1) for k in ks}


def variant_embeddings(raw_emb, bo_emb, seg_conf, thr):
    """Представление кадра при пороге thr: belly_oriented где seg_conf>=thr, иначе raw (fallback).
    thr=None → всё raw."""
    raw_emb = np.asarray(raw_emb, float)
    if thr is None:
        return raw_emb
    out = raw_emb.copy()
    use = np.asarray(seg_conf) >= thr
    out[use] = np.asarray(bo_emb, float)[use]
    return out


def summarize(hits, cohorts, ks=(1, 5)):
    """hits={k:bool}, cohorts — когорты probe. -> recall@k overall + Wilson CI + per-cohort."""
    cohorts = np.asarray(cohorts)
    n = len(cohorts)
    out = {"overall": {"n": int(n)}}
    for k in ks:
        h = np.asarray(hits[k])
        out["overall"][f"recall@{k}"] = float(h.mean()) if n else 0.0
        lo, hi = wilson_ci(int(h.sum()), n)
        out["overall"][f"ci@{k}"] = [round(lo, 4), round(hi, 4)]
    for c in sorted(set(cohorts.tolist())):
        m = cohorts == c
        out[c] = {"n": int(m.sum()),
                  **{f"recall@{k}": float(np.asarray(hits[k])[m].mean()) if m.any() else 0.0 for k in ks}}
    return out


def _paired_stats(base_hits, var_hits):
    """Парная стат по ОДНИМ И ТЕМ ЖЕ probe: McNemar (точный двусторонний) + bootstrap-CI разницы recall.
    -> {mcnemar_b, mcnemar_c, mcnemar_p, bootstrap_diff, bootstrap_ci95}. c = var попал там, где base нет."""
    b, c, p = mcnemar(base_hits, var_hits)
    diff, lo, hi = paired_bootstrap_recall_diff(base_hits, var_hits)
    return {"mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 4),
            "bootstrap_diff": round(diff, 4), "bootstrap_ci95": [round(lo, 4), round(hi, 4)]}


_PATTERN_SAFE = {"debend": True, "ribbon": True, "wnorm": False}  # ДЕФОЛТ; на реальном решении cli передаёт ИЗМЕРЕННЫЙ unroll.pattern_safe_methods(cfg)


def gate_passed(ab, k: int = 1, challenger: str = "belly_oriented", baseline: str = "raw"):
    """Гейт: challenger не хуже baseline по recall@k (overall, FULL probe set). Дефолт = belly_oriented≥raw."""
    return ab[challenger]["overall"][f"recall@{k}"] >= ab[baseline]["overall"][f"recall@{k}"]


def unroll_adopt_decision(ab, pattern_ok=None, baseline: str = "belly_oriented",
                          alpha: float = 0.05, primary_k: int = 5, allow_k1_significance: bool = False):
    """Решение Блока 3: какой вариант идёт во ВХОД ЭМБЕДДЕРА. Честное правило (значимость + non-regression).

    Кандидат принимается, ТОЛЬКО если он: (1) pattern-safe; (2) НЕ хуже baseline по recall@1 И recall@5
    (non-regression); (3) прирост СТАТИСТИЧЕСКИ ПОДДЕРЖАН — McNemar p<alpha по @primary_k (@5).
    allow_k1_significance=True дополнительно засчитывает значимость по @1 (по умолчанию ВЫКЛ — строго @5,
    т.к. Блок 3 канонизирует вход под top-K shortlist). Среди прошедших — максимум по @primary_k. Если ни
    один — baseline (распрямление НЕ принято; решение отложено к Блоку 4; для Блока 5 варианты сохранены)."""
    return _adopt(ab, pattern_ok, baseline, alpha, primary_k, allow_k1_significance)["decision"]


def unroll_adopt_explain(ab, pattern_ok=None, baseline: str = "belly_oriented",
                         alpha: float = 0.05, primary_k: int = 5, allow_k1_significance: bool = False):
    """Как unroll_adopt_decision, но возвращает обоснование (для сериализации в ab_metrics.json):
    {decision, rule, alpha, primary_k, allow_k1_significance, baseline, considered[]}."""
    return _adopt(ab, pattern_ok, baseline, alpha, primary_k, allow_k1_significance)


def _adopt(ab, pattern_ok, baseline, alpha, primary_k, allow_k1):
    pattern_ok = _PATTERN_SAFE if pattern_ok is None else pattern_ok
    base = ab[baseline]["overall"]
    considered = []
    for name, blk in ab.items():
        if not (isinstance(name, str) and name.startswith("unroll_")):
            continue
        if not (isinstance(blk, dict) and "overall" in blk):
            continue
        o = blk["overall"]
        s1 = ab.get(f"stats_{name}_vs_bo@1", {})
        sp = ab.get(f"stats_{name}_vs_bo@{primary_k}", {})
        non_regress = (o.get("recall@1", 0) >= base["recall@1"]) and (o.get("recall@5", 0) >= base["recall@5"])
        # СТРОГО @primary_k; @1 — только если явно разрешено (по умолчанию нет)
        supported = (sp.get("mcnemar_p", 1.0) < alpha) or (allow_k1 and s1.get("mcnemar_p", 1.0) < alpha)
        rec = {"variant": name, "recall@1": o.get("recall@1"), "recall@5": o.get("recall@5"),
               "pattern_safe": bool(pattern_ok.get(name[len("unroll_"):], False)),
               "p@1": s1.get("mcnemar_p"), f"p@{primary_k}": sp.get("mcnemar_p"),
               "non_regression": bool(non_regress), "significant": bool(supported)}
        rec["eligible"] = bool(rec["pattern_safe"] and non_regress and supported)
        considered.append(rec)
    eligible = [r for r in considered if r["eligible"]]
    if eligible:
        winner = max(eligible, key=lambda r: (r.get(f"recall@{primary_k}") or 0.0, r.get("recall@1") or 0.0))
        decision = winner["variant"]
        rule = (f"pattern-safe + non-regression + значим McNemar p<{alpha} по @{primary_k}"
                f"{' или @1' if allow_k1 else ''}; выбор по максимуму @{primary_k}")
    else:
        decision = baseline
        rule = (f"ни один unroll-вариант не показал ЗНАЧИМОГО (McNemar p<{alpha} по @{primary_k}) прироста "
                f"при non-regression @1&@5 — на пилоте (n мал, zero-shot, без cross-fit) различия в пределах "
                f"шума; распрямление НЕ принято в эмбеддер, решение отложено к Блоку 4")
    return {"decision": decision, "rule": rule, "alpha": alpha, "primary_k": primary_k,
            "allow_k1_significance": bool(allow_k1), "baseline": baseline, "considered": considered}


def run_ab_analysis(g_raw_e, g_bo_e, g_conf, g_ids, p_raw_e, p_bo_e, p_conf, p_ids, p_coh,
                    thr=0.15, sweep=(0.15, 0.20, 0.25, 0.30, 0.35), ks=(1, 5), extra_variants=None):
    """Вся ЧИСТАЯ аналитика A/B из готовых эмбеддингов. -> ab-словарь (+ ключ '_perprobe' для csv).

    extra_variants: {имя: (g_var_e, p_var_e)} — распрямлённые варианты Блока 3 (raw-fallback для
    отсутствующих кропов уже применён снаружи; порог seg_conf применяется как к belly_oriented).
    Для каждого: summarize + парная стат vs belly_oriented (stats_<имя>_vs_bo@1) + sensitivity.
    """
    g_conf, p_conf, p_coh = np.asarray(g_conf, float), np.asarray(p_conf, float), np.asarray(p_coh)

    def hits_of(g_var_e, p_var_e, t):
        ge = variant_embeddings(g_raw_e, g_var_e, g_conf, t)
        pe = variant_embeddings(p_raw_e, p_var_e, p_conf, t)
        return per_probe_hits(pe, p_ids, ge, g_ids, ks)

    raw_h = per_probe_hits(p_raw_e, p_ids, g_raw_e, g_ids, ks)
    crop_h = hits_of(g_bo_e, p_bo_e, thr)
    ab = {"raw": summarize(raw_h, p_coh, ks), "belly_oriented": summarize(crop_h, p_coh, ks)}

    in_g = np.array([pid in set(np.asarray(g_ids).tolist()) for pid in p_ids])
    ab["closed_set"] = {"n_probe": int(len(p_ids)), "n_true_id_in_gallery": int(in_g.sum()),
                        "n_excluded_no_true_id": int((~in_g).sum())}
    if in_g.any():
        ab["paired_success_only"] = {
            "raw": summarize({k: raw_h[k][in_g] for k in ks}, p_coh[in_g], ks),
            "belly_oriented": summarize({k: crop_h[k][in_g] for k in ks}, p_coh[in_g], ks)}

    for k in ks:    # парная стат belly_oriented vs raw на ВСЕХ k (closed-set подмножество)
        ab[f"stats_closed@{k}"] = _paired_stats(raw_h[k][in_g], crop_h[k][in_g])
    ab["sensitivity"] = [{"seg_conf_min": t, "probe_crop_coverage": round(float((p_conf >= t).mean()), 3),
                          **{f"overall@{k}": round(float(hits_of(g_bo_e, p_bo_e, t)[k].mean()), 4) for k in ks}}
                         for t in sweep]
    perprobe = {"raw": raw_h, "crop": crop_h, "in_gallery": in_g}

    for name, (g_var_e, p_var_e) in (extra_variants or {}).items():
        vh = hits_of(g_var_e, p_var_e, thr)
        ab[name] = summarize(vh, p_coh, ks)
        if in_g.any():
            ab.setdefault("paired_success_only", {})[name] = summarize(
                {k: vh[k][in_g] for k in ks}, p_coh[in_g], ks)
        for k in ks:    # парная стат variant vs belly_oriented на ВСЕХ k; c = variant лучше bo
            ab[f"stats_{name}_vs_bo@{k}"] = _paired_stats(crop_h[k][in_g], vh[k][in_g])
        ab[f"sensitivity_{name}"] = [{"seg_conf_min": t,
                                      **{f"overall@{k}": round(float(hits_of(g_var_e, p_var_e, t)[k].mean()), 4)
                                         for k in ks}} for t in sweep]
        perprobe[name] = vh

    ab["_perprobe"] = perprobe
    return ab
