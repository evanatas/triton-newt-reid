"""Строгий A/B на дообученном эмбеддере re-ID (Блок 4) — ЧИСТАЯ склейка поверх ab_harness.

Что добавляет к ab_harness.run_ab_analysis (которое переиспользуется как есть для belly_oriented↔ribbon
на finetuned-эмбеддингах + sensitivity + per-cohort):
  • ab_detailed     — богатый per-probe лог: rank истинного id, score top-1, margin top1−top2, pred_id
                      (для интерпретируемости/таблиц ВКР);
  • compare_models  — парное сравнение ДВУХ эмбеддеров на ОДНИХ пробах (zero-shot↔finetuned, Mega↔MiewID,
                      ±GCN-warmup) с McNemar/bootstrap;
  • adopt_finetune  — честное правило адопции (как Блок 3): принимаем challenger ТОЛЬКО при non-regression
                      по всем k И значимом McNemar (p<alpha, c>b) по primary_k; иначе baseline (Rosa-логика).
Герметично (numpy/pandas); реальные эмбеддинги приходят с Colab (Этап B).
"""
import numpy as np
import pandas as pd

from .ab_harness import (
    _paired_stats, identity_scores, per_probe_hits, per_probe_identity_hits, summarize,
)


def recall_split(oof, variant, ks=(1, 5), n_official_probe=None, n_official_gallery=None,
                 aggregation: str = "max") -> dict:
    """Разнести две честные метрики + coverage (identity-level, контракт ТЗ).

      • closedset_reid_recall@k — качество re-ID, когда истинная особь ЕСТЬ в OOF-галерее (знаменатель =
        n_true_id_in_gallery). Чистая способность узнавать.
      • pipeline_recall@k       — продуктовая честность: знаменатель = ВСЕ официальные probe (n_official_probe;
        дефолт = probe в OOF). Кадр без кропа / без истинной особи в галерее = ПРОМАХ (штраф за coverage).
      • coverage                — сколько кадров реально дошло (OOF) против официального сплита.
    """
    role = np.asarray(oof["role"]); g = role == "gallery"; p = role == "probe"
    ids = np.asarray(oof["individual_id"]); emb = np.asarray(oof[variant], float)
    hits = per_probe_identity_hits(emb[p], ids[p], emb[g], ids[g], ks, aggregation)
    gset = set(ids[g].tolist())
    in_gallery = np.array([pid in gset for pid in ids[p]])
    n_probe_oof, n_gal_oof = int(p.sum()), int(g.sum())
    n_off_probe = int(n_official_probe) if n_official_probe is not None else n_probe_oof
    out = {"coverage": {"n_gallery_oof": n_gal_oof, "n_probe_oof": n_probe_oof,
                        "n_official_gallery": n_official_gallery, "n_official_probe": n_off_probe,
                        "n_true_id_in_gallery": int(in_gallery.sum()),
                        "n_excluded_no_true_id": int((~in_gallery).sum())},
           "closedset_reid_recall": {}, "pipeline_recall": {}, "aggregation": aggregation}
    for k in ks:
        h = np.asarray(hits[k])
        out["closedset_reid_recall"][k] = float(h[in_gallery].mean()) if in_gallery.any() else 0.0
        out["pipeline_recall"][k] = float(h[in_gallery].sum()) / n_off_probe if n_off_probe else 0.0
    return out


def load_oof_npy(oof_dir, variants=("belly_oriented", "unroll_ribbon")) -> dict:
    """Загрузить OOF-выгрузку с Colab (Этап B → artifacts/embed/oof/): <variant>.npy (finetuned эмбеддинги,
    out-of-fold) + md5/role/individual_id/cohort.npy. -> dict, совместимый с to_ab_inputs/run_ab_analysis."""
    from pathlib import Path
    d = Path(oof_dir)
    out = {v: np.load(d / f"{v}.npy") for v in variants}
    for k in ("md5", "role", "individual_id", "cohort"):
        out[k] = np.load(d / f"{k}.npy", allow_pickle=True)
    return out


def ab_detailed(probe_emb, probe_ids, gallery_emb, gallery_ids, probe_md5=None, ks=(1, 5),
                aggregation: str = "max") -> pd.DataFrame:
    """Per-probe таблица для анализа ошибок/таблиц ВКР. Тот же ранжир, что per_probe_hits / identity_scores
    (probe и gallery обрабатываются одним путём). Содержит ДВА уровня:
      • PHOTO-level (технический proxy): pred_id, score_top1, margin_top1_top2, true_rank, hit@k;
      • IDENTITY-level (контракт продукта, top-K = особи): true_identity_rank, top1_identity,
        top5_identities, n_gallery_photos_true_id, max_true_id_score, best_wrong_id_score,
        margin_identity_top1_top2, hit_id@k."""
    pe, ge = np.asarray(probe_emb, float), np.asarray(gallery_emb, float)
    sim = pe @ ge.T
    gids, pids = np.asarray(gallery_ids), np.asarray(probe_ids)
    order = np.argsort(-sim, axis=1, kind="stable")
    agg, uniq = identity_scores(pe, ge, gids, aggregation)          # IDENTITY-level скоры
    iorder = np.argsort(-agg, axis=1, kind="stable")               # stable: детерминизм ранга при ничьих
    rows = []
    for i in range(len(sim)):
        # --- photo-level ---
        ri = order[i]; rsim, rid = sim[i, ri], gids[ri]
        hit = np.where(rid == pids[i])[0]
        true_rank = int(hit[0] + 1) if len(hit) else -1
        other = rid != rid[0]
        margin = float(rsim[0] - (rsim[other][0] if other.any() else rsim[0]))
        # --- identity-level ---
        io = iorder[i]; rids, rsc = uniq[io], agg[i, io]
        ihit = np.where(rids == pids[i])[0]
        true_id_rank = int(ihit[0] + 1) if len(ihit) else -1
        n_true_photos = int((gids == pids[i]).sum())
        max_true = float(agg[i, np.where(uniq == pids[i])[0][0]]) if n_true_photos else float("nan")
        wrong = rids != pids[i]
        best_wrong = float(rsc[wrong][0]) if wrong.any() else float("nan")
        id_margin = float(rsc[0] - rsc[1]) if len(rsc) >= 2 else 0.0
        row = {"md5": probe_md5[i] if probe_md5 is not None else i, "individual_id": pids[i],
               "pred_id": rid[0], "score_top1": float(rsim[0]), "margin_top1_top2": margin,
               "true_rank": true_rank,
               "top1_identity": rids[0], "true_identity_rank": true_id_rank,
               "top5_identities": list(rids[:5]), "n_gallery_photos_true_id": n_true_photos,
               "max_true_id_score": max_true, "best_wrong_id_score": best_wrong,
               "margin_identity_top1_top2": id_margin}
        for k in ks:
            row[f"hit@{k}"] = bool(0 < true_rank <= k)
            row[f"hit_id@{k}"] = bool(0 < true_id_rank <= k)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(g_a_emb, p_a_emb, g_b_emb, p_b_emb, g_ids, p_ids, p_coh,
                   ks=(1, 5), label_a: str = "zero_shot", label_b: str = "finetuned") -> dict:
    """Парное сравнение двух эмбеддеров на ОДНИХ пробах. -> {label_a, label_b (summarize),
    stats_<b>_vs_<a>@k (McNemar/bootstrap, c = b лучше a), _perprobe}. Каждый эмбеддер ранжирует по
    СВОЕЙ галерее (gallery и probe — одной сборкой)."""
    ha = per_probe_hits(p_a_emb, p_ids, g_a_emb, g_ids, ks)
    hb = per_probe_hits(p_b_emb, p_ids, g_b_emb, g_ids, ks)
    out = {label_a: summarize(ha, p_coh, ks), label_b: summarize(hb, p_coh, ks)}
    for k in ks:
        out[f"stats_{label_b}_vs_{label_a}@{k}"] = _paired_stats(ha[k], hb[k])
    out["_perprobe"] = {label_a: ha, label_b: hb}
    return out


def build_finetuned_ab(oof, zero_bo, zero_ribbon=None, embed_variant: str = "unroll_ribbon",
                       thr: float = 0.15, ks=(1, 5)) -> dict:
    """Строгий A/B Этапа C из OOF finetuned-эмбеддингов (oof) + zero-shot baseline на ТЕХ ЖЕ кропах.

    Один run_ab_analysis даёт сразу: raw = zero-shot belly_oriented, belly_oriented = finetuned bo,
    <embed_variant> = finetuned ribbon → stats_closed = finetuned↔zero-shot (bo), stats_<var>_vs_bo =
    belly_oriented↔ribbon на finetuned (повторный A/B Блока 3 на дообученном). Плюс честное правило
    адопции finetune (adopt_finetuned_bo/ribbon) и ribbon (unroll_adopt_decision). seg_conf=1.0 (кропы
    уже выбраны). -> ab-словарь (без _perprobe)."""
    from .ab_harness import run_ab_analysis, unroll_adopt_explain
    role = np.asarray(oof["role"]); g = role == "gallery"; p = role == "probe"
    ids = np.asarray(oof["individual_id"]); coh = np.asarray(oof["cohort"])
    bo = np.asarray(oof["belly_oriented"], float)
    var = np.asarray(oof[embed_variant], float)
    zbo = np.asarray(zero_bo, float)
    conf = np.ones(len(role))
    ab = run_ab_analysis(zbo[g], bo[g], conf[g], ids[g], zbo[p], bo[p], conf[p], ids[p], coh[p],
                         thr=thr, ks=ks, extra_variants={embed_variant: (var[g], var[p])})
    ab.pop("_perprobe", None)
    explain = unroll_adopt_explain(ab, baseline="belly_oriented")     # bo↔ribbon на finetuned
    ab["unroll_adopt_decision"] = explain["decision"]
    ab["unroll_adopt_rationale"] = explain
    cmp_bo = compare_models(zbo[g], zbo[p], bo[g], bo[p], ids[g], ids[p], coh[p], ks=ks,
                            label_a="zeroshot_bo", label_b="finetuned_bo")
    cmp_bo.pop("_perprobe", None)
    ab["finetuned_vs_zeroshot_bo"] = cmp_bo
    ab["adopt_finetuned_bo"] = adopt_finetune(cmp_bo, baseline="zeroshot_bo",
                                              challenger="finetuned_bo", ks=ks)
    if zero_ribbon is not None:
        zrib = np.asarray(zero_ribbon, float)
        cmp_r = compare_models(zrib[g], zrib[p], var[g], var[p], ids[g], ids[p], coh[p], ks=ks,
                               label_a="zeroshot_ribbon", label_b="finetuned_ribbon")
        cmp_r.pop("_perprobe", None)
        ab["finetuned_vs_zeroshot_ribbon"] = cmp_r
        ab["adopt_finetuned_ribbon"] = adopt_finetune(cmp_r, baseline="zeroshot_ribbon",
                                                      challenger="finetuned_ribbon", ks=ks)
    return ab


def adopt_finetune(cmp, baseline: str = "zero_shot", challenger: str = "finetuned",
                   alpha: float = 0.05, primary_k: int = 5, ks=(1, 5)) -> dict:
    """Честное правило адопции (Rosa-логика Блока 3): challenger принимается ТОЛЬКО если он (1) НЕ хуже
    baseline по всем k (non-regression) И (2) прирост статистически поддержан — McNemar p<alpha по
    primary_k И c>b (challenger выиграл больше проб, чем проиграл). Иначе baseline. -> {decision, ...}."""
    if primary_k not in ks:
        raise ValueError(f"adopt_finetune: primary_k={primary_k} не входит в ks={tuple(ks)}")
    base, chal = cmp[baseline]["overall"], cmp[challenger]["overall"]
    for k in ks:    # явная проверка: НЕ подставлять молчаливый дефолт (вакуумная non-regression)
        for nm, blk in ((baseline, base), (challenger, chal)):
            if f"recall@{k}" not in blk:
                raise ValueError(f"adopt_finetune: recall@{k} отсутствует в overall для {nm!r}")
    non_regress = all(chal[f"recall@{k}"] >= base[f"recall@{k}"] for k in ks)
    sp = cmp.get(f"stats_{challenger}_vs_{baseline}@{primary_k}", {})
    significant = (sp.get("mcnemar_p", 1.0) < alpha) and (sp.get("mcnemar_c", 0) > sp.get("mcnemar_b", 0))
    eligible = bool(non_regress and significant)
    return {
        "decision": challenger if eligible else baseline,
        "non_regression": bool(non_regress), "significant": bool(significant),
        f"recall@{primary_k}_{baseline}": base.get(f"recall@{primary_k}"),
        f"recall@{primary_k}_{challenger}": chal.get(f"recall@{primary_k}"),
        f"mcnemar_p@{primary_k}": sp.get("mcnemar_p"),
        "rule": (f"принять {challenger}, если non-regression по @{tuple(ks)} И McNemar p<{alpha} "
                 f"по @{primary_k} с c>b; иначе baseline={baseline}"),
    }
