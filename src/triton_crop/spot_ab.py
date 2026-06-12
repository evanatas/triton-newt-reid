"""A/B матчера созвездий поверх ГОТОВОЙ sim-матрицы (Блок 5, шаги 5.6–5.7) — ЧИСТО (reuse ab_harness).

Главный нюанс: `ab_harness.per_probe_*` стартуют с `sim = probe_emb @ gallery_emb.T` (эмбеддинги).
Матчер созвездий даёт ПРОИЗВОЛЬНУЮ sim-матрицу (доля совпавших пятен) — здесь тонкие функции, которые
повторяют identity-ранжир от готовой sim и переиспользуют сводку/парную стат из ab_harness. Так числа
матчера напрямую сравнимы с эмбеддером (baseline @5 0.375), и решение принимается по Rosa-правилу.
Контракт ТЗ: top-K = ОСОБИ (identity-level), матчер бьёт в top-1 → primary_k=1.
"""
import numpy as np


def identity_hits_from_sim(sim, probe_ids, gallery_ids, ks=(1, 5), aggregation="max") -> dict:
    """IDENTITY-level CMC из ГОТОВОЙ sim-матрицы (n_probe×n_gallery): агрегируем фото галереи до особей
    (max/mean) → ранжируем уникальные особи → попал ли true id в top-k ОСОБЕЙ. -> {k: bool-массив}."""
    sim = np.asarray(sim, float)
    gids, pids = np.asarray(gallery_ids), np.asarray(probe_ids)
    uniq = np.unique(gids)
    agg = np.empty((sim.shape[0], len(uniq)), float)
    for j, u in enumerate(uniq):
        cols = sim[:, gids == u]
        agg[:, j] = cols.max(axis=1) if aggregation == "max" else cols.mean(axis=1)
    ranked = uniq[np.argsort(-agg, axis=1, kind="stable")]   # stable: детерминизм ранга при ничьих
    pid = pids[:, None]
    return {k: (ranked[:, :k] == pid).any(axis=1) for k in ks}


def summarize_sim(sim, probe_ids, gallery_ids, cohorts, ks=(1, 5), aggregation="max") -> dict:
    """recall@k overall + Wilson CI + per-cohort из sim-матрицы (формат как у эмбеддера). Reuse ab_harness."""
    from .ab_harness import summarize
    return summarize(identity_hits_from_sim(sim, probe_ids, gallery_ids, ks, aggregation), cohorts, ks)


def recall_split_sim(sim, probe_ids, gallery_ids, ks=(1, 5), n_official_probe=None,
                     aggregation="max") -> dict:
    """pipeline_recall (все офиц. probe, нет пятен/нет особи в галерее = промах) vs closedset_reid_recall
    (только probe с истинной особью в галерее) + coverage. Аналог embed_ab.recall_split на готовой sim."""
    hits = identity_hits_from_sim(sim, probe_ids, gallery_ids, ks, aggregation)
    gset = set(np.asarray(gallery_ids).tolist())
    in_g = np.array([p in gset for p in np.asarray(probe_ids)])
    n_probe = len(np.asarray(probe_ids))
    n_off = int(n_official_probe) if n_official_probe is not None else n_probe
    out = {"coverage": {"n_probe": n_probe, "n_official_probe": n_off,
                        "n_true_id_in_gallery": int(in_g.sum()),
                        "n_excluded_no_true_id": int((~in_g).sum())},
           "closedset_reid_recall": {}, "pipeline_recall": {}, "aggregation": aggregation}
    for k in ks:
        h = np.asarray(hits[k])
        out["closedset_reid_recall"][k] = float(h[in_g].mean()) if in_g.any() else 0.0
        out["pipeline_recall"][k] = float(h[in_g].sum()) / n_off if n_off else 0.0
    return out


def compare_matcher_vs_embedder(sim_matcher, emb_probe, emb_gallery, probe_ids, gallery_ids, p_coh,
                                ks=(1, 5), aggregation="max", label_a="embedder", label_b="matcher") -> dict:
    """Парное McNemar/bootstrap: матчер (готовая sim) ↔ эмбеддер-baseline на ОДНИХ пробах (identity-level).
    -> {label_a, label_b (summarize), stats_<b>_vs_<a>@k, _perprobe}. Reuse ab_harness."""
    from .ab_harness import _paired_stats, per_probe_identity_hits, summarize
    ha = per_probe_identity_hits(emb_probe, probe_ids, emb_gallery, gallery_ids, ks, aggregation)
    hb = identity_hits_from_sim(sim_matcher, probe_ids, gallery_ids, ks, aggregation)
    out = {label_a: summarize(ha, p_coh, ks), label_b: summarize(hb, p_coh, ks)}
    for k in ks:
        out[f"stats_{label_b}_vs_{label_a}@{k}"] = _paired_stats(ha[k], hb[k])
    out["_perprobe"] = {label_a: ha, label_b: hb}
    return out


def adopt_matcher(cmp, baseline="embedder", challenger="matcher", alpha=0.05, primary_k=1, ks=(1, 5)) -> dict:
    """Rosa-правило (как embed_ab.adopt_finetune): challenger принят ТОЛЬКО при non-regression по всем k
    И значимом McNemar (p<alpha, c>b) по primary_k. Для матчера primary_k=1 (его цель — top-1). -> {decision,...}."""
    if primary_k not in ks:
        raise ValueError(f"adopt_matcher: primary_k={primary_k} не входит в ks={tuple(ks)}")
    base, chal = cmp[baseline]["overall"], cmp[challenger]["overall"]
    for k in ks:
        for nm, blk in ((baseline, base), (challenger, chal)):
            if f"recall@{k}" not in blk:
                raise ValueError(f"adopt_matcher: recall@{k} отсутствует в overall для {nm!r}")
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


def detailed_rows(sim_matcher, emb_probe, emb_gallery, probe_ids, gallery_ids, probe_md5, cohorts,
                  n_spots_probe=None, matched_counts=None, ks=(1, 5), aggregation="max") -> list:
    """Per-probe аудит: для каждой пробы — true_id, top1 матчера, ранг истинной особи у матчера
    и у эмбеддера (1-based, None если особи нет в галерее), hit@1/@5, число пятен/совпавших пар.
    emb_probe/emb_gallery → эмбеддер-ранжир (probe@gallery.T). -> list[dict] (для ab_spots_detailed.csv)."""
    sim_m = np.asarray(sim_matcher, float)
    gids = np.asarray(gallery_ids); pids = np.asarray(probe_ids)
    uniq = np.unique(gids)

    def _agg(sim):
        a = np.empty((sim.shape[0], len(uniq)), float)
        for j, u in enumerate(uniq):
            cols = sim[:, gids == u]
            a[:, j] = cols.max(axis=1) if aggregation == "max" else cols.mean(axis=1)
        return a

    am = _agg(sim_m)
    ae = _agg(np.asarray(emb_probe, float) @ np.asarray(emb_gallery, float).T)
    ranked_m = uniq[np.argsort(-am, axis=1, kind="stable")]   # stable: детерминизм ранга при ничьих
    ranked_e = uniq[np.argsort(-ae, axis=1, kind="stable")]
    rows = []
    for i in range(len(pids)):
        tid = pids[i]
        wm = np.where(ranked_m[i] == tid)[0]; we = np.where(ranked_e[i] == tid)[0]
        rm = int(wm[0]) + 1 if len(wm) else None
        re = int(we[0]) + 1 if len(we) else None
        rows.append({
            "probe_md5": probe_md5[i] if probe_md5 is not None else None,
            "true_id": tid, "cohort": cohorts[i] if cohorts is not None else None,
            "n_spots_probe": int(n_spots_probe[i]) if n_spots_probe is not None else None,
            "top1_id": ranked_m[i, 0], "top1_score": float(am[i].max()),
            "matcher_rank": rm, "embedder_rank": re,
            "hit@1": bool(rm == 1), "hit@5": bool(rm is not None and rm <= 5),
            "matched_pairs_count": int(matched_counts[i]) if matched_counts is not None else None,
        })
    return rows


def grid_ab(sims, probe_ids, gallery_ids, cohorts, ks=(1, 5), primary_k=1, aggregation="max") -> dict:
    """Сетка конфигураций (детектор×матчер×поверхность): {name: sim-матрица} → лучший по recall@primary_k
    (overall) на dev; тай-брейк — больший k. -> {best, primary_k, table{name:{recall@k}}}."""
    table = {}
    for name, sim in sims.items():
        o = summarize_sim(sim, probe_ids, gallery_ids, cohorts, ks, aggregation)["overall"]
        table[name] = {f"recall@{k}": o[f"recall@{k}"] for k in ks}
    best = max(table, key=lambda n: (table[n].get(f"recall@{primary_k}", 0.0),
                                     table[n].get(f"recall@{max(ks)}", 0.0)))
    return {"best": best, "primary_k": primary_k, "table": table}
