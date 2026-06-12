"""Гибрид Блока 6: эмбеддер top-K shortlist → матчер созвездия переранжирует внутри (ЧИСТО, numpy).

Идея (план проекта): эмбеддер закрывает top-K (грубый отбор), матчер пятен бьёт в top-1 ВНУТРИ малого
набора кандидатов — там спурьёзные дальние особи уже отсечены shortlist'ом, матчеру легче. Обе входные
матрицы — n_probe×n_gallery (эмбеддер: probe@gallery.T; матчер: доля совпавших пятен), выровнены по md5.

Безопасность (УСЛОВНАЯ): при method='rerank' нули/ничьи матчера внутри top-K сохраняют порядок
эмбеддера (стабильная сортировка) → гибрид == эмбеддер. При НЕнулевых ошибочных скорах матчера
(спурьёзные матчи — наш случай по результатам экспериментов) переранжир внутри top-K может опустить истинную особь
→ recall@5 может проседать; «не хуже» строго верно ТОЛЬКО для нулевого матчера. Матчер не вытаскивает
особь из-за пределов top-K (shortlist-гейт). Контракт ТЗ: ранжируем ОСОБИ (identity-level).
"""
import numpy as np


def check_sims_provenance(stored: dict, want: dict, keys=("scope", "surface", "method", "matcher",
                                                           "ransac_iters", "embed_variant",
                                                           "sha_crops_manifest")) -> None:
    """Сверить провенанс сохранённых sim (hybrid_sims.npz) с текущими аргументами. ValueError
    при рассинхроне — чтобы --reuse-sims не использовал молча stale sim и не перезаписывал метрики."""
    diff = {k: (stored.get(k), want.get(k)) for k in keys if str(stored.get(k)) != str(want.get(k))}
    if diff:
        raise ValueError(f"hybrid --reuse-sims: провенанс сохранённых sim не совпадает с аргументами {diff}. "
                         f"Перегенерируй sim БЕЗ --reuse-sims.")


def _identity_agg(sim, gallery_ids, aggregation="max"):
    """sim (P×G) + gallery_ids → (agg P×U, uniq_ids U). Агрегация фото галереи до особей (max/mean)."""
    sim = np.asarray(sim, float)
    gids = np.asarray(gallery_ids)
    uniq = np.unique(gids)
    agg = np.empty((sim.shape[0], len(uniq)), float)
    for j, u in enumerate(uniq):
        cols = sim[:, gids == u]
        agg[:, j] = cols.max(axis=1) if aggregation == "max" else cols.mean(axis=1)
    return agg, uniq


def _z(v):
    """z-нормировка вектора (для линейного слияния разно-масштабных скоров). Нулевой разброс → нули."""
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def embedder_hits(embed_sim, gallery_ids, probe_ids, ks=(1, 5), aggregation="max") -> dict:
    """Baseline: identity-level recall@k эмбеддера (ранжир особей по агрегированной sim). -> {k: bool}."""
    agg, uniq = _identity_agg(embed_sim, gallery_ids, aggregation)
    ranked = uniq[np.argsort(-agg, axis=1, kind="stable")]   # stable: согласован с fuse_ranked_ids
    pid = np.asarray(probe_ids)[:, None]
    return {k: (ranked[:, :k] == pid).any(axis=1) for k in ks}


def fuse_ranked_ids(embed_sim, matcher_sim, gallery_ids, k=10, method="rerank", alpha=0.5,
                    aggregation="max") -> np.ndarray:
    """Слияние → матрица ранжированных ОСОБЕЙ (P×U). top-K берётся по эмбеддеру; внутри K — переранжир
    матчером; хвост (>K) сохраняет порядок эмбеддера. method ∈ {rerank, linear}.
    Нули/ничьи матчера → порядок эмбеддера; ненулевой ошибочный матчер может ухудшить @5 внутри top-K."""
    ea, uniq = _identity_agg(embed_sim, gallery_ids, aggregation)
    ma, _ = _identity_agg(matcher_sim, gallery_ids, aggregation)
    P, U = ea.shape
    k = int(min(k, U))
    out = np.empty((P, U), dtype=uniq.dtype)
    for i in range(P):
        order_e = np.argsort(-ea[i], kind="stable")        # ранжир особей эмбеддером
        cand = order_e[:k]
        if method == "linear":
            comb = alpha * _z(ea[i, cand]) + (1.0 - alpha) * _z(ma[i, cand])
            ranked = cand[np.argsort(-comb, kind="stable")]
        else:                                               # 'rerank': стабильно по матчеру, ничьи/нули → порядок эмбеддера
            ranked = cand[np.argsort(-ma[i, cand], kind="stable")]
        out[i] = uniq[np.concatenate([ranked, order_e[k:]])]
    return out


def fuse_hits(embed_sim, matcher_sim, gallery_ids, probe_ids, k=10, method="rerank", alpha=0.5,
              ks=(1, 5), aggregation="max") -> dict:
    """recall@k гибрида (identity-level). -> {k: bool-массив}."""
    ranked = fuse_ranked_ids(embed_sim, matcher_sim, gallery_ids, k, method, alpha, aggregation)
    pid = np.asarray(probe_ids)[:, None]
    return {kk: (ranked[:, :kk] == pid).any(axis=1) for kk in ks}


def compare_hybrid_vs_embedder(embed_sim, matcher_sim, gallery_ids, probe_ids, cohorts, k=10,
                               method="rerank", alpha=0.5, ks=(1, 5), aggregation="max") -> dict:
    """Парное McNemar/bootstrap: гибрид ↔ эмбеддер-alone на ОДНИХ пробах (identity-level). Reuse ab_harness."""
    from .ab_harness import _paired_stats, summarize
    he = embedder_hits(embed_sim, gallery_ids, probe_ids, ks, aggregation)
    hh = fuse_hits(embed_sim, matcher_sim, gallery_ids, probe_ids, k, method, alpha, ks, aggregation)
    out = {"embedder": summarize(he, cohorts, ks), "hybrid": summarize(hh, cohorts, ks)}
    for kk in ks:
        out[f"stats_hybrid_vs_embedder@{kk}"] = _paired_stats(he[kk], hh[kk])
    out["_perprobe"] = {"embedder": he, "hybrid": hh}
    return out


def fuse_grid(embed_sim, matcher_sim, gallery_ids, probe_ids, cohorts, methods=("rerank", "linear"),
              ks_topk=(5, 10, 20), alphas=(0.3, 0.5, 0.7), recall_ks=(1, 5), aggregation="max") -> dict:
    """Сетка слияния (method × top-K × alpha) → лучшая по recall@1 (overall); тай-брейк @max. + baseline
    эмбеддера. -> {best{method,k,alpha,recall@k}, embedder{recall@k}, table[]}.
    Внимание: best выбирается по recall@1 БЕЗ проверки non-regression по @5 (в отличие от
    spot_ab.adopt_matcher) — exploratory, не adopt-правило."""
    base = embedder_hits(embed_sim, gallery_ids, probe_ids, recall_ks, aggregation)
    emb_row = {f"recall@{kk}": float(np.asarray(base[kk]).mean()) for kk in recall_ks}
    table = []
    for method in methods:
        al_list = alphas if method == "linear" else (None,)
        for k in ks_topk:
            for al in al_list:
                hh = fuse_hits(embed_sim, matcher_sim, gallery_ids, probe_ids, k, method,
                               al if al is not None else 0.5, recall_ks, aggregation)
                row = {"method": method, "k": int(k), "alpha": al,
                       **{f"recall@{kk}": float(np.asarray(hh[kk]).mean()) for kk in recall_ks}}
                table.append(row)
    pk = recall_ks[0]
    best = max(table, key=lambda r: (r[f"recall@{pk}"], r[f"recall@{max(recall_ks)}"]))
    return {"best": best, "embedder": emb_row, "table": table}
