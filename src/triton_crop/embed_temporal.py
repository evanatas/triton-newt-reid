"""Temporal-срез оценки эмбеддера для TK/LAB (эксперимент C2/C3/C4) — ЧИСТО (numpy).

temporal-проба = probe особи, у которой в галерее есть кадр той же особи в ДРУГОЙ сессии (перепоимка).
Считает recall@k на этом срезе + парный McNemar между конфигами (один и тот же набор проб → пары
выровнены). Reuse spot_ab.identity_hits_from_sim + stats.mcnemar. Sealed НЕ трогается.
"""
import numpy as np

from .spot_ab import identity_hits_from_sim
from .stats import mcnemar


def temporal_probe_mask(probe_ids, probe_sessions, gallery_ids, gallery_sessions) -> np.ndarray:
    """bool-маска по probe: True, если у этой особи в галерее есть кадр в ДРУГОЙ сессии (перепоимка)."""
    g_ids = np.asarray(gallery_ids); g_sess = np.asarray(gallery_sessions)
    by_id = {}
    for i, s in zip(g_ids.tolist(), g_sess.tolist()):
        by_id.setdefault(i, set()).add(s)
    out = []
    for pid, ps in zip(np.asarray(probe_ids).tolist(), np.asarray(probe_sessions).tolist()):
        sess = by_id.get(pid, set())
        out.append(bool(sess - {ps}))                    # есть галерейная сессия ≠ probe-сессии
    return np.array(out, bool)


def temporal_recall(oof, variant: str = "unroll_ribbon", cohort_filter: str = "TK",
                    ks=(1, 5)) -> tuple:
    """recall@k на temporal-пробах когорты cohort_filter из OOF-словаря (ключи: variant-массив,
    role, individual_id, session, cohort). -> (rep{recall@k, n}, hits{k: bool-массив выровненных проб})."""
    emb = np.asarray(oof[variant], float)
    role = np.asarray(oof["role"]); ids = np.asarray(oof["individual_id"])
    sess = np.asarray(oof["session"]); coh = np.asarray(oof["cohort"])
    g, p = role == "gallery", role == "probe"
    sim = emb[p] @ emb[g].T
    hits_all = identity_hits_from_sim(sim, ids[p], ids[g], ks)        # по ВСЕМ пробам (галерея полная)
    keep = temporal_probe_mask(ids[p], sess[p], ids[g], sess[g]) & (coh[p] == cohort_filter)
    rep = {"n": int(keep.sum())}
    hits = {}
    for k in ks:
        hk = np.asarray(hits_all[k])[keep]
        hits[k] = hk
        rep[f"recall@{k}"] = float(hk.mean()) if hk.size else 0.0
    return rep, hits


def build_ablation_report(configs: dict, variant: str = "unroll_ribbon", cohort_filter: str = "TK",
                          ks=(1, 5), primary_k: int = 1) -> dict:
    """configs = {имя_конфига: oof-словарь}. -> {per_config{имя: {recall@k, n}}, mcnemar{B_vs_A: {b,c,p}}}.

    McNemar строится для каждого конфига против ПЕРВОГО (базового) на primary_k. Пробы одинаковы между
    конфигами (один OOF-набор) → пары выровнены; если длины не совпали — пропуск (несравнимо)."""
    names = list(configs)
    per, hits_by = {}, {}
    for nm in names:
        rep, hits = temporal_recall(configs[nm], variant, cohort_filter, ks)
        per[nm], hits_by[nm] = rep, hits
    base = names[0]
    mc = {}
    for nm in names[1:]:
        ha, hb = hits_by[base].get(primary_k), hits_by[nm].get(primary_k)
        if ha is None or hb is None or len(ha) != len(hb):
            continue
        b, c, p = mcnemar(ha, hb)                         # b: base попал/nm нет; c: nm попал/base нет
        mc[f"{nm}_vs_{base}"] = {"b": b, "c": c, "p": p, "primary_k": primary_k}
    return {"per_config": per, "mcnemar": mc, "variant": variant,
            "cohort_filter": cohort_filter, "ks": list(ks)}
