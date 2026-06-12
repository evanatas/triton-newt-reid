"""Финальный sealed-test отчёт ВКР: identity-level recall + open-set из ГОТОВЫХ эмбеддингов (ЧИСТАЯ логика).

Композирует `spot_ab` (recall@k overall / per-cohort / coverage из sim-матрицы) и `embed_eval`
(open-set AUROC + политика known/new). Применяется ОДИН раз к запечатанным test/open_test на заявленной
лучшей системе (zero-shot MegaDescriptor + unroll_ribbon) — после вскрытия настройки НЕ тюнятся.
Чистый numpy, тестируется на синтетике; сами эмбеддинги считает CLI (`embed-test`) лениво.
"""
import numpy as np

SEALED_STAGES = ("test", "open_test")   # запечатанные срезы — вскрываются ОДИН раз (sealed-test), потом не тюнятся


def assert_unsealed(stages, unseal: bool) -> None:
    """Единый гейт C9 (анти-утечка): запрещает работу с запечатанными test/open_test без явного --unseal.
    Срабатывает ДО загрузки моделей. `stages` — итерабельное имён стадий. Источник истины для cmd_crop /
    cmd_embed_test / cmd_detect_spots / cmd_match (раньше логика дублировалась). ValueError при нарушении."""
    sealed = sorted({str(s).strip() for s in stages} & set(SEALED_STAGES))
    if sealed and not unseal:
        raise ValueError(f"Стадии {sealed} запечатаны (гейт C9): нужен явный флаг --unseal "
                         f"(необратимое вскрытие, финальный замер ВКР; после вскрытия настройки НЕ тюнить).")


def build_sealed_report(test_emb, test_ids, test_cohorts, open_emb, gallery_emb, gallery_ids,
                        ks=(1, 5), n_official_test=None, openset_threshold=None,
                        margin_min: float = 0.0, aggregation: str = "max") -> dict:
    """Финальные числа ВКР: identity-level recall@k (overall + per-cohort + coverage) + open-set (known/new).

    test_emb/open_emb/gallery_emb — L2-norm эмбеддинги (P×D / O×D / G×D); *_ids — individual_id;
    test_cohorts — когорта пробы (TK = temporal-срез перепоимок Карелины, PW — Ребристый).
    open_emb=None/пусто → open-set AUROC=nan. openset_threshold (опц.) калибруется на dev — НЕ на test.
    -> dict (JSON-сериализуем): {recall:{overall,per_cohort,closedset_reid_recall,pipeline_recall,coverage},
    openset:{auroc,n_known,n_new,threshold,known_kept_rate,new_falsely_known_rate}, ks, aggregation}.
    """
    from .embed_eval import apply_policy, known_new_scores, open_set_auroc
    from .spot_ab import recall_split_sim, summarize_sim

    test_emb = np.asarray(test_emb, float)
    gallery_emb = np.asarray(gallery_emb, float)
    gallery_ids = np.asarray(gallery_ids)
    test_ids = np.asarray(test_ids)
    test_cohorts = np.asarray(test_cohorts)

    sim = test_emb @ gallery_emb.T                                # (P×G) косинусы (эмбеддинги L2-norm)
    summ = summarize_sim(sim, test_ids, gallery_ids, test_cohorts, ks, aggregation)
    split = recall_split_sim(sim, test_ids, gallery_ids, ks, n_official_test, aggregation)
    recall = {"overall": summ["overall"],
              "per_cohort": {c: summ[c] for c in summ if c != "overall"},
              "closedset_reid_recall": split["closedset_reid_recall"],
              "pipeline_recall": split["pipeline_recall"], "coverage": split["coverage"]}

    n_new = 0 if open_emb is None else len(np.asarray(open_emb, float))
    openset = {"auroc": float("nan"), "n_known": int(len(test_emb)), "n_new": int(n_new),
               "threshold": None, "margin_min": float(margin_min),
               "known_kept_rate": None, "new_falsely_known_rate": None}
    if n_new:
        probe = np.vstack([test_emb, np.asarray(open_emb, float)])
        is_new = np.array([False] * len(test_emb) + [True] * n_new)
        sc = known_new_scores(probe, gallery_emb, gallery_ids)
        openset["auroc"] = open_set_auroc(sc["max_sim"], is_new)
        if openset_threshold is not None:
            known = apply_policy(sc["max_sim"], sc["margin"], float(openset_threshold), margin_min)
            openset["threshold"] = float(openset_threshold)
            openset["known_kept_rate"] = float(known[~is_new].mean()) if (~is_new).any() else None
            openset["new_falsely_known_rate"] = float(known[is_new].mean()) if is_new.any() else None

    return {"recall": recall, "openset": openset, "ks": list(ks), "aggregation": aggregation}
