"""Open-set оценка эмбеддера re-ID (Блок 4) — ЧИСТАЯ логика (numpy; sklearn lazy внутри функций).

Заказчик прямо назвал задачу open metric learning: систему нужно уметь говорить «этой особи нет в базе».
Здесь — два независимых сигнала уверенности на пробу (score = близость к лучшему кандидату; margin =
отрыв top-1 от лучшего ДРУГОГО id). AUROC и порог Юдена считаются по score (max_sim); margin — только
вторичный гейт apply_policy, причём при дефолте open_set_margin_min=0.0 условие margin>=0 тождественно
истинно (margin неотрицателен по построению) — политика фактически score-only. Порог калибруется на
open_dev (open_test НЕ вскрывать — это дисциплина cli/Этапа C, не этого модуля).
"""
import numpy as np


def known_new_scores(probe_emb, gallery_emb, gallery_ids) -> dict:
    """Для каждой пробы (L2-norm эмбеддинг) → сигналы относительно галереи.

    -> {max_sim, margin, pred_id}:
      max_sim — наибольшее косинус-сходство с галереей (уверенность «известна»);
      margin  — max_sim − лучшее сходство среди ДРУГОГО individual_id (отрыв top-1 от ближайшей чужой особи);
      pred_id — individual_id ближайшего галерейного кадра (top-1).
    """
    sim = np.asarray(probe_emb, float) @ np.asarray(gallery_emb, float).T   # (P, G)
    gids = np.asarray(gallery_ids)
    order = np.argsort(-sim, axis=1, kind="stable")
    max_sim = np.empty(len(sim)); margin = np.empty(len(sim)); pred_id = np.empty(len(sim), dtype=gids.dtype)
    for i in range(len(sim)):
        ri = order[i]
        rsim, rid = sim[i, ri], gids[ri]
        max_sim[i] = rsim[0]
        pred_id[i] = rid[0]
        other = rid != rid[0]
        margin[i] = rsim[0] - (rsim[other][0] if other.any() else rsim[0])
    return {"max_sim": max_sim, "margin": margin, "pred_id": pred_id}


def open_set_auroc(scores, is_new) -> float:
    """AUROC разделения known(score высок) / new(score низок). Нужны ОБА класса, иначе nan.
    1.0 — идеальное разделение; ~0.5 — распределения перекрываются (как whole-body 2.0: AUROC 0.63)."""
    from sklearn.metrics import roc_auc_score
    is_new = np.asarray(is_new, bool)
    if is_new.all() or (~is_new).all():
        return float("nan")
    return float(roc_auc_score((~is_new).astype(int), np.asarray(scores, float)))


def tune_threshold(scores, is_new) -> float:
    """Порог known/new по индексу Юдена (max TPR−FPR на ROC), где positive = known. score≥thr → known.
    Калибруется на open_dev (новые особи известны), применяется на проде.

    Защита от вырождения: при одном классе (пустой open_dev для scope) или неразделяющем score sklearn
    кладёт в thresholds первым +inf — наивный argmax выбрал бы его → apply_policy(score≥inf) молча
    помечает ВСЕ пробы как new. Здесь: при одном классе возвращаем медиану scores; иначе игнорируем
    бесконечный порог перед argmax (всегда конечный порог).
    """
    from sklearn.metrics import roc_curve
    scores = np.asarray(scores, float)
    is_new = np.asarray(is_new, bool)
    if is_new.all() or (~is_new).all():
        return float(np.median(scores)) if scores.size else 0.0
    fpr, tpr, thr = roc_curve((~is_new).astype(int), scores)
    finite = np.isfinite(thr)
    j = int(np.argmax((tpr - fpr)[finite]))
    return float(thr[finite][j])


def apply_policy(max_sim, margin, thr, margin_min: float = 0.0) -> np.ndarray:
    """Политика known/new: известна, ЕСЛИ score≥thr И margin≥margin_min (два сигнала). -> bool-массив (known)."""
    return (np.asarray(max_sim, float) >= thr) & (np.asarray(margin, float) >= margin_min)
