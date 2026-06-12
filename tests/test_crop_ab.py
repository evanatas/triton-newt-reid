"""Тесты A/B-аналитики (чисто, без моделей): попадания, переключение по порогу, сводка, полный прогон."""
import numpy as np

from triton_crop.ab_harness import (gate_passed, per_probe_hits, run_ab_analysis, summarize,
                                     unroll_adopt_decision, unroll_adopt_explain, variant_embeddings)


def _ab_min(bo1, bo5, variants, raw1=0.0, raw5=0.0):
    ab = {"raw": {"overall": {"recall@1": raw1, "recall@5": raw5}},
          "belly_oriented": {"overall": {"recall@1": bo1, "recall@5": bo5}}}
    for name, (r1, r5) in variants.items():
        ab[name] = {"overall": {"recall@1": r1, "recall@5": r5}}
    return ab


def _ab_full(bo1, bo5, variants):
    """variants: {name: (recall@1, recall@5, p@1, p@5)} — с парными p для теста правила адопции."""
    ab = {"raw": {"overall": {"recall@1": 0.0, "recall@5": 0.0}},
          "belly_oriented": {"overall": {"recall@1": bo1, "recall@5": bo5}}}
    for name, (r1, r5, p1, p5) in variants.items():
        ab[name] = {"overall": {"recall@1": r1, "recall@5": r5}}
        ab[f"stats_{name}_vs_bo@1"] = {"mcnemar_p": p1}
        ab[f"stats_{name}_vs_bo@5"] = {"mcnemar_p": p5}
    return ab


def test_per_probe_hits_topk():
    g = np.array([[1., 0], [0, 1.]]); gid = ["A", "B"]
    p = np.array([[0.9, 0.1], [0.1, 0.9]]); pid = ["A", "B"]
    assert per_probe_hits(p, pid, g, gid, ks=(1,))[1].tolist() == [True, True]


def test_variant_embeddings_threshold_switch():
    raw = np.array([[1., 0], [2., 0]]); bo = np.array([[0, 1.], [0, 2.]]); conf = np.array([0.3, 0.1])
    out = variant_embeddings(raw, bo, conf, 0.2)            # 0.3>=0.2→bo; 0.1<0.2→raw
    assert out[0].tolist() == [0, 1.] and out[1].tolist() == [2., 0]
    assert variant_embeddings(raw, bo, conf, None).tolist() == raw.tolist()   # raw


def test_summarize_recall_and_cohort():
    hits = {1: np.array([True, False, True]), 5: np.array([True, True, True])}
    s = summarize(hits, ["TK", "TK", "PW"])
    assert abs(s["overall"]["recall@1"] - 2 / 3) < 1e-9 and s["overall"]["recall@5"] == 1.0
    assert s["TK"]["recall@1"] == 0.5 and s["PW"]["recall@1"] == 1.0


def test_run_ab_analysis_crop_helps_and_stats():
    g_ids = ["A", "B"]; p_ids = ["A", "B"]; p_coh = ["TK", "TK"]
    g_raw = np.array([[1., 0], [0, 1.]]); p_raw = np.array([[0, 1.], [1., 0]])   # raw → промах (к чужому ближе)
    g_bo = np.array([[1., 0], [0, 1.]]); p_bo = np.array([[1., 0], [0, 1.]])     # crop → точно к своему
    conf = np.array([0.9, 0.9])
    ab = run_ab_analysis(g_raw, g_bo, conf, g_ids, p_raw, p_bo, conf, p_ids, p_coh, thr=0.15, sweep=(0.15,))
    assert ab["raw"]["overall"]["recall@1"] == 0.0
    assert ab["belly_oriented"]["overall"]["recall@1"] == 1.0
    assert gate_passed(ab)
    assert ab["closed_set"]["n_excluded_no_true_id"] == 0
    assert ab["stats_closed@1"]["mcnemar_c"] == 2          # crop попал там, где raw нет — 2 раза


def test_run_ab_excludes_probe_without_true_id():
    g_ids = ["A"]; p_ids = ["A", "Z"]; p_coh = ["TK", "TK"]   # Z нет в галерее
    g_raw = np.array([[1., 0]]); g_bo = g_raw
    p_raw = np.array([[1., 0], [0, 1.]]); p_bo = p_raw
    conf_g = np.array([0.9]); conf_p = np.array([0.9, 0.9])
    ab = run_ab_analysis(g_raw, g_bo, conf_g, g_ids, p_raw, p_bo, conf_p, p_ids, p_coh, thr=0.15, sweep=(0.15,))
    assert ab["closed_set"]["n_excluded_no_true_id"] == 1     # Z исключён из paired
    assert ab["paired_success_only"]["raw"]["overall"]["n"] == 1


# --- Блок 3: A/B N-way + правило принятия unroll ---
def test_run_ab_three_way_unroll():
    g_ids = ["A", "B"]; p_ids = ["A", "B"]; p_coh = ["TK", "TK"]
    g_raw = np.array([[1., 0], [0, 1.]]); p_raw = np.array([[0, 1.], [1., 0]])    # raw → промах
    g_bo = np.array([[1., 0], [0, 1.]]); p_bo = np.array([[1., 0], [0, 1.]])      # bo → точно
    conf = np.array([0.9, 0.9])
    ab = run_ab_analysis(g_raw, g_bo, conf, g_ids, p_raw, p_bo, conf, p_ids, p_coh, thr=0.15,
                         sweep=(0.15,), extra_variants={"unroll_debend": (g_bo, p_bo)})
    assert ab["unroll_debend"]["overall"]["recall@1"] == 1.0
    assert "stats_unroll_debend_vs_bo@1" in ab and "sensitivity_unroll_debend" in ab


def test_run_ab_backward_compatible_without_extra():
    g_ids = ["A"]; p_ids = ["A"]; p_coh = ["TK"]
    g = np.array([[1., 0]]); p = np.array([[1., 0]]); conf = np.array([0.9])
    ab = run_ab_analysis(g, g, conf, g_ids, p, p, conf, p_ids, p_coh, thr=0.15, sweep=(0.15,))
    assert "unroll_debend" not in ab and gate_passed(ab)        # старый контракт цел


def test_adopt_significant_at5_safe():
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.55, 0.86, 1.0, 0.01)})
    assert unroll_adopt_decision(ab) == "unroll_debend"          # @5 значим, non-regress, pattern-safe


def test_adopt_baseline_when_not_significant():                 # P0-фикс: прирост на шуме → НЕ принимаем
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.55, 0.86, 1.0, 0.20)})
    assert unroll_adopt_decision(ab) == "belly_oriented"


def test_adopt_baseline_on_at5_regression():
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.60, 0.78, 0.01, 0.01)})   # @5 ниже baseline
    assert unroll_adopt_decision(ab) == "belly_oriented"


def test_adopt_excludes_pattern_unsafe():
    ab = _ab_full(0.50, 0.80, {"unroll_wnorm": (0.60, 0.90, 0.01, 0.01)})    # значим, но wnorm не safe
    assert unroll_adopt_decision(ab) == "belly_oriented"
    assert unroll_adopt_decision(ab, pattern_ok={"wnorm": True}) == "unroll_wnorm"


def test_adopt_picks_best_at5_among_eligible():
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.55, 0.85, 0.01, 0.01),
                               "unroll_ribbon": (0.52, 0.90, 0.01, 0.01)})
    assert unroll_adopt_decision(ab) == "unroll_ribbon"          # оба значимы; ribbon лучший @5


def test_adopt_explain_rationale():
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.55, 0.86, 1.0, 1.0)})    # не значим → baseline
    ex = unroll_adopt_explain(ab)
    assert ex["decision"] == "belly_oriented" and ex["alpha"] == 0.05 and len(ex["considered"]) == 1
    assert ex["allow_k1_significance"] is False and ex["primary_k"] == 5


def test_adopt_strict_at5_rejects_k1_only():
    ab = _ab_full(0.50, 0.80, {"unroll_debend": (0.60, 0.82, 0.01, 0.20)})  # @1 значим, @5 — нет
    assert unroll_adopt_decision(ab) == "belly_oriented"                    # строго @5 → не принят
    assert unroll_adopt_decision(ab, allow_k1_significance=True) == "unroll_debend"   # явный opt-in


def test_gate_passed_parametrized():
    ab = _ab_min(0.55, 0.82, {"unroll_debend": (0.60, 0.83)}, raw1=0.40, raw5=0.70)
    assert gate_passed(ab)                                       # default belly_oriented ≥ raw
    assert gate_passed(ab, 1, "unroll_debend", "belly_oriented")
