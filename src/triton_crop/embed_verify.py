"""Read-only аудит артефактов эмбеддера (Блок 4).

Зачем: в Блоке 4 случились (а) путаница Mega/MiewID и (б) рассинхрон `ab_embed_metrics.json` ↔ текущих
`oof/*.npy`. Этот модуль — детерминированный гейт, который СВЕРЯЕТ JSON-метрики с пересчётом
по `.npy`, фото-порядок md5 между прогонами, отсутствие запечатанных (test/open_test) md5 в OOF и
размерность чекпойнта. Падение = «артефакты нельзя нести в ВКР как авторитетные».

ЧИСТО (numpy) — тестируется на mock_embedder; загрузку .npy/json/ckpt делает CLI (`embed-verify`).
"""
import numpy as np

from .ab_harness import per_probe_hits, per_probe_identity_hits


def recompute_oof_recall(oof, variants=("belly_oriented", "unroll_ribbon"), ks=(1, 5)) -> dict:
    """Пересчёт overall recall@k ИЗ загруженного OOF (gallery↔probe по role). Возвращает оба уровня:
    'photo' (как per_probe_hits / ab_embed_metrics) и 'identity' (контракт ТЗ). -> {variant:{level:{k:r}}}."""
    role = np.asarray(oof["role"]); g = role == "gallery"; p = role == "probe"
    ids = np.asarray(oof["individual_id"])
    out = {}
    for v in variants:
        if v not in oof:
            continue
        e = np.asarray(oof[v], float)
        ph = per_probe_hits(e[p], ids[p], e[g], ids[g], ks)
        ih = per_probe_identity_hits(e[p], ids[p], e[g], ids[g], ks)
        out[v] = {"photo": {k: float(ph[k].mean()) for k in ks},
                  "identity": {k: float(ih[k].mean()) for k in ks}}
    return out


def verify_embed_artifacts(oof, expected_recall=None, forbidden_md5=None, ckpt_embed_dim=None,
                           compare_md5=None, variants=("belly_oriented", "unroll_ribbon"),
                           ks=(1, 5), tol: float = 1e-4) -> dict:
    """Аудит OOF-словаря (+ опц. сверки). -> {ok, checks:[{name, ok, detail}]}.

    Проверки (включаются переданными аргументами):
      shapes_consistent          — N строк совпадает у всех вариантов и метаполей;
      md5_order_matches_compare   — порядок md5 == compare_md5 (фото-выравнивание Mega/MiewID);
      metrics_json_matches_npy    — overall recall@k из JSON (expected_recall) == пересчёт из .npy;
      no_sealed_md5_in_oof        — нет forbidden_md5 (test/open_test) в OOF;
      checkpoint_embed_dim_matches_oof — размерность .npy == embed_dim чекпойнта.
    """
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    md5 = np.asarray(oof["md5"])
    n = len(md5)
    det = {"n": n}
    shape_ok = True
    for k in ("role", "individual_id", "cohort"):
        det[k] = int(len(np.asarray(oof[k]))); shape_ok &= det[k] == n
    for v in variants:
        if v in oof:
            arr = np.asarray(oof[v]); det[v] = list(arr.shape); shape_ok &= arr.shape[0] == n
    add("shapes_consistent", shape_ok, det)

    if compare_md5 is not None:
        cmp = np.asarray(compare_md5)
        same = len(cmp) == n and bool((md5 == cmp).all())
        add("md5_order_matches_compare", same, {"n": n, "n_compare": int(len(cmp))})

    if expected_recall is not None:
        rc = recompute_oof_recall(oof, variants, ks)
        ok = True; diffs = {}
        for v, exp in expected_recall.items():
            for k in ks:
                e = exp.get(k) if isinstance(exp, dict) else None
                got = rc.get(v, {}).get("photo", {}).get(k)
                if e is None or got is None:
                    continue
                d = abs(float(got) - float(e))
                diffs[f"{v}@{k}"] = {"json": round(float(e), 4), "recompute": round(float(got), 4),
                                     "diff": round(d, 4)}
                ok &= d <= tol
        add("metrics_json_matches_npy", ok, diffs)

    if forbidden_md5 is not None:
        inter = sorted(set(md5.tolist()) & set(forbidden_md5))
        add("no_sealed_md5_in_oof", len(inter) == 0, {"n_intersection": len(inter), "sample": inter[:10]})

    if ckpt_embed_dim is not None:
        v0 = next((v for v in variants if v in oof), None)
        dim = int(np.asarray(oof[v0]).shape[1]) if v0 is not None else None
        add("checkpoint_embed_dim_matches_oof", dim == int(ckpt_embed_dim),
            {"oof_dim": dim, "ckpt_dim": int(ckpt_embed_dim)})

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
