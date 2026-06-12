"""Гейты Блока 2 + EDA по crops_manifest.

run_crop_gates — инварианты корректности/анти-утечки (стиль G1–G12 Блока 1), ValidationError при
нарушении. Здесь хермэтично проверяются C1 (покрытие), C2 (реконсиляция), C3 (нет зеркала),
C4 (идентичность пайплайна), C7 (бюджет fallback). C5 — детерминизм (юнит в pipeline-тесте);
C6 (head-up tolerance), C8 (A/B записан), C9 (анти-утечка test) — на реальных данных/модельном этапе.
"""
from pathlib import Path


class ValidationError(Exception):
    """Нарушение жёсткого инварианта Блока 2."""


def check(condition, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def run_crop_gates(crops_df, manifest_df, cfg=None, fallback_budget: float = 0.10,
                   ab_metrics=None, unsealed: bool = False) -> list[str]:
    """unsealed=True — финальный режим ПОСЛЕ легитимного sealed-test: манифест ЗАКОННО содержит test/open_test
    кропы, поэтому анти-утечка C9 не применяется (иначе ложный провал на корректном финальном манифесте)."""
    warnings: list[str] = []

    # C1 — покрытие: каждая kpi_core gallery+probe (dup_keep) в train/dev имеет кроп.
    # test-LOCK НЕ кропается до Блока 6 (sealed); open_new — отдельные стадии → исключаем.
    if {"dup_keep", "kpi_scope", "split_role", "md5"} <= set(manifest_df.columns):
        keep = manifest_df[manifest_df["dup_keep"] == True]  # noqa: E712
        req_rows = keep[(keep["kpi_scope"] == "kpi_core")
                        & (keep["split_role"].isin(["gallery", "probe"]))]
        if "split_fold" in req_rows.columns:
            req_rows = req_rows[req_rows["split_fold"].isin(["train", "dev"])]
        if "is_new_open" in req_rows.columns:
            req_rows = req_rows[~req_rows["is_new_open"].astype(bool)]
        missing = set(req_rows["md5"]) - set(crops_df["md5"])
        check(not missing, f"C1: нет кропов для {len(missing)} kpi_core (train/dev) строк")

    # C2 — реконсиляция: каждый кроп ссылается на существующую строку манифеста
    orphans = set(crops_df["md5"]) - set(manifest_df["md5"])
    check(not orphans, f"C2: кропы без строки в манифесте: {sorted(orphans)[:3]}")

    # C3 — нет зеркала (хиральность = идентичность)
    check((~crops_df["mirrored"].astype(bool)).all(), "C3: применён зеркальный поворот")

    # C4 — идентичность пайплайна (среди не-full, ВКЛ. unroll): один canon_size и pipeline_version
    real = crops_df[crops_df["variant"] != "full"]
    if len(real):
        check(real["canon_size"].nunique() == 1, "C4: разный canon_size у кропов")
        check(real["pipeline_version"].nunique() == 1, "C4: разный pipeline_version у кропов")
        if cfg is not None:
            check(int(real["canon_size"].iloc[0]) == int(cfg.canon_size),
                  f"C4: canon_size {int(real['canon_size'].iloc[0])} ≠ cfg.canon_size {cfg.canon_size}")

    # C7 — бюджет fallback на kpi_core. Распрямлённые варианты unroll_* НЕ считаем (это про seg-fallback,
    # а unroll-строки всегда ok → иначе они искусственно разбавляли бы долю).
    core = crops_df[crops_df["kpi_scope"] == "kpi_core"]
    core = core[~core["variant"].astype(str).str.startswith("unroll_")]
    if len(core):
        # clipped_after_rotation — аудит-метка (геометрия кропа прежняя), не fallback-ступень.
        frac = float((~core["crop_status"].isin(["ok", "clipped_after_rotation"])).mean())
        check(frac <= fallback_budget, f"C7: доля fallback {frac:.2f} > {fallback_budget}")

    # C6 (мягко) — доля belly_oriented головой ВВЕРХ (head_y < cloaca_y в каноне)
    bo = crops_df[crops_df["variant"] == "belly_oriented"]
    if len(bo) and {"head_y", "cloaca_y"}.issubset(bo.columns):
        up = float((bo["head_y"] < bo["cloaca_y"]).mean())
        if up < 0.85:
            warnings.append(f"C6: только {up:.0%} belly_oriented головой вверх (<85%)")

    # C9 — анти-утечка: запечатанный test-LOCK НЕ должен быть кропнут на этапе разработки
    # (в финальном режиме unsealed=True test ЗАКОННО присутствует — C9 не применяется).
    if not unsealed and "split_fold" in manifest_df.columns:
        leaked = set(manifest_df[(manifest_df["split_fold"] == "test")
                                 & (manifest_df["kpi_scope"] == "kpi_core")]["md5"]) & set(crops_df["md5"])
        check(not leaked, f"C9: кропы для запечатанного test ({len(leaked)}) — утечка")

    # C8 — A/B записан для распрямлённых вариантов (Блок 3): Rosa-гейт ОБЯЗАТЕЛЕН (хард, не warning)
    unroll_present = sorted(v for v in crops_df["variant"].astype(str).unique() if v.startswith("unroll_"))
    if unroll_present:
        check(ab_metrics is not None,
              "C8: есть unroll-варианты, но НЕТ ab_metrics.json — A/B обязателен (Rosa), запусти `cli ab`")
        missing_ab = [v for v in unroll_present if v not in ab_metrics]
        check(not missing_ab, f"C8: в ab_metrics нет вариантов {missing_ab}")
        decision = ab_metrics.get("unroll_adopt_decision")
        check(decision is not None, "C8: нет unroll_adopt_decision в ab_metrics")
        check(decision == "belly_oriented" or decision in unroll_present,
              f"C8: unroll_adopt_decision={decision!r} не из belly_oriented ∪ {unroll_present}")
        # Major1: пересчитать решение по самому ab_metrics (числа + сохранённый pattern_safe) и сверить —
        # ловит рассинхрон/ручную правку артефакта (защита от возврата прежней дыры с неверным decision).
        rat = ab_metrics.get("unroll_adopt_rationale")
        if isinstance(rat, dict) and isinstance(ab_metrics.get("belly_oriented"), dict) \
                and "overall" in ab_metrics["belly_oriented"]:
            from .ab_harness import unroll_adopt_decision
            pattern_ok = {str(c.get("variant", ""))[len("unroll_"):]: bool(c.get("pattern_safe", False))
                          for c in rat.get("considered", []) if str(c.get("variant", "")).startswith("unroll_")}
            recomputed = unroll_adopt_decision(
                ab_metrics, pattern_ok=pattern_ok, baseline=rat.get("baseline", "belly_oriented"),
                alpha=rat.get("alpha", 0.05), primary_k=rat.get("primary_k", 5),
                allow_k1_significance=rat.get("allow_k1_significance", False))
            check(recomputed == decision,
                  f"C8: decision={decision!r} ≠ пересчёт по ab_metrics ({recomputed!r}) — рассинхрон артефакта")

    return warnings


def run_spot_gates(spots_df, manifest_df=None, ab_metrics=None, min_spots: int = 3,
                   coverage_min: float = 0.5, unsealed: bool = False) -> list[str]:
    """Гейты Блока 5 (матчинг созвездия пятен). ValidationError при нарушении хард-инвариантов.

      S6 — детекция ТОЛЬКО на belly_oriented/unroll_debend (НЕ ribbon: сливает пятна, spot-QA Блока 3) [хард];
      S7 — анти-утечка: запечатанный test (kpi_core) не должен попадать в spots (как C9) [хард];
      S1 — покрытие детектора: доля кропов с >= min_spots пятен [мягкий warning];
      S5 — A/B записан + ПЕРЕСЧИТЫВАЕТСЯ по артефакту (Rosa, как C8): matcher_adopt_decision совпадает
           с пересчётом adopt_matcher по сохранённому compare [хард].
    (S2 нет-зеркала / S3 симметрия+детерминизм / S4 инвариантность — юнит-уровень в тестах матчера.)
    """
    warnings: list[str] = []

    if "detect_variant" in spots_df.columns:
        bad = sorted(set(spots_df["detect_variant"].astype(str)) - {"belly_oriented", "unroll_debend"})
        check(not bad, f"S6: детекция пятен на запрещённой поверхности {bad} "
                       f"(только belly_oriented/unroll_debend — ribbon сливает пятна)")

    if not unsealed and manifest_df is not None and {"split_fold", "kpi_scope", "md5"} <= set(manifest_df.columns) \
            and "md5" in spots_df.columns:
        leaked = set(manifest_df[(manifest_df["split_fold"] == "test")
                                 & (manifest_df["kpi_scope"] == "kpi_core")]["md5"]) & set(spots_df["md5"])
        check(not leaked, f"S7: пятна для запечатанного test ({len(leaked)}) — утечка")

    if "n_spots" in spots_df.columns and len(spots_df):
        cov = float((spots_df["n_spots"] >= min_spots).mean())
        if cov < coverage_min:
            warnings.append(f"S1: только {cov:.0%} кропов имеют >={min_spots} пятен "
                            f"(<{coverage_min:.0%}) — матчинг ограничен")

    if ab_metrics is not None and ("matcher_adopt_decision" in ab_metrics or "compare" in ab_metrics):
        dec = ab_metrics.get("matcher_adopt_decision")
        check(dec in ("embedder", "matcher"), f"S5: matcher_adopt_decision={dec!r} не из {{embedder, matcher}}")
        cmp = ab_metrics.get("compare")
        rat = ab_metrics.get("matcher_adopt_rationale") or {}
        if isinstance(cmp, dict) and "embedder" in cmp and "matcher" in cmp:
            from .spot_ab import adopt_matcher
            pk = int(rat.get("primary_k", 1))
            ks = tuple(ab_metrics.get("recall_ks", (1, 5)))
            recomputed = adopt_matcher(cmp, primary_k=pk, ks=ks)["decision"]
            check(recomputed == dec,
                  f"S5: matcher_adopt_decision={dec!r} ≠ пересчёт по compare ({recomputed!r}) — рассинхрон артефакта")

    return warnings


def write_crop_eda(crops_df, out_dir) -> None:
    """EDA: счётчики по variant/crop_status + ПОКРЫТИЕ распрямления (vs belly_oriented) → crop_eda.md."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_unroll = crops_df["variant"].astype(str).str.startswith("unroll_")
    lines = ["# EDA — Блоки 2–3 (кроп + распрямление)", "", "## Варианты кропа", ""]
    lines += [f"- {k}: {int(v)}" for k, v in crops_df.groupby("variant").size().items()]
    lines += ["", "## crop_status (базовый кроп, без unroll)", ""]
    lines += [f"- {k}: {int(v)}" for k, v in crops_df[~is_unroll].groupby("crop_status").size().items()]
    n_bo = int((crops_df["variant"] == "belly_oriented").sum())
    unroll_vs = sorted(crops_df.loc[is_unroll, "variant"].astype(str).unique())
    if unroll_vs and n_bo:
        lines += ["", "## Покрытие распрямления (vs belly_oriented)", ""]
        lines += [f"- {v}: {int((crops_df['variant'] == v).sum())}/{n_bo} "
                  f"({int((crops_df['variant'] == v).sum()) / n_bo:.1%})" for v in unroll_vs]
    if "kpi_scope" in crops_df.columns:
        core = crops_df[(crops_df["kpi_scope"] == "kpi_core") & (~is_unroll)]   # без unroll (не разбавлять)
        if len(core):
            ok = float((core["crop_status"] == "ok").mean())
            lines += ["", f"kpi_core (базовый кроп): доля ok = {ok:.2%}, fallback = {1 - ok:.2%}"]
    (out / "crop_eda.md").write_text("\n".join(lines), encoding="utf-8")
