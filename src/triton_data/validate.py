"""Жёсткие гейты + EDA-отчёт по манифесту.

run_gates — 12 инвариантов корректности и анти-утечки; падает (ValidationError) при нарушении.
write_eda — markdown-отчёт + фигуры (matplotlib/PIL) для глазной проверки.
"""
import re
from datetime import date as _date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .imageio import load_canonical  # noqa: E402

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")
_EXPECTED_DIMS = {"TK": {3024, 4032}, "PW": {3024, 4032}, "LAB": {3000, 4000}}


class ValidationError(Exception):
    """Нарушение жёсткого инварианта манифеста (в отличие от assert не вырезается под -O)."""


def check(condition, message: str) -> None:
    """Жёсткая проверка инварианта; при провале — ValidationError."""
    if not condition:
        raise ValidationError(message)


# ─────────────────────────── ГЕЙТЫ ───────────────────────────

def run_gates(target: pd.DataFrame, external: pd.DataFrame, cfg) -> list[str]:
    """Проверить 12 инвариантов. Вернуть список мягких предупреждений (G10).
    Жёсткие нарушения → ValidationError."""
    warnings: list[str] = []
    allrows = pd.concat([target, external], ignore_index=True)
    keep = allrows[allrows["dup_keep"] == True]      # noqa: E712
    tkeep = target[target["dup_keep"] == True]        # noqa: E712

    # G1 — нет md5 одновременно в gallery и probe
    g_md5 = set(tkeep[tkeep["split_role"] == "gallery"]["md5"])
    p_md5 = set(tkeep[tkeep["split_role"] == "probe"]["md5"])
    check(g_md5.isdisjoint(p_md5), f"G1: одна фотография в gallery и probe: {sorted(g_md5 & p_md5)[:3]}")

    # G2 — каждая closed-set probe-особь присутствует в галерее
    closed_probe = set(tkeep[(tkeep["split_role"] == "probe") & (~tkeep["is_new_open"])]["individual_id"])
    gallery_ids = set(tkeep[tkeep["split_role"] == "gallery"]["individual_id"])
    check(closed_probe <= gallery_ids, f"G2: probe-особи без галереи: {sorted(closed_probe - gallery_ids)[:5]}")

    # G3 — external никогда не gallery/probe/в target
    check((external["role"] == "external").all(), "G3: во внешнем манифесте есть не-external")
    check("GCN" not in set(target["cohort"]), "G3: GCN попал в target-манифест")

    # G4 — ровно один выживший на группу дублей
    for gi, grp in allrows[allrows["dup_group"] >= 0].groupby("dup_group"):
        n_keep = int((grp["dup_keep"] == True).sum())  # noqa: E712
        check(n_keep == 1, f"G4: в группе {gi} выживших {n_keep} (должен быть 1)")

    # G5 — счётчики сходятся; каждый dup_keep target имеет схему
    n_keep = int((allrows["dup_keep"] == True).sum())   # noqa: E712
    n_drop = int((allrows["dup_keep"] == False).sum())  # noqa: E712
    check(n_keep + n_drop == len(allrows), "G5: dup_keep не покрывает все строки")
    check(tkeep["split_scheme"].notna().all(), "G5: есть выжившая target-строка без split_scheme")

    # G6 — probe ⇒ fold∈{dev,test}; gallery ⇒ train (probe НИКОГДА не в обучении)
    probe_folds = tkeep[tkeep["split_role"] == "probe"]["split_fold"]
    check(probe_folds.isin(["dev", "test"]).all(), "G6: probe вне dev/test (риск утечки в train)")
    gal_folds = tkeep[tkeep["split_role"] == "gallery"]["split_fold"]
    check((gal_folds == "train").all(), "G6: gallery вне train")

    # G7 — пути относительные и существуют
    ws = Path(cfg.workspace_root)
    for rel in allrows["rel_path"]:
        check(not str(rel).startswith("/"), f"G7: абсолютный путь: {rel}")
    for rel, md5 in zip(allrows["rel_path"], allrows["md5"]):
        if not str(md5).startswith("MISSING:"):
            check((ws / rel).exists(), f"G7: файл не найден: {rel}")

    # G8 — неймспейсы id не пересекаются между когортами; префикс = когорта
    id_cohort = allrows.groupby("individual_id")["cohort"]
    check((id_cohort.nunique() == 1).all(), "G8: individual_id встречается в нескольких когортах")
    first_cohort = id_cohort.agg(lambda s: s.iloc[0])
    bad = [iid for iid, ch in first_cohort.items() if iid.split("-")[0] != ch]
    check(not bad, f"G8: префикс ≠ когорта: {bad[:5]}")

    # G9 — даты валидны (формат, диапазон, реальный календарь — 2025-02-31 не пройдёт)
    for d in target["date"].dropna().unique():
        m = _DATE_RE.match(str(d))
        check(m is not None, f"G9: неверный формат даты: {d}")
        year, month = int(m.group(1)), int(m.group(2))
        check(2023 <= year <= 2026 and 1 <= month <= 12, f"G9: дата вне диапазона: {d}")
        if m.group(3):
            try:
                _date.fromisoformat(str(d))
            except ValueError:
                raise ValidationError(f"G9: несуществующая календарная дата: {d}")
    check(set(target["date_source"].dropna()) <= {"filename", "subfolder", "none", "unparsed"},
          "G9: неизвестный date_source")

    # G10 (warn) — EXIF-aware разрешение
    for cohort, dims in _EXPECTED_DIMS.items():
        sub = target[target["cohort"] == cohort]
        off = sub[sub.apply(lambda r: {int(r["width"]), int(r["height"])} != dims, axis=1)]
        if len(off):
            warnings.append(f"G10: {cohort}: {len(off)} фото с разрешением ≠ {sorted(dims)}")

    # G11 — md5 уникальны среди выживших
    check(keep["md5"].is_unique, "G11: дубли md5 среди выживших (dup_keep)")

    # G12 — целостность fold/scope: train только из gallery; open_new только probe и вне галереи;
    #        dev и test не пересекаются по md5; kpi_scope согласован с когортой/ролью.
    check(set(tkeep[tkeep["split_fold"] == "train"]["split_role"]) <= {"gallery"},
          "G12: в train-fold попало не-gallery")
    new_rows = tkeep[tkeep["is_new_open"]]
    check((new_rows["split_role"] == "probe").all(), "G12: open_new не всё probe")
    check(set(new_rows["individual_id"]).isdisjoint(gallery_ids), "G12: open_new-особь есть в галерее")
    dev_md5 = set(tkeep[tkeep["split_fold"] == "dev"]["md5"])
    test_md5 = set(tkeep[tkeep["split_fold"] == "test"]["md5"])
    check(dev_md5.isdisjoint(test_md5), "G12: dev и test пересекаются по md5")
    bad_scope = allrows[
        ((allrows["cohort"].isin(["TK", "PW"])) & (allrows["role"] == "target") & (allrows["kpi_scope"] != "kpi_core"))
        | ((allrows["cohort"] == "LAB") & (allrows["role"] == "target") & (allrows["kpi_scope"] != "temporal_aux"))
        | ((allrows["role"] == "external") & (allrows["kpi_scope"] != "external"))
    ]
    check(len(bad_scope) == 0, f"G12: неверный kpi_scope у {len(bad_scope)} строк")

    return warnings


# ─────────────────────────── EDA ───────────────────────────

def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_нет данных_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    return f"{head}\n{sep}\n{body}\n"


def _counts_table(target, external) -> str:
    rows = []
    for cohort, sub in target.groupby("cohort"):
        k = sub[sub.dup_keep == True]  # noqa: E712
        rows.append({
            "когорта": cohort, "вид": sub["species"].iloc[0],
            "особей": k["individual_id"].nunique(), "файлов": len(sub), "после дедупа": len(k),
            "gallery": int((k.split_role == "gallery").sum()), "probe": int((k.split_role == "probe").sum()),
            "train": int((k.split_fold == "train").sum()),
            "dev": int((k.split_fold == "dev").sum()),
            "test": int((k.split_fold == "test").sum()),
        })
    if len(external):
        ek = external[external.dup_keep == True]  # noqa: E712
        rows.append({"когорта": "GCN (ext)", "вид": external["species"].iloc[0],
                     "особей": ek["individual_id"].nunique(), "файлов": len(external),
                     "после дедупа": len(ek), "gallery": 0, "probe": 0, "train": 0, "dev": 0, "test": 0})
    return _md_table(pd.DataFrame(rows))


def _photos_per_individual(target, figs) -> list[str]:
    L = []
    keep = target[target.dup_keep == True]  # noqa: E712
    for cohort, sub in keep.groupby("cohort"):
        counts = sub.groupby("individual_id").size()
        L.append(f"- **{cohort}**: мин {counts.min()}, медиана {int(counts.median())}, "
                 f"макс {counts.max()}, особей {counts.size}")
        try:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(counts.values, bins=range(1, int(counts.max()) + 2))
            ax.set_title(f"Фото на особь — {cohort}")
            ax.set_xlabel("фото"); ax.set_ylabel("особей")
            fig.tight_layout(); fig.savefig(figs / f"hist_photos_{cohort}.png", dpi=90); plt.close(fig)
            L.append(f"  ![hist {cohort}](figs/hist_photos_{cohort}.png)")
        except Exception as ex:  # график не критичен
            L.append(f"  _(график не построен: {ex})_")
    return L


def _date_coverage(target, figs) -> list[str]:
    L = []
    dated = target[(target.dup_keep == True) & target["date"].notna()]  # noqa: E712
    for cohort, sub in dated.groupby("cohort"):
        by_date = sub.groupby("date").size()
        L.append(f"- **{cohort}**: " + ", ".join(f"{d}×{n}" for d, n in by_date.items()))
    lab = dated[dated.cohort == "LAB"]
    if len(lab):
        try:
            piv = lab.pivot_table(index="individual_id", columns="date", values="md5",
                                  aggfunc="count", fill_value=0)
            fig, ax = plt.subplots(figsize=(6, max(2, 0.25 * len(piv))))
            ax.imshow(piv.values, aspect="auto", cmap="Greens")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index, fontsize=6)
            ax.set_title("LAB: покрытие особь × сессия")
            fig.tight_layout(); fig.savefig(figs / "lab_timeline.png", dpi=90); plt.close(fig)
            L.append("  ![LAB timeline](figs/lab_timeline.png)")
        except Exception as ex:
            L.append(f"  _(LAB timeline не построен: {ex})_")
    return L


def _resolution_table(target, external) -> str:
    allr = pd.concat([target, external], ignore_index=True)
    allr = allr.assign(res=allr.apply(
        lambda r: f"{int(r['width'])}×{int(r['height'])}" if pd.notna(r["width"]) else "—", axis=1))
    vc = allr.groupby(["cohort", "res"]).size().reset_index(name="n")
    ori = allr.groupby(["cohort", "orientation"], dropna=False).size().reset_index(name="n")
    return _md_table(vc) + "\n**Ориентация (EXIF):**\n\n" + _md_table(ori)


def _dup_table(target, external) -> str:
    allr = pd.concat([target, external], ignore_index=True)
    dups = allr[allr.dup_group >= 0]
    if not len(dups):
        return "_точных дублей не найдено_\n"
    rows, n_cross = [], 0
    for gi, grp in dups.groupby("dup_group"):
        cohorts = sorted(set(grp["cohort"]))
        is_cross = len(cohorts) > 1
        n_cross += int(is_cross)
        surv = grp[grp.dup_keep == True]  # noqa: E712
        rows.append({"группа": gi, "членов": len(grp), "когорты": ",".join(cohorts),
                     "кросс-когортная": "ДА" if is_cross else "нет",
                     "выживший": Path(surv["rel_path"].iloc[0]).name if len(surv) else "—"})
    return f"Групп дублей: {len(rows)}; кросс-когортных: {n_cross}.\n\n" + _md_table(pd.DataFrame(rows))


def _split_table(target) -> str:
    keep = target[target.dup_keep == True]  # noqa: E712
    t = keep.groupby(["cohort", "split_scheme", "split_role"], dropna=False).size().reset_index(name="n")
    return _md_table(t)


def _anomaly_log(target, external) -> str:
    allr = pd.concat([target, external], ignore_index=True)
    note_str = allr["notes"].astype(str).str.strip()
    # после CSV-раунд-трипа пустое значение читается как NaN → строка "nan": исключаем
    anom = allr[allr["notes"].notna() & ~note_str.str.lower().isin(["", "nan", "none"])]
    extra = ""
    if "mask_empty" in external.columns and len(external):
        extra = f"\nGCN пустых масок: {int((external['mask_empty'] == True).sum())}."  # noqa: E712
    if not len(anom):
        return "_аномалий не отмечено_\n" + extra
    rows = [{"файл": Path(r["rel_path"]).name, "когорта": r["cohort"], "заметка": r["notes"]}
            for _, r in anom.head(100).iterrows()]
    tail = f"\n_…и ещё {len(anom) - 100}_" if len(anom) > 100 else ""
    return _md_table(pd.DataFrame(rows)) + tail + extra


def _contact_sheets(target, cfg, figs) -> list[str]:
    from PIL import Image
    L = []
    ws = Path(cfg.workspace_root)
    keep = target[target.dup_keep == True]  # noqa: E712
    cell = 160
    for cohort, sub in keep.groupby("cohort"):
        thumbs = []
        for _, r in sub.head(8).iterrows():
            try:
                im = load_canonical(ws / r["rel_path"]); im.thumbnail((cell, cell)); thumbs.append(im)
            except Exception:
                pass
        if not thumbs:
            continue
        cols = min(4, len(thumbs)); nrows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, nrows * cell), (255, 255, 255))
        for i, im in enumerate(thumbs):
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        sheet.save(figs / f"contact_{cohort}.png")
        L.append(f"- **{cohort}**: ![contact {cohort}](figs/contact_{cohort}.png)")
    return L


def write_eda(target: pd.DataFrame, external: pd.DataFrame, cfg, out_dir) -> None:
    """Собрать reports/eda.md + figs/ для глазной проверки данных."""
    out = Path(out_dir)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    L = ["# EDA — Блок 1 (Данные), triton 3.0", ""]
    L += ["## 1. Счётчики по когортам", "", _counts_table(target, external)]
    L += ["## 2. Фото на особь", ""] + _photos_per_individual(target, figs)
    L += ["", "## 3. Покрытие по датам", ""] + _date_coverage(target, figs)
    L += ["", "## 4. Разрешение и ориентация", "", _resolution_table(target, external)]
    L += ["## 5. Дубли (md5)", "", _dup_table(target, external)]
    L += ["## 6. Сводка сплитов (target)", "", _split_table(target)]
    L += ["## 7. Лог аномалий", "", _anomaly_log(target, external)]
    L += ["", "## 8. Контакт-листы", ""] + _contact_sheets(target, cfg, figs)
    (out / "eda.md").write_text("\n".join(L), encoding="utf-8")
