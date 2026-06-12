"""md5-дедуп: группировка по байтовому хешу + детерминированный выбор «выжившего».

Дедуп несёт ДВЕ функции:
1) анти-утечка: одинаковая фотография не должна попасть и в gallery, и в probe (баг 2.0);
2) корректная идентичность: выживший в группе определяет, какой local_id/дата/сессия
   попадут в манифест. Поэтому правило выбора выжившего — часть корректности, а не косметика.

Все строки сохраняются (ничего не выбрасываем) — не-выжившие помечаются dup_keep=False
и в сплиты/KPI не идут, но остаются видимыми в EDA-отчёте по дублям.
"""
import re
from collections import defaultdict
from pathlib import Path


def _survivor_priority(rec: dict) -> tuple:
    """Ключ сортировки; МЕНЬШЕ = предпочтительнее как выживший.

    Порядок правил:
    0) LAB: одиночная цифра в стеме («1») — аномалия (валидные id — 2-значные 01..25),
       проигрывает 2-значным («10») → чинит ошибочную метку 1.jpg ≡ копия 10.jpg;
    1) «именованный» файл важнее дампа камеры IMG_* и LAB-варианта «.k»;
    2) меньшая глубина пути (верхний уровень важнее подпапки «Доп фото»);
    3) лексикографика имени — финальный детерминированный разрыв ничьей.
    """
    name = Path(rec["rel_path"]).name
    stem = Path(rec["rel_path"]).stem
    cohort = rec.get("cohort", "")
    depth = rec["rel_path"].count("/")
    lab_bare_single = 1 if (cohort == "LAB" and re.fullmatch(r"\d", stem)) else 0
    generic = name.upper().startswith("IMG_")
    lab_variant = cohort == "LAB" and "." in stem
    rank_named = 1 if (generic or lab_variant) else 0
    return (lab_bare_single, rank_named, depth, name)


def deduplicate(records: list[dict]) -> list[dict]:
    """Проставляет каждой записи dup_group (−1 если уникальна) и dup_keep (выживший).

    Порядок входных записей сохраняется. Индексы групп стабильны: нумеруются по
    сортировке md5 среди групп с дублями (воспроизводимо между запусками).
    """
    by_md5: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)

    dup_md5s = sorted(md5 for md5, grp in by_md5.items() if len(grp) > 1)
    group_index = {md5: i for i, md5 in enumerate(dup_md5s)}

    for md5, grp in by_md5.items():
        if len(grp) == 1:
            grp[0]["dup_group"] = -1
            grp[0]["dup_keep"] = True
            continue
        gi = group_index[md5]
        survivor = min(grp, key=_survivor_priority)
        for r in grp:
            r["dup_group"] = gi
            r["dup_keep"] = r is survivor
            if r is not survivor:
                note = f"md5-дубль группы {gi}; выживший {Path(survivor['rel_path']).name}"
                r["notes"] = (r.get("notes", "") + "; " + note) if r.get("notes") else note
    return records
