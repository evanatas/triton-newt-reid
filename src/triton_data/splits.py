"""Детерминированный сплиттер (seed=42): роль · 3-частный fold · схема · open-set · kpi_scope.

Оси:
  split_role  ∈ {gallery, probe}             — эталон базы vs запрос «кто это».
  split_fold  ∈ {train, dev, test}           — учим / тюним / ЗАПЕЧАТАНО:
                gallery → train (учим эмбеддер + эталоны);
                probe   → dev (A-B и тюнинг блоков 2–6) ИЛИ test (headline KPI, вскрыть 1 раз).
                Probe НИКОГДА не train → нельзя «выучить» проверочные фото (грабли 2.0).
  split_scheme∈ {temporal, random, gallery_only, external, open_new}
  kpi_scope   ∈ {kpi_core (TK+PW — официальный 75/95), temporal_aux (LAB), external (GCN)}

Headline-KPI = kpi_scope==kpi_core & split_fold==test & split_role==probe (по срезам overall/temporal).
Детерминизм: выбор open_new и dev/test зависит только от individual_id/rel_path и seed (blake2b),
не от порядка строк.
"""
import hashlib
import math

import numpy as np
import pandas as pd


def _hash_hex(key: str) -> str:
    return hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).hexdigest()


def _frac01(key: str) -> float:
    """Детерминированное число в [0,1) из ключа (для порога dev/test)."""
    h = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def _id_rng(individual_id, seed: int) -> np.random.Generator:
    h = hashlib.blake2b(str(individual_id).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big") ^ int(seed))


def _select_fraction(ids: list, frac: float, salt) -> set:
    if frac <= 0 or not ids:
        return set()
    k = math.ceil(frac * len(ids))
    return set(sorted(ids, key=lambda x: _hash_hex(f"{salt}:{x}"))[:k])


def _pick_random_probe(group: pd.DataFrame, seed: int):
    rng = _id_rng(group["individual_id"].iloc[0], seed)
    sessions = group["session"].dropna()
    if sessions.nunique() >= 2:
        # наименее частая сессия; при равных счётчиках — лексикографически меньшее имя
        # (не зависит от порядка строк, в отличие от value_counts().index[-1])
        least = min(sessions.unique(), key=lambda s: (int((sessions == s).sum()), str(s)))
        cand = group[group["session"] == least]
    else:
        cand = group
    cand = cand.sort_values("rel_path")
    return cand.index[int(rng.integers(len(cand)))]


def _kpi_scope(cohort: str, role: str) -> object:
    if role == "external":
        return "external"
    if cohort in ("TK", "PW"):
        return "kpi_core"
    if cohort == "LAB":
        return "temporal_aux"
    return pd.NA


def assign_splits(df: pd.DataFrame, seed: int = 42,
                  test_frac: float = 0.5, open_new_frac: float = 0.15) -> pd.DataFrame:
    df = df.copy()
    df["split_role"] = pd.NA
    df["split_fold"] = pd.NA
    df["split_scheme"] = pd.NA
    df["is_new_open"] = False
    df["kpi_scope"] = [_kpi_scope(c, r) for c, r in zip(df["cohort"], df["role"])]

    df.loc[df["role"] == "external", "split_scheme"] = "external"
    work = df[(df["role"] == "target") & (df["dup_keep"] == True)]  # noqa: E712

    # --- классификация особей: temporal-eligible (≥2 дат) или нет (по когортам) ---
    ind_dates: dict = {}
    nontemporal_by_cohort: dict = {}
    for ind, g in work.groupby("individual_id"):
        dates = sorted(d for d in g["date"].dropna().unique())
        ind_dates[ind] = dates
        if len(dates) < 2:
            nontemporal_by_cohort.setdefault(g["cohort"].iloc[0], []).append(ind)

    open_new: set = set()
    for cohort, ids in nontemporal_by_cohort.items():
        open_new |= _select_fraction(ids, open_new_frac, f"{seed}-open")

    # --- роль + схема по каждой особи ---
    for ind, g in work.groupby("individual_id"):
        idx = g.index
        if ind in open_new:  # open-set NEW: целиком вне галереи, всё — пробы на отвержение
            df.loc[idx, "split_scheme"] = "open_new"
            df.loc[idx, "split_role"] = "probe"
            df.loc[idx, "is_new_open"] = True
            continue
        dates = ind_dates[ind]
        if len(dates) >= 2:
            df.loc[idx, "split_scheme"] = "temporal"
            df.loc[g.index[(g["date"] == dates[0]) | (g["date"].isna())], "split_role"] = "gallery"
            df.loc[g.index[g["date"].isin(dates[1:])], "split_role"] = "probe"
        elif len(g) >= 2:
            df.loc[idx, "split_scheme"] = "random"
            df.loc[idx, "split_role"] = "gallery"
            df.loc[_pick_random_probe(g, seed), "split_role"] = "probe"
        else:
            df.loc[idx, "split_scheme"] = "gallery_only"
            df.loc[idx, "split_role"] = "gallery"

    # --- 3-частный fold: gallery → train; probe → dev/test (запечатанный test) ---
    df.loc[df["split_role"] == "gallery", "split_fold"] = "train"
    probe_idx = df.index[df["split_role"] == "probe"]
    if len(probe_idx):
        fr = df.loc[probe_idx, "rel_path"].map(lambda r: _frac01(f"{seed}-fold:{r}"))
        df.loc[probe_idx, "split_fold"] = np.where(fr.to_numpy() < test_frac, "test", "dev")

    df["is_new_open"] = df["is_new_open"].astype(bool)
    return df
