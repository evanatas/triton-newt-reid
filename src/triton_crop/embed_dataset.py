"""Данные для дообучения эмбеддера re-ID (Блок 4) — ЧИСТАЯ логика (без torch/картинок).

Готовит обучающие записи и схему честной оценки:
  • build_records       — кропы под выбранные строки (loader.select) с fallback-протоколом A/B;
  • LabelEncoder        — individual_id ↔ class-idx (стабильный сорт, для головы ArcFace);
  • kfold_by_individual — out-of-fold ПО ОСОБЯМ (анти-утечка: все кадры особи в одном фолде);
  • PKSampler           — батчи metric learning P особей × K кадров (детерминированы seed).

Картинки не читаются (это делает embed_train на Colab) — здесь только пути и метаданные, поэтому
модуль герметичен и быстр. Кропы берём ТОЛЬКО через select_crops_with_fallback (контракт Блока 4).
"""
import numpy as np
import pandas as pd

from .crops_manifest import select_crops_with_fallback


def build_records(crops_df, rows_df, variant: str = "unroll_ribbon",
                  fallback: str = "belly_oriented", drop_missing: bool = True) -> pd.DataFrame:
    """Строки loader.select → по записи на md5: какой кроп брать для эмбеддера (variant, иначе fallback).

    Тот же fallback-протокол, что в A/B (cmd_ab) — НЕ молчаливо меньше строк. drop_missing=True
    убирает md5 без единого кропа (нечего эмбеддить). -> DataFrame[md5, crop_path, individual_id,
    cohort, kpi_scope, variant_used].
    """
    sel = select_crops_with_fallback(crops_df, rows_df, variant=variant, fallback=fallback)
    if drop_missing:
        sel = sel[sel["variant_used"].notna()].reset_index(drop=True)
    cols = [c for c in ("md5", "crop_path", "individual_id", "cohort", "kpi_scope", "variant_used")
            if c in sel.columns]
    return sel[cols].reset_index(drop=True)


class LabelEncoder:
    """individual_id ↔ class-idx. Классы = отсортированные уникальные (стабильно, seed-независимо)."""

    def __init__(self, labels):
        self.classes_ = sorted({str(x) for x in labels})
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, labels) -> np.ndarray:
        return np.array([self._to_idx[str(x)] for x in labels], dtype=np.int64)

    def inverse_transform(self, codes) -> list:
        return [self.classes_[int(i)] for i in codes]

    def __len__(self) -> int:
        return len(self.classes_)


def kfold_by_individual(individual_ids, n_folds: int = 5, seed: int = 42):
    """k фолдов ПО ОСОБЯМ: все кадры одной особи попадают в ОДИН фолд (анти-утечка re-ID).

    -> list[(train_idx, val_idx)] по кадрам. Особи детерминированно перемешиваются (seed) и
    раскидываются по фолдам round-robin → балансировка. val_idx фолда f = кадры особей фолда f.
    """
    ids = np.asarray([str(x) for x in individual_ids])
    uniq = sorted(set(ids.tolist()))
    if not (2 <= n_folds <= len(uniq)):
        raise ValueError(f"kfold_by_individual: n_folds={n_folds} вне [2, {len(uniq)}] (число особей)")
    perm = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = {uniq[u]: rank % n_folds for rank, u in enumerate(perm)}
    fold_by_frame = np.array([fold_of[x] for x in ids])
    all_idx = np.arange(len(ids))
    return [(all_idx[fold_by_frame != f], all_idx[fold_by_frame == f]) for f in range(n_folds)]


class PKSampler:
    """Сэмплер батчей metric learning: P особей × K кадров на батч (детерминирован seed).

    Итерация по особям (перемешаны seed), по P за раз; на особь берётся K кадров (без повтора, иначе
    добор с повтором для long-tail). Каждый __iter__ создаёт свежий RandomState(seed) → один seed даёт
    идентичную последовательность (контракт детерминизма сэмплера). drop_last=True: батчей = n_classes // P (хвост < P отбрасывается).
    """

    def __init__(self, labels, p: int, k: int, seed: int = 42, drop_last: bool = True):
        self.labels = np.asarray([str(x) for x in labels])
        self.p, self.k, self.seed, self.drop_last = p, k, seed, drop_last
        self.by_label = {u: np.where(self.labels == u)[0]
                         for u in sorted(set(self.labels.tolist()))}
        self.classes = list(self.by_label)
        if p > len(self.classes):
            raise ValueError(f"PKSampler: p={p} > числа особей {len(self.classes)}")

    def __len__(self) -> int:
        n = len(self.classes)
        return n // self.p if self.drop_last else int(np.ceil(n / self.p))

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        classes = list(self.classes)
        rng.shuffle(classes)
        for b in range(len(self)):
            chosen = classes[b * self.p:(b + 1) * self.p]
            if len(chosen) < self.p:                              # добор хвоста (только при drop_last=False)
                pool = [c for c in self.classes if c not in chosen]
                chosen = chosen + list(rng.choice(pool, self.p - len(chosen), replace=False))
            batch = []
            for c in chosen:
                idx = self.by_label[c]
                replace = len(idx) < self.k                       # long-tail: добор с повтором
                batch.extend(rng.choice(idx, self.k, replace=replace).tolist())
            yield batch


class SessionAwarePKSampler:
    """P×K-сэмплер с cross-session позитивами: K кадров особи берутся из РАЗНЫХ сессий —
    явные temporal hard-positives, которых не давал session-agnostic PKSampler.

    На особь: индексы кадров группируются по сессии; сессии перемешиваются (seed) и обходятся
    round-robin, из каждой берётся кадр без повтора, пока не наберётся K. Если разных сессий < K —
    добор из уже использованных сессий (с повтором при нужде): для одно-сессионных особей вырождается
    в поведение PKSampler (graceful fallback). Детерминирован seed. |batch| = P*K, P разных особей.
    """

    def __init__(self, labels, sessions, p: int, k: int, seed: int = 42, drop_last: bool = True):
        self.labels = np.asarray([str(x) for x in labels])
        self.sessions = np.asarray([str(x) for x in sessions])
        self.p, self.k, self.seed, self.drop_last = p, k, seed, drop_last
        self.by_label = {u: np.where(self.labels == u)[0]
                         for u in sorted(set(self.labels.tolist()))}
        self.classes = list(self.by_label)
        if p > len(self.classes):
            raise ValueError(f"SessionAwarePKSampler: p={p} > числа особей {len(self.classes)}")

    def __len__(self) -> int:
        n = len(self.classes)
        return n // self.p if self.drop_last else int(np.ceil(n / self.p))

    def _pick_k(self, label, rng):
        """K индексов кадров особи, максимально покрывая разные сессии (round-robin по сессиям)."""
        idx = self.by_label[label]
        by_sess = {}
        for i in idx:
            by_sess.setdefault(self.sessions[i], []).append(int(i))
        sess_keys = list(by_sess)
        rng.shuffle(sess_keys)
        pools = {s: list(rng.permutation(by_sess[s])) for s in sess_keys}   # перемешанные кадры в сессии
        chosen, si = [], 0
        while len(chosen) < self.k:
            s = sess_keys[si % len(sess_keys)]
            if pools[s]:
                chosen.append(pools[s].pop())
            else:                                          # сессия исчерпана → добор любым кадром (с повтором)
                chosen.append(int(rng.choice(idx)))
            si += 1
        return chosen[:self.k]

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        classes = list(self.classes)
        rng.shuffle(classes)
        for b in range(len(self)):
            chosen = classes[b * self.p:(b + 1) * self.p]
            if len(chosen) < self.p:                       # добор хвоста (только при drop_last=False)
                pool = [c for c in self.classes if c not in chosen]
                chosen = chosen + list(rng.choice(pool, self.p - len(chosen), replace=False))
            batch = []
            for c in chosen:
                batch.extend(self._pick_k(c, rng))
            yield batch
