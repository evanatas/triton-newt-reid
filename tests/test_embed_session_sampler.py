"""TDD SessionAwarePKSampler (C2): K кадров особи из РАЗНЫХ сессий (temporal hard-positives).

Инвариант: батч = P особей × K кадров (как PKSampler); НО для особи с ≥K сессий все K кадров —
из K разных сессий; для особи с s<K сессий покрыты все s сессий. Детерминизм seed=42.
"""
import numpy as np
import pytest

from triton_crop.embed_dataset import SessionAwarePKSampler


def _sessions_of(labels, sessions, batch, label):
    idx = [i for i in batch if labels[i] == label]
    return [sessions[i] for i in idx]


def test_session_aware_shape_and_cross_session():
    # 4 особи, у каждой 3 сессии по 2 кадра
    labels, sessions = [], []
    for ind in ["A", "B", "C", "D"]:
        for s in ["s1", "s2", "s3"]:
            labels += [ind, ind]
            sessions += [s, s]
    labels, sessions = np.array(labels), np.array(sessions)
    s = SessionAwarePKSampler(labels, sessions, p=2, k=3, seed=42)
    batches = list(s)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 2 * 3                      # |batch| = P*K
        labs = labels[np.array(batch)]
        assert len(set(labs.tolist())) == 2             # ровно P особей
        for u in set(labs.tolist()):
            sess = _sessions_of(labels, sessions, batch, u)
            assert len(sess) == 3                        # ровно K кадров особи
            assert len(set(sess)) == 3                   # K кадров — из 3 РАЗНЫХ сессий (cross-session)


def test_session_aware_single_session_fallback():
    # особь с одной сессией: cross-session невозможен → берём K кадров из неё (не падаем)
    labels = np.array(["A"] * 4 + ["B"] * 4)
    sessions = np.array(["s1"] * 4 + ["s1"] * 4)        # у всех одна сессия
    s = SessionAwarePKSampler(labels, sessions, p=2, k=3, seed=42)
    batches = list(s)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 6
        for u in ("A", "B"):
            assert len(_sessions_of(labels, sessions, batch, u)) == 3   # K кадров есть, fallback сработал


def test_session_aware_deterministic_seed():
    labels = np.array([x for ind in "ABCD" for x in [ind] * 6])
    sessions = np.array([s for _ in "ABCD" for s in (["s1"] * 2 + ["s2"] * 2 + ["s3"] * 2)])
    a = list(SessionAwarePKSampler(labels, sessions, p=2, k=2, seed=42))
    b = list(SessionAwarePKSampler(labels, sessions, p=2, k=2, seed=42))
    assert a == b                                        # один seed → одинаковая последовательность


def test_session_aware_rejects_p_gt_classes():
    labels = np.array(["A", "A", "B", "B"])
    sessions = np.array(["s1", "s2", "s1", "s2"])
    with pytest.raises(ValueError):
        SessionAwarePKSampler(labels, sessions, p=5, k=2, seed=42)
