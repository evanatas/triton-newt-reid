"""Тесты embed_dataset (Блок 4): build_records / LabelEncoder / kfold по особям / PKSampler.

Гейты: детерминизм seed=42 (фолды и порядок батчей воспроизводимы); cross-fit
анти-утечка (все кадры особи в одном фолде, train∩val по особям = ∅). Герметично (table-only).
"""
import numpy as np
import pandas as pd
import pytest

from triton_crop.embed_dataset import (
    LabelEncoder,
    PKSampler,
    build_records,
    kfold_by_individual,
)


def test_build_records_covers_rows_with_fallback(synthetic_embed_dataset):
    d = synthetic_embed_dataset
    rec = build_records(d.crops_df, d.gallery_rows, variant="unroll_ribbon", fallback="belly_oriented")
    # покрытие == len(rows): у каждого md5 есть хотя бы belly_oriented → fallback закрывает 100%
    assert len(rec) == len(d.gallery_rows)
    assert set(rec["md5"]) == set(d.gallery_rows["md5"])
    # variant_used: ribbon где есть, иначе belly_oriented (тот же fallback-протокол, что в A/B)
    assert set(rec["variant_used"]) <= {"unroll_ribbon", "belly_oriented"}
    assert (rec["variant_used"] == "unroll_ribbon").any()
    assert (rec["variant_used"] == "belly_oriented").any()        # fallback хотя бы раз сработал
    assert rec["individual_id"].notna().all()                     # личность протянута (для обучения)
    assert rec["crop_path"].notna().all()


def test_build_records_drops_missing_crops():
    rows = pd.DataFrame({"md5": ["a", "b"], "individual_id": ["X", "Y"],
                         "cohort": ["TK", "TK"], "kpi_scope": ["kpi_core", "kpi_core"]})
    crops = pd.DataFrame({"md5": ["a"], "variant": ["belly_oriented"], "crop_path": ["c/a.png"],
                          "individual_id": ["X"], "cohort": ["TK"], "kpi_scope": ["kpi_core"]})
    rec = build_records(crops, rows, variant="unroll_ribbon", fallback="belly_oriented")
    assert list(rec["md5"]) == ["a"]                              # b без единого кропа → отброшен
    assert list(rec["variant_used"]) == ["belly_oriented"]


def test_label_encoder_stable_and_invertible(synthetic_embed_dataset):
    d = synthetic_embed_dataset
    le = LabelEncoder(d.individuals)
    assert len(le) == len(set(d.individuals))
    assert le.classes_ == sorted(set(d.individuals))             # seed-независимый стабильный сорт
    codes = le.transform(["TK-01", "PW-04", "TK-01"])
    assert codes[0] == codes[2] and codes[0] != codes[1]
    assert le.inverse_transform(codes) == ["TK-01", "PW-04", "TK-01"]
    # независимая пересборка кодирует одинаково (детерминизм, без скрытого состояния)
    assert list(le.transform(d.individuals)) == list(LabelEncoder(d.individuals).transform(d.individuals))


def test_kfold_by_individual_no_leak(synthetic_embed_dataset):
    d = synthetic_embed_dataset
    ids = build_records(d.crops_df, d.gallery_rows)["individual_id"].to_numpy()
    folds = kfold_by_individual(ids, n_folds=5, seed=42)
    assert len(folds) == 5
    seen_val = set()
    for tr, va in folds:
        assert set(ids[tr]).isdisjoint(set(ids[va]))             # анти-утечка: особь не в train И val одного фолда
        assert len(tr) + len(va) == len(ids)                     # разбиение полное
        assert len(va) > 0
        seen_val |= set(ids[va].tolist())
    assert seen_val == set(ids.tolist())                         # каждая особь в val ровно один раз


def test_kfold_deterministic(synthetic_embed_dataset):
    d = synthetic_embed_dataset
    ids = build_records(d.crops_df, d.gallery_rows)["individual_id"].to_numpy()
    f1 = kfold_by_individual(ids, n_folds=5, seed=42)
    f2 = kfold_by_individual(ids, n_folds=5, seed=42)
    for (t1, v1), (t2, v2) in zip(f1, f2):
        assert np.array_equal(t1, t2) and np.array_equal(v1, v2)


def test_kfold_rejects_bad_nfolds(synthetic_embed_dataset):
    ids = synthetic_embed_dataset.gallery_rows["individual_id"].to_numpy()
    with pytest.raises(ValueError):
        kfold_by_individual(ids, n_folds=1)                      # < 2 фолдов
    with pytest.raises(ValueError):
        kfold_by_individual(ids, n_folds=999)                    # больше, чем особей


def test_pksampler_shape_and_determinism(synthetic_embed_dataset):
    d = synthetic_embed_dataset
    labels = build_records(d.crops_df, d.gallery_rows)["individual_id"].to_numpy()
    s = PKSampler(labels, p=3, k=2, seed=42)
    b1, b2 = list(s), list(s)
    assert b1 == b2 and len(b1) > 0                              # один seed → одинаковая последовательность
    for batch in b1:
        assert len(batch) == 3 * 2                              # |batch| = P*K
        labs = labels[batch]
        uniq = set(labs.tolist())
        assert len(uniq) == 3                                   # ровно P разных особей
        for u in uniq:
            assert int((labs == u).sum()) == 2                  # ровно K кадров каждой
