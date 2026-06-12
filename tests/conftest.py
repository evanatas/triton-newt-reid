"""Фикстуры: крошечная синтетическая мастерская, повторяющая причуды реальных данных.

Позволяет гонять реальные ingest/dedup/split без данных заказчика и быстро.
"""
import hashlib
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from triton_data.config import Config, DatasetSpec

TK_DIR = "tk_cohort"
LAB_DIR = "lab_cohort_dates"
PW_DIR = "pw_cohort"
GCN_DIR = "gcn_external"


def _img(path, color, size=(20, 16), orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if orientation is not None:
        ex = im.getexif()
        ex[0x0112] = orientation
        im.save(path, exif=ex.tobytes())
    else:
        im.save(path)


@pytest.fixture
def synthetic_workspace(tmp_path):
    """Создаёт 4 когорты с реальными причудами; возвращает Config на эту мастерскую."""
    ws = tmp_path / "ws"

    # --- TK: даты в именах + в подпапках, skip «99 (skip)», typo-подпапка ---
    tk = ws / TK_DIR
    _img(tk / "1" / "01-01-1224.JPG", (10, 20, 30))
    _img(tk / "1" / "01-02-0125.JPG", (11, 21, 31))
    _img(tk / "1" / "Дополнительные фото от 0225" / "IMG_1000.JPG", (12, 22, 32))
    _img(tk / "2" / "02-01-1224.JPG", (13, 23, 33))
    (tk / "99 (skip)").mkdir(parents=True, exist_ok=True)
    _img(tk / "99 (skip)" / "Иллюстрация.jpg", (99, 99, 99))  # должна быть ПРОПУЩЕНА
    (tk / "99 (skip)" / "пояснение.txt").write_text("заметка", encoding="utf-8")
    _img(tk / "66" / "66-01-1224.JPG", (14, 24, 34))
    _img(tk / "66" / "Дополниетльные фото от 0125" / "IMG_2000.JPG", (15, 25, 35))  # опечатка

    # --- LAB: подпапки-даты, вариант «.k», байт-дубли (03.1≡03, 1≡10) ---
    lab = ws / LAB_DIR
    _img(lab / "05.08.2025" / "03.jpg", (40, 50, 60))
    shutil.copyfile(lab / "05.08.2025" / "03.jpg", lab / "05.08.2025" / "03.1.jpg")  # дубль
    _img(lab / "05.08.2025" / "пояснение.jpg", (10, 20, 30))  # имя НЕ вида локального id → ingest_lab пропускает (не падает)
    _img(lab / "25.10.2025" / "03.jpg", (41, 51, 61))  # другая сессия — другой контент
    _img(lab / "25.10.2025" / "10.jpg", (42, 52, 62))
    shutil.copyfile(lab / "25.10.2025" / "10.jpg", lab / "25.10.2025" / "1.jpg")  # дубль (ошибочная метка)

    # --- PW: без дат, "{id}-{session} ({shot})" ---
    pw = ws / PW_DIR
    _img(pw / "1" / "01-01 (1).JPG", (70, 80, 90))
    _img(pw / "1" / "01-01 (2).JPG", (71, 81, 91))
    _img(pw / "2" / "02-01 (1).JPG", (72, 82, 92))

    # --- GCN: metadata.csv + Raw_Data, RLE-префикс HxW (2 варианта), пустая маска ---
    gcn = ws / GCN_DIR
    _img(gcn / "Raw_Data" / "1" / "IMG_2530.JPEG", (100, 110, 120))
    _img(gcn / "Raw_Data" / "10" / "IMG_2550.JPEG", (101, 111, 121))
    (gcn / "metadata.csv").write_text(
        ",identity,file_name,recapture_id,survey,bbox,segmentation_mask_rle\n"
        '0,1,IMG_2530.JPEG,1,2,"[1.0, 2.0, 3.0, 4.0]",2048x1536:abcd\n'
        '1,10,IMG_2550.JPEG,10,2,"[5.0, 6.0, 7.0, 8.0]",1536x2048:\n',
        encoding="utf-8",
    )

    specs = (
        DatasetSpec("karelinii", TK_DIR, "Triturus karelinii", "TK", "target",
                    "folder_per_id", ("99 (skip)",)),
        DatasetSpec("lab", LAB_DIR, "Triturus_unconfirmed", "LAB", "target",
                    "date_subdirs", (), {"id_from": "filename_stem"}),
        DatasetSpec("pleurodeles", PW_DIR, "Pleurodeles waltl", "PW", "target",
                    "folder_per_id"),
        DatasetSpec("gcnid", GCN_DIR, "Triturus cristatus", "GCN", "external",
                    "metadata_csv", (), {"metadata_csv": "metadata.csv", "raw_subdir": "Raw_Data"}),
    )
    return Config(workspace_root=ws, seed=42, image_extensions=(".jpg", ".jpeg"),
                  datasets=specs, merge_individuals={})


@pytest.fixture
def synthetic_newt():
    """Синтетическая «тушка» тритона для тестов геометрии/псевдо-меток (Блок 2).

    Тело-эллипс (вертикально) + широкая «голова» сверху (малый y) + узкий «хвост» снизу
    + 4 боковые «лапы»-выступа. Голова ВВЕРХУ. Возвращает SimpleNamespace с маской (HxW bool),
    RGB-кадром и точной GT-геометрией (head/cloaca/tail/belly_polygon) → строгие ассерты.
    """
    H, W, cx = 220, 130, 65
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, 110), (20, 70), 0, 0, 360, 255, -1)          # тело
    cv2.circle(m, (cx, 45), 27, 255, -1)                             # голова (шире тела)
    cv2.fillPoly(m, [np.array([[cx - 6, 170], [cx + 6, 170],         # хвост (узкий клин)
                               [cx + 2, 205], [cx - 2, 205]])], 255)
    for (dx, y) in [(-22, 100), (22, 100), (-22, 140), (22, 140)]:   # лапы (боковые выступы)
        cv2.circle(m, (cx + dx, y), 10, 255, -1)
    mask = m > 0
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[mask] = (180, 140, 90)
    belly_polygon = np.array([[cx - 16, 72], [cx + 16, 72],          # центральный торс без лап
                              [cx + 16, 158], [cx - 16, 158]], float)
    return SimpleNamespace(mask=mask, rgb=rgb, head_xy=(cx, 20), cloaca_xy=(cx, 162),
                           tail_xy=(cx, 203), belly_polygon=belly_polygon)


@pytest.fixture
def synthetic_curved_newt():
    """Изогнутое (C/банан) пузо в каноне + вшитые пятна известных центроидов — для тестов Блока 3.

    Центральная ось — парабола x(y)=cx+A·((y−ymid)/R)² (изгиб вбок); тело — полоса постоянной
    ширины вдоль оси (голова вверху, малый y). На теле — контрастные пятна-кружки с известными
    центроидами (spot_centroids) → проверяем, что распрямление выпрямляет тело, НО сохраняет узор.
    Возвращает: mask (HxW bool), rgb (uint8), axis_start/end (внутри маски), spot_centroids,
    axis_x (callable), base_color/spot_color.
    """
    H, W = 200, 160
    cx, ymid, R, A, hw = 70, 100, 80, 45.0, 18
    y0, y1 = 30, 176
    base, spot = (170, 130, 80), (20, 20, 230)

    def axis_x(y):
        return cx + A * ((y - ymid) / R) ** 2

    mask = np.zeros((H, W), np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    ys = list(range(y0, y1))
    left = [(int(round(axis_x(y) - hw)), y) for y in ys]            # полоса с плоскими торцами:
    right = [(int(round(axis_x(y) + hw)), y) for y in ys]          # центроид строки = axis_x(y)
    band = np.array(left + right[::-1], np.int32)
    cv2.fillPoly(mask, [band], 255)
    cv2.fillPoly(rgb, [band], base)
    mask = mask > 0
    spot_centroids = []
    for ys, off in [(55, -8), (90, 7), (125, -5), (160, 6)]:
        xs = int(round(axis_x(ys) + off))
        cv2.circle(rgb, (xs, ys), 4, spot, -1)
        spot_centroids.append((xs, ys))
    axis_start = (int(round(axis_x(y0 + 4))), y0 + 4)
    axis_end = (int(round(axis_x(y1 - 6))), y1 - 6)
    return SimpleNamespace(mask=mask, rgb=rgb, axis_start=axis_start, axis_end=axis_end,
                           spot_centroids=spot_centroids, head_xy=axis_start, cloaca_xy=axis_end,
                           axis_x=axis_x, base_color=base, spot_color=spot)


# ───────────────────────── Блок 5 (матчинг созвездия пятен) ─────────────────────────

@pytest.fixture
def synthetic_two_sessions():
    """Чистые созвездия центроидов пятен для TDD матчера (БЕЗ картинок — геометрия).

    Одна особь X в ДВУХ «сессиях»: pts_a и pts_b = pts_a, преобразованное ИЗВЕСТНЫМ similarity
    (поворот theta + масштаб scale + сдвиг t) — растяжение/поза/ориентация изменились, но ТОПОЛОГИЯ
    созвездия та же (устойчивый признак: центр пятна не двигается). pts_other — другая особь (другое облако).
    pts_a_mirror — ЗЕРКАЛО pts_a (хиральность нарушена; матчер ОБЯЗАН штрафовать — это другая особь).
    Облако АСИММЕТРИЧНО (нет случайной зеркальной симметрии). Детерминизм seed=42.
    Возвращает: pts_a/pts_b/pts_other/pts_a_mirror (N,2 float), theta_deg, scale, translation, n.
    """
    rng = np.random.RandomState(42)
    n = 8
    pts_a = rng.uniform(-1.0, 1.0, size=(n, 2)) * np.array([1.0, 2.2])   # вытянуто по y (как тело)
    pts_a = pts_a - pts_a.mean(0)
    theta_deg, scale, t = 37.0, 1.6, np.array([12.0, -5.0])
    th = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts_b = (pts_a @ rot.T) * scale + t                                  # similarity (det>0, без зеркала)
    pts_other = rng.uniform(-1.0, 1.0, size=(n, 2)) * np.array([1.0, 2.2])
    pts_other = pts_other - pts_other.mean(0)
    pts_a_mirror = pts_a * np.array([-1.0, 1.0])                         # отражение по x → другая особь
    return SimpleNamespace(pts_a=pts_a.astype(float), pts_b=pts_b.astype(float),
                           pts_other=pts_other.astype(float), pts_a_mirror=pts_a_mirror.astype(float),
                           theta_deg=theta_deg, scale=scale, translation=t, n=n)


# ───────────────────────── Блок 4 (эмбеддер re-ID) ─────────────────────────

@pytest.fixture
def synthetic_embed_dataset():
    """Табличный синтетический набор для Блока 4 (embed_dataset / embed_ab) — БЕЗ картинок на диске.

    12 особей (8 TK + 4 PW), long-tail кадров (повторяет реальность: TK ~3–4, PW больше). На каждый md5
    есть `belly_oriented`-кроп; на ~70 % также `unroll_ribbon` (остальные → fallback на belly_oriented,
    как в проде). Кадры разделены на gallery (split_fold=train, чем учить эмбеддер) и probe (dev, для A/B).
    Возвращает crops_df (схема crops_manifest), gallery_rows / probe_rows (как loader.select), individuals,
    ks. Детерминизм seed=42 (table-only → быстро и герметично).
    """
    rng = np.random.RandomState(42)
    inds = [f"TK-{i:02d}" for i in range(1, 9)] + [f"PW-{i:02d}" for i in range(1, 5)]
    crops, grows, prows = [], [], []
    n = 0
    for ind in inds:
        cohort = ind.split("-")[0]
        n_gal = int(rng.randint(2, 5))            # 2..4 галерейных кадра (>=2 → kfold по особям корректен)
        n_prb = int(rng.randint(1, 3))            # 1..2 probe-кадра (dev)
        for j in range(n_gal + n_prb):
            md5 = f"{n:032x}"; n += 1
            role = "gallery" if j < n_gal else "probe"
            fold = "train" if role == "gallery" else "dev"
            base = {"md5": md5, "individual_id": ind, "cohort": cohort, "split_role": role,
                    "split_fold": fold, "kpi_scope": "kpi_core", "seg_conf": round(float(rng.uniform(0.2, 0.9)), 3)}
            (grows if role == "gallery" else prows).append(dict(base))
            crops.append({**base, "variant": "belly_oriented", "crop_status": "ok",
                          "crop_path": f"crops/{md5}.png"})
            if rng.rand() < 0.7:                  # ~70 % имеют ribbon → fallback тестируем на остальных
                crops.append({**base, "variant": "unroll_ribbon", "crop_status": "ok",
                              "crop_path": f"crops/{md5}__ribbon.png"})
    return SimpleNamespace(
        crops_df=pd.DataFrame(crops),
        gallery_rows=pd.DataFrame(grows).reset_index(drop=True),
        probe_rows=pd.DataFrame(prows).reset_index(drop=True),
        individuals=inds, ks=(1, 5))


@pytest.fixture
def mock_embedder():
    """Детерминированный «эмбеддер» БЕЗ timm/torch: кадры одной особи → близкие L2-norm векторы, разные
    особи разнесены. Для A/B (recall) и open-set (AUROC known≫new) на синтетике — без тяжёлых моделей.

    Возвращает callable embed(individual_ids, dim=64, sep=1.0, noise=0.05, seed=42) -> (N, dim) float32.
    Центр особи детерминирован hashlib (стабилен между запусками; НЕ встроенный hash — тот зависит от
    PYTHONHASHSEED). sep — разнесённость центров (больше → выше recall); noise — внутриклассовый разброс.
    """
    def _embed(individual_ids, dim=64, sep=1.0, noise=0.05, seed=42):
        ids = list(individual_ids)
        centers = {}
        for u in set(ids):
            s = int(hashlib.md5(str(u).encode("utf-8")).hexdigest()[:8], 16)
            centers[u] = np.random.RandomState(s).randn(dim) * sep
        rng = np.random.RandomState(seed)
        out = np.stack([centers[i] + rng.randn(dim) * noise for i in ids]).astype(np.float32)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
    return _embed
