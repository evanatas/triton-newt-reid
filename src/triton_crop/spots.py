"""Детекция пятен на брюшке тритона (Блок 5, шаг 5.3) — ЧИСТЫЙ CV (numpy/cv2/skimage).

Пятна = локальные области, цвет которых сильно отличается от тела (тёмные на жёлто-оранжевом у реальных
тритонов; контрастные — у синтетики). Детектор ПОДКЛЮЧАЕМЫЙ (cfg.detect_method): лучший метод+параметры
выбираются честным A/B на dev (cli spot-ab). Работает ВНУТРИ маски пуза (фон/лапы не считаем).

Возвращает центроиды пятен — устойчивый признак: «центр пятна не двигается» при росте/растяжении особи,
поэтому именно центроид, а не контур, идёт в созвездие (constellation.py). Отбор салиентных (min-area,
top-N) против over-detection (на худших реальных кадрах простой порог даёт 76–121 «пятен»).

Методы:
  • deviation (дефолт) — отклонение цвета пикселя от МЕДИАНЫ тела > k·MAD (робастно; ловит и тёмные
    реальные, и контрастные синтетические пятна);
  • darkness — относительная темнота gray < darkness_frac·mean(тело) (ранний прототип,
    для реальных тёмных пятен на светлом теле);
  • log / dog — skimage.feature.blob_log / blob_dog по карте «пятнистости» (масштаб-инвариантно).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spot:
    """Одно салиентное пятно. Координаты — в каноне кропа (пиксели; x=столбец, y=строка)."""
    x: float          # центроид по x (столбец) — устойчивый признак (не двигается при росте пятна)
    y: float          # центроид по y (строка)
    area: float       # площадь, px (для top-N отбора и веса)
    score: float      # салиентность: контраст пятна к телу (>1 = ярче порога), для фильтра/сортировки


def spots_to_array(spots) -> np.ndarray:
    """list[Spot] -> (N,2) float-массив центроидов (x,y). Пустой -> (0,2)."""
    if not spots:
        return np.zeros((0, 2), float)
    return np.array([[s.x, s.y] for s in spots], float)


def _normalize_illum(rgb, cfg):
    """CLAHE-нормализация освещения/контраста (LAB, выравнивание L). Делает бледные и яркие кадры одной
    особи между сессиями сопоставимее (главный барьер temporal — см. визуальный QA TK-004). -> RGB uint8."""
    import cv2
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip),
                            tileGridSize=(int(cfg.clahe_grid), int(cfg.clahe_grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _spotness(rgb, mask):
    """Карта «пятнистости» = евклидово расстояние цвета пикселя от МЕДИАНЫ тела (внутри маски).
    -> (d[H,W] float, dmed, dmad) (робастные медиана/MAD расстояния по пикселям тела)."""
    rgb = np.asarray(rgb, float)
    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(rgb.shape[:2], float), 0.0, 0.0
    med = np.median(rgb[m], axis=0)
    d = np.sqrt(((rgb - med) ** 2).sum(axis=2))
    dm = d[m]
    dmed = float(np.median(dm))
    dmad = float(np.median(np.abs(dm - dmed)))
    return d, dmed, dmad


def _components(binary, mask, rgb, dmap, cfg):
    """connectedComponents по бинарной карте пятен ВНУТРИ маски -> list[Spot] (до отбора top-N).
    score = СРЕДНЯЯ «пятнистость» компоненты (по всем её пикселям), нормированная на (медиана+MAD) тела
    (контраст). Средняя, а не в одном пикселе-центроиде (центроид может попасть на слабый пиксель)."""
    import cv2
    binary = (np.asarray(binary, bool) & np.asarray(mask, bool)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    _, dmed, dmad = _spotness(rgb, mask)
    denom = dmed + dmad + 1e-6
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.spot_min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        comp = labels == i
        score = float(dmap[comp].mean() / denom) if dmap is not None and comp.any() else 1.0
        out.append(Spot(x=cx, y=cy, area=float(area), score=score))
    return out


def _select(spots, cfg):
    """Отбор салиентных: фильтр score + top-N по САЛИЕНТНОСТИ (area×contrast) или площади (детерминированно).
    select_by='salience' (дефолт): крупная слабоконтрастная тень не вытесняет маленькое яркое пятно."""
    import math
    spots = [s for s in spots if s.score >= cfg.spot_min_score]
    if getattr(cfg, "select_by", "salience") == "salience":
        # СУБЛИНЕЙНО по площади (√area·contrast): крупная клякса не топит мелкие яркие
        key = lambda s: (-(math.sqrt(max(s.area, 0.0)) * s.score), -s.score, -s.area, s.x, s.y)
    else:
        key = lambda s: (-s.area, s.x, s.y)                                  # старое поведение: по площади
    spots = sorted(spots, key=key)
    return spots[: cfg.spot_top_n]


def detect_spots(rgb, mask, cfg) -> list[Spot]:
    """Детектировать пятна внутри маски пуза. -> list[Spot] (отсортирован по убыванию салиентности
    √area·contrast при дефолтном select_by='salience'; по убыванию площади — при select_by='area',
    см. _select).

    Метод — cfg.detect_method ∈ {deviation, darkness, log, dog}. Детерминирован (рандома нет)."""
    import cv2
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    erode = int(getattr(cfg, "mask_erode_px", 0))
    if erode > 0:                                    # убрать краевое кольцо (letterbox/обрез → ложные тёмные пятна)
        k = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        if not mask.any():
            return []
    if getattr(cfg, "illum_norm", False):            # выровнять освещение/контраст между сессиями (CLAHE)
        rgb = _normalize_illum(rgb, cfg)
    method = cfg.detect_method
    dmap, dmed, dmad = _spotness(rgb, mask)

    if method == "deviation":
        thr = dmed + cfg.deviation_k * dmad
        binary = dmap > thr
        return _select(_components(binary, mask, rgb, dmap, cfg), cfg)

    if method == "darkness":
        gray = rgb.astype(float).mean(axis=2)
        thr = cfg.darkness_frac * float(gray[mask].mean())
        binary = (gray < thr) & mask
        return _select(_components(binary, mask, rgb, dmap, cfg), cfg)

    if method in ("log", "dog"):
        from skimage.feature import blob_dog, blob_log
        s = dmap / (dmap[mask].max() + 1e-6)            # карта пятнистости в [0,1] (тёмное тело → 0)
        s = np.where(mask, s, 0.0)
        fn = blob_log if method == "log" else blob_dog
        blobs = fn(s, min_sigma=cfg.log_min_sigma, max_sigma=cfg.log_max_sigma,
                   num_sigma=cfg.log_num_sigma, threshold=cfg.log_threshold) if method == "log" else \
            fn(s, min_sigma=cfg.log_min_sigma, max_sigma=cfg.log_max_sigma, threshold=cfg.log_threshold)
        out = []
        for b in blobs:
            yy, xx, sig = float(b[0]), float(b[1]), float(b[-1])
            if not mask[int(round(yy)), int(round(xx))]:
                continue
            area = float(np.pi * (np.sqrt(2.0) * sig) ** 2)
            if area < cfg.spot_min_area:
                continue
            out.append(Spot(x=xx, y=yy, area=area, score=float(s[int(round(yy)), int(round(xx))])))
        return _select(out, cfg)

    raise ValueError(f"detect_spots: неизвестный detect_method={method!r} (deviation|darkness|log|dog)")
