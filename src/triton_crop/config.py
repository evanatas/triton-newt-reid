"""Конфиги: CropConfig (Блок 2, кроп брюшка) + EmbedConfig (Блок 4, эмбеддер re-ID).

Оба — frozen dataclass + load_*_config(yaml) (лишние ключи игнорируются, tuple-поля кастятся).
Источник истины прогонов — yaml в configs/ (crop.yaml пилотно расходится только в seg/pose_conf_min);
dataclass-дефолты — fallback; рассинхрон ловят тесты-стражи в tests/test_crop_config.py.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_PATH = _CONFIGS / "crop.yaml"
_DEFAULT_EMBED_PATH = _CONFIGS / "embed.yaml"
_DEFAULT_SPOT_PATH = _CONFIGS / "spot.yaml"


@dataclass(frozen=True)
class CropConfig:
    canon_size: int = 384              # сторона канонического кропа (квадрат, letterbox)
    margin_frac: float = 0.06          # доля стороны bbox пуза в tight-кроп
    seg_conf_min: float = 0.35         # ниже → fallback
    seg_area_frac_min: float = 0.01    # пустая/мелкая маска → fallback
    pose_conf_min: float = 0.30        # ниже → ось из PCA + арбитр, флаг
    mask_background: str = "black"     # заливка фона: black|mean|none
    unroll_halfwidth_frac: float = 0.55
    # --- Блок 3 (распрямление / unroll) ---
    unroll_methods: tuple = ("debend", "ribbon", "wnorm")  # какие методы строить и A/B; debend — безопасный дефолт
    unroll_poly_deg: int = 3              # степень полинома центральной линии (ловит S-изгиб)
    unroll_min_rows: int = 24             # минимум строк тела для распрямления, иначе fallback
    unroll_max_shift_frac: float = 0.45   # кламп сдвига строки (доля стороны) — защита от выезда узора
    unroll_pattern_tol_frac: float = 0.04   # допуск смещения центроидов пятен (измеренный гейт узора)
    clip_area_tol_frac: float = 0.02        # доля потери площади маски после поворота, выше которой кроп помечается clipped
    fallback_ladder: tuple = ("belly_oriented", "belly_mask", "full")  # рунги, реально реализованные в pipeline
    seed: int = 42
    proxy_model: str = "hf-hub:BVRA/MegaDescriptor-L-384"
    seg_weights: str = "artifacts/belly_seg.pt"
    pose_weights: str = "artifacts/belly_pose.pt"


def load_crop_config(path=_DEFAULT_PATH) -> CropConfig:
    """Прочитать crop.yaml поверх дефолтов CropConfig (лишние ключи игнорируются)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "fallback_ladder" in raw:
        raw["fallback_ladder"] = tuple(raw["fallback_ladder"])
    if "unroll_methods" in raw:
        raw["unroll_methods"] = tuple(raw["unroll_methods"])
    known = set(CropConfig.__dataclass_fields__)
    return CropConfig(**{k: v for k, v in raw.items() if k in known})


@dataclass(frozen=True)
class EmbedConfig:
    """Ручки Блока 4 (дообучение эмбеддера re-ID). Дефолты = рецепт MegaDescriptor (Čermák WACV-2024)
    с поправкой на малый датасет (~600 кадров, оверфит Swin-L — урок 2.0: freeze + аугментация + cross-fit).
    """
    # --- эмбеддер и его вход ---
    base_model: str = "hf-hub:BVRA/MegaDescriptor-L-384"  # timm zero-shot baseline → finetune
    image_size: int = 384                      # сторона входа (= сторона канонического кропа)
    embed_variant: str = "unroll_ribbon"       # принятый вход (Блок 3: значимый @5)
    fallback_variant: str = "belly_oriented"   # если нет ribbon-кропа (тот же fallback-протокол A/B)
    # --- MiewID (2-й кандидат A/B; Rosa: MiewID 73/93 > MegaDescriptor 63/83 на hard-split) ---
    miewid_model: str = "conservationxlabs/miewid-msv3"  # грузится transformers AutoModel; выход 2152-d
    miewid_revision: str = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"  # пин = состояние main на 2026-06-10 (= embed_model.MIEWID_REVISION, страж в тестах); MiewID — dev-сравнение Блока 4, НЕ sealed-система
    miewid_image_size: int = 440               # вход MiewID 440² + ImageNet-norm (НЕ 384, как MegaDescriptor)
    # --- ArcFace (Čermák: лучший сеттинг m=0.5, s=64) ---
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    # --- оптимизация: рецепт ЗАФИКСИРОВАН в коде (embed_train всегда строит SGD + cosine-annealing с warmup;
    #     раннего стопа НЕТ — вместо val-early-stop честность даёт cross-fit OOF). Поля ниже = действующие гиперпараметры. ---
    optimizer: str = "sgd"                     # для лога/прозрачности; код всегда строит SGD
    momentum: float = 0.9
    lr_head: float = 1e-3                       # голова ArcFace учится быстрее backbone
    lr_backbone: float = 1e-5                   # backbone дотюнивается мягко (анти-оверфит)
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_epochs: int = 3
    # --- P×K-сэмплер metric learning (P особей × K кадров на батч) ---
    batch_p: int = 8
    batch_k: int = 4
    # --- регуляризация против оверфита Swin-L на малом датасете (урок 2.0) ---
    freeze_backbone_stages: int = 3            # морозим patch_embed+layers[0..2] (~68% Swin-L): регуляризация малой выборки + экономия VRAM
    augment: bool = True
    # --- temporal-эксперимент (C2/C3/C4; дефолты OFF → финальная система и Блок 4 без изменений) ---
    session_aware_sampling: bool = False       # C2: K кадров особи из РАЗНЫХ сессий (cross-session позитивы)
    drift_augment: bool = False                # C3: целевая drift-аугментация (включается после сигнала C2) (зарезервировано — в коде не потребляется)
    train_cohorts: tuple = ("TK", "PW")        # когорты в обучении; +("LAB",) = C4 (зарезервировано — в коде не потребляется)
    train_all_sessions: bool = False           # A1: в обучение брать все сессии train-особей (gallery+probe) → cross-session пары для session-aware
    # --- честная оценка (cross-fit out-of-fold ПО ОСОБЯМ; анти-утечка train→gallery) ---
    cross_fit_folds: int = 5
    recall_ks: tuple = (1, 5)                  # какие top-k мерить (KPI: @1 и @5)
    # --- open-set (known/new + калибровка порога) ---
    open_set_youden: bool = True               # порог по индексу Юдена на open_dev
    open_set_margin_min: float = 0.0           # доп. порог margin top1−top2 (0 = только score)
    # --- воспроизводимость / артефакты ---
    seed: int = 42
    ckpt_dir: str = "artifacts/embed"          # чекпойнты finetune (Colab → Drive каждую эпоху)


def load_embed_config(path=_DEFAULT_EMBED_PATH) -> EmbedConfig:
    """Прочитать embed.yaml поверх дефолтов EmbedConfig (лишние ключи игнорируются)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "recall_ks" in raw:
        raw["recall_ks"] = tuple(raw["recall_ks"])
    if "train_cohorts" in raw:
        raw["train_cohorts"] = tuple(raw["train_cohorts"])
    known = set(EmbedConfig.__dataclass_fields__)
    return EmbedConfig(**{k: v for k, v in raw.items() if k in known})


@dataclass(frozen=True)
class SpotConfig:
    """Ручки Блока 5 (детекция пятен → созвездие центроидов → матчинг → top-1 + оверлей).

    Детектор и матчер ПОДКЛЮЧАЕМЫЕ (detect_method / match_method) — лучший выбирается честным A/B на dev
    (цель — взять KPI top-1). Матчер НЕ обучается (классический CV) → cross-fit не нужен; параметры тюнятся
    на dev/синтетике (test/open_test НЕ вскрывать).
    """
    # --- поверхность детекции (НЕ ribbon: spot-QA Блока 3 — ribbon сливает пятна, медиана Δ=−4) ---
    detect_variant: str = "belly_oriented"     # belly_oriented | unroll_debend
    fallback_variant: str = "belly_oriented"   # если нет debend-кропа — тот же fallback-протокол A/B
    # --- детектор пятен (подключаемый; лучший — по A/B на dev) ---
    detect_method: str = "deviation"           # deviation (отклонение от цвета тела — реал И синтетика) | darkness | log | dog
    illum_norm: bool = False                   # CLAHE-нормализация освещения/контраста ДО детекции (выровнять бледные↔яркие кадры между сессиями — главный барьер temporal)
    clahe_clip: float = 2.0                    # сила CLAHE (clipLimit)
    clahe_grid: int = 8                        # сетка CLAHE (tileGridSize)
    deviation_k: float = 4.0                   # пятно: ||rgb − median_body|| > k·MAD (робастный порог)
    darkness_frac: float = 0.62                # darkness: gray < darkness_frac * mean(fg внутри маски) — для реальных тёмных пятен
    mask_erode_px: int = 4                     # эрозия маски перед детекцией — убрать краевые артефакты (letterbox/обрез)
    spot_min_area: int = 6                     # мин площадь компоненты (px) — против шума
    spot_top_n: int = 20                       # top-N по площади — против over-detection (76–121 «пятен»)
    spot_min_score: float = 0.0               # порог салиентности (контраст пятна к локальному фону)
    select_by: str = "salience"               # отбор top-N: salience (area×contrast) | area
    bg_sum_thr: int = 30                       # foreground_from_crop: сумма каналов <= → near-чёрный фон
    log_min_sigma: float = 2.0                 # blob_log/dog: масштабы (px)
    log_max_sigma: float = 8.0
    log_num_sigma: int = 5
    log_threshold: float = 0.05                # порог отклика blob-детектора
    # --- матчер созвездия (similarity, det>0, БЕЗ зеркала) ---
    match_method: str = "guided"               # guided (дескриптор-кандидаты + исчерпывающий перебор, детерминирован) | ransac | nn
    descriptor_knn: int = 5                    # соседей в локальном similarity-инвариантном дескрипторе (guided)
    match_knn: int = 4                         # кандидатов-соответствий на пятно по дескриптору (guided)
    ransac_iters: int = 500                    # итераций случайного RANSAC (для sensitivity-сравнения; детерм. seed)
    match_inlier_tol: float = 0.08             # допуск невязки inlier (доля масштаба созвездия)
    score_norm: str = "max"                    # max (n_in/max(a,b) — устойчиво к спурьёзным мелким созвездиям) | min | mean | jaccard
    min_inliers: int = 3                       # < min_inliers совпавших пятен → score=0 (коинцидентное 2-3-точечное ≠ матч)
    min_spots_for_match: int = 3               # < N пятен в созвездии → score=0 (нечего матчить)
    # --- метрики / воспроизводимость ---
    recall_ks: tuple = (1, 5)
    seed: int = 42


def load_spot_config(path=_DEFAULT_SPOT_PATH) -> SpotConfig:
    """Прочитать spot.yaml поверх дефолтов SpotConfig (лишние ключи игнорируются, recall_ks → tuple)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "recall_ks" in raw:
        raw["recall_ks"] = tuple(raw["recall_ks"])
    known = set(SpotConfig.__dataclass_fields__)
    return SpotConfig(**{k: v for k, v in raw.items() if k in known})
