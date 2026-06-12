"""Тесты CropConfig + EmbedConfig + чтения configs/crop.yaml / embed.yaml (TDD)."""
import pytest

from triton_crop.config import (
    CropConfig,
    EmbedConfig,
    SpotConfig,
    load_crop_config,
    load_embed_config,
    load_spot_config,
)


def test_cropconfig_defaults():
    c = CropConfig()
    assert c.canon_size == 384 and c.seed == 42
    assert c.fallback_ladder[0] == "belly_oriented"


def test_load_crop_config_reads_yaml():
    c = load_crop_config()  # configs/crop.yaml
    assert c.canon_size == 384
    assert c.mask_background == "black"
    assert "belly_oriented" in c.fallback_ladder


def test_crop_yaml_diverges_from_dataclass_only_in_conf_thresholds():
    # страж doc↔код: crop.yaml — источник истины (пилотные значения); по комментарию в YAML расхождение
    # с дефолтами CropConfig допустимо ТОЛЬКО в seg_conf_min/pose_conf_min. Ловит будущий рассинхрон.
    y = load_crop_config()           # из configs/crop.yaml (пилот)
    d = CropConfig()                 # дефолты dataclass (целевой fallback)
    assert y.seg_conf_min == 0.15 and y.pose_conf_min == 0.15       # источник истины — YAML
    assert d.seg_conf_min == 0.35 and d.pose_conf_min == 0.30       # целевой fallback dataclass
    diff = {k for k in CropConfig.__dataclass_fields__ if getattr(y, k) != getattr(d, k)}
    assert diff == {"seg_conf_min", "pose_conf_min"}
    assert "bbox" not in y.fallback_ladder and "bbox" not in d.fallback_ladder  # мёртвый рунг убран


def test_embed_and_spot_yaml_match_dataclass_defaults():
    # страж двойного источника истины: embed.yaml/spot.yaml заявляют дублирование дефолтов
    # dataclass — ловим тихий рассинхрон значений (crop.yaml легально расходится в conf-порогах,
    # его страж — выше). frozen dataclass → поэлементное сравнение.
    assert load_embed_config() == EmbedConfig()
    assert load_spot_config() == SpotConfig()


# --- Блок 4: EmbedConfig (эмбеддер re-ID, finetune) ---

def test_embedconfig_defaults():
    c = EmbedConfig()
    # эмбеддер и вход (принятый Блоком 3 unroll_ribbon + fallback belly_oriented)
    assert c.base_model == "hf-hub:BVRA/MegaDescriptor-L-384"
    assert c.embed_variant == "unroll_ribbon"
    assert c.fallback_variant == "belly_oriented"
    # ArcFace по Čermák WACV-2024 (m=0.5, s=64)
    assert c.arcface_margin == 0.5 and c.arcface_scale == 64.0
    # честная оценка + воспроизводимость
    assert c.cross_fit_folds == 5 and c.seed == 42
    assert c.recall_ks == (1, 5)
    # open-set: калибровка порога по Youden
    assert c.open_set_youden is True
    # P×K-сэмплер metric learning
    assert c.batch_p == 8 and c.batch_k == 4


def test_load_embed_config_reads_yaml():
    c = load_embed_config()  # configs/embed.yaml
    assert c.base_model.startswith("hf-hub:")
    assert c.embed_variant == "unroll_ribbon"
    assert isinstance(c.recall_ks, tuple) and c.recall_ks == (1, 5)
    assert c.arcface_margin == 0.5 and c.arcface_scale == 64.0


def test_embed_yaml_has_no_decorative_keys():
    # страж: embed.yaml не должен рекламировать ключи, которых нет в EmbedConfig (ловит мёртвый конфиг)
    import yaml

    from triton_crop.config import _DEFAULT_EMBED_PATH
    keys = set(yaml.safe_load(_DEFAULT_EMBED_PATH.read_text(encoding="utf-8")))
    extra = keys - set(EmbedConfig.__dataclass_fields__)
    assert not extra, f"декоративные ключи в embed.yaml: {extra}"


def test_embedconfig_miewid_fields():
    c = EmbedConfig()
    assert c.miewid_model == "conservationxlabs/miewid-msv3"
    assert c.miewid_image_size == 440                  # MiewID: вход 440×440 (НЕ 384, как MegaDescriptor)
    assert hasattr(c, "miewid_revision")               # пин commit для воспроизводимости ВКР


def test_miewid_revision_pinned_and_matches_model_pin():
    # пин ревизии MiewID (trust_remote_code): config-дефолт обязан совпадать с пином
    # embed_model.MIEWID_REVISION — один и тот же commit, без тихого расхождения
    from triton_crop.embed_model import MIEWID_REVISION
    c = EmbedConfig()
    assert c.miewid_revision == MIEWID_REVISION and c.miewid_revision != ""


def test_load_embed_config_ignores_unknown_and_casts_tuple(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(
        "arcface_margin: 0.4\n"
        "unknown_key: 123\n"          # лишний ключ → игнор (как load_crop_config)
        "recall_ks: [1, 3, 5]\n",     # list из yaml → tuple
        encoding="utf-8",
    )
    c = load_embed_config(p)
    assert c.arcface_margin == 0.4
    assert c.recall_ks == (1, 3, 5)
    assert not hasattr(c, "unknown_key")


# --- Блок 5: SpotConfig (детекция пятен + матчинг созвездия) ---

def test_spotconfig_defaults():
    c = SpotConfig()
    # поверхность детекции — belly_oriented/debend, НЕ ribbon (spot-QA Блока 3)
    assert c.detect_variant == "belly_oriented"
    assert c.fallback_variant == "belly_oriented"
    assert c.detect_method in ("deviation", "darkness", "log", "dog")
    # матчер: дефолт guided (дескриптор-кандидаты + исчерпывающий перебор, детерминирован)
    assert c.match_method in ("guided", "ransac", "nn")
    assert c.score_norm in ("max", "min", "mean", "jaccard")
    assert c.min_spots_for_match >= 2
    # детерминизм + KPI-метрики
    assert c.seed == 42
    assert c.recall_ks == (1, 5)


def test_spotconfig_frozen():
    c = SpotConfig()
    with pytest.raises(Exception):                 # frozen dataclass — присваивание запрещено
        c.darkness_frac = 0.5                       # type: ignore[misc]


def test_load_spot_config_reads_yaml():
    c = load_spot_config()                          # configs/spot.yaml
    assert c.detect_variant in ("belly_oriented", "unroll_debend")
    assert c.detect_variant != "unroll_ribbon"      # хард-запрет (ribbon сливает пятна)
    assert isinstance(c.recall_ks, tuple)


def test_load_spot_config_ignores_unknown_and_casts_tuple(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "darkness_frac: 0.55\n"
        "match_method: nn\n"
        "unknown_key: 1\n"                           # лишний ключ → игнор
        "recall_ks: [1, 3, 5]\n",                    # list из yaml → tuple
        encoding="utf-8",
    )
    c = load_spot_config(p)
    assert c.darkness_frac == 0.55
    assert c.match_method == "nn"
    assert c.recall_ks == (1, 3, 5)
    assert not hasattr(c, "unknown_key")


def test_spot_config_new_fields_defaults():
    # инварианты Блока 5: guided-матчер (детерминирован), салиентный отбор, foreground-порог
    c = SpotConfig()
    assert c.match_method == "guided"          # дефолт-матчер теперь guided (устраняет iteration-starvation)
    assert c.select_by == "salience"           # отбор top-N по салиентности (area×contrast), не площади
    assert c.bg_sum_thr == 30                  # foreground_from_crop: порог near-чёрного фона
    assert c.descriptor_knn == 5               # соседей в локальном similarity-инвариантном дескрипторе
