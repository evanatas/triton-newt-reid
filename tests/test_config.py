"""Тесты загрузки конфига. ГЕРМЕТИЧНЫ: не требуют данных заказчика (CI-safe)."""
from pathlib import Path

import pytest

from triton_data.config import ConfigError, load_config

_EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "configs" / "paths.example.yaml"


def test_repo_config_parses_hermetically():
    # парсинг публичного шаблона configs/paths.example.yaml БЕЗ требования наличия папок данных
    # (реальный paths.yaml в .gitignore → на свежем clone его нет; шаблон всегда в репозитории)
    cfg = load_config(_EXAMPLE_YAML, validate_dirs=False)
    assert cfg.seed == 42
    by = {d.name: d for d in cfg.datasets}
    assert set(by) == {"karelinii", "lab", "pleurodeles", "gcnid"}
    assert by["gcnid"].role == "external" and by["lab"].role == "target"
    assert by["karelinii"].id_prefix == "TK"
    assert len(by["karelinii"].skip_dirs) == 1  # шаблонный плейсхолдер папки без снимков
    assert cfg.merge_individuals == {}
    prefixes = [d.id_prefix for d in cfg.datasets]
    assert len(prefixes) == len(set(prefixes))


def test_load_config_missing_file_friendly(tmp_path):
    # свежий clone: configs/paths.yaml в .gitignore → вместо сырого FileNotFoundError
    # дружелюбная ConfigError с подсказкой скопировать шаблон paths.example.yaml
    with pytest.raises(ConfigError, match="paths.example.yaml"):
        load_config(tmp_path / "нет.yaml", validate_dirs=False)


def _write_yaml(path, workspace, dirname):
    path.write_text(
        f'workspace_root: "{workspace}"\n'
        "seed: 7\n"
        "datasets:\n"
        f"  - {{name: a, dir: {dirname}, species: X, id_prefix: A, role: target, layout: folder_per_id}}\n",
        encoding="utf-8",
    )


def test_validate_dirs_ok_on_existing(tmp_path):
    (tmp_path / "ws" / "d1").mkdir(parents=True)
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(yaml_path, tmp_path / "ws", "d1")
    cfg = load_config(yaml_path, validate_dirs=True)
    assert cfg.seed == 7 and cfg.datasets[0].id_prefix == "A"


def test_validate_dirs_raises_on_missing(tmp_path):
    (tmp_path / "ws").mkdir()
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(yaml_path, tmp_path / "ws", "nope")
    with pytest.raises(ConfigError):
        load_config(yaml_path, validate_dirs=True)


def test_embed_config_temporal_flags_default_off():
    from triton_crop.config import EmbedConfig
    c = EmbedConfig()
    assert c.session_aware_sampling is False             # дефолт OFF → Блок 4 воспроизводим
    assert c.drift_augment is False
    assert c.train_cohorts == ("TK", "PW")


def test_embed_config_train_all_sessions_default_off():
    from triton_crop.config import EmbedConfig
    assert EmbedConfig().train_all_sessions is False     # дефолт OFF → Блок 4 без изменений


def test_load_embed_config_picks_session_flag(tmp_path):
    from triton_crop.config import load_embed_config
    p = tmp_path / "embed_C2.yaml"
    p.write_text("session_aware_sampling: true\ntrain_cohorts: [TK, PW, LAB]\n", encoding="utf-8")
    c = load_embed_config(p)
    assert c.session_aware_sampling is True
    assert c.train_cohorts == ("TK", "PW", "LAB")        # список из yaml → tuple
