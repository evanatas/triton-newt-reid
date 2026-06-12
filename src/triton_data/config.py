"""Загрузка и валидация configs/paths.yaml в типизированные объекты."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Ошибка конфигурации (отсутствует ключ/папка, недопустимое значение)."""


@dataclass(frozen=True)
class DatasetSpec:
    """Описание одной когорты-датасета."""

    name: str
    dir: str
    species: str
    id_prefix: str
    role: str            # "target" | "external"
    layout: str          # "folder_per_id" | "date_subdirs" | "metadata_csv"
    skip_dirs: tuple = ()
    extra: dict = field(default_factory=dict)  # layout-зависимое (metadata_csv, raw_subdir, id_from…)

    def abs_dir(self, workspace_root) -> Path:
        return Path(workspace_root) / self.dir


@dataclass(frozen=True)
class Config:
    workspace_root: Path
    seed: int
    image_extensions: tuple
    datasets: tuple
    merge_individuals: dict


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"
_SPEC_KEYS = {"name", "dir", "species", "id_prefix", "role", "layout", "skip_dirs"}


def load_config(path=_DEFAULT_PATH, validate_dirs: bool = True) -> Config:
    """Прочитать paths.yaml → Config. При validate_dirs проверяет существование папок."""
    path = Path(path)
    if not path.exists():    # paths.yaml в .gitignore → на свежем clone его нет; подсказываем шаблон
        raise ConfigError(f"{path} не найден — скопируйте configs/paths.example.yaml "
                          f"в configs/paths.yaml и подставьте свои пути.")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    workspace_root = Path(raw["workspace_root"])
    if validate_dirs and not workspace_root.is_dir():
        raise ConfigError(f"workspace_root не найден: {workspace_root}")

    exts = tuple(e.lower() for e in raw.get("image_extensions", [".jpg", ".jpeg"]))

    specs: list[DatasetSpec] = []
    prefixes: set[str] = set()
    for d in raw["datasets"]:
        extra = {k: v for k, v in d.items() if k not in _SPEC_KEYS}
        spec = DatasetSpec(
            name=d["name"], dir=d["dir"], species=d["species"],
            id_prefix=d["id_prefix"], role=d["role"], layout=d["layout"],
            skip_dirs=tuple(d.get("skip_dirs", [])), extra=extra,
        )
        if spec.role not in ("target", "external"):
            raise ConfigError(f"role должен быть target/external: {spec.name}={spec.role}")
        if spec.id_prefix in prefixes:
            raise ConfigError(f"дублирующийся id_prefix: {spec.id_prefix}")
        prefixes.add(spec.id_prefix)
        if validate_dirs and not spec.abs_dir(workspace_root).is_dir():
            raise ConfigError(f"папка датасета не найдена: {spec.abs_dir(workspace_root)}")
        specs.append(spec)

    return Config(
        workspace_root=workspace_root,
        seed=int(raw["seed"]),
        image_extensions=exts,
        datasets=tuple(specs),
        merge_individuals=dict(raw.get("merge_individuals") or {}),
    )
