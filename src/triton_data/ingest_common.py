"""Общие помощники ингесторов: базовая запись манифеста и фильтр изображений."""
from pathlib import Path

from .imageio import file_md5, read_image_stats


def is_image(path: Path, exts) -> bool:
    """Файл является изображением нужного расширения и не скрытый (не .DS_Store и т.п.)."""
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in exts


def rel_path(abs_path, workspace_root) -> str:
    """Путь относительно workspace_root (в манифесте — только относительные пути)."""
    return str(Path(abs_path).resolve().relative_to(Path(workspace_root).resolve()))


def base_record(abs_path, spec, cfg) -> dict:
    """Базовая запись target-когорты: id-неймспейс, путь, md5, display-размеры, заготовки полей."""
    st = read_image_stats(abs_path)
    return {
        "cohort": spec.id_prefix,
        "species": spec.species,
        "role": spec.role,
        "local_id": None,
        "rel_path": rel_path(abs_path, cfg.workspace_root),
        "md5": file_md5(abs_path),
        "width": st.width,
        "height": st.height,
        "orientation": st.orientation,
        "date": None,
        "date_source": "none",
        "session": None,
        "shot": None,
        "notes": "",
    }
