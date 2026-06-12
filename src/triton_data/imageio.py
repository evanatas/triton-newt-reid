"""Единственный канонический ридер изображений проекта.

Дисциплина единого препроцессинга (урок предыдущей версии проекта): пиксели, размеры и md5
читаются ТОЛЬКО здесь. Блоки 2–6 импортируют эти же функции — расхождений препроцессинга
между gallery/probe/train быть не может.
"""
import hashlib
from dataclasses import dataclass

from PIL import Image, ImageOps

_ORIENTATION_TAG = 0x0112  # 274 — EXIF Orientation


@dataclass(frozen=True)
class ImageStats:
    """Свойства изображения: размеры — УЖЕ с учётом EXIF-поворота (display-размеры)."""

    width: int
    height: int
    orientation: int | None  # сырой тег EXIF (для воспроизводимости transpose)
    mode: str


def file_md5(path, chunk_size: int = 1 << 20) -> str:
    """md5 по БАЙТАМ файла (потоково). Байт-идентичные файлы дают одинаковый хеш
    независимо от EXIF — это корректная основа дедупа."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image_stats(path) -> ImageStats:
    """Display-размеры (с учётом EXIF-поворота) + сырой тег ориентации, БЕЗ декодирования пикселей.

    Размеры берутся из заголовка; для ориентаций 5–8 (повороты на 90°/270°) стороны меняются
    местами — как сделал бы exif_transpose, но без дорогого декодирования всего изображения.
    """
    with Image.open(path) as img:
        exif = img.getexif()
        orientation = exif.get(_ORIENTATION_TAG) if exif else None
        width, height = img.size
        mode = img.mode
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return ImageStats(width=width, height=height, orientation=orientation, mode=mode)


def load_canonical(path) -> Image.Image:
    """Загрузить изображение в каноническом виде: EXIF-поворот применён, режим RGB.

    Это единственный загрузчик пикселей для последующих блоков (сегментация, эмбеддер).
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
