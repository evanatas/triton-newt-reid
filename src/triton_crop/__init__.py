"""triton_crop — Блок 2: сегментация брюшка + ориентация + канонический кроп.

Зависит от triton_data (loader, imageio). Модели (YOLO/SAM2) — ТОЛЬКО в модулях
sam_bootstrap/predict/train_*/proxy_embed; геометрия/маски/манифест — чистый numpy,
покрыты тестами на синтетике (как Блок 1).
"""

__version__ = "0.1.0"
