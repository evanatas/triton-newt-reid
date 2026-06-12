"""Аугментации для дообучения эмбеддера re-ID (Блок 4) — достоверные, детерминируемые.

Что мотивирует выбор (постановка заказчика + специфика тритонов как объекта):
  • Affine scale 0.9–1.1 — особь поела/переварила (узор растягивается), рост — но НЕ flip;
  • Affine rotate/translate малые — кадр уже ориентирован (голова вверх), нужна лишь устойчивость;
  • ColorJitter — съёмка «в темноте с фонариком на разных субстратах» (демо с заказчиком): блики/тень/фон;
  • ElasticTransform — нелинейное растяжение узора (изгиб тела); при дефолте alpha=1.0 (sigma=50) амплитуда
    смещений ~1px на 384²-кропе — фактически выключена (замерено: меняется ~0.2 % пикселей против ~79 % при
    alpha=50); рабочие значения — десятки‑сотни, в прогонах Блока 4 не использовались;
  • GaussNoise — сенсорный шум телефона.
БЕЗ зеркала (HorizontalFlip/Flip/Transpose) — зеркальный тритон = ДРУГАЯ особь (заказчик); зеркало
допустимо лишь как новый искусственный ID, что делается вне этого пайплайна. albumentations — lazy
(CPU локально; на Colab ElasticTransform лучше выносить на GPU/kornia — урок 2.0).
"""
import numpy as np


def build_train_augment(scale=(0.9, 1.1), translate: float = 0.05, rotate: float = 8.0,
                        brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.1,
                        hue: float = 0.02, elastic_alpha: float = 1.0, elastic_sigma: float = 50.0,
                        noise_std=(0.01, 0.05), p_affine: float = 0.9, p_color: float = 0.7,
                        p_elastic: float = 0.3, p_noise: float = 0.3, seed=None):
    """Собрать albumentations.Compose тренировочных аугментаций (БЕЗ зеркала). seed (если задан) делает
    применение детерминированным. Вероятности p_* позволяют изолировать группы (геометрия/цвет/упругость)."""
    import albumentations as A
    return A.Compose([
        A.Affine(scale=scale, translate_percent=(0.0, translate), rotate=(-rotate, rotate), p=p_affine),
        A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=p_color),
        A.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, p=p_elastic),
        A.GaussNoise(std_range=noise_std, p=p_noise),
    ], seed=seed)


def augment_image(image, seed: int, **params) -> np.ndarray:
    """Детерминированно аугментировать ОДИН RGB-кадр (HWC uint8): один seed → один результат
    (Compose пересобирается с этим seed). Возвращает RGB uint8 того же размера. **params → build_train_augment."""
    aug = build_train_augment(seed=seed, **params)
    return aug(image=np.asarray(image))["image"]
