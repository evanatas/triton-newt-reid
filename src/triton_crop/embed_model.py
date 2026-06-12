"""Модельная граница эмбеддера re-ID (Блок 4): ArcFaceHead + timm-backbone + единый embed-контракт.

МОДЕЛЬНЫЙ модуль: torch — на уровне модуля (нужен для nn.Module); timm (скачивание весов
MegaDescriptor/MiewID) — lazy внутри build_embedder. Чистые модули (embed_dataset/eval/ab) и cli
его на уровне модуля НЕ импортируют → не тянут torch зря.

ArcFace — свой, ~30 строк (Deng 2019; гиперпараметры Čermák WACV-2024: m=0.5, s=64). Не тащим
хрупкий wildlife-tools локально (урок 2.0: конфликт numpy2/pandas3). embed_images повторяет КОНТРАКТ
proxy_embed.embed: RGB uint8 → timm-transform → model → L2-norm float32 — один путь train/eval.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace, Deng 2019). forward(emb, labels) → логиты для CrossEntropy;
    forward(emb) → масштабированный косинус (инференс). Эмбеддинги и веса L2-нормируются → косинус.
    К истинному классу добавляется угловой зазор margin, всё масштабируется на scale (радиус гиперсферы).

    Канонический guard: phi = cos(θ+m), пока θ < π−m; за порогом (трудные пары, θ>π−m) — линейное
    продолжение penalty cos−m·sin(m). Без него cos(θ+m) немонотонен за π−m (margin перестаёт
    ужесточать границу, градиент истинного логита меняет знак → дестабилизация). Считаем через формулу
    сложения (cos·cos_m − sin·sin_m), БЕЗ acos — у acos градиент 1/sin(θ) рвётся у θ→0,π.
    """

    def __init__(self, in_features: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)             # порог по cos: θ = π−m
        self._mm = math.sin(math.pi - self.margin) * self.margin   # penalty за порогом (= sin(m)·m)

    def forward(self, embeddings, labels=None):
        cos = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1)).clamp(-1.0, 1.0)
        if labels is None:
            return cos * self.scale                            # инференс: scaled cos, без зазора
        sin = torch.sqrt((1.0 - cos * cos).clamp_min(0.0))
        phi = cos * self._cos_m - sin * self._sin_m            # cos(θ+m) через формулу сложения
        phi = torch.where(cos > self._th, phi, cos - self._mm)  # hard-margin guard за θ>π−m (монотонность)
        target = F.one_hot(labels, num_classes=cos.size(1)).bool()
        return torch.where(target, phi, cos) * self.scale


class EmbedNet(nn.Module):
    """Дообучаемая модель re-ID: timm-backbone (num_classes=0) + ArcFaceHead.

    train: forward(x, labels) → логиты для CrossEntropy. В голову идёт СЫРОЙ выход backbone (БЕЗ no_grad,
    БЕЗ внешней L2-norm — ArcFaceHead нормирует сам). eval: embed(x) → L2-norm эмбеддинг в no_grad (тот же
    контракт, что embed_images/proxy_embed.embed). embed_dim берётся из backbone.num_features (не хардкод).
    """

    def __init__(self, backbone, num_classes: int, embed_dim=None, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.backbone = backbone
        d = embed_dim if embed_dim is not None else getattr(backbone, "num_features", None)
        if d is None:
            raise ValueError("EmbedNet: укажи embed_dim (у backbone нет атрибута num_features)")
        self.embed_dim = int(d)
        self.head = ArcFaceHead(self.embed_dim, num_classes, margin, scale)

    def forward(self, x, labels=None):
        return self.head(self.backbone(x), labels)            # backbone с градиентом; голова нормирует сама

    @torch.no_grad()
    def embed(self, x):
        return F.normalize(self.backbone(x).float(), dim=1)   # eval-путь (gallery/probe), тот же контракт, что embed_images


def freeze_backbone_stages(backbone, n_stages: int) -> None:
    """Заморозить первые n_stages стадий Swin-backbone (patch_embed + layers[:n]). 0 → ничего.

    ВНИМАНИЕ (разведка Этапа B): у Swin-L-384 layers[2]≈66 % параметров, layers[3]≈31 %; заморозка
    1–2 стадий фризит лишь ~2.4 % — для реальной регуляризации малой выборки (~112 особей) морозить
    стоит n_stages=3 (включая layers[2], ~68 %). n_stages — число СТАДИЙ (0..4), НЕ блоков внутри стадии.
    """
    if n_stages <= 0:
        return
    frozen = 0
    if hasattr(backbone, "patch_embed"):
        for p in backbone.patch_embed.parameters():
            p.requires_grad_(False)
            frozen += 1
    layers = getattr(backbone, "layers", None)
    if layers is not None:
        for i in range(min(n_stages, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad_(False)
                frozen += 1
    if frozen == 0:                                          # хард-фейл вместо тихого no-op
        raise ValueError(
            "freeze_backbone_stages: n_stages>0, но не заморожено ни одного параметра — у backbone нет "
            "patch_embed/layers (timm-Swin). Для MiewID (transformers AutoModel) заморозка стадий не "
            "реализована: весь backbone остался бы обучаемым (риск оверфита на ~112 особях). Либо "
            "base_model=Mega, либо freeze_backbone_stages=0 осознанно.")


def build_param_groups(net, lr_backbone: float = 1e-5, lr_head: float = 1e-3, weight_decay: float = 1e-4):
    """Две группы оптимизатора: backbone (только requires_grad, малый LR + WD) и голова ArcFace
    (LR ~10×, без WD — учится с нуля). Фильтр requires_grad обязателен, иначе замороженные веса попадут
    в оптимизатор. -> list[dict] для torch.optim."""
    head_ids = {id(p) for p in net.head.parameters()}
    backbone = [p for p in net.backbone.parameters() if p.requires_grad and id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": list(net.head.parameters()), "lr": lr_head, "weight_decay": 0.0},
    ]


def _is_miewid(name) -> bool:
    """Имя относится к MiewID (грузится transformers AutoModel, НЕ timm)?"""
    return "miewid" in str(name).lower()


# Пин коммита conservationxlabs/miewid-msv3: защита trust_remote_code от подмены апстрима + воспроизводимость.
MIEWID_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"


def miewid_transform(image_size: int = 440):
    """Препроцессинг MiewID: Resize(image_size²) + ToTensor + ImageNet-нормировка (вход 440, НЕ 384)."""
    import torchvision.transforms as T
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_embedder(name: str = "hf-hub:BVRA/MegaDescriptor-L-384", device: str = "mps",
                   num_classes: int = 0, image_size: int = 440, revision=None):
    """Эмбеддер-backbone + его «родной» transform (один на probe и gallery). Lazy (скачивание весов). {model, transform, device}.

    Два пути: (1) timm-backbone без головы (num_classes=0) — MegaDescriptor (Swin-L, 1536-d), родной
    resolve_data_config transform; (2) MiewID ('miewid' в имени) — transformers AutoModel(trust_remote_code,
    revision), выход 2152-d ненормирован (L2-norm делает embed_images — единый путь), вход 440² + ImageNet-norm.
    revision по умолчанию запинен MIEWID_REVISION (воспроизводимость + безопасность trust_remote_code);
    переопределяется параметром. gallery и probe эмбеддятся ОДНОЙ сборкой. Для finetune backbone —
    feature-extractor под ArcFaceHead.
    """
    if _is_miewid(name):
        from transformers import AutoModel
        repo = name.split("miewid:", 1)[-1] if str(name).startswith("miewid:") else name
        model = AutoModel.from_pretrained(repo, trust_remote_code=True,
                                          revision=(revision or MIEWID_REVISION)).eval().to(device)
        return {"model": model, "transform": miewid_transform(image_size), "device": device}
    import timm
    model = timm.create_model(name, pretrained=True, num_classes=num_classes).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"model": model, "transform": timm.data.create_transform(**cfg), "device": device}


def embed_images(embedder, images, batch_size: int = 16) -> np.ndarray:
    """RGB-кадры (HWC uint8) → L2-нормированные эмбеддинги (N, D) float32. Контракт proxy_embed.embed
    (probe и gallery обрабатываются одним путём).

    На eval — БЕЗ аугментаций, тот же transform, что у gallery. Модель в no_grad/eval не переключаем
    насильно (вызывающий сам ставит .eval()) — но градиенты отключены.
    """
    from PIL import Image
    if not images:
        return np.zeros((0, 1), np.float32)
    model, transform, device = embedder["model"], embedder["transform"], embedder["device"]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.stack([transform(Image.fromarray(im)) for im in images[i:i + batch_size]]).to(device)
            chunks.append(model(x).float().cpu().numpy())
    e = np.concatenate(chunks, 0)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
