"""Proxy-эмбеддер для A/B-гейта Rosa + метрики recall.

recall_metrics — ЧИСТАЯ (numpy), тестируется. build_proxy/embed (timm/torch) — модельные,
лениво импортируются и пишутся на модельном этапе (за маркером model).
"""
import numpy as np


def recall_metrics(sim, gallery_ids, probe_ids, ks=(1, 5)) -> dict:
    """Closed-set Recall@k по особям: probe «попал», если в top-k галереи есть его individual_id.

    sim — матрица сходства (n_probe × n_gallery), большее = ближе.
    """
    gallery_ids = np.asarray(gallery_ids)
    probe_ids = np.asarray(probe_ids)
    out = {"n_probe": int(len(probe_ids))}
    if len(probe_ids) == 0:
        return {**out, **{f"recall@{k}": 0.0 for k in ks}}
    order = np.argsort(-np.asarray(sim), axis=1, kind="stable")   # индексы галереи по убыванию (stable при ничьих)
    ranked = gallery_ids[order]                    # (n_probe × n_gallery) id-шники по рангу
    for k in ks:
        hit = (ranked[:, :k] == probe_ids[:, None]).any(axis=1)
        out[f"recall@{k}"] = float(hit.mean())
    return out


def build_proxy(model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384", device: str = "mps"):
    """Зеро-шот эмбеддер (timm) → {model, transform, device}. Модельная граница (lazy import)."""
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"model": model, "transform": timm.data.create_transform(**cfg), "device": device}


def embed(proxy, images, batch_size: int = 16):
    """Список RGB-кадров (HWC uint8) → L2-нормированные эмбеддинги (N, D) numpy."""
    import torch
    from PIL import Image
    if not images:
        return np.zeros((0, 1), np.float32)
    model, transform, device = proxy["model"], proxy["transform"], proxy["device"]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.stack([transform(Image.fromarray(im)) for im in images[i:i + batch_size]]).to(device)
            chunks.append(model(x).float().cpu().numpy())
    e = np.concatenate(chunks, 0)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
