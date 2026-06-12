"""Тесты embed_model (Блок 4): ArcFaceHead (свой, чистый torch) + embed_images (единый embed-контракт).

Гейты: ArcFace корректен — forward-формы, margin>0 ужесточает истинный класс, loss падает на
сепарабельной синтетике; embed_images отдаёт L2-norm float32 (один путь eval).
torch/torchvision импортируются ВНУТРИ функций — общая коллекция сьюта не тянет тяжёлые либы.
build_embedder (timm-скачивание MegaDescriptor) не тестируем (как build_proxy) — это Этап B (Colab).
"""
import numpy as np


def test_arcface_forward_shapes():
    import torch
    from triton_crop.embed_model import ArcFaceHead
    torch.manual_seed(42)
    head = ArcFaceHead(16, 4, margin=0.5, scale=64.0)
    emb = torch.randn(8, 16)
    labels = torch.randint(0, 4, (8,))
    assert head(emb, labels).shape == (8, 4)            # train: логиты с угловым зазором
    assert head(emb).shape == (8, 4)                    # eval (labels=None): масштабированный косинус
    assert torch.allclose(head(emb), head(emb))         # forward детерминирован


def test_arcface_margin_lowers_true_logit():
    import torch
    from triton_crop.embed_model import ArcFaceHead
    torch.manual_seed(1)
    B, D, C = 16, 12, 5
    emb, labels = torch.randn(B, D), torch.randint(0, C, (B,))
    head_m = ArcFaceHead(D, C, margin=0.5, scale=30.0)
    head_0 = ArcFaceHead(D, C, margin=0.0, scale=30.0)
    head_0.load_state_dict(head_m.state_dict())         # те же веса, отличается только margin
    lm, l0 = head_m(emb, labels), head_0(emb, labels)
    tm = lm[torch.arange(B), labels]
    t0 = l0[torch.arange(B), labels]
    assert torch.all(tm <= t0 + 1e-4)                   # margin НЕ повышает истинный логит
    assert torch.any(tm < t0 - 1e-3)                    # и реально понижает — эффект есть
    mask = torch.ones_like(lm, dtype=torch.bool)
    mask[torch.arange(B), labels] = False
    assert torch.allclose(lm[mask], l0[mask], atol=1e-5)  # нецелевые логиты не тронуты


def test_arcface_margin_monotone_past_pi_minus_m():
    # Канонический ArcFace (Deng 2019): логит истинного класса phi(θ) МОНОТОННО убывает по θ на всём
    # [0, π] и НИКОГДА не превышает cos(θ). Наивный cos(θ+m) ломается за θ>π−m (косинус немонотонен):
    # margin начинает ОСЛАБЛЯТЬ границу и градиент меняет знак. Гейт ловит именно закритическую зону.
    import math

    import torch
    from triton_crop.embed_model import ArcFaceHead
    head = ArcFaceHead(2, 2, margin=0.5, scale=1.0)         # scale=1 → логит == phi
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    thetas = torch.linspace(0.0, math.pi - 1e-3, 60)        # угол эмбеддинга к своему центроиду w0
    embs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
    labels = torch.zeros(60, dtype=torch.long)
    phi = head(embs, labels)[torch.arange(60), 0]           # логит истинного класса (scale=1)
    cos_true = torch.cos(thetas)
    assert torch.all(phi <= cos_true + 1e-5)                # margin только ужесточает (penalty), даже за π−m
    assert torch.all(phi[1:] <= phi[:-1] + 1e-5)            # phi монотонно убывает на ВСЁМ [0,π]


def test_arcface_loss_decreases_on_separable():
    import torch
    import torch.nn.functional as F
    from triton_crop.embed_model import ArcFaceHead
    torch.manual_seed(0)
    C, D, per = 4, 8, 20
    centers = F.normalize(torch.randn(C, D), dim=1)
    emb = torch.cat([centers[c] + 0.05 * torch.randn(per, D) for c in range(C)])
    labels = torch.tensor([c for c in range(C) for _ in range(per)])
    head = ArcFaceHead(D, C, margin=0.5, scale=16.0)
    opt = torch.optim.SGD(head.parameters(), lr=0.5)
    losses = []
    for _ in range(80):
        loss = F.cross_entropy(head(emb, labels), labels)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5                 # сепарабельные кластеры → loss падает


def _tiny_backbone():
    """Крошечный backbone (num_features=16) для CPU-тестов finetune без timm/MegaDescriptor."""
    import torch
    class TinyBackbone(torch.nn.Module):
        num_features = 16
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8, 16)
        def forward(self, x):
            return self.fc(x.flatten(1))
    return TinyBackbone()


def test_embednet_train_and_embed_paths():
    import torch
    from triton_crop.embed_model import EmbedNet
    net = EmbedNet(_tiny_backbone(), num_classes=4, margin=0.5, scale=32.0)
    x = torch.randn(6, 3, 8, 8)
    labels = torch.randint(0, 4, (6,))
    logits = net(x, labels)                                   # train: логиты с градиентом (НЕ no_grad)
    assert logits.shape == (6, 4) and logits.requires_grad
    e = net.embed(x)                                          # eval: L2-norm эмбеддинг, no_grad (один путь eval)
    assert e.shape == (6, 16) and not e.requires_grad
    assert torch.allclose(e.norm(dim=1), torch.ones(6), atol=1e-5)


def test_freeze_backbone_stages_swinlike():
    import torch
    from triton_crop.embed_model import freeze_backbone_stages
    m = torch.nn.Module()
    m.patch_embed = torch.nn.Linear(4, 4)
    m.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    freeze_backbone_stages(m, 2)                              # заморозить patch_embed + layers[0,1]
    assert all(not p.requires_grad for p in m.patch_embed.parameters())
    assert all(not p.requires_grad for p in m.layers[0].parameters())
    assert all(not p.requires_grad for p in m.layers[1].parameters())
    assert all(p.requires_grad for p in m.layers[2].parameters())   # 3-я стадия учится
    freeze_backbone_stages(m, 0)                             # 0 — ничего не морозит (идемпотентно)


def test_freeze_backbone_stages_raises_when_nothing_frozen():
    # backbone без patch_embed/layers (как MiewID-AutoModel): n_stages>0 раньше тихо ничего не морозил —
    # весь backbone остался бы обучаемым. Теперь — явный ValueError. n_stages=0 по-прежнему ранний return.
    import pytest
    import torch
    from triton_crop.embed_model import freeze_backbone_stages
    m = torch.nn.Linear(4, 4)                                # нет ни patch_embed, ни layers
    with pytest.raises(ValueError):
        freeze_backbone_stages(m, 2)
    freeze_backbone_stages(m, 0)                             # осознанный 0 не падает (ранний return)


def test_build_param_groups_splits_head_backbone():
    from triton_crop.embed_model import EmbedNet, build_param_groups
    net = EmbedNet(_tiny_backbone(), num_classes=3)
    groups = build_param_groups(net, lr_backbone=1e-5, lr_head=1e-3, weight_decay=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-3
    assert groups[1]["weight_decay"] == 0.0                   # ArcFace.weight без WD
    head_ids = {id(p) for p in net.head.parameters()}
    assert all(id(p) in head_ids for p in groups[1]["params"])           # голова → 2-я группа
    assert all(id(p) not in head_ids for p in groups[0]["params"])       # backbone → 1-я


def test_miewid_detection_and_transform():
    # MiewID — 2-й кандидат (Rosa: MiewID 73/93 > Mega 63/83). Грузится иначе (AutoModel, вход 440,
    # выход 2152-d) — отдельная ветка build_embedder; embed_images/L2-norm универсальны. Реальную
    # загрузку (сеть/веса) локально НЕ тестируем (как MegaDescriptor) — только детектор и transform.
    from PIL import Image
    from triton_crop.embed_model import MIEWID_REVISION, _is_miewid, miewid_transform
    assert _is_miewid("conservationxlabs/miewid-msv3") and _is_miewid("miewid:foo")
    assert not _is_miewid("hf-hub:BVRA/MegaDescriptor-L-384")
    x = miewid_transform(440)(Image.fromarray(np.zeros((100, 120, 3), np.uint8)))
    assert tuple(x.shape) == (3, 440, 440)             # Resize(440)+ToTensor+ImageNet-norm
    assert len(MIEWID_REVISION) == 40 and all(c in "0123456789abcdef" for c in MIEWID_REVISION)  # пин commit hash


def test_embed_images_l2norm_contract():
    import torch
    import torchvision.transforms as T
    from triton_crop.embed_model import embed_images
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 5)).eval()
    transform = T.Compose([T.Resize((8, 8)), T.ToTensor()])
    embedder = {"model": model, "transform": transform, "device": "cpu"}
    imgs = [np.random.default_rng(i).integers(0, 255, (10, 12, 3), np.uint8) for i in range(7)]
    e = embed_images(embedder, imgs, batch_size=4)
    assert e.shape == (7, 5) and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)   # контракт: L2-нормировка
    assert embed_images(embedder, []).shape[0] == 0                 # пустой вход не падает
