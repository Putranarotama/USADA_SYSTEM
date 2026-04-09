# model.py
# ============================================================
#  Arsitektur Hybrid:
#    EfficientNetV2-B0  (CNN — local features)
#  + Swin-Tiny          (Transformer — global features)
#  + Cross-Attention Fusion
#  + Progressive Fine-Tuning support
#
#  Referensi:
#  - Aboelenin et al. 2025, Complex & Intelligent Systems
#  - Liu et al. 2021, ICCV (Swin Transformer)
#  - Tan & Le 2021, ICML (EfficientNetV2)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CrossAttentionFusion(nn.Module):
    """
    Swin (Query) aktif memilih fitur relevan dari EfficientNet (Key-Value).
    Lebih kuat dari simple concat karena ada seleksi fitur dinamis.
    """

    def __init__(self, dim_cnn: int, dim_swin: int,
                 hidden: int = 256, heads: int = 4):
        super().__init__()
        self.proj_cnn  = nn.Linear(dim_cnn,  hidden)
        self.proj_swin = nn.Linear(dim_swin, hidden)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=heads,
            dropout=0.1, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, cnn_feat: torch.Tensor,
                swin_feat: torch.Tensor) -> torch.Tensor:
        q = self.proj_swin(swin_feat).unsqueeze(1)  # (B,1,H)
        k = self.proj_cnn(cnn_feat).unsqueeze(1)    # (B,1,H)
        attn_out, _ = self.cross_attn(q, k, k)
        x = self.norm1(q + attn_out).squeeze(1)     # (B,H)
        x = self.norm2(x + self.ffn(x))
        return x


class UsadaHybridModel(nn.Module):

    def __init__(self, num_classes: int, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── EfficientNetV2-B0 ──────────────────────────────────────────────
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=pretrained,
            num_classes=0, global_pool="")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat  = self.effnet(dummy)
        self.cnn_dim = feat.shape[1]
        self.gap     = nn.AdaptiveAvgPool2d(1)

        # Store last conv for Grad-CAM
        self._gradcam_feat = None
        self._gradcam_grad = None
        self._last_conv = self._find_last_conv(self.effnet)
        if self._last_conv:
            self._last_conv.register_forward_hook(self._fwd_hook)
            self._last_conv.register_full_backward_hook(self._bwd_hook)

        # ── Swin-Tiny ──────────────────────────────────────────────────────
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained,
            num_classes=0, global_pool="avg")
        with torch.no_grad():
            dummy2 = torch.zeros(1, 3, 224, 224)
            feat2  = self.swin(dummy2)
        self.swin_dim = feat2.shape[1]

        # ── Cross-Attention Fusion ─────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            dim_cnn=self.cnn_dim, dim_swin=self.swin_dim,
            hidden=256, heads=4)

        # ── Classifier Head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Mulai dengan backbone frozen
        self._freeze_all_backbone()

    # ── Grad-CAM hooks ─────────────────────────────────────────────────────
    def _find_last_conv(self, module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def _fwd_hook(self, m, inp, out):
        self._gradcam_feat = out

    def _bwd_hook(self, m, grad_in, grad_out):
        self._gradcam_grad = grad_out[0]

    # ── Progressive Fine-Tuning ────────────────────────────────────────────
    def _freeze_all_backbone(self):
        for p in self.effnet.parameters(): p.requires_grad = False
        for p in self.swin.parameters():   p.requires_grad = False

    def unfreeze_partial(self):
        """Unfreeze 50% layer terakhir kedua backbone."""
        eff_ch = list(self.effnet.children())
        for ch in eff_ch[len(eff_ch)//2:]:
            for p in ch.parameters(): p.requires_grad = True
        for name, p in self.swin.named_parameters():
            if any(s in name for s in ["layers.2","layers.3","norm"]):
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN branch
        cnn_map = self.effnet(x)
        if cnn_map.dim() == 4:
            cnn_f = self.gap(cnn_map).flatten(1)
        else:
            cnn_f = cnn_map

        # Transformer branch
        swin_f = self.swin(x)

        # Fusion + classify
        fused = self.fusion(cnn_f, swin_f)
        return self.head(fused)


def get_model(num_classes, dropout_rate=0.3,
              pretrained=True, device="cpu") -> UsadaHybridModel:
    model = UsadaHybridModel(num_classes, dropout_rate, pretrained)
    model = model.to(device)
    total   = sum(p.numel() for p in model.parameters())
    trainable = model.get_trainable_params()
    print(f"[MODEL] EfficientNetV2-B0 + Swin-Tiny + CrossAttentionFusion")
    print(f"        Total params    : {total:,}")
    print(f"        Trainable (P1)  : {trainable:,}")
    return model
