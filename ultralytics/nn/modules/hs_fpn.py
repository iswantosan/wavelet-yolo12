"""HS-FPN building blocks (CA + SFF) for small-object detection.

Reference: Xu et al., "MFDS-DETR: Multi-frequency dual-domain stable representation
for medical detection" (2024) — proposes HS-FPN with Channel Attention (CA) and
Selective Feature Fusion (SFF) modules for small-object detection in medical
microscopy.

CA  : per-scale channel-wise gating (CBAM-style avg+max pool through MLP).
SFF : selective fusion of high-res (spatial) and low-res (semantic) features
      via learned gating on the upsampled low-res branch.

These are exposed as independent modules so a YAML can compose an HS-FPN-style
neck by stacking CA on each pyramid level and using SFF in place of the standard
top-down add/concat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CA", "SFF")


class CA(nn.Module):
    """Channel Attention module (avg+max pool through shared MLP).

    Multiplies the input feature map by a per-channel sigmoid gate. Useful to
    suppress irrelevant channels (e.g. background staining artefacts).

    Args:
        c1: input channels.
        c2: output channels (must equal c1).
        reduction: bottleneck ratio in the MLP (default 8).
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 8):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        assert c1 == c2, f"CA requires c1 == c2, got c1={c1}, c2={c2}"
        hidden = max(c1 // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * gate


class SFF(nn.Module):
    """Selective Feature Fusion: gated top-down fusion of low-res into high-res.

    Takes a list of two feature maps `[high_res, low_res]` where low_res has
    half the spatial size of high_res. low_res is upsampled (nearest), gated
    by a 1x1-sigmoid attention computed on the concatenation, then added to
    high_res. Channels of low_res are aligned to high_res via a 1x1 conv if
    needed.

    Args:
        c1: list/tuple of input channels [high_res_ch, low_res_ch].
        c2: output channels (defaults to high_res_ch).
    """

    def __init__(self, c1, c2: int | None = None):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            assert len(c1) == 2, f"SFF expects 2 input channels, got {c1}"
            ch_high, ch_low = int(c1[0]), int(c1[1])
        else:
            ch_high = ch_low = int(c1)
        c2 = c2 if c2 is not None else ch_high

        self.align = nn.Conv2d(ch_low, c2, 1) if ch_low != c2 else nn.Identity()
        self.reduce_high = nn.Conv2d(ch_high, c2, 1) if ch_high != c2 else nn.Identity()
        self.gate = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        high, low = feats
        low_aligned = self.align(low)
        low_up = F.interpolate(low_aligned, size=high.shape[-2:], mode="nearest")
        high_r = self.reduce_high(high)
        gate = self.gate(torch.cat([high_r, low_up], dim=1))
        return high_r + low_up * gate
