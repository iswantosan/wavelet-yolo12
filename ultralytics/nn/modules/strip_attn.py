"""Strip attention modules for elongated object detection.

Reference: Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
Semantic Segmentation" (NeurIPS 2022) — Multi-Scale Convolutional Attention (MSCA).

For elongated objects (rod-shaped AFB), anisotropic depthwise kernels (1, k) and
(k, 1) capture features along horizontal/vertical axes more effectively than
isotropic square kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("StripAttention", "StripAttnBlock")


class StripAttention(nn.Module):
    """Multi-scale strip attention (SegNeXt-style MSCA).

    Applies a depthwise 5x5 conv followed by parallel multi-scale strip
    branches: kernel sizes (1, k) and (k, 1) for k in `scales`. Output is
    used as an attention gate multiplied with the input.

    Args:
        c1: input channels (must equal c2 — attention preserves channels).
        c2: output channels.
        scales: tuple of strip lengths (default (7, 11, 21)).
    """

    def __init__(self, c1: int, c2: int | None = None, scales=(7, 11, 21)):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        assert c1 == c2, f"StripAttention requires c1 == c2, got c1={c1}, c2={c2}"
        self.dim = c1

        self.conv0 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        self.strips = nn.ModuleList()
        for k in scales:
            self.strips.append(
                nn.ModuleDict({
                    "h": nn.Conv2d(c1, c1, (1, k), padding=(0, k // 2), groups=c1),
                    "v": nn.Conv2d(c1, c1, (k, 1), padding=(k // 2, 0), groups=c1),
                })
            )
        self.proj = nn.Conv2d(c1, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.conv0(x)
        for s in self.strips:
            attn = attn + s["h"](attn) + s["v"](attn)
        attn = self.proj(attn)
        return u * attn


class StripAttnBlock(nn.Module):
    """Strip attention block: 1x1 reduce -> StripAttention -> 1x1 expand + residual.

    Drop-in feature refinement block. Useful in the neck before detection heads.

    Args:
        c1: input channels.
        c2: output channels (typically same as c1 for residual).
        scales: tuple of strip lengths.
    """

    def __init__(self, c1: int, c2: int | None = None, scales=(7, 11, 21)):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.same_ch = c1 == c2
        self.reduce = nn.Conv2d(c1, c2, 1) if not self.same_ch else nn.Identity()
        self.norm1 = nn.BatchNorm2d(c2)
        self.attn = StripAttention(c2, c2, scales=scales)
        self.norm2 = nn.BatchNorm2d(c2)
        self.proj = nn.Conv2d(c2, c2, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.reduce(x)
        out = self.norm1(identity)
        out = self.attn(out)
        out = self.act(self.norm2(out))
        out = self.proj(out)
        return out + identity
