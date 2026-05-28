# Ultralytics 🚀 AGPL-3.0 License
"""Wavelet-based modules for Wavelet-YOLOv12.

Hypothesis: small medical objects (e.g. acid-fast bacilli) carry strong
high-frequency signal that vanilla stride-2 conv attenuates. Replacing the
downsampling step with a 2D Haar discrete wavelet transform (DWT) yields a
lossless multi-band decomposition that preserves edge / texture information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_kernels():
    # 2x2 Haar basis, normalised so output preserves energy.
    k = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],   # LL  (approximation)
            [[1.0, 1.0], [-1.0, -1.0]], # LH  (horizontal edges)
            [[1.0, -1.0], [1.0, -1.0]], # HL  (vertical edges)
            [[1.0, -1.0], [-1.0, 1.0]], # HH  (diagonal edges)
        ]
    ) / 2.0
    return k  # (4, 2, 2)


class HaarDWT(nn.Module):
    """2D Haar DWT applied per-channel via depthwise stride-2 conv.

    Input  : (B, C, H, W)
    Output : (B, 4, C, H/2, W/2)  ordered as [LL, LH, HL, HH]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        kernels = _haar_kernels()                       # (4, 2, 2)
        weight = kernels.repeat(in_channels, 1, 1)      # (4*C, 2, 2)
        weight = weight.unsqueeze(1)                    # (4*C, 1, 2, 2)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # groups=C → each input channel produces 4 outputs (LL, LH, HL, HH).
        y = F.conv2d(x, self.weight, stride=2, groups=c)  # (B, 4C, H/2, W/2)
        # Re-order from interleaved (c0_LL c0_LH ...) to (subband, channel).
        y = y.view(b, c, 4, h // 2, w // 2).permute(0, 2, 1, 3, 4).contiguous()
        return y  # (B, 4, C, H/2, W/2)


class WaveDown(nn.Module):
    """Wavelet downsampling block — drop-in replacement for stride-2 Conv.

    Splits the feature into a low-frequency stream (LL) and a high-frequency
    stream (concatenation of LH, HL, HH), projects both, and fuses them.

    Args:
        c1: input channels.
        c2: output channels.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dwt = HaarDWT(c1)
        self.proj_ll = nn.Conv2d(c1, c2, 1, bias=False)
        self.proj_hf = nn.Conv2d(3 * c1, c2, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub = self.dwt(x)                                  # (B, 4, C, H/2, W/2)
        ll = sub[:, 0]                                     # (B, C, H/2, W/2)
        hf = sub[:, 1:].flatten(1, 2)                      # (B, 3C, H/2, W/2)
        return self.fuse(torch.cat([self.proj_ll(ll), self.proj_hf(hf)], dim=1))


__all__ = ("HaarDWT", "WaveDown")
