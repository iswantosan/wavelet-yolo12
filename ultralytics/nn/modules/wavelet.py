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
        # Warm init: at step 0 the block reduces to SiLU(BN(AvgPool(X))).
        # - proj_ll = identity (zero-padded if c1 != c2), so Y_LL = S_LL.
        # - proj_hf = 0, so Y_HF = 0.
        # - fuse's LL columns = identity, HF columns = small Kaiming, so the
        #   step-0 output is exactly Y_LL while gradients still flow back to
        #   proj_hf during training (allowing the high-frequency stream to grow).
        self._identity_init_1x1(self.proj_ll)
        nn.init.zeros_(self.proj_hf.weight)
        self._warm_init_fuse_conv(self.fuse[0], lf_channels=c2, hf_scale=1e-2)

    @staticmethod
    def _identity_init_1x1(conv: nn.Conv2d) -> None:
        """Identity-like init for a 1×1 conv: top-left min(c_in, c_out) block is the
        identity matrix, rest is zero. For c_in == c_out this is exact identity."""
        with torch.no_grad():
            conv.weight.zero_()
            n = min(conv.in_channels, conv.out_channels)
            for i in range(n):
                conv.weight[i, i, 0, 0] = 1.0

    @staticmethod
    def _warm_init_fuse_conv(conv: nn.Conv2d, lf_channels: int, hf_scale: float) -> None:
        """Warm-init for the fusion 1×1 conv. The input layout is
        concat(Y_LL, Y_HF) with Y_LL occupying the first ``lf_channels`` channels.
        The LL block is initialised to identity so Y_LL passes through unchanged
        at step 0, while the HF block receives a small Kaiming init so that
        ∂Y/∂Y_HF is non-zero and gradients can still flow back to proj_hf."""
        with torch.no_grad():
            conv.weight.zero_()
            c_out = conv.out_channels
            # LL columns (first lf_channels of the input) → identity to c_out.
            n = min(lf_channels, c_out)
            for i in range(n):
                conv.weight[i, i, 0, 0] = 1.0
            # HF columns (remaining input channels) → small Kaiming for grad flow.
            hf_block = conv.weight[:, lf_channels:, :, :]
            if hf_block.numel() > 0:
                nn.init.kaiming_uniform_(hf_block, a=5 ** 0.5)
                hf_block.mul_(hf_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub = self.dwt(x)                                  # (B, 4, C, H/2, W/2)
        ll = sub[:, 0]                                     # (B, C, H/2, W/2)
        hf = sub[:, 1:].flatten(1, 2)                      # (B, 3C, H/2, W/2)
        return self.fuse(torch.cat([self.proj_ll(ll), self.proj_hf(hf)], dim=1))


__all__ = ("HaarDWT", "WaveDown")
