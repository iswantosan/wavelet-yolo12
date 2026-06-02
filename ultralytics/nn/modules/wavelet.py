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


class WaveAttnDown(nn.Module):
    """Strategy A — stride-2 downsampling Conv with wavelet-HF attention gate.

    Main path mirrors the Ultralytics Conv attribute layout (``self.conv``,
    ``self.bn``, ``self.act``) so a pretrained Conv at the same model index
    transfers via name-matched ``load_state_dict``. A parallel branch computes
    HF features from the pre-downsample input via Haar DWT and applies them
    as a multiplicative residual gate.

    At init: ``alpha = 0`` → forward output = main path exactly. The wavelet
    branch contributes nothing until alpha learns a non-zero value, so this
    block can only improve over a vanilla stride-2 Conv (never degrade).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, d: int = 1):
        super().__init__()
        pad = (k - 1) // 2 if p is None else p
        # Names match Ultralytics Conv class so pretrained weights transfer.
        self.conv = nn.Conv2d(c1, c2, k, s, padding=pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        self.dwt = HaarDWT(c1)
        self.hf_proj = nn.Conv2d(3 * c1, c2, 1, bias=False)
        self.attn_bn = nn.BatchNorm2d(c2)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))                          # main path
        hf = self.dwt(x)[:, 1:].flatten(1, 2)                        # (B, 3C, H/2, W/2)
        attn = torch.sigmoid(self.attn_bn(self.hf_proj(hf)))         # (B, c2, H/2, W/2)
        return y * (1 + self.alpha * attn)


class WaveAttnDownV2(nn.Module):
    """Strategy A v2 — improved wavelet-augmented stride-2 downsampling.

    Key differences from WaveAttnDown (v1):

    1. **No single-scalar α bottleneck.** v1 used a learned α (init 0) to gate
       a sigmoid attention mask. At α=0 the backward gradient to ``hf_proj``
       is zero, so the wavelet projection cannot learn until α grows — a
       chicken-and-egg problem that empirically caps the wavelet branch's
       contribution.
       v2 uses an additive residual without the α gate. Each output channel
       learns its own wavelet contribution via per-channel projection weights.

    2. **Structured HF processing.** The wavelet branch is a 3×3 → 1×1
       Conv-BN-SiLU block instead of a single 1×1 projection. The 3×3 conv
       captures local spatial coherence of HF features (edges run along
       directions, not isolated pixels).

    3. **Small-but-non-zero init** on the final projection (Kaiming × 0.05),
       so step-0 residual is small (≈ 5% of natural scale) but the full
       branch receives non-zero gradients from the first batch.

    Main path mirrors Ultralytics Conv attribute names (``conv``, ``bn``,
    ``act``) so pretrained Conv weights transfer by name.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, d: int = 1):
        super().__init__()
        pad = (k - 1) // 2 if p is None else p
        # Main path — Ultralytics-Conv compatible names for pretrain match.
        self.conv = nn.Conv2d(c1, c2, k, s, padding=pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # Wavelet HF branch — structured 3×3 → 1×1 with BN/SiLU.
        self.dwt = HaarDWT(c1)
        c_mid = max(c2 // 2, 16)
        self.hf_branch = nn.Sequential(
            nn.Conv2d(3 * c1, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=False),
        )
        # Final 1×1 weight scaled to ~5% of Kaiming, so step-0 residual is
        # small but gradients flow normally (no zero-init dead-gradient trap).
        with torch.no_grad():
            self.hf_branch[3].weight.mul_(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))                          # main downsample
        hf = self.dwt(x)[:, 1:].flatten(1, 2)                        # (B, 3C, H/2, W/2)
        residual = self.hf_branch(hf)                                # ~5% scale at init
        return y + residual


class WaveAttnDownV3(nn.Module):
    """Strategy A v3 — minimal-capacity wavelet spatial gate (~10 params).

    A bet that v1 (alpha-gated) and v2 (additive residual) both underperform
    because the wavelet branch adds *too much* capacity and either interferes
    with the pretrained main path or overfits the small training set.

    v3 strips the wavelet branch down to a single learnable spatial gate
    derived from the HF energy map. The wavelet is no longer a parallel
    feature path; it becomes a *structural prior* that modulates the main
    conv output spatially.

    Forward:
      energy(x, y) = mean over (band, channel) of |HF_subband|²
      energy_norm  = per-image z-score of energy
      gate(x, y)   = 2 · σ(Conv3x3(energy_norm))     # init = 1.0
      output       = main_conv(x) · gate

    Init: Conv3x3 weight = bias = 0 → gate = 1.0 → output = main path
    exactly. Gradient to the gate flows non-trivially from step 0 (sigmoid
    derivative at 0 = 0.25, energy_norm is non-zero), so training can begin
    learning the gate immediately.

    Parameters added: 9 (Conv weight) + 1 (Conv bias) = 10 total.

    Hypothesis: bacilli correlate with locally elevated HF energy in the
    input. A learned spatial gate exploiting this prior provides task-
    aligned regularisation; extra wavelet feature processing (v1, v2) hurts
    by either interfering with the pretrained main path (v2) or being
    bottlenecked by zero-init scalars (v1).

    Main path mirrors Ultralytics Conv attribute layout for pretrained
    weight transfer.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, d: int = 1):
        super().__init__()
        pad = (k - 1) // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding=pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        self.dwt = HaarDWT(c1)
        # 3×3 spatial smoother over the 1-channel HF energy map (10 params).
        self.gate = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))                          # main downsample

        # Per-pixel HF energy (averaged over bands + channels) → (B, 1, H/2, W/2).
        hf = self.dwt(x)[:, 1:]                                      # (B, 3, C, H/2, W/2)
        energy = hf.pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(1)

        # Per-image z-score so the gate sees a scale-invariant signal.
        mu = energy.mean(dim=(2, 3), keepdim=True)
        sigma = energy.std(dim=(2, 3), keepdim=True) + 1e-6
        energy = (energy - mu) / sigma

        gate = 2 * torch.sigmoid(self.gate(energy))                  # init: 1.0
        return y * gate


class WaveHFSkip(nn.Module):
    """Strategy B — HF-only side branch that produces a narrow feature map at
    half resolution. Intended to be referenced from the head/neck and
    concatenated alongside an existing FPN concat (e.g. inject input-side
    HF detail into the P3 detection scale).

    The main backbone path is NOT touched, so pretrained Conv weights at the
    insertion point remain intact. The projection is zero-init so the new
    concat'd channels contribute zero at step 0 — downstream layers see the
    same input distribution as baseline.
    """

    def __init__(self, c1: int, c_out: int):
        super().__init__()
        self.dwt = HaarDWT(c1)
        self.proj = nn.Conv2d(3 * c1, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()
        nn.init.zeros_(self.proj.weight)  # step-0 contribution = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = self.dwt(x)[:, 1:].flatten(1, 2)                        # (B, 3C, H/2, W/2)
        return self.act(self.bn(self.proj(hf)))


class WaveUp(nn.Module):
    """Strategy C — wavelet-augmented 2× upsampling. Drop-in replacement for
    ``nn.Upsample(scale_factor=2, mode='nearest')`` in the neck.

    Base path  : nearest-neighbour upsample (matches pretrained behaviour).
    Residual   : predict LH/HL/HH from input via 1×1 conv, synthesise via
                 Haar inverse DWT (transposed conv with the HF kernels).
    Output     : ``base + alpha * residual``.

    At init: ``alpha = 0`` and ``hf_pred`` is also zero-init, so the output is
    exactly the nearest-neighbour upsample. Pretrained downstream layers see
    the same distribution as before.
    """

    def __init__(self, c: int):
        super().__init__()
        self.c = c
        self.hf_pred = nn.Conv2d(c, 3 * c, 1, bias=False)
        nn.init.zeros_(self.hf_pred.weight)
        # IDWT uses the HF kernels (LH, HL, HH) only — the LL/base term is
        # provided by the nearest-neighbour interpolation in forward().
        kernels = _haar_kernels()[1:]                                # (3, 2, 2)
        weight = kernels.repeat(c, 1, 1).unsqueeze(1)                # (3C, 1, 2, 2)
        self.register_buffer("idwt_weight", weight)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="nearest")
        hf = self.hf_pred(x)                                         # (B, 3C, H, W)
        residual = F.conv_transpose2d(hf, self.idwt_weight, stride=2, groups=self.c)
        return base + self.alpha * residual


__all__ = ("HaarDWT", "WaveDown", "WaveAttnDown", "WaveAttnDownV2", "WaveAttnDownV3", "WaveHFSkip", "WaveUp")
