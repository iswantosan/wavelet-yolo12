# Ultralytics 🚀 AGPL-3.0 License
"""Alterable Kernel Convolution (AKConv).

Reference: Ai et al., "AKConv: Convolutional Kernel with Arbitrary Sampled
Shapes and Arbitrary Number of Parameters", 2023.

AKConv generalises the standard convolution kernel along two axes:
  1. The number of sampling points N is arbitrary (not constrained to k²).
  2. Each sampling point has a learned 2D offset per output position, so the
     effective "kernel" can take an arbitrary shape (asymmetric, elongated,
     curved) and adapts to the input.

For rod-shaped targets (e.g. acid-fast bacilli), this gives the kernel the
freedom to align with object orientation rather than being constrained to
the axis-aligned k×k grid of a regular Conv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AKConv(nn.Module):
    """Alterable Kernel Convolution layer.

    Args:
        c1: input channels
        c2: output channels
        num_param: number of sampling points N (paper default 9). Larger N
                   gives more spatial flexibility at increased param cost.
        stride: spatial stride (1 = same-res, 2 = downsample-by-2).
    """

    def __init__(self, c1: int, c2: int, num_param: int = 9, stride: int = 1):
        super().__init__()
        if num_param < 1:
            raise ValueError(f"num_param must be >= 1, got {num_param}")
        self.c1 = c1
        self.c2 = c2
        self.num_param = num_param
        self.stride = stride

        # Main feature conv after sampling. Output of bilinear sampling is
        # arranged as (B, C, H_out * N, W_out); a (N, 1) stride-(N, 1) conv
        # then linearly combines the N samples per output position.
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=(num_param, 1), stride=(num_param, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # Offset prediction. Output channels = 2*N (x and y offset per sample).
        self.p_conv = nn.Conv2d(c1, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.zeros_(self.p_conv.weight)
        if self.p_conv.bias is not None:
            nn.init.zeros_(self.p_conv.bias)

    @staticmethod
    def _init_sampling_pattern(num_param: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Base offset pattern for the N sampling points.

        Fills a k×k grid (k = floor(sqrt(N))), then places any remaining
        points in an additional row beyond the grid. The pattern matches the
        original AKConv reference implementation and degenerates to a regular
        3×3 grid for N = 9.

        Returns:
            Tensor of shape (1, 2*N, 1, 1) with x-offsets stacked first, then
            y-offsets.
        """
        N = num_param
        k = int(N**0.5)
        # k×k regular grid centred at origin.
        coords = torch.arange(0, k, dtype=dtype, device=device) - (k - 1) / 2.0
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        xs = gx.flatten()
        ys = gy.flatten()

        # Remainder: extra row at y = k - (k-1)/2 with x linearly spaced.
        rem = N - k * k
        if rem > 0:
            extra_x = torch.arange(0, rem, dtype=dtype, device=device) - (rem - 1) / 2.0
            extra_y = torch.full((rem,), float(k) - (k - 1) / 2.0, dtype=dtype, device=device)
            xs = torch.cat([xs, extra_x])
            ys = torch.cat([ys, extra_y])

        return torch.cat([xs, ys]).view(1, 2 * N, 1, 1)

    @staticmethod
    def _base_grid(h: int, w: int, num_param: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Per-output-position centre coordinates, repeated for N samples.

        Returns:
            Tensor of shape (1, 2*N, h, w) with x-grid stacked first.
        """
        N = num_param
        ys = torch.arange(0, h, dtype=dtype, device=device)
        xs = torch.arange(0, w, dtype=dtype, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)
        gx = gx.view(1, 1, h, w).expand(1, N, h, w)
        gy = gy.view(1, 1, h, w).expand(1, N, h, w)
        return torch.cat([gx, gy], dim=1)

    def _bilinear_sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample x at p_0 + p_n + offset using bilinear interpolation.

        Args:
            x: (B, C, H_in, W_in)
            offset: (B, 2*N, H_out, W_out) — learned per-position 2D offsets

        Returns:
            (B, C, H_out, W_out, N) — features sampled at each of the N
            adaptive positions.
        """
        B, C, H_in, W_in = x.shape
        N = self.num_param
        H_out, W_out = offset.shape[2], offset.shape[3]

        # Sampling position = stride·(grid center) + base offset pattern + learned offset.
        p_n = self._init_sampling_pattern(N, offset.dtype, offset.device)  # (1, 2N, 1, 1)
        p_0 = self._base_grid(H_out, W_out, N, offset.dtype, offset.device)  # (1, 2N, H_out, W_out)
        if self.stride > 1:
            p_0 = p_0 * float(self.stride)
        p = p_0 + p_n + offset  # (B, 2N, H_out, W_out)

        p_x = p[:, :N]  # (B, N, H_out, W_out)
        p_y = p[:, N:]

        # Normalise to grid_sample's [-1, 1] convention. align_corners=True
        # is required so that integer pixel positions map exactly to grid
        # indices; otherwise base_grid positions would be off by half a pixel.
        denom_x = max(W_in - 1, 1)
        denom_y = max(H_in - 1, 1)
        nx = 2.0 * p_x / float(denom_x) - 1.0
        ny = 2.0 * p_y / float(denom_y) - 1.0

        # Stack the N grids along the batch dimension for a single grid_sample
        # call: (B*N, C, H_in, W_in) input and (B*N, H_out, W_out, 2) grid.
        # The repeat broadcasts each batch element N times.
        grid = torch.stack([nx, ny], dim=-1)  # (B, N, H_out, W_out, 2)
        grid = grid.reshape(B * N, H_out, W_out, 2)
        x_rep = x.unsqueeze(1).expand(B, N, C, H_in, W_in).reshape(B * N, C, H_in, W_in)

        sampled = F.grid_sample(
            x_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B*N, C, H_out, W_out)

        return sampled.view(B, N, C, H_out, W_out).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.p_conv(x)                                        # (B, 2N, H_out, W_out)
        x_offset = self._bilinear_sample(x, offset)                    # (B, C, H_out, W_out, N)
        B, C, H_out, W_out, N = x_offset.shape
        # Reshape so the N sample dim runs along height for the (N, 1) conv.
        x_offset = x_offset.permute(0, 1, 2, 4, 3).reshape(B, C, H_out * N, W_out)
        return self.act(self.bn(self.conv(x_offset)))


__all__ = ("AKConv",)
