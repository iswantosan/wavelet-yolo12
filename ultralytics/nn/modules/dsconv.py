"""Dynamic Snake Convolution (DSConv) for tubular/elongated object detection.

Reference: Qi et al., "Dynamic Snake Convolution based on Topological Geometric
Constraints for Tubular Structure Segmentation" (ICCV 2023).

DSConv generalises standard convolution by learning per-pixel offsets that are
constrained to form a continuous snake-like curve. For elongated/tubular
structures (rod-shaped AFB), this lets each output activation aggregate
features along the rod axis instead of a fixed square grid.

Two morphs:
    morph=0 : snake along x-axis (horizontal sweep, shifts in y)
    morph=1 : snake along y-axis (vertical sweep, shifts in x)

Each pixel predicts K offsets (bounded via tanh * extend_scope) that are
cumulatively summed → smooth curve. The K sample points are interpolated via
F.grid_sample and reduced by a Conv2d of kernel (K, 1) or (1, K).

This is a clean pure-PyTorch port — slower than the official CUDA build but
correct and dependency-free.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("DSConv", "DSConvBlock")


class DSConv(nn.Module):
    """Single-axis Dynamic Snake Convolution.

    Args:
        c1: input channels.
        c2: output channels.
        k:  snake length (kernel size along the snake axis). Default 9.
        morph: 0 = x-axis snake, 1 = y-axis snake.
        extend_scope: bound on per-step offset magnitude (default 1.0).
    """

    def __init__(self, c1: int, c2: int, k: int = 9, morph: int = 0, extend_scope: float = 1.0):
        super().__init__()
        assert k >= 3 and k % 2 == 1, f"DSConv k must be odd >=3, got {k}"
        assert morph in (0, 1), f"DSConv morph must be 0 or 1, got {morph}"
        self.c1, self.c2 = c1, c2
        self.k = k
        self.morph = morph
        self.extend_scope = extend_scope

        # Predict 2K offsets per pixel: (dx_i, dy_i) for i in 0..K-1.
        # Use one shared 3x3 to keep params low; bn + tanh stabilises training.
        self.offset_conv = nn.Conv2d(c1, 2 * k, 3, padding=1)
        self.offset_bn = nn.BatchNorm2d(2 * k)

        # Conv along the snake dimension after sampling.
        if morph == 0:
            self.dsc_conv = nn.Conv2d(c1, c2, kernel_size=(k, 1), stride=(k, 1))
        else:
            self.dsc_conv = nn.Conv2d(c1, c2, kernel_size=(1, k), stride=(1, k))
        self.gn = nn.GroupNorm(min(32, c2), c2) if c2 >= 4 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    # --------------------------------------------------------------------- #
    def _build_snake_grid(self, offset: torch.Tensor) -> torch.Tensor:
        """Build (B, H*k or k*H, W or W*k, 2) grid in normalized coords for grid_sample."""
        B, _, H, W = offset.shape
        K = self.k
        device = offset.device
        dtype = offset.dtype
        half = (K - 1) // 2

        # Split offsets: shape (B, K, H, W) each
        d = offset.view(B, 2, K, H, W) * self.extend_scope
        dx = torch.tanh(d[:, 0])  # (B, K, H, W)
        dy = torch.tanh(d[:, 1])  # (B, K, H, W)

        # Base coordinates (H, W) → flat grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )  # (H, W)
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, K, H, W)  # (B, K, H, W)
        xs = xs.unsqueeze(0).unsqueeze(0).expand(B, K, H, W)

        # Step offsets along the snake axis (k = -half ... +half)
        step = torch.arange(-half, half + 1, device=device, dtype=dtype).view(1, K, 1, 1)

        if self.morph == 0:
            # x-axis sweep: step along x (constant), accumulate dy along k
            x_coords = xs + step  # (B, K, H, W)
            # Cumulative dy gives a smooth curve; recenter so shift==0 at k=half
            cumdy = torch.cumsum(dy, dim=1)
            y_shift = cumdy - cumdy[:, half:half + 1]
            y_coords = ys + y_shift
        else:
            # y-axis sweep
            y_coords = ys + step
            cumdx = torch.cumsum(dx, dim=1)
            x_shift = cumdx - cumdx[:, half:half + 1]
            x_coords = xs + x_shift

        # Normalize to [-1, 1] for grid_sample
        x_norm = 2.0 * x_coords / max(W - 1, 1) - 1.0
        y_norm = 2.0 * y_coords / max(H - 1, 1) - 1.0

        # Stack into grid shape (B, K, H, W, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)
        if self.morph == 0:
            # Output H' = K * H, W' = W → conv (K, 1) stride (K, 1) reduces H'.
            # Want order: for each (h, w), K consecutive samples interleaved in H.
            grid = grid.permute(0, 2, 1, 3, 4).contiguous().view(B, K * H, W, 2)
        else:
            # Output H' = H, W' = K * W → conv (1, K) stride (1, K) reduces W'.
            grid = grid.permute(0, 2, 3, 1, 4).contiguous().view(B, H, K * W, 2)
        return grid

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_bn(self.offset_conv(x))
        grid = self._build_snake_grid(offset)
        # Sample features along snake curve.
        # grid: (B, H*K, W, 2) for morph=0, or (B, H, K*W, 2) for morph=1
        sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B, C, H*K, W) or (B, C, H, K*W)
        out = self.dsc_conv(sampled)
        return self.act(self.gn(out))


class DSConvBlock(nn.Module):
    """Dual-morph DSConv block: x-snake and y-snake in parallel, concat + fuse.

    Captures elongated patterns along both axes, which matches AFB rods that
    may appear in any orientation in microscopy images.

    Args:
        c1: input channels.
        c2: output channels.
        k:  snake length (default 9).
        extend_scope: offset magnitude bound (default 1.0).
    """

    def __init__(self, c1: int, c2: int, k: int = 9, extend_scope: float = 1.0):
        super().__init__()
        half = c2 // 2
        self.dsc_x = DSConv(c1, half, k=k, morph=0, extend_scope=extend_scope)
        self.dsc_y = DSConv(c1, c2 - half, k=k, morph=1, extend_scope=extend_scope)
        self.fuse = nn.Conv2d(c2, c2, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.dsc_x(x)
        fy = self.dsc_y(x)
        out = self.fuse(torch.cat([fx, fy], dim=1))
        return self.act(self.bn(out))
