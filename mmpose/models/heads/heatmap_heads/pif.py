"""Pose Information Fusion modes used by the MambaPose paper experiments."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


PIF_MODES = frozenset({'full', 'disabled', 'no_prior', 'no_cycling'})


_FULL_SCAN = {
    14: (
        [0, 14, 13, 14, 0, 1, 3, 5, 3, 1, 0, 7, 9, 11, 9, 7, 0,
         8, 10, 12, 10, 8, 0, 2, 4, 6, 4, 2, 0],
        [28, 9, 23, 8, 24, 7, 25, 11, 17, 12, 18, 13, 19, 2, 3]),
    17: (
        [0, 7, 5, 3, 1, 2, 4, 6, 0, 6, 8, 10, 8, 6, 0, 12, 14,
         16, 14, 12, 0, 13, 15, 17, 15, 13, 0, 7, 9, 11, 9, 7, 0],
        [32, 4, 5, 3, 6, 2, 13, 27, 12, 28, 11, 29, 19, 21, 18, 22,
         17, 23]),
}


_NO_CYCLING_SCAN = {
    # Exact CrowdPose ablation left commented by the authors.
    14: (
        [0, 14, 13, 1, 3, 5, 7, 9, 11, 8, 10, 12, 2, 4, 6],
        [0, 3, 12, 4, 13, 5, 14, 6, 9, 7, 10, 8, 11, 2, 1]),
    # The paper evaluates no-cycling only on 14-keypoint CrowdPose. This
    # 17-keypoint path is the structural extension of the same rule: visit
    # each anatomical branch once and reconstruct every token exactly once.
    17: (
        [0, 7, 5, 3, 1, 2, 4, 6, 6, 8, 10, 12, 14, 16, 13, 15, 17,
         7, 9, 11],
        [0, 4, 5, 3, 6, 2, 7, 1, 9, 18, 10, 19, 11, 14, 12, 15, 13,
         16]),
}


def build_scan_indices(
        num_keypoints: int, mode: str) -> tuple[Tensor, Tensor]:
    """Return scan indices and positions that reconstruct mean plus joints."""
    if mode not in PIF_MODES:
        raise ValueError(f'unknown pif mode: {mode}')
    if num_keypoints not in (14, 17):
        raise ValueError(
            'MambaPose defines pose-prior scans only for 14 or 17 keypoints')
    if mode in {'disabled', 'no_prior'}:
        indices = list(range(num_keypoints + 1))
        return torch.tensor(indices), torch.tensor(indices)
    source = _FULL_SCAN if mode == 'full' else _NO_CYCLING_SCAN
    scan, restore = source[num_keypoints]
    return torch.tensor(scan), torch.tensor(restore)


class PoseInteraction(nn.Module):
    """Paper PIF plus the exact ablation modes described by the authors."""

    def __init__(
            self,
            dim: int,
            num_keypoints: int,
            mode: str = 'full',
            block_factory: Callable[..., nn.Module] | None = None):
        super().__init__()
        if mode not in PIF_MODES:
            raise ValueError(f'unknown pif mode: {mode}')
        if num_keypoints not in (14, 17):
            raise ValueError(
                'MambaPose supports PIF only for 14 or 17 keypoints')
        if block_factory is None:
            # Delayed to avoid a module cycle: tokenbase owns the author block
            # implementation and imports this class for its head.
            from .tokenbase import create_block
            block_factory = create_block

        self.dim = dim
        self.num_keypoints = num_keypoints
        self.mode = mode
        # Preserve the author's construction order and attribute names so the
        # full-mode regression can compare operation-for-operation.
        self.MambaBlock = block_factory(d_model=dim)
        self.MambaBlock2 = block_factory(d_model=dim)
        self.within_dropout = nn.Dropout(0.1)
        self.within_norm = nn.LayerNorm(dim)
        self.within_dropout_2 = nn.Dropout(0.1)
        self.within_norm_2 = nn.LayerNorm(dim)
        self.Mamba_selfScanBlock = block_factory(d_model=dim)
        self.param = nn.Parameter(torch.tensor(0.1))

        scan, restore = build_scan_indices(num_keypoints, mode)
        self.register_buffer('scan_indices', scan, persistent=False)
        self.register_buffer('restore_indices', restore, persistent=False)

    def _global_fusion(self, tokens: Tensor) -> Tensor:
        topk_num = 5
        similarity = torch.matmul(tokens, tokens.transpose(1, 2))
        topk_indices = torch.topk(
            similarity, k=topk_num, dim=-1).indices
        batch_indices = torch.arange(
            tokens.size(0), device=tokens.device).view(-1, 1, 1)
        expanded = tokens[batch_indices, topk_indices]
        expanded = expanded.view(-1, topk_num, tokens.size(2))
        result, _ = self.Mamba_selfScanBlock(
            torch.flip(expanded, [1]), None, inference_params=None)
        result = result.view(tokens.size(0), -1, tokens.size(2))
        # This global flip is intentional: it is the exact active author
        # implementation used for all reported primary configurations.
        expanded_new = torch.flip(result, [1])[:, 0::topk_num, :]
        tokens = tokens + self.param * self.within_dropout_2(expanded_new)
        return self.within_norm_2(tokens)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (
                self.num_keypoints, self.dim):
            raise ValueError(
                'PIF expects [batch, num_keypoints, dim], got '
                f'{tuple(tokens.shape)}')
        # Paper Table V: without PIF, Transformer keypoint tokens are mapped
        # directly to heatmaps.
        if self.mode == 'disabled':
            return tokens

        tokens = self._global_fusion(tokens)
        mean_token = tokens.mean(dim=1, keepdim=True)
        tokens = torch.cat((mean_token, tokens), dim=1)

        hidden = tokens[:, self.scan_indices, :]
        normal, _ = self.MambaBlock(hidden, None, inference_params=None)
        reverse, _ = self.MambaBlock2(
            torch.flip(hidden, [1]), None, inference_params=None)
        hidden = normal + torch.flip(reverse, [1])
        hidden = hidden[:, self.restore_indices, :]
        tokens = self.within_norm(tokens + self.within_dropout(hidden))
        return tokens[:, 1:, :]
