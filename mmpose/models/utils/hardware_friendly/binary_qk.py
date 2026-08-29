"""Head-local deterministic Binary Q/K helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def ste_sign(value: Tensor) -> Tensor:
    """Return {-1,+1} with zero defined as +1 and identity STE gradient."""
    signed = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
    return value + (signed - value).detach()


def binary_qk_logits(query: Tensor, key: Tensor) -> Tensor:
    """Compute an unscaled signed Q/K dot product; softmax remains external."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError('binary Q/K expects [batch, head, token, channel] tensors')
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError('binary Q/K query and key dimensions disagree')
    return torch.einsum('bhid,bhjd->bhij', ste_sign(query), ste_sign(key))


def binary_scaled_qk_logits(query: Tensor, key: Tensor) -> Tensor:
    """Compute BinaryAttention-style scaled signed Q/K logits.

    The activation scales are runtime per-batch, per-head means over both the
    token and channel dimensions.  The attention module applies its original
    head-dimension scale separately.
    """
    signed_dot = binary_qk_logits(query, key)
    query_scale = query.abs().mean(dim=-2, keepdim=True).mean(
        dim=-1, keepdim=True)
    key_scale = key.abs().mean(dim=-2, keepdim=True).mean(
        dim=-1, keepdim=True)
    return signed_dot * query_scale * key_scale


@dataclass(frozen=True)
class BinaryQKOperationReport:
    layers: int
    heads: int
    query_tokens: int
    key_tokens: int
    head_dim: int
    theoretical_changed_multiplies: int
    measured_latency: bool = False


def binary_qk_operation_report(
        *, query_tokens: int, key_tokens: int, head_dim: int, heads: int,
        layers: int = 6) -> BinaryQKOperationReport:
    values = (query_tokens, key_tokens, head_dim, heads, layers)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError('operation dimensions must be positive integers')
    if layers != 6:
        raise ValueError('Binary Q/K is admitted only for the six head layers')
    return BinaryQKOperationReport(
        layers=layers,
        heads=heads,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        head_dim=head_dim,
        theoretical_changed_multiplies=(
            layers * heads * query_tokens * key_tokens * head_dim),
    )
