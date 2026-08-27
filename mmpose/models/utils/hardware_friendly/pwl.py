"""Deterministic bounded piecewise-linear nonlinear approximations."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn


_FUNCTIONS = frozenset({'silu', 'gelu', 'softplus', 'exp'})


class PiecewiseLinearApproximation(nn.Module):
    """Continuous PWL function with clamp saturation and QAT gradients."""

    def __init__(
            self,
            breakpoints: Sequence[float] | Tensor | None,
            slopes: Sequence[float] | Tensor | None,
            intercepts: Sequence[float] | Tensor | None,
            *, function_name: str | None = None,
            max_error: float = 0.0,
            mean_error: float = 0.0,
            enabled: bool = True):
        super().__init__()
        self.enabled = bool(enabled)
        self.function_name = function_name
        self.saturation = 'identity' if not enabled else 'clamp'
        self.max_error = float(max_error)
        self.mean_error = float(mean_error)
        if not enabled:
            if any(value is not None for value in (breakpoints, slopes,
                                                    intercepts)):
                raise ValueError('disabled PWL must not carry approximation state')
            return
        points = torch.as_tensor(breakpoints, dtype=torch.float64)
        segment_slopes = torch.as_tensor(slopes, dtype=torch.float64)
        segment_intercepts = torch.as_tensor(intercepts, dtype=torch.float64)
        if points.ndim != 1 or points.numel() < 2:
            raise ValueError('PWL requires at least two finite breakpoints')
        if not torch.isfinite(points).all():
            raise ValueError('PWL breakpoints must be finite')
        if not torch.all(points[1:] > points[:-1]):
            raise ValueError('PWL breakpoints must be strictly increasing')
        segments = points.numel() - 1
        if segment_slopes.shape != (segments,) or segment_intercepts.shape != (
                segments,):
            raise ValueError('PWL slope/intercept count must equal segment count')
        if not torch.isfinite(segment_slopes).all() or not torch.isfinite(
                segment_intercepts).all():
            raise ValueError('PWL coefficients must be finite')
        if segments > 1:
            joints = points[1:-1]
            left = segment_slopes[:-1] * joints + segment_intercepts[:-1]
            right = segment_slopes[1:] * joints + segment_intercepts[1:]
            if not torch.allclose(left, right, rtol=1e-10, atol=1e-12):
                raise ValueError('PWL segments must be continuous')
        if not math.isfinite(self.max_error) or not math.isfinite(
                self.mean_error) or self.max_error < self.mean_error \
                or self.mean_error < 0:
            raise ValueError('PWL error statistics must be finite and ordered')
        self.register_buffer('breakpoints', points)
        self.register_buffer('slopes', segment_slopes)
        self.register_buffer('intercepts', segment_intercepts)

    @classmethod
    def identity(cls) -> 'PiecewiseLinearApproximation':
        return cls(None, None, None, enabled=False)

    @property
    def domain(self) -> tuple[float, float] | None:
        if not self.enabled:
            return None
        return float(self.breakpoints[0]), float(self.breakpoints[-1])

    @property
    def segments(self) -> int:
        return 0 if not self.enabled else self.slopes.numel()

    def forward(self, value: Tensor) -> Tensor:
        if not self.enabled:
            return value
        points = self.breakpoints.to(device=value.device, dtype=value.dtype)
        slopes = self.slopes.to(device=value.device, dtype=value.dtype)
        intercepts = self.intercepts.to(device=value.device, dtype=value.dtype)
        bounded = value.clamp(points[0], points[-1])
        indices = torch.bucketize(bounded, points[1:-1], right=True)
        return slopes[indices] * bounded + intercepts[indices]


def fit_pwl(
        reference_fn: Callable[[Tensor], Tensor],
        domain: tuple[float, float],
        segments: int,
        grid_points: int,
        *, function_name: str) -> PiecewiseLinearApproximation:
    """Fit deterministic endpoint interpolation and measure it on a grid."""
    if function_name not in _FUNCTIONS:
        raise ValueError(
            f'function_name must select exactly one of {sorted(_FUNCTIONS)}')
    if (not isinstance(domain, tuple) or len(domain) != 2
            or not all(isinstance(item, (int, float)) and math.isfinite(item)
                       for item in domain) or domain[0] >= domain[1]):
        raise ValueError('PWL domain must be a finite increasing pair')
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError('PWL segments must be a positive integer')
    if (isinstance(grid_points, bool) or not isinstance(grid_points, int)
            or grid_points < segments + 1):
        raise ValueError('grid_points must cover every segment boundary')

    points = torch.linspace(
        float(domain[0]), float(domain[1]), segments + 1,
        dtype=torch.float64)
    with torch.no_grad():
        values = reference_fn(points)
    if values.shape != points.shape or not torch.isfinite(values).all():
        raise ValueError('reference function must return finite pointwise values')
    slopes = (values[1:] - values[:-1]) / (points[1:] - points[:-1])
    intercepts = values[:-1] - slopes * points[:-1]
    candidate = PiecewiseLinearApproximation(
        points, slopes, intercepts, function_name=function_name)

    grid = torch.linspace(
        float(domain[0]), float(domain[1]), grid_points,
        dtype=torch.float64)
    with torch.no_grad():
        reference = reference_fn(grid)
        approximate = candidate(grid)
        error = (approximate - reference).abs()
    if not torch.isfinite(error).all():
        raise ValueError('PWL error grid contains non-finite values')
    candidate.max_error = float(error.max())
    candidate.mean_error = float(error.mean())
    return candidate
