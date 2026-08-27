"""Deterministic, bounded activation range observers."""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

import torch
from torch import Tensor, nn


class ActivationRangeObserver(nn.Module):
    """Collect max/range statistics without changing the observed tensor.

    Percentiles use a fixed 256-bin log2 histogram spanning ``[2^-32, 2^32]``.
    Storage is constant and mergeable; a reported quantile is the upper edge of
    its bin, so the multiplicative over-estimation is bounded by one bin width.
    Zeros are counted separately and exactly.
    """

    _BINS = 256
    _MIN_EXP = -32.0
    _MAX_EXP = 32.0

    def __init__(
            self,
            granularity: Literal['tensor', 'channel', 'token'] = 'tensor'):
        super().__init__()
        if granularity not in {'tensor', 'channel', 'token'}:
            raise ValueError(f'unsupported observer granularity: {granularity!r}')
        self.granularity = granularity
        self.register_buffer('max_abs', torch.empty(0, dtype=torch.float32))
        self.register_buffer(
            'histogram', torch.zeros(self._BINS, dtype=torch.int64))
        self.register_buffer('sample_count', torch.zeros((), dtype=torch.int64))
        self.register_buffer('zero_count', torch.zeros((), dtype=torch.int64))
        self.register_buffer('minimum', torch.full((), float('inf')))
        self.register_buffer('maximum', torch.full((), -float('inf')))
        self._observed_shape: tuple[int, ...] | None = None
        self._token_ids: tuple[str, ...] | None = None

    @property
    def token_ids(self) -> tuple[str, ...] | None:
        return self._token_ids

    def get_extra_state(self) -> dict[str, object]:
        return {
            'schema_version': 1,
            'granularity': self.granularity,
            'observed_shape': self._observed_shape,
            'token_ids': self._token_ids,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict) or state.get('schema_version') != 1:
            raise ValueError('invalid observer serialization state')
        if state.get('granularity') != self.granularity:
            raise ValueError('observer granularity changed during resume')
        shape = state.get('observed_shape')
        token_ids = state.get('token_ids')
        self._observed_shape = tuple(shape) if shape is not None else None
        self._token_ids = tuple(token_ids) if token_ids is not None else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'max_abs'
        if key in state_dict:
            self.max_abs = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _shape_for(self, value: Tensor) -> tuple[int, ...]:
        if value.ndim == 0:
            if self.granularity != 'tensor':
                raise ValueError(
                    f'{self.granularity} observer requires a non-scalar tensor')
            return ()
        if self.granularity == 'tensor':
            return ()
        if self.granularity == 'channel':
            channel_axis = 1 if value.ndim >= 4 else value.ndim - 1
            return (value.shape[channel_axis],)
        if value.ndim < 2:
            raise ValueError('token observer requires token and channel dimensions')
        return tuple(value.shape[:-1])

    def _current_max(self, absolute: Tensor) -> Tensor:
        if self.granularity == 'tensor':
            return absolute.max().to(torch.float32)
        if self.granularity == 'channel':
            channel_axis = 1 if absolute.ndim >= 4 else absolute.ndim - 1
            axes = tuple(axis for axis in range(absolute.ndim)
                         if axis != channel_axis)
            return absolute.amax(dim=axes).to(torch.float32)
        return absolute.amax(dim=-1).to(torch.float32)

    def _validate_identity(
            self, shape: tuple[int, ...], token_ids: Sequence[str] | None) -> None:
        normalized_ids = tuple(token_ids) if token_ids is not None else None
        if self.granularity != 'token' and normalized_ids is not None:
            raise ValueError('token identity is only valid for token observers')
        if normalized_ids is not None:
            if not normalized_ids or not all(
                    isinstance(item, str) and item for item in normalized_ids):
                raise ValueError('token identity must contain non-empty strings')
            if len(normalized_ids) != shape[-1]:
                raise ValueError('token identity count disagrees with token shape')
        if self._observed_shape is not None and shape != self._observed_shape:
            label = 'token shape' if self.granularity == 'token' else 'observer shape'
            raise ValueError(f'{label} drift: {self._observed_shape} -> {shape}')
        if self._token_ids is not None and normalized_ids != self._token_ids:
            raise ValueError('token identity drift')
        if self._observed_shape is not None and (
                self._token_ids is None) != (normalized_ids is None):
            raise ValueError('token identity drift')

    def forward(
            self, value: Tensor,
            *, token_ids: Sequence[str] | None = None) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError('observer input must be a tensor')
        if value.numel() == 0:
            raise ValueError('observer input must not be empty')
        if not torch.is_floating_point(value):
            raise TypeError('observer input must be floating point')
        if not torch.isfinite(value).all():
            raise ValueError('observer input must contain only finite values')
        shape = self._shape_for(value)
        self._validate_identity(shape, token_ids)

        absolute = value.detach().abs().to(torch.float32)
        current = self._current_max(absolute)
        nonzero = absolute[absolute != 0]
        histogram = torch.zeros_like(self.histogram)
        if nonzero.numel():
            width = (self._MAX_EXP - self._MIN_EXP) / self._BINS
            indices = torch.floor(
                (torch.log2(nonzero.clamp(2 ** self._MIN_EXP,
                                           2 ** self._MAX_EXP))
                 - self._MIN_EXP) / width).to(torch.int64)
            indices.clamp_(0, self._BINS - 1)
            histogram = torch.bincount(
                indices, minlength=self._BINS).to(self.histogram.device)

        if self._observed_shape is None:
            self._observed_shape = shape
            self._token_ids = tuple(token_ids) if token_ids is not None else None
            self.max_abs = current.to(self.histogram.device)
        else:
            self.max_abs = torch.maximum(
                self.max_abs, current.to(self.max_abs.device))
        self.histogram.add_(histogram)
        self.sample_count.add_(value.numel())
        self.zero_count.add_((absolute == 0).sum().to(self.zero_count.device))
        self.minimum.copy_(torch.minimum(
            self.minimum, value.detach().amin().to(self.minimum)))
        self.maximum.copy_(torch.maximum(
            self.maximum, value.detach().amax().to(self.maximum)))
        return value

    def merge(self, other: 'ActivationRangeObserver') -> 'ActivationRangeObserver':
        if not isinstance(other, ActivationRangeObserver):
            raise TypeError('can only merge another ActivationRangeObserver')
        if other.granularity != self.granularity:
            raise ValueError('cannot merge observers with different granularity')
        if other._observed_shape is None:
            return self
        self._validate_identity(other._observed_shape, other._token_ids)
        if self._observed_shape is None:
            self._observed_shape = other._observed_shape
            self._token_ids = other._token_ids
            self.max_abs = other.max_abs.detach().clone().to(self.histogram.device)
        else:
            self.max_abs = torch.maximum(
                self.max_abs, other.max_abs.to(self.max_abs.device))
        self.histogram.add_(other.histogram.to(self.histogram.device))
        self.sample_count.add_(other.sample_count.to(self.sample_count.device))
        self.zero_count.add_(other.zero_count.to(self.zero_count.device))
        self.minimum.copy_(torch.minimum(
            self.minimum, other.minimum.to(self.minimum)))
        self.maximum.copy_(torch.maximum(
            self.maximum, other.maximum.to(self.maximum)))
        return self

    def _percentile(self, percentile: float) -> float:
        if not 0 < percentile <= 1:
            raise ValueError('percentiles must be in (0, 1]')
        total = int(self.sample_count.item())
        if total == 0:
            raise ValueError('observer has no samples')
        rank = max(1, math.ceil(percentile * total))
        zeros = int(self.zero_count.item())
        if rank <= zeros:
            return 0.0
        cumulative = torch.cumsum(self.histogram.cpu(), dim=0)
        index = int(torch.searchsorted(
            cumulative, torch.tensor(rank - zeros), right=False).item())
        index = min(index, self._BINS - 1)
        width = (self._MAX_EXP - self._MIN_EXP) / self._BINS
        return float(2 ** (self._MIN_EXP + (index + 1) * width))

    def summary(self, *, percentiles: Iterable[float] = (0.5, 0.9, 0.99,
                                                          0.999)) -> dict:
        width = (self._MAX_EXP - self._MIN_EXP) / self._BINS
        return {
            'granularity': self.granularity,
            'sample_count': int(self.sample_count.item()),
            'zero_count': int(self.zero_count.item()),
            'max_abs': self.max_abs.detach().cpu().tolist(),
            'range': [float(self.minimum), float(self.maximum)],
            'percentiles': {
                str(value): self._percentile(float(value))
                for value in percentiles
            },
            'algorithm': 'fixed-log2-histogram-v1',
            'histogram_bins': self._BINS,
            'histogram_domain': [2 ** self._MIN_EXP, 2 ** self._MAX_EXP],
            'relative_error_bound': 2 ** width - 1,
            'outlier_ratio_above_p99_bin': self._outlier_ratio(0.99),
            'token_ids': list(self._token_ids) if self._token_ids else None,
            'observed_shape': list(self._observed_shape)
            if self._observed_shape is not None else None,
        }

    def _outlier_ratio(self, percentile: float) -> float:
        total = int(self.sample_count.item())
        if total == 0:
            raise ValueError('observer has no samples')
        rank = max(1, math.ceil(percentile * total))
        zeros = int(self.zero_count.item())
        if rank <= zeros:
            return float((total - zeros) / total)
        cumulative = torch.cumsum(self.histogram.cpu(), dim=0)
        index = int(torch.searchsorted(
            cumulative, torch.tensor(rank - zeros), right=False).item())
        above = int(self.histogram[index + 1:].sum().item())
        return above / total
