"""CUDA-event latency measurement with injectable CPU-only test timers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence


class LatencyError(ValueError):
    """Raised when a latency protocol or observed sample is invalid."""


LEASE_MAX_AGE_SECONDS = 300
LEASE_MAX_FUTURE_SKEW_SECONDS = 30
_LEASE_ID = re.compile(r'^[0-9a-f]{64}$')


class LatencyTimer(Protocol):
    def synchronize(self) -> None: ...

    def measure_ms(self, callable_: Callable[[], Any]) -> float: ...


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True)
class LatencySummary:
    median_ms: float
    p90_ms: float
    p95_ms: float
    sample_count: int

    @classmethod
    def from_samples_ms(cls, samples: Sequence[float]) -> 'LatencySummary':
        if not samples:
            raise LatencyError('at least one latency sample is required')
        normalized = []
        for sample in samples:
            if (
                    isinstance(sample, bool)
                    or not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or float(sample) < 0.0):
                raise LatencyError(
                    'latency samples must be finite non-negative numbers')
            normalized.append(float(sample))
        normalized.sort()
        return cls(
            median_ms=_percentile(normalized, 0.5),
            p90_ms=_percentile(normalized, 0.9),
            p95_ms=_percentile(normalized, 0.95),
            sample_count=len(normalized),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            'median_ms': self.median_ms,
            'p90_ms': self.p90_ms,
            'p95_ms': self.p95_ms,
            'sample_count': self.sample_count,
        }


class _CudaEventTimer:
    """Production timer; this is the only path that accesses CUDA timing."""

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def measure_ms(self, callable_: Callable[[], Any]) -> float:
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        self._torch.cuda.synchronize()
        return float(start.elapsed_time(end))


def measure_latency(
        callable_: Callable[[], Any], warmup: int = 50, repeats: int = 200,
        *, timer: LatencyTimer | None = None) -> LatencySummary:
    """Measure one callable after warmup using synchronized CUDA events."""
    return LatencySummary.from_samples_ms(measure_latency_samples(
        callable_, warmup=warmup, repeats=repeats, timer=timer))


def measure_latency_samples(
        callable_: Callable[[], Any], warmup: int = 50, repeats: int = 200,
        *, timer: LatencyTimer | None = None) -> tuple[float, ...]:
    """Collect validated synchronized samples for protocol artifacts."""
    if (
            isinstance(warmup, bool) or not isinstance(warmup, int)
            or warmup < 0):
        raise LatencyError('warmup must be a non-negative integer')
    if (
            isinstance(repeats, bool) or not isinstance(repeats, int)
            or repeats <= 0):
        raise LatencyError('repeats must be a positive integer')
    effective_timer = timer if timer is not None else _CudaEventTimer()
    for _ in range(warmup):
        callable_()
    effective_timer.synchronize()
    samples = tuple(
        effective_timer.measure_ms(callable_) for _ in range(repeats))
    LatencySummary.from_samples_ms(samples)
    return samples


def validate_gpu_lease(value: object) -> dict[str, Any]:
    required = {
        'stage_id', 'pid', 'boot_id', 'timestamp', 'device_index',
        'allowed_pids', 'lease_id',
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise LatencyError('GPU lease provenance has invalid fields')
    if not isinstance(value['stage_id'], str) or not value['stage_id']:
        raise LatencyError('GPU lease stage_id is invalid')
    if (
            isinstance(value['pid'], bool) or not isinstance(value['pid'], int)
            or value['pid'] <= 0):
        raise LatencyError('GPU lease PID is invalid')
    allowed = value['allowed_pids']
    if not isinstance(allowed, (list, tuple)) or not allowed or not all(
            isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            for pid in allowed):
        raise LatencyError('GPU lease allowed_pids are invalid')
    if len(set(allowed)) != len(allowed) or value['pid'] not in allowed:
        raise LatencyError('GPU lease owner must be included once in allowed_pids')
    try:
        uuid.UUID(value['boot_id'])
    except (AttributeError, TypeError, ValueError) as error:
        raise LatencyError('GPU lease boot_id must be a UUID string') from error
    try:
        timestamp = datetime.fromisoformat(value['timestamp'])
    except (TypeError, ValueError) as error:
        raise LatencyError('GPU lease timestamp must be ISO-8601') from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise LatencyError('GPU lease timestamp must include a timezone')
    device = value['device_index']
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise LatencyError('GPU lease device_index is invalid')
    if (
            not isinstance(value['lease_id'], str)
            or not _LEASE_ID.fullmatch(value['lease_id'])):
        raise LatencyError('GPU lease lease_id must be lowercase 64-hex')
    return dict(value)


_lease = validate_gpu_lease


def build_latency_result(
        *, flip: Sequence[float], no_flip: Sequence[float], warmup: int,
        repeats: int, gpu_lease: Mapping[str, Any]) -> dict[str, Any]:
    """Build the batch-one dual-mode latency result for a stage envelope."""
    if len(flip) != repeats or len(no_flip) != repeats:
        raise LatencyError('latency sample counts must equal repeats')
    return {
        'protocol': {
            'batch_size': 1,
            'warmup': warmup,
            'iterations': repeats,
            'timer': 'torch.cuda.Event',
            'synchronize': True,
            'scope': 'full_topdown_model',
            'lease_max_age_seconds': LEASE_MAX_AGE_SECONDS,
            'lease_max_future_skew_seconds': LEASE_MAX_FUTURE_SKEW_SECONDS,
        },
        'modes': {
            'flip': LatencySummary.from_samples_ms(flip).to_dict(),
            'no_flip': LatencySummary.from_samples_ms(no_flip).to_dict(),
        },
        'gpu_lease': validate_gpu_lease(gpu_lease),
    }
