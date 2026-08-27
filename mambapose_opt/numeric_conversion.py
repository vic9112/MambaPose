"""Stable public entry point for explicit numeric model conversion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from mmengine.hooks import Hook
from torch import nn

from mmpose.registry import HOOKS

from mmpose.models.utils.hardware_friendly.fake_quant import (
    ConversionReport, QuantPolicy, QuantSpec, convert_for_fake_quant,
    export_int8_state)


class NumericBindingError(ValueError):
    """Raised when a downstream numeric artifact no longer matches its input."""


@dataclass(frozen=True)
class NumericInputBinding:
    paths: tuple[tuple[str, str], ...]
    sha256: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind_numeric_inputs(paths: Mapping[str, Path]) -> NumericInputBinding:
    if not paths:
        raise NumericBindingError('numeric inputs must not be empty')
    normalized: list[tuple[str, str]] = []
    hashes: list[tuple[str, str]] = []
    for role, supplied in sorted(paths.items()):
        if not isinstance(role, str) or not role:
            raise NumericBindingError('numeric input roles must be non-empty')
        path = Path(supplied).resolve()
        if not path.is_file():
            raise NumericBindingError(f'{role} input is missing: {path}')
        normalized.append((role, str(path)))
        hashes.append((role, _sha256(path)))
    return NumericInputBinding(tuple(normalized), tuple(hashes))


def verify_numeric_inputs(binding: NumericInputBinding) -> None:
    expected = dict(binding.sha256)
    for role, raw_path in binding.paths:
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected.get(role):
            raise NumericBindingError(f'{role} input hash changed')


def numeric_stage_plan(
        kind: str, *, conditional: bool) -> tuple[str, ...]:
    """Return the immutable Route 3 order without admitting invasive work."""
    if kind == 'observer':
        return ('calibrate', 'compare')
    if kind == 'weight-only':
        return ('convert', 'export', 'profile', 'evaluate', 'latency', 'compare')
    if kind == 'w8a8':
        return ('calibrate', 'convert', 'train', 'profile', 'evaluate',
                'latency', 'compare')
    if kind in {'pwl', 'binary-qk'}:
        if not conditional:
            raise ValueError(f'{kind} requires explicit conditional admission')
        return ('train', 'profile', 'evaluate', 'latency', 'compare')
    raise ValueError(f'unsupported numeric candidate kind: {kind!r}')


def quant_policy_from_config(value: Mapping[str, object]) -> QuantPolicy:
    if not isinstance(value, Mapping) or set(value) != {'allow', 'deny', 'spec'}:
        raise NumericBindingError(
            'quant policy config must contain allow, deny, and spec')
    spec_value = value['spec']
    if not isinstance(spec_value, Mapping):
        raise NumericBindingError('quant policy spec must be a mapping')
    scale = spec_value.get('activation_scale')
    if scale == 'runtime-calibration-artifact':
        raise NumericBindingError(
            'W8A8 conversion requires a verified calibration artifact')
    normalized = dict(spec_value)
    if isinstance(scale, list):
        normalized['activation_scale'] = tuple(scale)
    try:
        spec = QuantSpec(**normalized)
        return QuantPolicy(
            allow=tuple(value['allow']), deny=tuple(value['deny']), spec=spec)
    except (TypeError, ValueError) as error:
        raise NumericBindingError(f'invalid quant policy: {error}') from error


def apply_numeric_runtime(
        model: nn.Module, numeric_optimization: Mapping[str, Any]
        ) -> ConversionReport | None:
    """Apply the config-bound fake-QDQ policy once to an instantiated model."""
    if not isinstance(numeric_optimization, Mapping):
        raise NumericBindingError('numeric runtime config must be a mapping')
    kind = numeric_optimization.get('candidate_kind')
    if kind not in {'weight-only', 'w8a8'}:
        return None
    existing = getattr(model, '_numeric_conversion_report', None)
    if existing is not None:
        if not isinstance(existing, ConversionReport):
            raise NumericBindingError('numeric runtime marker is invalid')
        return existing
    policy_value = numeric_optimization.get('quant_policy')
    if not isinstance(policy_value, Mapping):
        raise NumericBindingError('numeric runtime quant policy is missing')
    report = convert_for_fake_quant(
        model, quant_policy_from_config(policy_value))
    model._numeric_conversion_report = report
    return report


@HOOKS.register_module()
class NumericRuntimeHook(Hook):
    """Apply a manifest-configured numeric policy before runner execution."""

    priority = 'VERY_HIGH'

    @staticmethod
    def _apply(runner) -> None:
        model = getattr(runner.model, 'module', runner.model)
        config = runner.cfg.get('numeric_optimization')
        apply_numeric_runtime(model, config)

    def before_train(self, runner) -> None:
        self._apply(runner)

    def before_val(self, runner) -> None:
        self._apply(runner)

    def before_test(self, runner) -> None:
        self._apply(runner)


__all__ = [
    'ConversionReport', 'NumericBindingError', 'NumericInputBinding',
    'NumericRuntimeHook', 'QuantPolicy', 'QuantSpec', 'apply_numeric_runtime',
    'bind_numeric_inputs', 'convert_for_fake_quant', 'export_int8_state',
    'numeric_stage_plan', 'quant_policy_from_config', 'verify_numeric_inputs']
