"""Stable public entry point for explicit numeric model conversion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mmengine.hooks import Hook
import torch
from torch import nn

from mmpose.registry import HOOKS

from mmpose.models.utils.hardware_friendly.fake_quant import (
    ConversionReport, QuantPolicy, QuantSpec, convert_for_fake_quant,
    export_int8_state)
from mmpose.models.utils.hardware_friendly.pwl import (
    PiecewiseLinearApproximation)
from mambapose_opt.pwl_artifacts import (
    PWLArtifactError, PWLInstallationReport, load_pwl_fit_reference,
    load_pwl_installation_reference, validate_pwl_fit_report)


class NumericBindingError(ValueError):
    """Raised when a downstream numeric artifact no longer matches its input."""


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        kind: str, *, conditional: bool,
        recovery: bool = False) -> tuple[str, ...]:
    """Return the immutable Route 3 order without admitting invasive work."""
    if recovery:
        if not conditional or kind not in {'w8a8', 'pwl', 'binary-qk'}:
            raise ValueError(
                'numeric recovery requires an explicit conditional candidate')
        return ('train', 'profile', 'evaluate', 'latency')
    if kind == 'observer':
        return ('calibrate',)
    if kind == 'weight-only':
        return ('convert', 'export', 'profile', 'evaluate', 'latency')
    if kind == 'w8a8':
        return ('calibrate', 'convert', 'profile', 'evaluate', 'latency')
    if kind == 'pwl':
        if not conditional:
            raise ValueError(f'{kind} requires explicit conditional admission')
        return ('calibrate', 'convert', 'profile', 'evaluate', 'latency')
    if kind == 'binary-qk':
        if not conditional:
            raise ValueError(f'{kind} requires explicit conditional admission')
        return ('smoke-stage-a', 'profile', 'evaluate', 'latency')
    raise ValueError(f'unsupported numeric candidate kind: {kind!r}')


def quant_policy_from_config(
        value: Mapping[str, object], *,
        calibration_artifact: Mapping[str, Any] | None = None) -> QuantPolicy:
    allowed_fields = {
        'allow', 'deny', 'spec', 'activation_observers',
        'calibration_artifact'}
    measured_fields = {
        'allow', 'deny', 'spec', 'activation_observers'}
    if (not isinstance(value, Mapping)
            or set(value) not in (
                {'allow', 'deny', 'spec'}, measured_fields, allowed_fields)):
        raise NumericBindingError(
            'quant policy config must contain allow, deny, and spec')
    spec_value = value['spec']
    if not isinstance(spec_value, Mapping):
        raise NumericBindingError('quant policy spec must be a mapping')
    scale = spec_value.get('activation_scale')
    if scale == 'runtime-calibration-artifact':
        if calibration_artifact is None:
            raise NumericBindingError(
                'W8A8 conversion requires a verified calibration artifact')
        from mambapose_opt.numeric_calibration import (
            CalibrationContractError, validate_calibration_artifact)
        try:
            validate_calibration_artifact(calibration_artifact)
        except CalibrationContractError as error:
            raise NumericBindingError(
                f'W8A8 calibration artifact is invalid: {error}') from error
        observers = value.get('activation_observers')
        if not isinstance(observers, Mapping):
            raise NumericBindingError(
                'W8A8 policy requires explicit per-role activation observers')
        allow = tuple(value['allow'])
        if set(observers) != set(allow):
            raise NumericBindingError(
                'activation observers must exactly cover allowed roles')
        measured = calibration_artifact['hooks'].get('activation_scales')
        if not isinstance(measured, Mapping) or set(measured) != set(allow):
            raise NumericBindingError(
                'calibration activation scales must exactly cover allowed roles')
        common = dict(spec_value)
        common['activation_bits'] = None
        common['activation_scale'] = None
        try:
            base_spec = QuantSpec(**common)
            role_specs = []
            for role in allow:
                record = measured[role]
                if (not isinstance(record, Mapping)
                        or set(record) != {'source_record', 'granularity', 'scale'}
                        or record['source_record'] != observers[role]
                        or record['granularity'] not in {'tensor', 'channel'}):
                    raise NumericBindingError(
                        f'activation scale binding is invalid for {role}')
                role_scale = record['scale']
                if isinstance(role_scale, list):
                    role_scale = tuple(role_scale)
                source_record = calibration_artifact['hooks']['records'][
                    record['source_record']]
                _validate_measured_activation_scale(
                    role, record['granularity'], role_scale, source_record)
                role_value = dict(spec_value)
                role_value['activation_scale'] = role_scale
                role_specs.append((role, QuantSpec(**role_value)))
            return QuantPolicy(
                allow=allow, deny=tuple(value['deny']), spec=base_spec,
                role_specs=tuple(role_specs))
        except (TypeError, ValueError) as error:
            if isinstance(error, NumericBindingError):
                raise
            raise NumericBindingError(
                f'invalid measured W8A8 policy: {error}') from error
    normalized = dict(spec_value)
    if isinstance(scale, list):
        normalized['activation_scale'] = tuple(scale)
    try:
        spec = QuantSpec(**normalized)
        return QuantPolicy(
            allow=tuple(value['allow']), deny=tuple(value['deny']), spec=spec)
    except (TypeError, ValueError) as error:
        raise NumericBindingError(f'invalid quant policy: {error}') from error


def _validate_measured_activation_scale(
        role: str, granularity: object, scale: object,
        source_record: Mapping[str, Any]) -> None:
    if granularity != source_record.get('granularity'):
        raise NumericBindingError(
            f'measured activation granularity disagrees for {role}')
    if granularity == 'tensor':
        numeric_range = source_record.get('range')
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
                or not math.isfinite(float(scale)) or float(scale) <= 0
                or not isinstance(numeric_range, list)
                or len(numeric_range) != 2):
            raise NumericBindingError(
                f'measured activation scale is invalid for {role}')
        maximum = max(abs(float(item)) for item in numeric_range)
        expected = maximum / 127.0 if maximum else 1.0
        if not math.isclose(float(scale), expected, rel_tol=1e-12, abs_tol=0.0):
            raise NumericBindingError(
                f'activation scale is not derived from measured range for {role}')
        return
    if granularity != 'channel' or not isinstance(scale, tuple):
        raise NumericBindingError(
            f'measured activation scale is invalid for {role}')
    maximum = source_record.get('max_abs')
    shape = source_record.get('observed_shape')
    if (not isinstance(maximum, list) or not maximum
            or not isinstance(shape, list) or shape != [len(maximum)]
            or len(scale) != len(maximum)):
        raise NumericBindingError(
            f'measured activation scale width disagrees for {role}')
    expected = tuple(float(item) / 127.0 if float(item) else 1.0
                     for item in maximum)
    if (any(not isinstance(item, (int, float)) or isinstance(item, bool)
            or not math.isfinite(float(item)) or float(item) <= 0
            for item in scale)
            or any(not math.isclose(float(actual), wanted, rel_tol=1e-12,
                                    abs_tol=0.0)
                   for actual, wanted in zip(scale, expected))):
        raise NumericBindingError(
            f'activation scale is not derived from measured channels for {role}')


def apply_numeric_runtime(
        model: nn.Module, numeric_optimization: Mapping[str, Any]
        ) -> ConversionReport | PWLInstallationReport | None:
    """Apply the config-bound fake-QDQ policy once to an instantiated model."""
    if not isinstance(numeric_optimization, Mapping):
        raise NumericBindingError('numeric runtime config must be a mapping')
    kind = numeric_optimization.get('candidate_kind')
    if kind == 'pwl':
        return _install_pwl_runtime(model, numeric_optimization)
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
    calibration = None
    reference = policy_value.get('calibration_artifact')
    if reference is not None:
        if (not isinstance(reference, Mapping)
                or set(reference) != {'path', 'sha256'}):
            raise NumericBindingError('calibration artifact reference is invalid')
        relative = Path(str(reference['path']))
        root = Path(__file__).resolve().parents[1]
        if (relative.is_absolute() or any(
                part in {'.', '..'} for part in relative.parts)
                or relative.parts[:2] != ('work_dirs', 'optimization')):
            raise NumericBindingError('calibration artifact path is unsafe')
        path = root / relative
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise NumericBindingError(
                    'calibration artifact path must not use symlinks')
        if (not path.is_file() or not isinstance(reference['sha256'], str)
                or _sha256(path) != reference['sha256']):
            raise NumericBindingError('calibration artifact hash changed')
        try:
            calibration = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise NumericBindingError(
                'calibration artifact is not valid JSON') from error
    report = convert_for_fake_quant(
        model, quant_policy_from_config(
            policy_value, calibration_artifact=calibration))
    model._numeric_conversion_report = report
    return report


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = name.rpartition('.')
    return (model.get_submodule(parent_name) if parent_name else model), leaf


def install_pwl_fit(
        model: nn.Module, *, fit: Mapping[str, Any],
        expected_report: PWLInstallationReport) -> PWLInstallationReport:
    """Install only coefficients authenticated by a measured fit report."""
    fitted = validate_pwl_fit_report(fit)
    expected = {
        'function_name': fitted['function_name'],
        'source': fitted['source'],
        'roles': tuple(fitted['operation_roles']),
        'input_roles': tuple(
            item['exact_input_role'] for item in fitted['input_roles']),
        'domain': tuple(fitted['domain']),
        'segments': fitted['segments'],
        'in_domain_max_error': fitted['in_domain_error']['max'],
        'in_domain_mean_error': fitted['in_domain_error']['mean'],
        'observed_range': tuple(fitted['observed_range']),
        'observed_range_max_error': fitted['observed_range_error']['max'],
        'observed_range_mean_error': fitted['observed_range_error']['mean'],
        'clamp_ratio': fitted['clamp']['ratio'],
        'saturation': fitted['saturation'],
        'qat_form': fitted['qat_form'],
        'hardware_latency_claimed': False,
        'exact_comparator': fitted['exact_comparator'],
    }
    if any(getattr(expected_report, name) != value
           for name, value in expected.items()):
        raise NumericBindingError(
            'PWL installation report disagrees with measured fit')
    existing = getattr(model, '_numeric_pwl_installation_report', None)
    if existing is not None:
        if existing != expected_report:
            raise NumericBindingError('PWL runtime marker is invalid')
        return existing
    coefficients = fitted['coefficients']

    def approximation() -> PiecewiseLinearApproximation:
        return PiecewiseLinearApproximation(
            coefficients['breakpoints'], coefficients['slopes'],
            coefficients['intercepts'],
            function_name=fitted['function_name'],
            max_error=fitted['in_domain_error']['max'],
            mean_error=fitted['in_domain_error']['mean'])

    roles = tuple(fitted['operation_roles'])
    modules = dict(model.named_modules())
    missing = tuple(role for role in roles if role not in modules)
    if missing:
        raise NumericBindingError(f'PWL roles are missing: {missing}')
    if fitted['source'] == 'module':
        expected_type = (
            nn.SiLU if fitted['function_name'] == 'silu' else nn.GELU)
        invalid = tuple(
            role for role in roles if type(modules[role]) is not expected_type)
        if invalid:
            raise NumericBindingError(
                f'PWL module roles have incompatible function: {invalid}')
        for role in roles:
            parent, leaf = _parent_and_leaf(model, role)
            setattr(parent, leaf, approximation())
    else:
        invalid = tuple(
            role for role in roles
            if not callable(getattr(modules[role], 'install_numeric_pwl', None)))
        if invalid:
            raise NumericBindingError(
                f'PWL functional roles do not expose SS2D installation: '
                f'{invalid}')
        for role in roles:
            modules[role].install_numeric_pwl(
                fitted['function_name'], approximation())
    model._numeric_pwl_installation_report = expected_report
    return expected_report


def _install_pwl_runtime(
        model: nn.Module,
        numeric_optimization: Mapping[str, Any]) -> PWLInstallationReport:
    value = numeric_optimization.get('pwl')
    required = {
        'enabled_function', 'source', 'roles', 'domain', 'segments',
        'grid_points', 'saturation', 'qat_form', 'selection_policy',
        'candidate_id', 'fit_artifact', 'installation_manifest'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise NumericBindingError(
            'PWL runtime requires fit artifact and installation manifest')
    candidate_id = value['candidate_id']
    if not isinstance(candidate_id, str) or not candidate_id:
        raise NumericBindingError('PWL runtime candidate identity is invalid')
    policy = {
        name: value[name] for name in (
            'enabled_function', 'source', 'roles', 'domain', 'segments',
            'grid_points', 'saturation', 'qat_form', 'selection_policy')}
    try:
        fit = load_pwl_fit_reference(
            value['fit_artifact'], repository_root=REPOSITORY_ROOT,
            expected_candidate_id=candidate_id, expected_policy=policy)
        installation = load_pwl_installation_reference(
            value['installation_manifest'], repository_root=REPOSITORY_ROOT,
            expected_candidate_id=candidate_id,
            expected_fit_reference=value['fit_artifact'], expected_fit=fit)
    except PWLArtifactError as error:
        raise NumericBindingError(
            f'PWL measured deployment binding is invalid: {error}') from error
    return install_pwl_fit(
        model, fit=fit, expected_report=installation['report_object'])


@HOOKS.register_module()
class NumericRuntimeHook(Hook):
    """Apply a manifest-configured numeric policy before runner execution."""

    priority = 'VERY_HIGH'

    @staticmethod
    def apply_to_model(model: nn.Module, config: Mapping[str, Any]):
        """Run the same registered hook operation outside an MMEngine Runner."""
        return apply_numeric_runtime(model, config)

    @staticmethod
    def _apply(runner) -> None:
        model = getattr(runner.model, 'module', runner.model)
        config = runner.cfg.get('numeric_optimization')
        NumericRuntimeHook.apply_to_model(model, config)

    def before_train(self, runner) -> None:
        self._apply(runner)

    def before_val(self, runner) -> None:
        self._apply(runner)

    def before_test(self, runner) -> None:
        self._apply(runner)


__all__ = [
    'ConversionReport', 'NumericBindingError', 'NumericInputBinding',
    'PWLInstallationReport',
    'NumericRuntimeHook', 'QuantPolicy', 'QuantSpec', 'apply_numeric_runtime',
    'bind_numeric_inputs', 'convert_for_fake_quant', 'export_int8_state',
    'install_pwl_fit', 'numeric_stage_plan', 'quant_policy_from_config',
    'verify_numeric_inputs']
