"""Strict measured PWL fit and installation artifacts for Route 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch

from .pwl_paths import canonical_relative_path
import torch.nn.functional as F

from mmpose.models.utils.hardware_friendly.pwl import (
    PiecewiseLinearApproximation, fit_pwl)


class PWLArtifactError(ValueError):
    """Raised when measured PWL evidence is incomplete or mutable."""


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_FUNCTIONS = frozenset({'silu', 'gelu', 'softplus', 'exp'})
_TAIL_FUNCTIONS = frozenset({'silu', 'gelu', 'softplus'})
_TAIL_SATURATION = 'continuous-asymptotic-tail-v1'
_SELECTION_POLICY = 'observed-range-max-then-mean-v1'
_TAIL_PERCENTILES = (0.9, 0.99, 0.999)
_TAIL_BINS = 256
_TAIL_MIN_EXP = -32.0
_TAIL_MAX_EXP = 32.0
_INSTALLATION_REPORT_FIELDS = {
    'function_name', 'source', 'roles', 'input_roles', 'domain', 'segments',
    'in_domain_max_error', 'in_domain_mean_error', 'observed_range',
    'observed_range_max_error', 'observed_range_mean_error',
    'out_of_domain_ratio', 'fit_artifact_path', 'fit_artifact_sha256',
    'saturation', 'qat_form', 'hardware_latency_claimed', 'exact_comparator',
}


@dataclass(frozen=True)
class PWLInstallationReport:
    """Serializable operation-level proof of one installed PWL function."""

    function_name: str
    source: str
    roles: tuple[str, ...]
    input_roles: tuple[str, ...]
    domain: tuple[float, float]
    segments: int
    in_domain_max_error: float
    in_domain_mean_error: float
    observed_range: tuple[float, float]
    observed_range_max_error: float
    observed_range_mean_error: float
    out_of_domain_ratio: float
    fit_artifact_path: str
    fit_artifact_sha256: str
    saturation: str = 'clamp'
    qat_form: str = 'differentiable'
    hardware_latency_claimed: bool = False
    exact_comparator: Mapping[str, Any] | None = None


def _reference(function_name: str):
    try:
        return {
            'silu': F.silu,
            'gelu': F.gelu,
            'softplus': F.softplus,
            'exp': torch.exp,
        }[function_name]
    except KeyError as error:
        raise PWLArtifactError(
            f'PWL function must be one of {sorted(_FUNCTIONS)}') from error


def _policy(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        'enabled_function', 'source', 'roles', 'domain', 'segments',
        'grid_points', 'saturation', 'qat_form', 'selection_policy'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise PWLArtifactError('PWL policy fields are invalid')
    function_name = value['enabled_function']
    if function_name not in _FUNCTIONS:
        raise PWLArtifactError('PWL policy function is invalid')
    source = value['source']
    expected_source = (
        'module' if function_name in {'silu', 'gelu'} else 'ss2d-transition')
    if source != expected_source:
        raise PWLArtifactError('PWL policy source disagrees with function')
    roles = value['roles']
    if (not isinstance(roles, (list, tuple)) or not roles
            or any(not isinstance(role, str) or not role for role in roles)
            or len(set(roles)) != len(roles)):
        raise PWLArtifactError('PWL policy roles are invalid')
    domain = value['domain']
    if (not isinstance(domain, (list, tuple)) or len(domain) != 2
            or any(isinstance(item, bool)
                   or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) for item in domain)
            or float(domain[0]) >= float(domain[1])):
        raise PWLArtifactError('PWL policy domain is invalid')
    segments = value['segments']
    grid_points = value['grid_points']
    if (isinstance(segments, bool) or not isinstance(segments, int)
            or segments < 1
            or isinstance(grid_points, bool) or not isinstance(grid_points, int)
            or grid_points < segments + 1):
        raise PWLArtifactError('PWL fit resolution is invalid')
    expected_saturation = (
        _TAIL_SATURATION if function_name in _TAIL_FUNCTIONS else 'clamp')
    if value['saturation'] != expected_saturation:
        raise PWLArtifactError(
            'PWL policy saturation disagrees with function')
    if value['qat_form'] != 'differentiable':
        raise PWLArtifactError('PWL policy QAT form must be differentiable')
    if value['selection_policy'] != _SELECTION_POLICY:
        raise PWLArtifactError(
            'PWL policy must defer selection to measured artifact ranking')
    return {
        **dict(value),
        'roles': tuple(roles),
        'domain': (float(domain[0]), float(domain[1])),
    }


def _domain_handling(saturation: str) -> dict[str, str]:
    if saturation == _TAIL_SATURATION:
        return {
            'kind': _TAIL_SATURATION,
            'left': 'constant-endpoint',
            'right': 'identity-plus-endpoint-offset',
        }
    if saturation == 'clamp':
        return {'kind': 'clamp'}
    raise PWLArtifactError('PWL domain handling is invalid')


def exact_input_role(
        operation_role: str, *, function_name: str, source: str) -> str:
    if source == 'module':
        return f'{operation_role}.input'
    return f'{operation_role}.transition_{function_name}_input'


def _metric(error: torch.Tensor) -> dict[str, float | int]:
    if error.numel() < 1 or not torch.isfinite(error).all():
        raise PWLArtifactError('PWL error measurement is non-finite')
    return {
        'max': float(error.max()),
        'mean': float(error.mean()),
        'samples': int(error.numel()),
    }


def _range_error(
        approximation: PiecewiseLinearApproximation, reference,
        bounds: tuple[float, float], samples: int) -> dict[str, float | int]:
    grid = torch.linspace(*bounds, samples, dtype=torch.float64)
    with torch.no_grad():
        error = (approximation(grid) - reference(grid)).abs()
    return _metric(error)


def _new_tail_record() -> dict[str, Any]:
    return {
        'histogram': torch.zeros(_TAIL_BINS, dtype=torch.int64),
        'zero_count': 0, 'underflow_count': 0, 'overflow_count': 0,
    }


def _observe_tail(record: dict[str, Any], observed: torch.Tensor) -> None:
    absolute = observed.abs()
    record['zero_count'] += int((absolute == 0).sum())
    nonzero = absolute[absolute != 0]
    if not nonzero.numel():
        return
    lower = 2 ** _TAIL_MIN_EXP
    upper = 2 ** _TAIL_MAX_EXP
    underflow = nonzero < lower
    overflow = nonzero > upper
    record['underflow_count'] += int(underflow.sum())
    record['overflow_count'] += int(overflow.sum())
    bounded = nonzero[~(underflow | overflow)]
    if bounded.numel():
        width = (_TAIL_MAX_EXP - _TAIL_MIN_EXP) / _TAIL_BINS
        indices = torch.floor(
            (torch.log2(bounded) - _TAIL_MIN_EXP) / width
        ).to(torch.int64).clamp(0, _TAIL_BINS - 1)
        record['histogram'].add_(torch.bincount(
            indices, minlength=_TAIL_BINS).cpu())


def _tail_percentile(
        histogram: Sequence[int], *, zero_count: int,
        total: int, percentile: float) -> float:
    rank = max(1, math.ceil(percentile * total))
    if rank <= zero_count:
        return 0.0
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank - zero_count:
            width = (_TAIL_MAX_EXP - _TAIL_MIN_EXP) / _TAIL_BINS
            return float(2 ** (_TAIL_MIN_EXP + (index + 1) * width))
    raise PWLArtifactError('PWL tail histogram does not cover its samples')


def _tail_report(record: Mapping[str, Any], *, total: int) -> dict[str, Any]:
    histogram = [int(item) for item in record['histogram'].tolist()]
    bounded = not (record['underflow_count'] or record['overflow_count'])
    return {
        'algorithm': 'fixed-log2-absolute-histogram-v1',
        'sample_count': total,
        'zero_count': record['zero_count'],
        'underflow_count': record['underflow_count'],
        'overflow_count': record['overflow_count'],
        'histogram_bins': _TAIL_BINS,
        'histogram_domain': [2 ** _TAIL_MIN_EXP, 2 ** _TAIL_MAX_EXP],
        'histogram': histogram,
        'percentile_bound_valid': bounded,
        'absolute_percentiles': {
            str(percentile): (
                _tail_percentile(
                    histogram, zero_count=record['zero_count'], total=total,
                    percentile=percentile)
                if bounded else None)
            for percentile in _TAIL_PERCENTILES
        },
    }


def _validate_tail_statistics(value: object, *, expected_total: int) -> bool:
    fields = {
        'algorithm', 'sample_count', 'zero_count', 'underflow_count',
        'overflow_count', 'histogram_bins', 'histogram_domain', 'histogram',
        'percentile_bound_valid', 'absolute_percentiles'}
    if not isinstance(value, Mapping) or set(value) != fields:
        return False
    counts = ('sample_count', 'zero_count', 'underflow_count', 'overflow_count')
    if (value['algorithm'] != 'fixed-log2-absolute-histogram-v1'
            or value['sample_count'] != expected_total
            or any(not isinstance(value[name], int)
                   or isinstance(value[name], bool) or value[name] < 0
                   for name in counts)
            or value['histogram_bins'] != _TAIL_BINS
            or value['histogram_domain'] != [
                2 ** _TAIL_MIN_EXP, 2 ** _TAIL_MAX_EXP]
            or not isinstance(value['histogram'], list)
            or len(value['histogram']) != _TAIL_BINS
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0
                   for item in value['histogram'])
            or sum(value['histogram']) + value['zero_count']
            + value['underflow_count'] + value['overflow_count']
            != expected_total):
        return False
    bounded = not (value['underflow_count'] or value['overflow_count'])
    if value['percentile_bound_valid'] is not bounded:
        return False
    percentiles = value['absolute_percentiles']
    if not isinstance(percentiles, Mapping) or set(percentiles) != {
            str(item) for item in _TAIL_PERCENTILES}:
        return False
    if not bounded:
        return all(item is None for item in percentiles.values())
    expected = {
        str(percentile): _tail_percentile(
            value['histogram'], zero_count=value['zero_count'],
            total=expected_total, percentile=percentile)
        for percentile in _TAIL_PERCENTILES}
    return all(
        isinstance(percentiles[name], (int, float))
        and not isinstance(percentiles[name], bool)
        and math.isclose(
            float(percentiles[name]), wanted, rel_tol=0.0, abs_tol=0.0)
        for name, wanted in expected.items())


class PWLObservationAccumulator:
    """Streaming exact-input observation without retaining calibration data."""

    def __init__(self, policy: Mapping[str, Any]):
        self.policy = _policy(policy)
        function_name = self.policy['enabled_function']
        self.reference = _reference(function_name)
        self.approximation = fit_pwl(
            self.reference, self.policy['domain'], self.policy['segments'],
            self.policy['grid_points'], function_name=function_name,
            saturation=self.policy['saturation'])
        self._records = {
            role: {
                'count': 0, 'minimum': math.inf, 'maximum': -math.inf,
                'below': 0, 'above': 0, 'error_sum': 0.0,
                'error_max': 0.0, 'tail': _new_tail_record(),
            }
            for role in self.policy['roles']
        }

    def observe(self, operation_role: str, value: torch.Tensor) -> None:
        if operation_role not in self._records:
            raise PWLArtifactError(
                f'PWL observation role is not declared: {operation_role}')
        if not isinstance(value, torch.Tensor) or value.numel() < 1:
            raise PWLArtifactError('PWL observation must be a non-empty tensor')
        observed = value.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
        if not torch.isfinite(observed).all():
            raise PWLArtifactError(
                f'PWL observation is non-finite: {operation_role}')
        with torch.no_grad():
            error = (self.approximation(observed)
                     - self.reference(observed)).abs()
        if not torch.isfinite(error).all():
            raise PWLArtifactError(
                f'PWL observed error is non-finite: {operation_role}')
        record = self._records[operation_role]
        lower, upper = self.policy['domain']
        record['count'] += int(observed.numel())
        record['minimum'] = min(record['minimum'], float(observed.min()))
        record['maximum'] = max(record['maximum'], float(observed.max()))
        record['below'] += int((observed < lower).sum())
        record['above'] += int((observed > upper).sum())
        record['error_sum'] += float(error.sum())
        record['error_max'] = max(record['error_max'], float(error.max()))
        _observe_tail(record['tail'], observed)

    def report(self, *, candidate_id: str) -> dict[str, Any]:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise PWLArtifactError('PWL candidate identity is invalid')
        missing = tuple(
            role for role, record in self._records.items()
            if record['count'] < 1)
        if missing:
            raise PWLArtifactError(
                f'PWL exact input roles were not observed: {missing}')
        policy = self.policy
        rows = []
        total = below = above = 0
        error_sum = 0.0
        error_max = 0.0
        observed_min = math.inf
        observed_max = -math.inf
        for role, record in self._records.items():
            count = record['count']
            bounds = (record['minimum'], record['maximum'])
            role_below = record['below']
            role_above = record['above']
            observed_error = {
                'max': record['error_max'],
                'mean': record['error_sum'] / count,
                'samples': count,
            }
            rows.append({
                'operation_role': role,
                'exact_input_role': exact_input_role(
                    role, function_name=policy['enabled_function'],
                    source=policy['source']),
                'observed_range': list(bounds),
                'observed_range_error': _range_error(
                    self.approximation, self.reference, bounds,
                    policy['grid_points']),
                'observed_samples_error': observed_error,
                'tail_statistics': _tail_report(
                    record['tail'], total=count),
                'domain_coverage': {
                    'below': role_below,
                    'above': role_above,
                    'total': count,
                    'ratio': (role_below + role_above) / count,
                    'handling': policy['saturation'],
                },
            })
            total += count
            below += role_below
            above += role_above
            error_sum += record['error_sum']
            error_max = max(error_max, record['error_max'])
            observed_min = min(observed_min, bounds[0])
            observed_max = max(observed_max, bounds[1])
        approximation = self.approximation
        exact_comparator = None
        if policy['enabled_function'] == 'exp':
            exact_comparator = {
                'kind': 'exact-export-time-constant-folding',
                'applicable_source': 'static-parameter',
                'runtime_nonlinear_operations': 0,
                'max_error': 0.0,
                'mean_error': 0.0,
                'preferred_over_pwl_when_exportable': True,
            }
        reasons = []
        if policy['saturation'] == 'clamp':
            if below + above:
                reasons.append('clamp-count-nonzero')
            if (observed_min < policy['domain'][0]
                    or observed_max > policy['domain'][1]):
                reasons.append('observed-range-outside-domain')
        return {
            'schema_version': 2,
            'candidate_id': candidate_id,
            'function_name': policy['enabled_function'],
            'source': policy['source'],
            'operation_roles': list(policy['roles']),
            'input_roles': [{
                'operation_role': role,
                'exact_input_role': exact_input_role(
                    role, function_name=policy['enabled_function'],
                    source=policy['source']),
            } for role in policy['roles']],
            'domain': list(policy['domain']),
            'segments': policy['segments'],
            'grid_points': policy['grid_points'],
            'saturation': policy['saturation'],
            'qat_form': policy['qat_form'],
            'selection_policy': policy['selection_policy'],
            'coefficients': {
                'breakpoints': approximation.breakpoints.tolist(),
                'slopes': approximation.slopes.tolist(),
                'intercepts': approximation.intercepts.tolist(),
            },
            'observed_range': [observed_min, observed_max],
            'in_domain_error': _range_error(
                approximation, self.reference, policy['domain'],
                policy['grid_points']),
            'observed_range_error': _range_error(
                approximation, self.reference,
                (observed_min, observed_max), policy['grid_points']),
            'observed_samples_error': {
                'max': error_max,
                'mean': error_sum / total,
                'samples': total,
            },
            'domain_coverage': {
                'below': below, 'above': above, 'total': total,
                'ratio': (below + above) / total,
                'handling': policy['saturation'],
            },
            'admission': {
                'decision': 'rejected' if reasons else 'passed',
                'reasons': reasons,
            },
            'role_observations': rows,
            'exact_comparator': exact_comparator,
        }


def fit_pwl_observations(
        *, candidate_id: str, policy: Mapping[str, Any],
        observations: Mapping[str, Iterable[torch.Tensor]]) -> dict[str, Any]:
    """Fit the declared domain and measure exact supplied role inputs."""
    accumulator = PWLObservationAccumulator(policy)
    if not isinstance(observations, Mapping):
        raise PWLArtifactError('PWL observations must be a role mapping')
    if set(observations) != set(accumulator.policy['roles']):
        raise PWLArtifactError(
            'PWL observations must exactly cover declared operation roles')
    for role in accumulator.policy['roles']:
        for value in observations[role]:
            accumulator.observe(role, value)
    result = accumulator.report(candidate_id=candidate_id)
    validate_pwl_fit_report(
        result, expected_candidate_id=candidate_id,
        expected_policy=accumulator.policy)
    return result


def _finite_metric(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {'max', 'mean', 'samples'}
        and all(isinstance(value[name], (int, float))
                and not isinstance(value[name], bool)
                and math.isfinite(float(value[name]))
                for name in ('max', 'mean'))
        and 0 <= float(value['mean']) <= float(value['max'])
        and isinstance(value['samples'], int)
        and not isinstance(value['samples'], bool)
        and value['samples'] > 0)


def _finite_range(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item)) for item in value)
        and float(value[0]) <= float(value[1]))


def _metric_matches(
        actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual['samples'] == expected['samples']
        and math.isclose(
            float(actual['max']), float(expected['max']),
            rel_tol=1e-12, abs_tol=1e-15)
        and math.isclose(
            float(actual['mean']), float(expected['mean']),
            rel_tol=1e-12, abs_tol=1e-15))


def _domain_coverage(value: object, *, expected_handling: str) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
            'below', 'above', 'total', 'ratio', 'handling'}:
        return False
    if (any(not isinstance(value[name], int) or isinstance(value[name], bool)
            or value[name] < 0 for name in ('below', 'above', 'total'))
            or value['total'] < 1
            or value['below'] + value['above'] > value['total']
            or not isinstance(value['ratio'], (int, float))
            or isinstance(value['ratio'], bool)
            or not math.isfinite(float(value['ratio']))
            or value['handling'] != expected_handling):
        return False
    return math.isclose(
        float(value['ratio']),
        (value['below'] + value['above']) / value['total'],
        rel_tol=1e-12, abs_tol=1e-15)


def validate_pwl_fit_report(
        value: Mapping[str, Any], *, expected_candidate_id: str | None = None,
        expected_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fields = {
        'schema_version', 'candidate_id', 'function_name', 'source',
        'operation_roles', 'input_roles', 'domain', 'segments', 'grid_points',
        'saturation', 'qat_form', 'selection_policy', 'coefficients',
        'observed_range', 'in_domain_error', 'observed_range_error',
        'observed_samples_error', 'domain_coverage', 'role_observations',
        'admission', 'exact_comparator'}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PWLArtifactError('PWL fit artifact fields are invalid')
    if value.get('schema_version') != 2:
        raise PWLArtifactError('PWL fit schema version is invalid')
    candidate_id = value.get('candidate_id')
    if not isinstance(candidate_id, str) or not candidate_id:
        raise PWLArtifactError('PWL fit candidate identity is invalid')
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise PWLArtifactError('PWL fit candidate identity disagrees')
    policy = _policy(expected_policy or {
        'enabled_function': value.get('function_name'),
        'source': value.get('source'),
        'roles': value.get('operation_roles'),
        'domain': value.get('domain'),
        'segments': value.get('segments'),
        'grid_points': value.get('grid_points'),
        'saturation': value.get('saturation'),
        'qat_form': value.get('qat_form'),
        'selection_policy': value.get('selection_policy'),
    })
    expected = {
        'function_name': policy['enabled_function'],
        'source': policy['source'],
        'operation_roles': list(policy['roles']),
        'domain': list(policy['domain']),
        'segments': policy['segments'],
        'grid_points': policy['grid_points'],
        'saturation': policy['saturation'],
        'qat_form': policy['qat_form'],
        'selection_policy': policy['selection_policy'],
    }
    if any(value[name] != wanted for name, wanted in expected.items()):
        raise PWLArtifactError('PWL fit artifact disagrees with target policy')
    wanted_input_roles = [{
        'operation_role': role,
        'exact_input_role': exact_input_role(
            role, function_name=policy['enabled_function'],
            source=policy['source']),
    } for role in policy['roles']]
    if value['input_roles'] != wanted_input_roles:
        raise PWLArtifactError('PWL exact input roles disagree with policy')
    coefficients = value['coefficients']
    if not isinstance(coefficients, Mapping) or set(coefficients) != {
            'breakpoints', 'slopes', 'intercepts'}:
        raise PWLArtifactError('PWL coefficients are invalid')
    try:
        approximation = PiecewiseLinearApproximation(
            coefficients['breakpoints'], coefficients['slopes'],
            coefficients['intercepts'],
            function_name=policy['enabled_function'],
            saturation=policy['saturation'])
    except (TypeError, ValueError) as error:
        raise PWLArtifactError(f'PWL coefficients are invalid: {error}') from error
    if (approximation.domain != policy['domain']
            or approximation.segments != policy['segments']):
        raise PWLArtifactError('PWL coefficients disagree with fit policy')
    canonical = fit_pwl(
        _reference(policy['enabled_function']), policy['domain'],
        policy['segments'], policy['grid_points'],
        function_name=policy['enabled_function'],
        saturation=policy['saturation'])
    if any(not torch.equal(actual, wanted) for actual, wanted in (
            (approximation.breakpoints, canonical.breakpoints),
            (approximation.slopes, canonical.slopes),
            (approximation.intercepts, canonical.intercepts))):
        raise PWLArtifactError(
            'PWL coefficients disagree with canonical tracked-policy fit')
    observed_range = value['observed_range']
    if not _finite_range(observed_range):
        raise PWLArtifactError('PWL observed range is invalid')
    if (not _finite_metric(value['in_domain_error'])
            or value['in_domain_error']['samples'] != policy['grid_points']
            or not _finite_metric(value['observed_range_error'])
            or value['observed_range_error']['samples'] != policy['grid_points']
            or not _finite_metric(value['observed_samples_error'])
            or not _domain_coverage(
                value['domain_coverage'],
                expected_handling=policy['saturation'])):
        raise PWLArtifactError('PWL fit errors or domain coverage are invalid')
    rows = value['role_observations']
    if (not isinstance(rows, list) or len(rows) != len(policy['roles'])
            or [row.get('operation_role') if isinstance(row, Mapping) else None
                for row in rows] != list(policy['roles'])):
        raise PWLArtifactError('PWL role observations are invalid')
    role_ranges = []
    role_clamps = []
    role_sample_errors = []
    reference = _reference(policy['enabled_function'])
    for row, input_role in zip(rows, wanted_input_roles):
        if (set(row) != {
                'operation_role', 'exact_input_role', 'observed_range',
                'observed_range_error', 'observed_samples_error',
                'tail_statistics', 'domain_coverage'}
                or row['exact_input_role'] != input_role['exact_input_role']
                or not _finite_range(row['observed_range'])
                or not _finite_metric(row['observed_range_error'])
                or row['observed_range_error']['samples'] != policy['grid_points']
                or not _finite_metric(row['observed_samples_error'])
                or not _domain_coverage(
                    row['domain_coverage'],
                    expected_handling=policy['saturation'])
                or row['observed_samples_error']['samples'] !=
                row['domain_coverage']['total']):
            raise PWLArtifactError('PWL role observation schema is invalid')
        if not _validate_tail_statistics(
                row['tail_statistics'],
                expected_total=row['observed_samples_error']['samples']):
            raise PWLArtifactError(
                'PWL role tail statistics are invalid or not recomputable')
        recomputed = _range_error(
            approximation, reference, tuple(row['observed_range']),
            policy['grid_points'])
        if not _metric_matches(row['observed_range_error'], recomputed):
            raise PWLArtifactError(
                'PWL role observation disagrees with recomputed error')
        role_ranges.append(row['observed_range'])
        role_clamps.append(row['domain_coverage'])
        role_sample_errors.append(row['observed_samples_error'])
    aggregate_total = sum(item['total'] for item in role_clamps)
    aggregate_below = sum(item['below'] for item in role_clamps)
    aggregate_above = sum(item['above'] for item in role_clamps)
    aggregate_sample_max = max(item['max'] for item in role_sample_errors)
    aggregate_sample_mean = sum(
        item['mean'] * item['samples'] for item in role_sample_errors
    ) / aggregate_total
    if (observed_range != [
                min(item[0] for item in role_ranges),
                max(item[1] for item in role_ranges)]
            or value['domain_coverage']['total'] != aggregate_total
            or value['domain_coverage']['below'] != aggregate_below
            or value['domain_coverage']['above'] != aggregate_above
            or value['observed_samples_error']['samples'] != aggregate_total
            or not math.isclose(
                float(value['observed_samples_error']['max']),
                float(aggregate_sample_max), rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(
                float(value['observed_samples_error']['mean']),
                float(aggregate_sample_mean), rel_tol=1e-12, abs_tol=1e-15)):
        raise PWLArtifactError(
            'PWL aggregate observations disagree with role observations')
    expected_reasons = []
    if policy['saturation'] == 'clamp':
        if aggregate_below + aggregate_above:
            expected_reasons.append('clamp-count-nonzero')
        if (float(observed_range[0]) < policy['domain'][0]
                or float(observed_range[1]) > policy['domain'][1]):
            expected_reasons.append('observed-range-outside-domain')
    expected_admission = {
        'decision': 'rejected' if expected_reasons else 'passed',
        'reasons': expected_reasons,
    }
    if value['admission'] != expected_admission:
        raise PWLArtifactError(
            'PWL fit admission disagrees with measured domain evidence')
    recomputed_domain = _range_error(
        approximation, reference, policy['domain'], policy['grid_points'])
    recomputed_observed = _range_error(
        approximation, reference, tuple(observed_range),
        policy['grid_points'])
    if (not _metric_matches(value['in_domain_error'], recomputed_domain)
            or not _metric_matches(
                value['observed_range_error'], recomputed_observed)):
        raise PWLArtifactError(
            'PWL fit artifact disagrees with recomputed grid error')
    comparator = value['exact_comparator']
    expected_comparator = ({
        'kind': 'exact-export-time-constant-folding',
        'applicable_source': 'static-parameter',
        'runtime_nonlinear_operations': 0,
        'max_error': 0.0,
        'mean_error': 0.0,
        'preferred_over_pwl_when_exportable': True,
    } if policy['enabled_function'] == 'exp' else None)
    if comparator != expected_comparator:
        raise PWLArtifactError('PWL exact comparator is invalid')
    return dict(value)


def require_pwl_fit_admitted(value: Mapping[str, Any]) -> dict[str, Any]:
    fitted = validate_pwl_fit_report(value)
    if fitted['admission'] != {'decision': 'passed', 'reasons': []}:
        raise PWLArtifactError(
            'PWL measured fit is not admitted for ranking or installation')
    return fitted


def require_pwl_runtime_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Admit a runtime PWL only when no exact static comparator supersedes it."""
    fitted = require_pwl_fit_admitted(value)
    comparator = fitted['exact_comparator']
    if (fitted['function_name'] == 'exp'
            and isinstance(comparator, Mapping)
            and comparator.get('preferred_over_pwl_when_exportable') is True
            and comparator.get('applicable_source') == 'static-parameter'
            and comparator.get('runtime_nonlinear_operations') == 0):
        raise PWLArtifactError(
            'exp PWL is superseded by exact export-time constant folding')
    return fitted


def rank_pwl_fit_artifacts(
        values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Rank by observed-range max, then mean and domain coverage ratio."""
    validated = [require_pwl_runtime_candidate(value) for value in values]
    return tuple(sorted(validated, key=lambda item: (
        float(item['observed_range_error']['max']),
        float(item['observed_range_error']['mean']),
        float(item['domain_coverage']['ratio']),
        item['candidate_id'])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_reference(
        reference: Mapping[str, Any], *, repository_root: Path,
        label: str) -> Path:
    if (not isinstance(reference, Mapping)
            or set(reference) != {'path', 'sha256'}
            or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str)
            or not _SHA256.fullmatch(reference['sha256'])):
        raise PWLArtifactError(f'{label} reference is invalid')
    try:
        relative = canonical_relative_path(reference['path'], label=label)
    except ValueError as error:
        raise PWLArtifactError(str(error)) from error
    if relative.parts[:2] != ('work_dirs', 'optimization'):
        raise PWLArtifactError(
            f'{label} path must be under work_dirs/optimization')
    root = Path(repository_root).resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PWLArtifactError(f'{label} path must not use symlink')
    if not cursor.is_file() or _sha256(cursor) != reference['sha256']:
        raise PWLArtifactError(f'{label} hash changed')
    return cursor


def load_pwl_fit_reference(
        reference: Mapping[str, Any], *, repository_root: Path,
        expected_candidate_id: str,
        expected_policy: Mapping[str, Any]) -> dict[str, Any]:
    path = _strict_reference(
        reference, repository_root=repository_root, label='PWL fit artifact')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise PWLArtifactError('PWL fit artifact is invalid JSON') from error
    fit = value.get('pwl_fit') if isinstance(value, Mapping) else None
    return validate_pwl_fit_report(
        fit, expected_candidate_id=expected_candidate_id,
        expected_policy=expected_policy)


def _installation_report_from_fit(
        fitted: Mapping[str, Any],
        reference: Mapping[str, str]) -> PWLInstallationReport:
    require_pwl_runtime_candidate(fitted)
    return PWLInstallationReport(
        function_name=fitted['function_name'], source=fitted['source'],
        roles=tuple(fitted['operation_roles']),
        input_roles=tuple(
            item['exact_input_role'] for item in fitted['input_roles']),
        domain=tuple(fitted['domain']), segments=fitted['segments'],
        in_domain_max_error=fitted['in_domain_error']['max'],
        in_domain_mean_error=fitted['in_domain_error']['mean'],
        observed_range=tuple(fitted['observed_range']),
        observed_range_max_error=fitted['observed_range_error']['max'],
        observed_range_mean_error=fitted['observed_range_error']['mean'],
        out_of_domain_ratio=fitted['domain_coverage']['ratio'],
        fit_artifact_path=reference['path'],
        fit_artifact_sha256=reference['sha256'],
        saturation=fitted['saturation'],
        qat_form=fitted['qat_form'],
        exact_comparator=fitted['exact_comparator'])


def build_pwl_installation_manifest(
        *, candidate_id: str, fit: Mapping[str, Any],
        fit_reference: Mapping[str, str]) -> dict[str, Any]:
    fitted = validate_pwl_fit_report(
        fit, expected_candidate_id=candidate_id)
    require_pwl_runtime_candidate(fitted)
    reference = dict(fit_reference)
    if (set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str)
            or not _SHA256.fullmatch(str(reference['sha256']))):
        raise PWLArtifactError('PWL installation fit reference is invalid')
    report = _installation_report_from_fit(fitted, reference)
    return {
        'schema_version': 2,
        'candidate_id': candidate_id,
        'fit_artifact': reference,
        'report': asdict(report),
        'operation_manifest': {
            'function': fitted['function_name'],
            'source': fitted['source'],
            'operation_roles': list(fitted['operation_roles']),
            'exact_input_roles': [
                item['exact_input_role'] for item in fitted['input_roles']],
            'segments_per_role': fitted['segments'],
            'domain_handling': _domain_handling(fitted['saturation']),
            'hardware_latency_claimed': False,
        },
    }


def validate_pwl_installation_manifest(
        value: Mapping[str, Any], *, expected_candidate_id: str,
        expected_fit_reference: Mapping[str, str],
        expected_fit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) != {
                'schema_version', 'candidate_id', 'fit_artifact', 'report',
                'operation_manifest'}
            or value.get('schema_version') != 2
            or value.get('candidate_id') != expected_candidate_id
            or value.get('fit_artifact') != dict(expected_fit_reference)):
        raise PWLArtifactError('PWL installation manifest identity is invalid')
    report_value = value['report']
    if (not isinstance(report_value, Mapping)
            or set(report_value) != _INSTALLATION_REPORT_FIELDS):
        raise PWLArtifactError('PWL installation report is invalid')
    try:
        normalized = dict(report_value)
        for name in ('roles', 'input_roles', 'domain', 'observed_range'):
            normalized[name] = tuple(normalized[name])
        report = PWLInstallationReport(**normalized)
    except (KeyError, TypeError, ValueError) as error:
        raise PWLArtifactError('PWL installation report is invalid') from error
    expected_saturation = (
        _TAIL_SATURATION
        if report.function_name in _TAIL_FUNCTIONS else 'clamp')
    if (report.function_name not in _FUNCTIONS
            or report.saturation != expected_saturation):
        raise PWLArtifactError(
            'PWL installation report domain handling is invalid')
    if (report.fit_artifact_path != expected_fit_reference['path']
            or report.fit_artifact_sha256 != expected_fit_reference['sha256']
            or report.hardware_latency_claimed is not False):
        raise PWLArtifactError('PWL installation report fit binding is invalid')
    if expected_fit is None:
        raise PWLArtifactError(
            'PWL installation measured fit authority is required')
    fitted = validate_pwl_fit_report(
        expected_fit, expected_candidate_id=expected_candidate_id)
    expected_report = _installation_report_from_fit(
        fitted, expected_fit_reference)
    if report != expected_report:
        raise PWLArtifactError(
            'PWL installation report disagrees with measured fit')
    expected_operation = {
        'function': report.function_name,
        'source': report.source,
        'operation_roles': list(report.roles),
        'exact_input_roles': list(report.input_roles),
        'segments_per_role': report.segments,
        'domain_handling': _domain_handling(report.saturation),
        'hardware_latency_claimed': False,
    }
    if value['operation_manifest'] != expected_operation:
        raise PWLArtifactError('PWL operation manifest is invalid')
    result = dict(value)
    result['report_object'] = report
    return result


def load_pwl_installation_reference(
        reference: Mapping[str, Any], *, repository_root: Path,
        expected_candidate_id: str,
        expected_fit_reference: Mapping[str, str],
        expected_fit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = _strict_reference(
        reference, repository_root=repository_root,
        label='PWL installation manifest')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise PWLArtifactError(
            'PWL installation manifest is invalid JSON') from error
    return validate_pwl_installation_manifest(
        value, expected_candidate_id=expected_candidate_id,
        expected_fit_reference=expected_fit_reference,
        expected_fit=expected_fit)


__all__ = [
    'PWLArtifactError', 'PWLInstallationReport',
    'PWLObservationAccumulator', 'build_pwl_installation_manifest',
    'exact_input_role', 'fit_pwl_observations',
    'load_pwl_fit_reference', 'load_pwl_installation_reference',
    'rank_pwl_fit_artifacts', 'require_pwl_fit_admitted',
    'require_pwl_runtime_candidate',
    'validate_pwl_fit_report',
    'validate_pwl_installation_manifest']
