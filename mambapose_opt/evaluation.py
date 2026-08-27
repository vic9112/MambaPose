"""Strict COCO metric normalization and candidate-result loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')
_SOURCE_KEYS = {
    'AP': ('coco/AP',),
    'AP50': ('coco/AP .5', 'coco/AP50'),
    'AP75': ('coco/AP .75', 'coco/AP75'),
    'APM': ('coco/AP (M)', 'coco/APM'),
    'APL': ('coco/AP (L)', 'coco/APL'),
    'AR': ('coco/AR',),
}
_PROVENANCE_FIELDS = {
    'checkpoint_sha256', 'config_sha256', 'data_inventory_sha256',
    'git_commit',
}


class MetricError(ValueError):
    """Raised when metrics or their provenance cannot support a gate."""


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_provenance(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise MetricError(
            'provenance must contain exact checkpoint/config/data hashes and commit')
    normalized = dict(value)
    for field in _PROVENANCE_FIELDS - {'git_commit'}:
        item = normalized[field]
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise MetricError(f'{field} must be a lowercase 64-hex SHA-256')
    commit = normalized['git_commit']
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise MetricError('git_commit must be a lowercase 40-hex commit')
    return normalized


@dataclass(frozen=True)
class CocoMetrics:
    """COCO metrics in the one canonical internal unit: percentage points."""

    ap: float
    ap50: float
    ap75: float
    apm: float
    apl: float
    ar: float
    unit: str = 'percentage_points'

    def __post_init__(self) -> None:
        if self.unit != 'percentage_points':
            raise MetricError('COCO metric unit must be percentage_points')
        for name in ('ap', 'ap50', 'ap75', 'apm', 'apl', 'ar'):
            value = getattr(self, name)
            if not _finite_number(value) or not 0.0 <= float(value) <= 100.0:
                raise MetricError(
                    f'{name.upper()} must be finite and in range 0..100')

    def to_dict(self) -> dict[str, float | str]:
        return {
            'unit': self.unit,
            'AP': float(self.ap),
            'AP50': float(self.ap50),
            'AP75': float(self.ap75),
            'APM': float(self.apm),
            'APL': float(self.apl),
            'AR': float(self.ar),
        }

    @classmethod
    def from_dict(cls, value: object) -> 'CocoMetrics':
        fields = {'unit', 'AP', 'AP50', 'AP75', 'APM', 'APL', 'AR'}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MetricError('normalized COCO metrics have invalid fields')
        return cls(
            ap=value['AP'], ap50=value['AP50'], ap75=value['AP75'],
            apm=value['APM'], apl=value['APL'], ar=value['AR'],
            unit=value['unit'])


def load_coco_metrics(
        path: Path | str, *,
        provenance: Mapping[str, str] | None = None) -> CocoMetrics:
    """Load raw MMPose fractions and normalize them exactly once to AP points."""
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot load COCO metrics: {error}') from error
    if not isinstance(raw, Mapping):
        raise MetricError('COCO metrics must be an object')
    selected: dict[str, float] = {}
    for canonical, aliases in _SOURCE_KEYS.items():
        present = [key for key in aliases if key in raw]
        if not present:
            raise MetricError(f'COCO metric {canonical} is required')
        if len(present) != 1:
            raise MetricError(f'COCO metric {canonical} has duplicate aliases')
        value = raw[present[0]]
        if not _finite_number(value):
            raise MetricError(f'COCO metric {canonical} must be finite')
        selected[canonical] = float(value)
    if any(value < 0.0 for value in selected.values()):
        raise MetricError('COCO metric fraction is outside range 0..1')
    if any(value > 1.0 for value in selected.values()):
        qualifier = 'mixed units or ' if any(
            value <= 1.0 for value in selected.values()) else ''
        raise MetricError(
            f'raw MMPose metrics use {qualifier}non-fraction values; '
            'expected fractions in range 0..1')
    if provenance is None:
        raise MetricError(
            'metric provenance with checkpoint/config/data hashes is required')
    validate_provenance(provenance)
    return CocoMetrics(
        ap=selected['AP'] * 100.0,
        ap50=selected['AP50'] * 100.0,
        ap75=selected['AP75'] * 100.0,
        apm=selected['APM'] * 100.0,
        apl=selected['APL'] * 100.0,
        ar=selected['AR'] * 100.0,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_determinism(
        value: object, provenance: Mapping[str, str]) -> Mapping[str, Any]:
    required = {
        'python_seed', 'numpy_seed', 'torch_seed', 'worker_count', 'workers',
        'persistent_workers', 'order_hashes', 'provenance',
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise MetricError('determinism record has invalid fields')
    seeds = (value['python_seed'], value['numpy_seed'], value['torch_seed'])
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise MetricError('determinism root seeds must be integers')
    worker_count = value['worker_count']
    workers = value['workers']
    if (
            isinstance(worker_count, bool) or not isinstance(worker_count, int)
            or worker_count < 0 or not isinstance(workers, list)
            or len(workers) != worker_count):
        raise MetricError('determinism worker records do not match worker_count')
    worker_fields = {
        'worker_id', 'python_seed', 'numpy_seed', 'torch_seed'}
    if any(
            not isinstance(worker, Mapping) or set(worker) != worker_fields
            or any(
                isinstance(worker[field], bool)
                or not isinstance(worker[field], int)
                for field in worker_fields)
            for worker in workers):
        raise MetricError('determinism worker seed record is invalid')
    if value['persistent_workers'] is not False:
        raise MetricError('determinism requires disabled persistent workers')
    hashes = value['order_hashes']
    if (
            not isinstance(hashes, Mapping) or not hashes
            or any(
                not isinstance(epoch, str) or not epoch.isdigit()
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                for epoch, digest in hashes.items())):
        raise MetricError('determinism order hashes are invalid')
    nested = validate_provenance(value['provenance'])
    if nested != dict(provenance):
        raise MetricError('determinism provenance disagrees with result provenance')
    return _freeze(value)


@dataclass(frozen=True)
class CandidateResult:
    """Immutable, validated normalized row from an evaluation artifact."""

    candidate_id: str
    route: str
    metrics: CocoMetrics
    flip_test: bool
    provenance: Mapping[str, str]
    determinism: Mapping[str, Any]
    protocol: Mapping[str, Any]
    calibration_split: str | None
    profile: Mapping[str, Any] | None
    latency: Mapping[str, Any] | None
    gpu_lease: Mapping[str, Any] | None
    artifact_paths: Mapping[str, Path]
    evaluation_artifact: Path

    @classmethod
    def from_artifacts(cls, root: Path | str) -> 'CandidateResult':
        root = Path(root)
        path = root if root.is_file() else root / 'evaluate/evaluate.json'
        try:
            envelope = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise MetricError(f'cannot load evaluation artifact: {error}') from error
        required = {'schema_version', 'candidate_id', 'stage', 'result'}
        if not isinstance(envelope, Mapping) or set(envelope) != required:
            raise MetricError('evaluation artifact must use the stage envelope')
        if envelope['schema_version'] != 1:
            raise MetricError('evaluation schema_version must be 1')
        if envelope['stage'] != 'evaluate':
            raise MetricError('evaluation artifact stage must be evaluate')
        candidate_id = envelope['candidate_id']
        if not isinstance(candidate_id, str) or not candidate_id:
            raise MetricError('evaluation candidate_id must be non-empty')
        result = envelope['result']
        result_fields = {
            'route', 'flip_test', 'metrics', 'provenance', 'determinism',
            'protocol', 'calibration_split',
        }
        if not isinstance(result, Mapping) or set(result) != result_fields:
            raise MetricError('evaluation result has invalid fields')
        route = result['route']
        if not isinstance(route, str) or not route:
            raise MetricError('evaluation route must be non-empty')
        if not isinstance(result['flip_test'], bool):
            raise MetricError('evaluation flip_test must be boolean')
        calibration_split = result['calibration_split']
        if calibration_split not in {None, 'train2017'}:
            raise MetricError(
                'calibration_split must be absent or train2017, never val2017')
        provenance = validate_provenance(result['provenance'])
        determinism = _validate_determinism(result['determinism'], provenance)
        protocol = result['protocol']
        if not isinstance(protocol, Mapping):
            raise MetricError('evaluation protocol must be an object')
        if (
                protocol.get('dataset') != 'coco'
                or protocol.get('split') != 'val2017'
                or protocol.get('complete_split') is not True
                or isinstance(protocol.get('batch_size'), bool)
                or not isinstance(protocol.get('batch_size'), int)
                or protocol['batch_size'] <= 0):
            raise MetricError('evaluation protocol is not complete COCO val2017')
        profile: Mapping[str, Any] | None = None
        latency: Mapping[str, Any] | None = None
        gpu_lease: Mapping[str, Any] | None = None
        artifact_paths: dict[str, Path] = {'evaluation': path.resolve()}
        if root.is_dir():
            profile_path = root / 'profile/profile.json'
            latency_path = root / 'latency/latency.json'
            profile = _load_profile(
                profile_path, candidate_id=candidate_id,
                provenance=provenance)
            latency, gpu_lease = _load_latency(
                latency_path, candidate_id=candidate_id, route=route,
                provenance=provenance)
            artifact_paths.update({
                'profile': profile_path.resolve(),
                'latency': latency_path.resolve(),
            })
        return cls(
            candidate_id=candidate_id,
            route=route,
            metrics=CocoMetrics.from_dict(result['metrics']),
            flip_test=result['flip_test'],
            provenance=MappingProxyType(provenance),
            determinism=determinism,
            protocol=_freeze(protocol),
            calibration_split=calibration_split,
            profile=profile,
            latency=latency,
            gpu_lease=gpu_lease,
            artifact_paths=MappingProxyType(artifact_paths),
            evaluation_artifact=path.resolve(),
        )


def _load_profile(
        path: Path, *, candidate_id: str,
        provenance: Mapping[str, str]) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot load profile artifact: {error}') from error
    fields = {
        'schema_version', 'git_commit', 'candidate', 'config', 'checkpoint',
        'checkpoint_sha256', 'input_shapes', 'output_shapes', 'parameters',
        'modules',
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MetricError('profile artifact has invalid fields')
    if value['schema_version'] != 1 or value['candidate'] != candidate_id:
        raise MetricError('profile artifact identity mismatch')
    if (
            value['git_commit'] != provenance['git_commit']
            or value['checkpoint_sha256'] != provenance['checkpoint_sha256']):
        raise MetricError('profile artifact provenance mismatch')
    parameters = value['parameters']
    if (
            not isinstance(parameters, Mapping)
            or set(parameters) != {
                'total', 'trainable', 'bytes_by_dtype', 'by_prefix'}
            or isinstance(parameters['total'], bool)
            or not isinstance(parameters['total'], int)
            or parameters['total'] < 0
            or isinstance(parameters['trainable'], bool)
            or not isinstance(parameters['trainable'], int)
            or not 0 <= parameters['trainable'] <= parameters['total']):
        raise MetricError('profile parameter summary is invalid')
    if not isinstance(value['modules'], list) or not value['modules']:
        raise MetricError('profile module inventory is invalid')
    return _freeze(value)


def _load_latency(
        path: Path, *, candidate_id: str, route: str,
        provenance: Mapping[str, str]
        ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        envelope = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot load latency artifact: {error}') from error
    if (
            not isinstance(envelope, Mapping)
            or set(envelope) != {
                'schema_version', 'candidate_id', 'stage', 'result'}
            or envelope['schema_version'] != 1
            or envelope['candidate_id'] != candidate_id
            or envelope['stage'] != 'latency'):
        raise MetricError('latency artifact envelope identity mismatch')
    result = envelope['result']
    if (
            not isinstance(result, Mapping)
            or set(result) != {
                'route', 'provenance', 'protocol', 'modes', 'gpu_lease'}
            or result['route'] != route):
        raise MetricError('latency artifact provenance or route mismatch')
    latency_provenance = validate_provenance(result['provenance'])
    shared_fields = {
        'checkpoint_sha256', 'data_inventory_sha256', 'git_commit'}
    if any(
            latency_provenance[field] != provenance[field]
            for field in shared_fields):
        raise MetricError('latency artifact provenance or route mismatch')
    protocol = result['protocol']
    if (
            not isinstance(protocol, Mapping)
            or protocol.get('batch_size') != 1
            or protocol.get('timer') != 'torch.cuda.Event'
            or protocol.get('synchronize') is not True
            or protocol.get('scope') != 'full_topdown_model'
            or isinstance(protocol.get('warmup'), bool)
            or not isinstance(protocol.get('warmup'), int)
            or protocol['warmup'] < 0
            or isinstance(protocol.get('iterations'), bool)
            or not isinstance(protocol.get('iterations'), int)
            or protocol['iterations'] <= 0):
        raise MetricError('latency protocol is invalid')
    modes = result['modes']
    summary_fields = {'median_ms', 'p90_ms', 'p95_ms', 'sample_count'}
    if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
        raise MetricError('latency must report flip and no_flip separately')
    for name, summary in modes.items():
        if (
                not isinstance(summary, Mapping)
                or set(summary) != summary_fields
                or summary['sample_count'] != protocol['iterations']
                or any(
                    not _finite_number(summary[field])
                    or float(summary[field]) < 0.0
                    for field in ('median_ms', 'p90_ms', 'p95_ms'))
                or not summary['median_ms'] <= summary['p90_ms'] <= summary['p95_ms']):
            raise MetricError(f'latency {name} summary is invalid')
    lease = result['gpu_lease']
    lease_fields = {
        'stage_id', 'pid', 'boot_id', 'timestamp', 'device_index',
        'allowed_pids',
    }
    if (
            not isinstance(lease, Mapping) or set(lease) != lease_fields
            or lease['stage_id'] != f'{candidate_id}:latency'
            or isinstance(lease['pid'], bool) or not isinstance(lease['pid'], int)
            or lease['pid'] <= 0):
        raise MetricError('latency GPU lease provenance is invalid')
    return _freeze(result), _freeze(lease)


def stage_envelope(
        candidate_id: str, stage: str, result: Mapping[str, Any]
        ) -> dict[str, Any]:
    """Return the strict non-profile artifact envelope consumed by Task 2."""
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('candidate_id must be non-empty')
    if not isinstance(stage, str) or not stage:
        raise ValueError('stage must be non-empty')
    return {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'stage': stage,
        'result': dict(result),
    }
