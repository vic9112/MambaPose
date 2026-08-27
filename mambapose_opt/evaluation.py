"""Strict COCO metric normalization and candidate-result loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from mambapose_opt.latency import LatencyError, validate_gpu_lease


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


_COCO_ANNOTATION = 'data/coco/annotations/person_keypoints_val2017.json'
_COCO_DETECTIONS = (
    'data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json')


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_coco_val_protocol(
        config: Mapping[str, Any], *, repository_root: Path,
        expected_image_count: int = 5000) -> dict[str, Any]:
    """Fail closed unless config and local assets prove official COCO val2017."""
    try:
        loader = config['test_dataloader']
        dataset = loader['dataset']
        sampler = loader['sampler']
        evaluator = config['test_evaluator']
    except (KeyError, TypeError) as error:
        raise MetricError('COCO val protocol configuration is incomplete') from error
    if dataset.get('type') != 'CocoDataset':
        raise MetricError('evaluation dataset must be CocoDataset')
    if dataset.get('data_mode') != 'topdown' or dataset.get('test_mode') is not True:
        raise MetricError('CocoDataset must be top-down and in test_mode')
    if (
            sampler.get('type') != 'DefaultSampler'
            or sampler.get('shuffle') is not False
            or sampler.get('round_up') is not False
            or loader.get('drop_last') is not False):
        raise MetricError('COCO val sampler must be complete and nonshuffling')
    data_root = Path(str(dataset.get('data_root', '')))
    annotation_relative = (data_root / str(dataset.get('ann_file', ''))).as_posix()
    if annotation_relative != _COCO_ANNOTATION:
        raise MetricError('evaluation annotation must be COCO val2017 keypoints')
    if dataset.get('bbox_file') != _COCO_DETECTIONS:
        raise MetricError('evaluation must use official COCO val detection boxes')
    if (
            evaluator.get('type') != 'CocoMetric'
            or evaluator.get('ann_file') != _COCO_ANNOTATION):
        raise MetricError('evaluation must use CocoMetric on COCO val2017')
    image_prefix = dataset.get('data_prefix')
    if not isinstance(image_prefix, Mapping) or image_prefix.get('img') != 'val2017/':
        raise MetricError('CocoDataset image prefix must be val2017')

    root = Path(repository_root).resolve()
    annotation_path = root / _COCO_ANNOTATION
    detection_path = root / _COCO_DETECTIONS
    inventory_path = root / 'data/inventory.json'
    try:
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        detections = json.loads(detection_path.read_text(encoding='utf-8'))
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot verify COCO val assets: {error}') from error
    images = annotation.get('images') if isinstance(annotation, Mapping) else None
    annotations = annotation.get('annotations') if isinstance(annotation, Mapping) else None
    if (
            not isinstance(images, list) or len(images) != expected_image_count
            or not isinstance(annotations, list)):
        raise MetricError(
            f'COCO val annotation must contain exactly {expected_image_count} images')
    ids: set[int] = set()
    image_dir = root / data_root / 'val2017'
    for row in images:
        if (
                not isinstance(row, Mapping)
                or isinstance(row.get('id'), bool)
                or not isinstance(row.get('id'), int)
                or row['id'] in ids
                or not isinstance(row.get('file_name'), str)
                or not (image_dir / row['file_name']).is_file()):
            raise MetricError('COCO val image inventory is incomplete or invalid')
        ids.add(row['id'])
    if (
            not isinstance(detections, list) or not detections
            or any(
                not isinstance(row, Mapping) or row.get('image_id') not in ids
                for row in detections)):
        raise MetricError('COCO val detections are empty or reference unknown images')
    assets = inventory.get('assets') if isinstance(inventory, Mapping) else None
    entries = [row for row in assets or [] if (
        isinstance(row, Mapping) and row.get('id') == 'coco-val-detections')]
    if len(entries) != 1 or entries[0].get('path') != _COCO_DETECTIONS:
        raise MetricError('data inventory lacks the official COCO val detections')
    detection_sha256 = _file_sha256(detection_path)
    if entries[0].get('sha256') != detection_sha256:
        raise MetricError('data inventory detection hash does not match actual asset')
    return {
        'dataset': 'coco', 'split': 'val2017', 'complete_split': True,
        'annotation_sha256': _file_sha256(annotation_path),
        'detection_sha256': detection_sha256,
        'inventory_detection_sha256': entries[0]['sha256'],
        'annotation_image_count': len(images),
        'annotation_record_count': len(annotations),
        'detection_record_count': len(detections),
        'verified_image_count': len(images),
    }


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
        'worker_id', 'torch_seed_source', 'python_seed_derivation',
        'numpy_seed_derivation'}
    if any(
            not isinstance(worker, Mapping) or set(worker) != worker_fields
            or isinstance(worker['worker_id'], bool)
            or not isinstance(worker['worker_id'], int)
            or worker['worker_id'] < 0
            or worker['torch_seed_source'] != 'torch.initial_seed()'
            or worker['python_seed_derivation'] != 'torch_seed % 2**32'
            or worker['numpy_seed_derivation'] != 'torch_seed % 2**32'
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


def _validate_recorded_coco_protocol(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricError('evaluation protocol must be an object')
    if (
            value.get('dataset') != 'coco'
            or value.get('split') != 'val2017'
            or value.get('complete_split') is not True
            or isinstance(value.get('batch_size'), bool)
            or not isinstance(value.get('batch_size'), int)
            or value['batch_size'] <= 0):
        raise MetricError('evaluation protocol is not complete COCO val2017')
    for name in (
            'annotation_sha256', 'detection_sha256',
            'inventory_detection_sha256'):
        if not isinstance(value.get(name), str) or not _SHA256.fullmatch(
                value[name]):
            raise MetricError(f'evaluation protocol {name} is invalid')
    if value['detection_sha256'] != value['inventory_detection_sha256']:
        raise MetricError('evaluation detection hash disagrees with inventory')
    counts = (
        value.get('annotation_image_count'),
        value.get('annotation_record_count'),
        value.get('detection_record_count'),
        value.get('verified_image_count'))
    if (
            any(isinstance(item, bool) or not isinstance(item, int)
                or item < 0 for item in counts)
            or value['annotation_image_count'] != 5000
            or value['verified_image_count'] != 5000
            or value['detection_record_count'] == 0):
        raise MetricError('evaluation protocol COCO asset counts are invalid')
    for name in ('source_config', 'checkpoint', 'data_inventory'):
        item = value.get(name)
        if (
                not isinstance(item, str) or not item
                or Path(item).is_absolute()
                or any(part in {'.', '..'} for part in Path(item).parts)):
            raise MetricError(f'evaluation protocol {name} path is invalid')
    if value['data_inventory'] != 'data/inventory.json':
        raise MetricError('evaluation protocol data inventory path is invalid')
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
    def from_artifacts(
            cls, root: Path | str, *, mode: str = 'flip') -> 'CandidateResult':
        root = Path(root)
        if root.is_file() or not root.is_dir():
            raise MetricError('candidate artifact root must be a directory')
        if mode not in {'flip', 'no_flip'}:
            raise MetricError('evaluation mode must be flip or no_flip')
        path = root / 'evaluate/evaluate.json'
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
        result_fields = {'route', 'calibration_split', 'modes'}
        if not isinstance(result, Mapping) or set(result) != result_fields:
            raise MetricError('evaluation result has invalid fields')
        route = result['route']
        if not isinstance(route, str) or not route:
            raise MetricError('evaluation route must be non-empty')
        calibration_split = result['calibration_split']
        if calibration_split not in {None, 'train2017'}:
            raise MetricError(
                'calibration_split must be absent or train2017, never val2017')
        modes = result['modes']
        if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
            raise MetricError('evaluation must contain both flip and no_flip modes')
        normalized_modes: dict[str, tuple[CocoMetrics, dict[str, str],
                                          Mapping[str, Any], Mapping[str, Any]]] = {}
        mode_fields = {'metrics', 'provenance', 'determinism', 'protocol'}
        for name, row in modes.items():
            if not isinstance(row, Mapping) or set(row) != mode_fields:
                raise MetricError(f'evaluation {name} mode has invalid fields')
            row_provenance = validate_provenance(row['provenance'])
            row_determinism = _validate_determinism(
                row['determinism'], row_provenance)
            row_protocol = _validate_recorded_coco_protocol(row['protocol'])
            normalized_modes[name] = (
                CocoMetrics.from_dict(row['metrics']), row_provenance,
                row_determinism, row_protocol)
        for field in ('checkpoint_sha256', 'data_inventory_sha256', 'git_commit'):
            if normalized_modes['flip'][1][field] != normalized_modes['no_flip'][1][field]:
                raise MetricError(f'evaluation mode provenance disagrees on {field}')
        for field in (
                'annotation_sha256', 'detection_sha256',
                'inventory_detection_sha256', 'annotation_image_count',
                'annotation_record_count', 'detection_record_count',
                'verified_image_count', 'source_config', 'checkpoint',
                'data_inventory'):
            if normalized_modes['flip'][3][field] != normalized_modes['no_flip'][3][field]:
                raise MetricError(f'evaluation mode protocol disagrees on {field}')
        metrics, provenance, determinism, protocol = normalized_modes[mode]
        profile: Mapping[str, Any] | None = None
        latency: Mapping[str, Any] | None = None
        gpu_lease: Mapping[str, Any] | None = None
        artifact_paths: dict[str, Path] = {'evaluation': path.resolve()}
        profile_path = root / 'profile/profile.json'
        latency_path = root / 'latency/latency.json'
        profile = _load_profile(
            profile_path, candidate_id=candidate_id,
            provenance=provenance)
        latency, gpu_lease = _load_latency(
            latency_path, candidate_id=candidate_id, route=route,
            provenance=provenance)
        if (
                profile['config'] != protocol['source_config']
                or profile['checkpoint'] != protocol['checkpoint']):
            raise MetricError('profile paths disagree with evaluation provenance')
        latency_protocol = latency['protocol']
        for field in ('source_config', 'checkpoint', 'data_inventory'):
            if latency_protocol.get(field) != protocol[field]:
                raise MetricError(
                    f'latency {field} disagrees with evaluation provenance')
        for field in (
                'annotation_sha256', 'detection_sha256',
                'inventory_detection_sha256', 'annotation_image_count',
                'annotation_record_count', 'detection_record_count',
                'verified_image_count'):
            if latency_protocol['data'][field] != protocol[field]:
                raise MetricError(
                    f'latency data {field} disagrees with evaluation provenance')
        artifact_paths.update({
            'profile': profile_path.resolve(),
            'latency': latency_path.resolve(),
        })
        try:
            for artifact in artifact_paths.values():
                artifact.relative_to(root.resolve())
        except ValueError as error:
            raise MetricError('candidate artifact path escapes its directory') from error
        return cls(
            candidate_id=candidate_id,
            route=route,
            metrics=metrics,
            flip_test=mode == 'flip',
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
    def valid_count(item: object) -> bool:
        return (
            isinstance(item, int) and not isinstance(item, bool) and item >= 0)

    def valid_count_map(item: object) -> bool:
        return (
            isinstance(item, Mapping) and bool(item)
            and all(isinstance(key, str) and bool(key) and valid_count(count)
                    for key, count in item.items()))

    def valid_shape(item: object) -> bool:
        if isinstance(item, list):
            if not item:
                return False
            if all(valid_count(dimension) and dimension > 0 for dimension in item):
                return True
            return all(valid_shape(child) for child in item)
        if isinstance(item, Mapping):
            return bool(item) and all(
                isinstance(key, str) and valid_shape(child)
                for key, child in item.items())
        return isinstance(item, str) and bool(item)

    if (
            not isinstance(parameters, Mapping)
            or set(parameters) != {
                'total', 'trainable', 'bytes_by_dtype', 'by_prefix'}
            or not valid_count(parameters['total'])
            or not valid_count(parameters['trainable'])
            or not 0 <= parameters['trainable'] <= parameters['total']):
        raise MetricError('profile parameter summary is invalid')
    if (
            not valid_count_map(parameters['bytes_by_dtype'])
            or not valid_count_map(parameters['by_prefix'])
            or not valid_shape(value['input_shapes'])
            or not valid_shape(value['output_shapes'])):
        raise MetricError('profile shape or parameter inventory is invalid')
    if not isinstance(value['modules'], list) or not value['modules']:
        raise MetricError('profile module inventory is invalid')
    if not all(
            isinstance(record, Mapping)
            and set(record) == {'name', 'kind', 'parameters', 'hazard'}
            and isinstance(record['name'], str)
            and isinstance(record['kind'], str) and bool(record['kind'])
            and isinstance(record['parameters'], int)
            and not isinstance(record['parameters'], bool)
            and record['parameters'] >= 0
            and (record['hazard'] is None or isinstance(record['hazard'], str))
            for record in value['modules']):
        raise MetricError('profile module inventory is invalid')
    if not isinstance(value['config'], str) or not value['config']:
        raise MetricError('profile config identity is invalid')
    if not isinstance(value['checkpoint'], str) or not value['checkpoint']:
        raise MetricError('profile checkpoint identity is invalid')
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
    protocol_fields = {
        'batch_size', 'warmup', 'iterations', 'timer', 'synchronize',
        'scope', 'source_config', 'checkpoint', 'data_inventory', 'data'}
    if (
            not isinstance(protocol, Mapping)
            or set(protocol) != protocol_fields
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
    latency_data = protocol['data']
    if not isinstance(latency_data, Mapping):
        raise MetricError('latency data protocol is invalid')
    _validate_recorded_coco_protocol({
        **latency_data,
        'batch_size': protocol['batch_size'],
        'source_config': protocol['source_config'],
        'checkpoint': protocol['checkpoint'],
        'data_inventory': protocol['data_inventory'],
    })
    modes = result['modes']
    summary_fields = {'median_ms', 'p90_ms', 'p95_ms', 'sample_count'}
    if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
        raise MetricError('latency must report flip and no_flip separately')
    for name, summary in modes.items():
        if (
                not isinstance(summary, Mapping)
                or set(summary) != summary_fields
                or isinstance(summary['sample_count'], bool)
                or not isinstance(summary['sample_count'], int)
                or summary['sample_count'] != protocol['iterations']
                or any(
                    not _finite_number(summary[field])
                    or float(summary[field]) < 0.0
                    for field in ('median_ms', 'p90_ms', 'p95_ms'))
                or not summary['median_ms'] <= summary['p90_ms'] <= summary['p95_ms']):
            raise MetricError(f'latency {name} summary is invalid')
    try:
        lease = validate_gpu_lease(result['gpu_lease'])
    except LatencyError as error:
        raise MetricError(f'latency GPU lease provenance is invalid: {error}') from error
    if lease['stage_id'] != f'{candidate_id}:latency':
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
