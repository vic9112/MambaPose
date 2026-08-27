"""Strict one-real-batch evidence harness for MambaPose distillation."""

from __future__ import annotations

from contextlib import suppress
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.evaluation import resolve_project_asset_root
from mambapose_opt.gpu_guard import (
    GpuLease, canonical_gpu_lock_path, exclusive_cuda_stage)
from mambapose_opt.latency import validate_gpu_lease
from mambapose_opt.source import (
    clean_git_commit, sha256_file, tracked_file_binding)


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')
_CONFIG_PREFIX = ('configs', 'optimization', 'accuracy_first')
_OUTPUT_PREFIX = ('work_dirs', 'optimization', 'accuracy-first')
_CORPUS_ALGORITHM = 'sha256-zip-member-and-extracted-content-v1'
_ARTIFACT_KIND = 'mambapose-real-batch-distillation-smoke'


@dataclass(frozen=True)
class TrainAssetBinding:
    inventory_path: str
    inventory_sha256: str
    image_archive_path: str
    image_archive_sha256: str
    annotation_archive_path: str
    annotation_archive_sha256: str
    annotation_path: str
    annotation_sha256: str
    corpus_sha256: str
    image_count: int


@dataclass(frozen=True)
class SmokePreflight:
    repository_root: Path
    asset_root: Path
    config_path: Path
    source: Mapping[str, Any]
    resolved_config_sha256: str
    experiment_id: str
    teacher_checkpoint: Path
    teacher_checkpoint_relative: str
    teacher_checkpoint_sha256: str
    student_checkpoint: Path
    student_checkpoint_relative: str
    student_checkpoint_sha256: str
    train_assets: TrainAssetBinding
    original_batch_size: int
    dataloader_override_sha256: str


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _mapping_fields(value: object, expected: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f'{label} has invalid fields')
    return value


def _relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} path is invalid')
    path = Path(value)
    if path.is_absolute() or any(part in {'.', '..'} for part in path.parts):
        raise ValueError(f'{label} path is invalid')
    return path


def _hex(value: object, *, label: str, commit: bool = False) -> str:
    pattern = _COMMIT if commit else _SHA256
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f'{label} is invalid')
    return value


def _file_binding(value: object, *, label: str) -> tuple[Path, str]:
    row = _mapping_fields(value, {'path', 'sha256'}, label)
    return (
        _relative_path(row['path'], label=label),
        _hex(row['sha256'], label=f'{label} sha256'),
    )


def _safe_asset_file(root: Path, value: object, *, label: str) -> Path:
    relative = _relative_path(value, label=label)
    lexical = root / relative
    try:
        effective = lexical.resolve(strict=True)
        effective.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} escapes the project asset root') from error
    if lexical.is_symlink() or not effective.is_file():
        raise ValueError(f'{label} must be a regular non-symlink file')
    return effective


def _authority_fields(value: object) -> Mapping:
    return _mapping_fields(value, {
        'inventory_path', 'inventory_sha256',
        'image_archive_path', 'image_archive_sha256',
        'image_inventory_asset_id', 'image_prefix', 'image_count',
        'annotation_archive_path', 'annotation_archive_sha256',
        'annotation_inventory_asset_id', 'annotation_path',
        'annotation_member',
    }, 'smoke_data_authority')


def _validate_train_dataset(dataset: object) -> Mapping:
    if not isinstance(dataset, Mapping):
        raise ValueError('train dataset config must be a mapping')
    prefix = dataset.get('data_prefix')
    pipeline = dataset.get('pipeline')
    if (
            dataset.get('type') != 'CocoDataset'
            or dataset.get('data_root') not in {'data/coco', 'data/coco/'}
            or dataset.get('data_mode') != 'topdown'
            or dataset.get('ann_file') != (
                'annotations/person_keypoints_train2017.json')
            or not isinstance(prefix, Mapping)
            or prefix.get('img') not in {'train2017', 'train2017/'}
            or not isinstance(pipeline, Sequence) or not pipeline
            or not isinstance(pipeline[-1], Mapping)
            or pipeline[-1].get('type') != 'PackPoseInputs'):
        raise ValueError(
            'smoke requires the production packed COCO train2017 pipeline')
    return dataset


def smoke_dataloader_config(source: Mapping) -> Mapping:
    """Clone the production pipeline with a one-sample local smoke loader."""
    if not isinstance(source, Mapping):
        raise TypeError('train_dataloader must be a mapping')
    _validate_train_dataset(source.get('dataset'))
    result = copy.deepcopy(source)
    result['batch_size'] = 1
    result['num_workers'] = 0
    result['persistent_workers'] = False
    result['sampler'] = dict(type='DefaultSampler', shuffle=False)
    result['drop_last'] = False
    result.pop('batch_sampler', None)
    result.pop('worker_init_fn', None)
    return result


def _inventory_asset(
        inventory: object, identifier: str, *, expected_path: str,
        expected_sha256: str) -> Mapping:
    if not isinstance(inventory, Mapping) or inventory.get('schema_version') != 1:
        raise ValueError('data inventory is malformed')
    assets = inventory.get('assets')
    if not isinstance(assets, list):
        raise ValueError('data inventory assets are malformed')
    matches = [row for row in assets
               if isinstance(row, Mapping) and row.get('id') == identifier]
    if len(matches) != 1:
        raise ValueError(f'data inventory must contain one {identifier!r} asset')
    row = matches[0]
    if (
            row.get('path') != expected_path
            or row.get('sha256') != expected_sha256
            or not isinstance(row.get('required_paths'), list)
            or not row['required_paths']):
        raise ValueError(f'data inventory {identifier!r} binding is invalid')
    return row


def _compare_streams_and_hash(first, second, digest: hashlib._Hash) -> None:
    while True:
        left = first.read(1024 * 1024)
        right = second.read(1024 * 1024)
        if left != right:
            raise ValueError('extracted train2017 corpus differs from archive')
        if not left:
            return
        digest.update(left)


def _verify_image_corpus(
        archive_path: Path, image_root: Path, *, prefix: str,
        expected_count: int) -> str:
    if prefix != 'train2017/':
        raise ValueError('image_prefix must be train2017/')
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            members = [item for item in members if item.filename.startswith(prefix)]
            names = [item.filename for item in members]
            if (
                    len(names) != len(set(names))
                    or any(
                        not name.endswith('.jpg')
                        or Path(name).is_absolute()
                        or any(part in {'.', '..'} for part in Path(name).parts)
                        for name in names)):
                raise ValueError('train2017 archive members are invalid')
            if len(members) != expected_count:
                raise ValueError('train2017 archive image count is invalid')
            extracted = {
                f'{prefix}{path.relative_to(image_root).as_posix()}'
                for path in image_root.rglob('*.jpg') if path.is_file()
            }
            if extracted != set(names):
                raise ValueError('extracted train2017 corpus file set is invalid')

            digest = hashlib.sha256()
            for item in sorted(members, key=lambda row: row.filename):
                target = image_root / Path(item.filename).relative_to(prefix)
                if target.is_symlink() or not target.is_file():
                    raise ValueError(
                        'extracted train2017 corpus contains unsafe images')
                digest.update(item.filename.encode('utf-8'))
                digest.update(b'\0')
                digest.update(str(item.file_size).encode('ascii'))
                digest.update(b'\0')
                if target.stat().st_size != item.file_size:
                    raise ValueError(
                        'extracted train2017 corpus differs from archive')
                with archive.open(item) as packed, target.open('rb') as unpacked:
                    _compare_streams_and_hash(packed, unpacked, digest)
            return digest.hexdigest()
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f'cannot verify train2017 archive: {error}') from error


def _verify_annotation(
        archive_path: Path, annotation_path: Path, *, member: str) -> str:
    relative = _relative_path(member, label='annotation_member')
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = [row for row in archive.infolist()
                     if not row.is_dir() and row.filename == relative.as_posix()]
            if len(infos) != 1:
                raise ValueError('annotation archive member is missing or duplicate')
            digest = hashlib.sha256()
            with (
                    archive.open(infos[0]) as packed,
                    annotation_path.open('rb') as unpacked):
                while True:
                    left = packed.read(1024 * 1024)
                    right = unpacked.read(1024 * 1024)
                    if left != right:
                        raise ValueError(
                            'extracted train2017 annotation differs from archive')
                    if not left:
                        break
                    digest.update(left)
            return digest.hexdigest()
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f'cannot verify train2017 annotation: {error}') from error


def validate_coco_train_assets_at_root(
        authority: object, dataset: object, asset_root: Path,
        ) -> TrainAssetBinding:
    """Bind every extracted COCO train image and annotation to official zips."""
    row = _authority_fields(authority)
    _validate_train_dataset(dataset)
    root = Path(asset_root).resolve(strict=True)
    count = row['image_count']
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('image_count must be a positive integer')
    for field in (
            'inventory_sha256', 'image_archive_sha256',
            'annotation_archive_sha256'):
        _hex(row[field], label=field)
    for field in ('image_inventory_asset_id', 'annotation_inventory_asset_id'):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f'{field} is invalid')

    inventory_path = _safe_asset_file(
        root, row['inventory_path'], label='inventory')
    image_archive = _safe_asset_file(
        root, row['image_archive_path'], label='image archive')
    annotation_archive = _safe_asset_file(
        root, row['annotation_archive_path'], label='annotation archive')
    annotation_path = _safe_asset_file(
        root, row['annotation_path'], label='annotation')
    observed = {
        'inventory': sha256_file(inventory_path),
        'image': sha256_file(image_archive),
        'annotation_archive': sha256_file(annotation_archive),
    }
    if observed['inventory'] != row['inventory_sha256']:
        raise ValueError('data inventory sha256 mismatch')
    if observed['image'] != row['image_archive_sha256']:
        raise ValueError('train2017 image archive sha256 mismatch')
    if observed['annotation_archive'] != row['annotation_archive_sha256']:
        raise ValueError('train2017 annotation archive sha256 mismatch')
    try:
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('data inventory is not valid JSON') from error
    _inventory_asset(
        inventory, row['image_inventory_asset_id'],
        expected_path=row['image_archive_path'],
        expected_sha256=row['image_archive_sha256'])
    _inventory_asset(
        inventory, row['annotation_inventory_asset_id'],
        expected_path=row['annotation_archive_path'],
        expected_sha256=row['annotation_archive_sha256'])

    image_root = root / 'data/coco/train2017'
    if image_root.is_symlink() or not image_root.is_dir():
        raise ValueError('extracted train2017 image root is invalid')
    corpus_sha256 = _verify_image_corpus(
        image_archive, image_root, prefix=row['image_prefix'],
        expected_count=count)
    annotation_sha256 = _verify_annotation(
        annotation_archive, annotation_path, member=row['annotation_member'])
    return TrainAssetBinding(
        inventory_path=row['inventory_path'],
        inventory_sha256=observed['inventory'],
        image_archive_path=row['image_archive_path'],
        image_archive_sha256=observed['image'],
        annotation_archive_path=row['annotation_archive_path'],
        annotation_archive_sha256=observed['annotation_archive'],
        annotation_path=row['annotation_path'],
        annotation_sha256=annotation_sha256,
        corpus_sha256=corpus_sha256,
        image_count=count,
    )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item)
                for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'config contains non-JSON value: {type(value).__name__}')


def _checkpoint_path(
        repository_root: Path, asset_root: Path, value: object, *, label: str,
        ) -> tuple[Path, str]:
    relative = _relative_path(value, label=label)
    if relative.parts[:2] != ('work_dirs', 'reproduction'):
        raise ValueError(f'{label} must be under work_dirs/reproduction')
    lexical = repository_root / relative
    try:
        effective = lexical.resolve(strict=True)
        effective.relative_to(asset_root / 'work_dirs/reproduction')
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} escapes reproduction artifacts') from error
    if not effective.is_file():
        raise ValueError(f'{label} is not a file')
    return effective, relative.as_posix()


def build_smoke_preflight(
        repository_root: Path, config_path: Path) -> SmokePreflight:
    """Validate immutable source, checkpoints, and all train data before CUDA."""
    root = Path(repository_root).resolve(strict=True)
    config_relative = _relative_path(config_path.as_posix(), label='config')
    if config_relative.parts[:3] != _CONFIG_PREFIX:
        raise ValueError('config must be under configs/optimization/accuracy_first')
    commit = clean_git_commit(root)
    config_binding = tracked_file_binding(
        root, config_relative, git_commit=commit)

    from mmengine.config import Config

    config = Config.fromfile(root / config_relative)
    model = config.get('model')
    if not isinstance(model, Mapping) or model.get('type') != (
            'MambaPoseHeatmapDistiller'):
        raise ValueError('config must build MambaPoseHeatmapDistiller')
    experiment_id = config.get('experiment_id')
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError('distillation config experiment_id is required')
    asset_root = resolve_project_asset_root(root)
    teacher, teacher_relative = _checkpoint_path(
        root, asset_root, model.get('teacher_checkpoint'),
        label='teacher checkpoint')
    student, student_relative = _checkpoint_path(
        root, asset_root, model.get('student_checkpoint'),
        label='student checkpoint')
    teacher_expected = _hex(
        model.get('teacher_checkpoint_sha256'),
        label='teacher checkpoint sha256')
    student_expected = _hex(
        model.get('student_checkpoint_sha256'),
        label='student checkpoint sha256')
    teacher_actual = sha256_file(teacher)
    student_actual = sha256_file(student)
    if teacher_actual != teacher_expected:
        raise ValueError('teacher checkpoint sha256 mismatch')
    if student_actual != student_expected:
        raise ValueError('student checkpoint sha256 mismatch')

    loader = config.get('train_dataloader')
    if not isinstance(loader, Mapping):
        raise ValueError('train_dataloader is missing')
    original_batch = loader.get('batch_size')
    if (
            isinstance(original_batch, bool)
            or not isinstance(original_batch, int)
            or original_batch <= 0):
        raise ValueError('production train batch_size is invalid')
    smoke_loader = smoke_dataloader_config(loader)
    assets = validate_coco_train_assets_at_root(
        config.get('smoke_data_authority'), loader.get('dataset'), asset_root)
    resolved = canonical_json_sha256(_canonicalize(config.to_dict()))
    override = canonical_json_sha256(_canonicalize(smoke_loader))
    return SmokePreflight(
        repository_root=root,
        asset_root=asset_root,
        config_path=config_relative,
        source={'git_commit': commit, 'config': config_binding},
        resolved_config_sha256=resolved,
        experiment_id=experiment_id,
        teacher_checkpoint=teacher,
        teacher_checkpoint_relative=teacher_relative,
        teacher_checkpoint_sha256=teacher_actual,
        student_checkpoint=student,
        student_checkpoint_relative=student_relative,
        student_checkpoint_sha256=student_actual,
        train_assets=assets,
        original_batch_size=original_batch,
        dataloader_override_sha256=override,
    )


def _validate_smoke_schema(value: object) -> Mapping:
    top = _mapping_fields(value, {
        'schema_version', 'kind', 'claims', 'execution', 'source', 'inputs',
        'data', 'gpu', 'checks', 'artifacts'}, 'smoke artifact')
    if top.get('schema_version') != 1 or top.get('kind') != _ARTIFACT_KIND:
        raise ValueError('smoke artifact identity is invalid')
    claims = _mapping_fields(
        top['claims'], {'coco_ap', 'latency', 'training_batch_vram'},
        'smoke claims')
    if claims != {
            'coco_ap': False, 'latency': False,
            'training_batch_vram': False}:
        raise ValueError('smoke artifact must not claim AP, latency, or training VRAM')
    execution = _mapping_fields(top['execution'], {
        'python_seed', 'numpy_seed', 'torch_seed',
        'deterministic_algorithms', 'python_dont_write_bytecode',
        'vram_scope'}, 'smoke execution')
    if execution != {
            'python_seed': 0,
            'numpy_seed': 0,
            'torch_seed': 0,
            'deterministic_algorithms': True,
            'python_dont_write_bytecode': True,
            'vram_scope': 'batch-1-smoke-not-training-batch'}:
        raise ValueError('smoke deterministic execution contract is invalid')

    source = _mapping_fields(
        top['source'], {'git_commit', 'config', 'resolved_config_sha256'},
        'smoke source')
    _hex(source['git_commit'], label='source git_commit', commit=True)
    _file_binding(source['config'], label='source config')
    _hex(source['resolved_config_sha256'], label='resolved config sha256')
    inputs = _mapping_fields(
        top['inputs'], {'teacher_checkpoint', 'student_checkpoint'},
        'smoke inputs')
    _file_binding(inputs['teacher_checkpoint'], label='teacher checkpoint')
    _file_binding(inputs['student_checkpoint'], label='student checkpoint')

    data = _mapping_fields(top['data'], {
        'dataset', 'split', 'pipeline_terminal', 'original_batch_size',
        'smoke_batch_size', 'smoke_num_workers',
        'smoke_persistent_workers', 'inventory', 'image_archive',
        'annotation_archive', 'annotation', 'corpus_digest_algorithm',
        'corpus_sha256', 'corpus_image_count',
        'dataloader_override_sha256', 'batch'}, 'smoke data')
    if (
            data['dataset'] != 'coco' or data['split'] != 'train2017'
            or data['pipeline_terminal'] != 'PackPoseInputs'
            or data['smoke_batch_size'] != 1
            or data['smoke_num_workers'] != 0
            or data['smoke_persistent_workers'] is not False
            or data['corpus_digest_algorithm'] != _CORPUS_ALGORITHM):
        raise ValueError('smoke data protocol is invalid')
    for field in ('original_batch_size', 'corpus_image_count'):
        if (
                isinstance(data[field], bool) or not isinstance(data[field], int)
                or data[field] <= 0):
            raise ValueError(f'{field} is invalid')
    for field in ('corpus_sha256', 'dataloader_override_sha256'):
        _hex(data[field], label=field)
    for field in ('inventory', 'image_archive', 'annotation_archive', 'annotation'):
        _file_binding(data[field], label=field)
    batch = _mapping_fields(data['batch'], {
        'size', 'image_ids', 'image_paths', 'image_sha256', 'packed_sha256'},
        'smoke batch')
    if batch['size'] != 1:
        raise ValueError('smoke batch size must be exactly one')
    ids = batch['image_ids']
    paths = batch['image_paths']
    hashes = batch['image_sha256']
    if (
            not isinstance(ids, list) or len(ids) != 1
            or isinstance(ids[0], bool) or not isinstance(ids[0], int)
            or ids[0] < 0
            or not isinstance(paths, list) or len(paths) != 1
            or not isinstance(hashes, list) or len(hashes) != 1):
        raise ValueError('smoke batch identity is invalid')
    image_path = _relative_path(paths[0], label='batch image')
    if image_path.parts[:3] != ('data', 'coco', 'train2017'):
        raise ValueError('batch image must be from COCO train2017')
    _hex(hashes[0], label='batch image sha256')
    _hex(batch['packed_sha256'], label='packed batch sha256')

    gpu = _mapping_fields(
        top['gpu'], {'device_index', 'lease', 'lease_sha256'}, 'smoke GPU')
    if (
            isinstance(gpu['device_index'], bool)
            or not isinstance(gpu['device_index'], int)
            or gpu['device_index'] < 0):
        raise ValueError('smoke GPU device_index is invalid')
    lease = validate_gpu_lease(gpu['lease'])
    if (
            lease['device_index'] != gpu['device_index']
            or not lease['stage_id'].startswith('distill-smoke:')):
        raise ValueError('smoke GPU lease identity is invalid')
    lease_hash = _hex(gpu['lease_sha256'], label='GPU lease sha256')
    if canonical_json_sha256(lease) != lease_hash:
        raise ValueError('GPU lease sha256 mismatch')

    checks = _mapping_fields(top['checks'], {
        'losses', 'teacher_frozen', 'teacher_gradient_tensors',
        'student_gradient_tensors', 'student_gradient_elements',
        'student_gradients_finite', 'peak_allocated_bytes',
        'peak_reserved_bytes', 'heatmap_shape',
        'reference_heatmaps_sha256', 'full_restore_heatmaps_sha256',
        'export_restore_heatmaps_sha256', 'full_restore_exact',
        'export_restore_exact'}, 'smoke checks')
    losses = _mapping_fields(checks['losses'], {
        'loss_kpt', 'loss_heatmap_distill', 'heatmap_distill_mse'},
        'smoke losses')
    if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in losses.values()):
        raise ValueError('smoke losses must be finite numbers')
    if (
            checks['teacher_frozen'] is not True
            or checks['teacher_gradient_tensors'] != 0
            or checks['student_gradients_finite'] is not True
            or checks['full_restore_exact'] is not True
            or checks['export_restore_exact'] is not True):
        raise ValueError('smoke model checks did not pass')
    for field in (
            'student_gradient_tensors', 'student_gradient_elements',
            'peak_allocated_bytes', 'peak_reserved_bytes'):
        item = checks[field]
        if (
                isinstance(item, bool) or not isinstance(item, int)
                or item <= 0):
            raise ValueError(f'{field} must be a positive integer')
    shape = checks['heatmap_shape']
    if shape != [1, 17, 64, 48]:
        raise ValueError('smoke heatmap shape is invalid')
    heatmap_hashes = (
        checks['reference_heatmaps_sha256'],
        checks['full_restore_heatmaps_sha256'],
        checks['export_restore_heatmaps_sha256'])
    for index, digest in enumerate(heatmap_hashes):
        _hex(digest, label=f'heatmap sha256 {index}')
    if len(set(heatmap_hashes)) != 1:
        raise ValueError('restored heatmaps do not match reference heatmaps')

    artifacts = _mapping_fields(
        top['artifacts'], {'distiller_checkpoint', 'student_checkpoint'},
        'smoke artifacts')
    _file_binding(artifacts['distiller_checkpoint'], label='distiller artifact')
    _file_binding(artifacts['student_checkpoint'], label='student artifact')
    return top


def _verify_bound_file(
        repository_root: Path, binding: object, *, label: str,
        artifact_root: Path | None = None,
        logical_artifact_root: Path | None = None) -> None:
    relative, expected = _file_binding(binding, label=label)
    if artifact_root is not None and relative.parts[:3] == _OUTPUT_PREFIX:
        logical = logical_artifact_root or artifact_root
        try:
            expected_parent = logical.resolve(strict=False).relative_to(
                repository_root.resolve(strict=True))
        except ValueError as error:
            raise ValueError(f'{label} artifact path is invalid') from error
        if relative.parent != expected_parent:
            raise ValueError(f'{label} artifact path is invalid')
        effective = artifact_root / relative.name
    else:
        effective = repository_root / relative
    try:
        resolved = effective.resolve(strict=True)
        if artifact_root is not None and relative.parts[:3] == _OUTPUT_PREFIX:
            resolved.relative_to(artifact_root.resolve(strict=True))
        else:
            try:
                resolved.relative_to(repository_root.resolve(strict=True))
            except ValueError:
                asset_root = resolve_project_asset_root(repository_root)
                resolved.relative_to(asset_root)
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} bound path is invalid') from error
    if effective.is_symlink() or not resolved.is_file():
        raise ValueError(f'{label} bound path must be a regular file')
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f'{label} sha256 mismatch')


def _verify_artifact_files(
        value: Mapping, *, repository_root: Path,
        physical_artifact_root: Path | None = None,
        logical_artifact_root: Path | None = None) -> None:
    source_binding = tracked_file_binding(
        repository_root, value['source']['config']['path'],
        git_commit=value['source']['git_commit'])
    if source_binding != value['source']['config']:
        raise ValueError('source config does not match its clean commit')
    for field in ('config',):
        _verify_bound_file(
            repository_root, value['source'][field], label=f'source {field}')
    for field in ('teacher_checkpoint', 'student_checkpoint'):
        _verify_bound_file(
            repository_root, value['inputs'][field], label=field)
    for field in ('inventory', 'image_archive', 'annotation_archive', 'annotation'):
        _verify_bound_file(repository_root, value['data'][field], label=field)
    for field in ('distiller_checkpoint', 'student_checkpoint'):
        _verify_bound_file(
            repository_root, value['artifacts'][field], label=f'artifact {field}',
            artifact_root=physical_artifact_root,
            logical_artifact_root=logical_artifact_root)
    image_path = repository_root / value['data']['batch']['image_paths'][0]
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError('batch image bound path is invalid')
    if sha256_file(image_path) != value['data']['batch']['image_sha256'][0]:
        raise ValueError('batch image sha256 mismatch')
    annotation_path = repository_root / value['data']['annotation']['path']
    try:
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('train2017 annotation is not valid JSON') from error
    images = annotation.get('images') if isinstance(annotation, Mapping) else None
    image_id = value['data']['batch']['image_ids'][0]
    file_name = Path(value['data']['batch']['image_paths'][0]).name
    matches = [row for row in images or ()
               if isinstance(row, Mapping) and row.get('id') == image_id]
    if len(matches) != 1 or matches[0].get('file_name') != file_name:
        raise ValueError('batch image identity disagrees with train2017 annotation')


def load_smoke_artifact(
        path: Path | str, *, repository_root: Path) -> dict[str, Any]:
    """Load and rehash every file bound by one strict smoke artifact."""
    root = Path(repository_root).resolve(strict=True)
    artifact = Path(path).resolve(strict=True)
    try:
        relative = artifact.relative_to(root)
    except ValueError as error:
        raise ValueError('smoke artifact must be inside the repository') from error
    if relative.parts[:3] != _OUTPUT_PREFIX or relative.name != 'smoke.json':
        raise ValueError('smoke artifact path is outside accuracy-first outputs')
    try:
        value = json.loads(artifact.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('smoke artifact is not valid JSON') from error
    validated = _validate_smoke_schema(value)
    _verify_artifact_files(
        validated, repository_root=root,
        physical_artifact_root=artifact.parent,
        logical_artifact_root=artifact.parent)
    return dict(validated)


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revalidate_preflight(preflight: SmokePreflight) -> None:
    commit = clean_git_commit(preflight.repository_root)
    if commit != preflight.source['git_commit']:
        raise RuntimeError('source commit changed after smoke preflight')
    binding = tracked_file_binding(
        preflight.repository_root, preflight.config_path,
        git_commit=commit)
    if binding != preflight.source['config']:
        raise RuntimeError('source config changed after smoke preflight')
    if sha256_file(preflight.teacher_checkpoint) != (
            preflight.teacher_checkpoint_sha256):
        raise RuntimeError('teacher checkpoint changed after smoke preflight')
    if sha256_file(preflight.student_checkpoint) != (
            preflight.student_checkpoint_sha256):
        raise RuntimeError('student checkpoint changed after smoke preflight')

    from mmengine.config import Config

    config = Config.fromfile(preflight.repository_root / preflight.config_path)
    observed = validate_coco_train_assets_at_root(
        config.get('smoke_data_authority'),
        config.train_dataloader.dataset,
        preflight.asset_root)
    if observed != preflight.train_assets:
        raise RuntimeError('COCO train2017 assets changed after smoke preflight')
    if canonical_json_sha256(_canonicalize(config.to_dict())) != (
            preflight.resolved_config_sha256):
        raise RuntimeError('resolved config changed after smoke preflight')
    if canonical_json_sha256(_canonicalize(
            smoke_dataloader_config(config.train_dataloader))) != (
                preflight.dataloader_override_sha256):
        raise RuntimeError('smoke dataloader override changed after preflight')


def _runtime_cuda_index(device_index: int) -> int:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None or not visible.strip():
        return device_index
    tokens = [item.strip() for item in visible.split(',') if item.strip()]
    if tokens != [str(device_index)]:
        raise RuntimeError(
            'CUDA_VISIBLE_DEVICES must expose exactly the leased physical GPU')
    return 0


def _update_tensor_hash(digest, label: str, tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(label.encode('utf-8'))
    digest.update(b'\0')
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(b'\0')
    digest.update(json.dumps(list(value.shape)).encode('ascii'))
    digest.update(b'\0')
    digest.update(value.numpy().tobytes())


def _packed_batch_binding(
        batch: object, *, asset_root: Path) -> dict[str, Any]:
    if not isinstance(batch, Mapping) or set(batch) != {'inputs', 'data_samples'}:
        raise RuntimeError('packed COCO batch must contain inputs and data_samples')
    inputs = batch['inputs']
    samples = batch['data_samples']
    if not isinstance(inputs, Sequence) or len(inputs) != 1:
        raise RuntimeError('smoke must load exactly one packed input')
    if not isinstance(samples, Sequence) or len(samples) != 1:
        raise RuntimeError('smoke must load exactly one packed data sample')
    sample = samples[0]
    metadata = getattr(sample, 'metainfo', None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError('packed COCO sample is missing metainfo')
    image_id = metadata.get('img_id')
    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
        raise RuntimeError('packed COCO sample has invalid img_id')
    image_value = metadata.get('img_path')
    if not isinstance(image_value, str):
        raise RuntimeError('packed COCO sample has no image path')
    try:
        image = Path(image_value).resolve(strict=True)
        relative_asset = image.relative_to(asset_root)
    except (OSError, ValueError) as error:
        raise RuntimeError('packed COCO image escapes the asset root') from error
    if (
            relative_asset.parts[:3] != ('data', 'coco', 'train2017')
            or image.is_symlink() or not image.is_file()):
        raise RuntimeError('packed COCO image is not from train2017')
    relative = relative_asset.as_posix()

    heatmaps = getattr(getattr(sample, 'gt_fields', None), 'heatmaps', None)
    weights = getattr(
        getattr(sample, 'gt_instance_labels', None), 'keypoint_weights', None)
    if heatmaps is None or weights is None:
        raise RuntimeError('packed COCO sample lacks heatmaps or keypoint weights')
    digest = hashlib.sha256()
    digest.update(f'img_id:{image_id}\0path:{relative}\0'.encode('utf-8'))
    _update_tensor_hash(digest, 'inputs[0]', inputs[0])
    _update_tensor_hash(digest, 'gt_fields.heatmaps', heatmaps)
    _update_tensor_hash(digest, 'keypoint_weights', weights)
    return {
        'size': 1,
        'image_ids': [image_id],
        'image_paths': [relative],
        'image_sha256': [sha256_file(image)],
        'packed_sha256': digest.hexdigest(),
    }


def _tensor_sha256(tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, 'heatmaps', tensor)
    return digest.hexdigest()


def _atomic_torch_save(torch, value: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _execute_smoke(
        preflight: SmokePreflight, staging_root: Path, output_root: Path,
        lease: GpuLease, device_index: int) -> dict[str, Any]:
    """Execute one production batch. Caller must already hold ``lease``."""
    _revalidate_preflight(preflight)
    runtime_index = _runtime_cuda_index(device_index)

    # Heavy model/CUDA imports are intentionally below every immutable-input
    # preflight and below acquisition of the canonical exclusive lease.
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.models.builder import build_pose_estimator
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        MambaPoseHeatmapDistiller, export_student_checkpoint,
        load_hash_validated_checkpoint)
    from mmpose.registry import MODELS
    from mmpose.utils import register_all_modules

    root = preflight.repository_root
    previous_cwd = Path.cwd()
    original_model = full_restored = exported_student = None
    try:
        os.chdir(root)
        register_all_modules()
        config = Config.fromfile(root / preflight.config_path)
        smoke_loader = smoke_dataloader_config(config.train_dataloader)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        loader = Runner.build_dataloader(
            smoke_loader, seed=0, diff_rank_seed=False)
        batch = next(iter(loader))
        batch_binding = _packed_batch_binding(
            batch, asset_root=preflight.asset_root)

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable after exclusive admission')
        if runtime_index < 0 or runtime_index >= torch.cuda.device_count():
            raise RuntimeError('leased CUDA device is not visible to PyTorch')
        device = torch.device(f'cuda:{runtime_index}')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device)

        # Absolute checkpoint paths avoid any service working-directory drift.
        config.model.teacher_checkpoint = str(preflight.teacher_checkpoint)
        config.model.student_checkpoint = str(preflight.student_checkpoint)
        original_model = MODELS.build(config.model)
        if not isinstance(original_model, MambaPoseHeatmapDistiller):
            raise RuntimeError('smoke config did not build the distiller')
        original_model.init_weights()
        original_model.to(device)
        original_model.train()
        processed = original_model.data_preprocessor(batch, training=True)
        losses = original_model(
            processed['inputs'], processed['data_samples'], mode='loss')
        expected_losses = {
            'loss_kpt', 'loss_heatmap_distill', 'heatmap_distill_mse'}
        if not isinstance(losses, Mapping) or set(losses) != expected_losses:
            raise RuntimeError('distiller returned an unexpected loss mapping')
        loss_values: dict[str, float] = {}
        for name in sorted(expected_losses):
            value = losses[name]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise RuntimeError(f'distiller loss is non-finite: {name}')
            loss_values[name] = float(value.detach().mean().cpu())
        total = sum(
            value.mean() for name, value in losses.items()
            if name.startswith('loss_'))
        if not torch.isfinite(total):
            raise RuntimeError('total distiller loss is non-finite')
        total.backward()

        teacher_parameters = tuple(original_model.teacher.parameters())
        if (
                original_model.teacher.training
                or any(parameter.requires_grad for parameter in teacher_parameters)):
            raise RuntimeError('teacher is not frozen in eval mode')
        teacher_gradients = sum(
            parameter.grad is not None for parameter in teacher_parameters)
        if teacher_gradients:
            raise RuntimeError('teacher received gradients')
        student_gradients = tuple(
            parameter.grad for parameter in original_model.student.parameters()
            if parameter.grad is not None)
        if not student_gradients:
            raise RuntimeError('student received no gradients')
        if any(not torch.isfinite(gradient).all() for gradient in student_gradients):
            raise RuntimeError('student gradients are non-finite')
        student_gradient_elements = sum(
            gradient.numel() for gradient in student_gradients)
        student_gradient_tensors = len(student_gradients)

        original_model.student.eval()
        with torch.no_grad():
            reference = MambaPoseHeatmapDistiller._heatmaps(
                original_model.student, processed['inputs']).detach().cpu()
        if list(reference.shape) != [1, 17, 64, 48]:
            raise RuntimeError('real student heatmap shape is unexpected')
        reference_hash = _tensor_sha256(reference)

        full_path = staging_root / 'distiller.pth'
        student_path = staging_root / 'student.pth'
        full_meta = {
            'schema_version': 1,
            'source': preflight.source,
            'resolved_config_sha256': preflight.resolved_config_sha256,
            'teacher_checkpoint_sha256': preflight.teacher_checkpoint_sha256,
            'student_checkpoint_sha256': preflight.student_checkpoint_sha256,
            'packed_batch_sha256': batch_binding['packed_sha256'],
            'gpu_lease_sha256': canonical_json_sha256(asdict(lease)),
            'execution': {
                'python_seed': 0,
                'numpy_seed': 0,
                'torch_seed': 0,
                'deterministic_algorithms': True,
                'python_dont_write_bytecode': True,
                'vram_scope': 'batch-1-smoke-not-training-batch',
            },
        }
        _atomic_torch_save(torch, {
            'state_dict': original_model.state_dict(), 'meta': full_meta},
            full_path)
        export_student_checkpoint(original_model, student_path)
        full_sha = sha256_file(full_path)
        student_sha = sha256_file(student_path)

        # Free the original graph before strict full and student-only restores.
        del losses, total, student_gradients, teacher_parameters, original_model
        original_model = None
        torch.cuda.empty_cache()

        full_restored = MODELS.build(config.model)
        full_restored.init_weights()
        load_hash_validated_checkpoint(
            full_restored, full_path, expected_sha256=full_sha, strict=True)
        full_restored.to(device).eval()
        with torch.no_grad():
            full_heatmaps = MambaPoseHeatmapDistiller._heatmaps(
                full_restored.student, processed['inputs']).detach().cpu()
        full_hash = _tensor_sha256(full_heatmaps)
        full_exact = torch.equal(reference, full_heatmaps)
        if not full_exact:
            raise RuntimeError('strict full checkpoint restore changed heatmaps')
        del full_restored
        full_restored = None
        torch.cuda.empty_cache()

        exported_student = build_pose_estimator(config.model.student)
        load_hash_validated_checkpoint(
            exported_student, student_path,
            expected_sha256=student_sha, strict=True)
        exported_student.to(device).eval()
        with torch.no_grad():
            export_heatmaps = MambaPoseHeatmapDistiller._heatmaps(
                exported_student, processed['inputs']).detach().cpu()
        export_hash = _tensor_sha256(export_heatmaps)
        export_exact = torch.equal(reference, export_heatmaps)
        if not export_exact:
            raise RuntimeError('student-only checkpoint restore changed heatmaps')

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_allocated <= 0 or peak_reserved <= 0:
            raise RuntimeError('CUDA peak memory counters are invalid')

        lease_record = asdict(lease)
        lease_record['allowed_pids'] = list(lease_record['allowed_pids'])
        output_full = output_root / 'distiller.pth'
        output_student = output_root / 'student.pth'
        assets = preflight.train_assets
        return {
            'schema_version': 1,
            'kind': _ARTIFACT_KIND,
            'claims': {
                'coco_ap': False,
                'latency': False,
                'training_batch_vram': False,
            },
            'execution': full_meta['execution'],
            'source': {
                **preflight.source,
                'resolved_config_sha256': preflight.resolved_config_sha256,
            },
            'inputs': {
                'teacher_checkpoint': {
                    'path': preflight.teacher_checkpoint_relative,
                    'sha256': preflight.teacher_checkpoint_sha256,
                },
                'student_checkpoint': {
                    'path': preflight.student_checkpoint_relative,
                    'sha256': preflight.student_checkpoint_sha256,
                },
            },
            'data': {
                'dataset': 'coco',
                'split': 'train2017',
                'pipeline_terminal': 'PackPoseInputs',
                'original_batch_size': preflight.original_batch_size,
                'smoke_batch_size': 1,
                'smoke_num_workers': 0,
                'smoke_persistent_workers': False,
                'inventory': {
                    'path': assets.inventory_path,
                    'sha256': assets.inventory_sha256,
                },
                'image_archive': {
                    'path': assets.image_archive_path,
                    'sha256': assets.image_archive_sha256,
                },
                'annotation_archive': {
                    'path': assets.annotation_archive_path,
                    'sha256': assets.annotation_archive_sha256,
                },
                'annotation': {
                    'path': assets.annotation_path,
                    'sha256': assets.annotation_sha256,
                },
                'corpus_digest_algorithm': _CORPUS_ALGORITHM,
                'corpus_sha256': assets.corpus_sha256,
                'corpus_image_count': assets.image_count,
                'dataloader_override_sha256': (
                    preflight.dataloader_override_sha256),
                'batch': batch_binding,
            },
            'gpu': {
                'device_index': device_index,
                'lease': lease_record,
                'lease_sha256': canonical_json_sha256(lease_record),
            },
            'checks': {
                'losses': loss_values,
                'teacher_frozen': True,
                'teacher_gradient_tensors': teacher_gradients,
                'student_gradient_tensors': student_gradient_tensors,
                'student_gradient_elements': student_gradient_elements,
                'student_gradients_finite': True,
                'peak_allocated_bytes': peak_allocated,
                'peak_reserved_bytes': peak_reserved,
                'heatmap_shape': list(reference.shape),
                'reference_heatmaps_sha256': reference_hash,
                'full_restore_heatmaps_sha256': full_hash,
                'export_restore_heatmaps_sha256': export_hash,
                'full_restore_exact': full_exact,
                'export_restore_exact': export_exact,
            },
            'artifacts': {
                'distiller_checkpoint': {
                    'path': output_full.relative_to(root).as_posix(),
                    'sha256': full_sha,
                },
                'student_checkpoint': {
                    'path': output_student.relative_to(root).as_posix(),
                    'sha256': student_sha,
                },
            },
        }
    finally:
        for model in (original_model, full_restored, exported_student):
            if model is not None:
                del model
        with suppress(Exception):
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        os.chdir(previous_cwd)


def run_distill_smoke(
        *, repository_root: Path, config_path: Path,
        output_relative: Path, device_index: int) -> Path:
    """Run one smoke under the canonical lease and atomically publish it."""
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1' \
            or not sys.dont_write_bytecode:
        raise RuntimeError('smoke requires PYTHONDONTWRITEBYTECODE=1')
    if isinstance(device_index, bool) or not isinstance(device_index, int) \
            or device_index < 0:
        raise ValueError('device_index must be a non-negative integer')
    root = Path(repository_root).resolve(strict=True)
    output = optimization_output_path(
        output_relative.as_posix(), repository_root=root)
    relative = output.relative_to(root)
    if relative.parts[:3] != _OUTPUT_PREFIX or len(relative.parts) < 4:
        raise ValueError(
            'output must be a candidate directory under '
            'work_dirs/optimization/accuracy-first')
    if os.path.lexists(output):
        raise FileExistsError(f'refusing to overwrite output: {relative}')

    preflight = build_smoke_preflight(root, config_path)
    lock_path = canonical_gpu_lock_path(root)
    stage_id = f'distill-smoke:{preflight.experiment_id}'
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    published = False
    with exclusive_cuda_stage(
            lock_path, device_index, {os.getpid()}, stage_id=stage_id) as lease:
        if os.path.lexists(output):
            raise FileExistsError(f'refusing to overwrite output: {relative}')
        staging = Path(tempfile.mkdtemp(
            prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent))
        try:
            record = _execute_smoke(
                preflight, staging, output, lease, device_index)
            validated = _validate_smoke_schema(record)
            _atomic_json(staging / 'smoke.json', validated)
            _verify_artifact_files(
                validated, repository_root=root,
                physical_artifact_root=staging,
                logical_artifact_root=output)
            staging.rename(output)
            published = True
            load_smoke_artifact(output / 'smoke.json', repository_root=root)
        except BaseException:
            if published and output.exists():
                with suppress(OSError):
                    output.rename(staging)
                    published = False
            raise
        finally:
            if not published and staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
    return output / 'smoke.json'
