"""Strict COCO metric normalization and candidate-result loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

from mambapose_opt.latency import LatencyError, validate_gpu_lease
from mambapose_opt.schema import (
    CandidateManifestError, CandidateSpec, parse_candidate_manifest)
from mambapose_opt.pwl_paths import canonical_relative_path


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


def _validate_pwl_stage_a_binding(
        value: object, *, expected: Mapping[str, str] | None,
        ) -> dict[str, str] | None:
    if value is None and expected is None:
        return None
    if (not isinstance(value, Mapping)
            or set(value) != {'path', 'sha256'}
            or not isinstance(value.get('sha256'), str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise MetricError('PWL Stage-A smoke binding is invalid')
    try:
        relative = canonical_relative_path(
            value.get('path'), label='PWL Stage-A smoke')
    except ValueError as error:
        raise MetricError(str(error)) from error
    if (relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name != 'smoke.json'
            or relative.parent.name != 'smoke-stage-a'):
        raise MetricError('PWL Stage-A smoke path is not canonical')
    normalized = {'path': relative.as_posix(), 'sha256': value['sha256']}
    if expected is None or normalized != dict(expected):
        raise MetricError('PWL Stage-A smoke binding mismatch')
    return normalized


_COCO_ANNOTATION = 'data/coco/annotations/person_keypoints_val2017.json'
_COCO_DETECTIONS = (
    'data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json')
_COCO_AUTHORITY = 'optimization/coco_val2017_authority.json'
_COCO_IMAGE_DIGEST_ALGORITHM = 'sha256-filename-size-content-v1'
_COCO_IMAGE_COUNT = 5000
_COCO_ANNOTATION_COUNT = 11004
_COCO_DETECTION_COUNT = 104125
_TRUSTED_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_BINDING_FIELDS = {
    'git_commit', 'manifest_path', 'manifest_sha256', 'config_path',
    'config_sha256', 'authority_path', 'authority_sha256'}


def resolve_shared_asset_exposure(
        repository_root: Path, asset_root: Path, relative_root: Path | str,
        *, label: str) -> Path:
    """Require one exact caller-visible link to a primary asset directory."""
    root = Path(repository_root).resolve(strict=True)
    primary = Path(asset_root).resolve(strict=True)
    relative = Path(relative_root)
    if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {'.', '..'} for part in relative.parts)):
        raise MetricError(f'{label} exposure root is invalid')
    expected = primary
    for part in relative.parts:
        expected = expected / part
        if expected.is_symlink():
            raise MetricError(
                f'primary checkout {label} asset root must not use symlinks')
    if not expected.is_dir():
        raise MetricError(f'primary checkout {label} asset root is missing')
    if root == primary:
        return expected

    exposed_parent = root
    for part in relative.parts[:-1]:
        exposed_parent = exposed_parent / part
        if exposed_parent.is_symlink() or not exposed_parent.is_dir():
            raise MetricError(
                f'linked worktree {label} exposure parent is invalid')
    exposed = exposed_parent / relative.parts[-1]
    try:
        target = Path(os.path.abspath(
            exposed.parent / os.readlink(exposed)))
    except OSError as error:
        raise MetricError(
            f'linked worktree {label} exposure is unreadable') from error
    if (
            not exposed.is_symlink()
            or target != expected.absolute()
            or exposed.resolve(strict=True) != expected.resolve(strict=True)):
        raise MetricError(
            f'linked worktree {label} exposure must target the Git common '
            f'checkout asset root')
    return expected


def resolve_project_asset_root(repository_root: Path) -> Path:
    """Resolve the one approved shared-asset checkout for a Git worktree.

    Linked worktrees may expose ignored ``data`` through a symlink, but that
    symlink must target the primary checkout identified by Git's common dir.
    An arbitrary caller-selected or symlink-selected asset tree is rejected.
    """
    root = Path(repository_root).resolve(strict=True)
    try:
        top = Path(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], cwd=root,
            text=True).strip()).resolve(strict=True)
        common_value = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'], cwd=root,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise MetricError('asset root requires a valid Git worktree') from error
    if top != root:
        raise MetricError('asset root must be resolved from the worktree root')
    common = Path(common_value)
    if not common.is_absolute():
        common = root / common
    common = common.resolve(strict=True)
    if common.name != '.git' or not common.is_dir():
        raise MetricError('Git common directory is not an approved checkout')
    asset_root = common.parent.resolve(strict=True)
    resolve_shared_asset_exposure(
        root, asset_root, 'data', label='shared data')
    return asset_root


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_repository_relative(
        path: Path | str, repository_root: Path, *, label: str) -> str:
    root = Path(repository_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise MetricError(f'{label} must be an existing repository file') from error
    if not resolved.is_file() or any(part in {'.', '..'} for part in relative.parts):
        raise MetricError(f'{label} must be a safe repository-relative file')
    return relative.as_posix()


def _git_blob(repository_root: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ['git', 'show', f'{commit}:{relative}'], cwd=repository_root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise MetricError(
            f'source path is not tracked at recorded commit: {relative}') from error


def build_source_binding(
        *, repository_root: Path, candidate: CandidateSpec,
        manifest_path: Path, git_commit: str) -> dict[str, str]:
    """Bind one candidate only to immutable, commit-addressed source."""
    if not isinstance(git_commit, str) or not _COMMIT.fullmatch(git_commit):
        raise MetricError('source git commit is invalid')
    root = Path(repository_root).resolve()
    manifest_relative = _safe_repository_relative(
        manifest_path, root, label='candidate manifest')
    config_relative = _safe_repository_relative(
        candidate.config, root, label='candidate config')
    authority_relative = _safe_repository_relative(
        _COCO_AUTHORITY, root, label='COCO authority')
    tracked = {
        'manifest': (manifest_relative, root / manifest_relative),
        'config': (config_relative, root / config_relative),
        'authority': (authority_relative, root / authority_relative),
    }
    hashes: dict[str, str] = {}
    for name, (relative, actual_path) in tracked.items():
        blob = _git_blob(root, git_commit, relative)
        actual = actual_path.read_bytes()
        if actual != blob:
            raise MetricError(
                f'{name} source differs from recorded clean Git commit')
        hashes[name] = _bytes_sha256(blob)
        if name == 'manifest':
            try:
                manifest_candidates = parse_candidate_manifest(
                    json.loads(blob))
            except (json.JSONDecodeError, CandidateManifestError) as error:
                raise MetricError(f'candidate manifest is invalid: {error}') from error
            if tuple(
                    row for row in manifest_candidates if row.id == candidate.id
                    ) != (candidate,):
                raise MetricError(
                    'candidate does not exactly match the recorded manifest')
    return {
        'git_commit': git_commit,
        'manifest_path': manifest_relative,
        'manifest_sha256': hashes['manifest'],
        'config_path': config_relative,
        'config_sha256': hashes['config'],
        'authority_path': authority_relative,
        'authority_sha256': hashes['authority'],
    }


def build_deterministic_evaluation_config(
        candidate: CandidateSpec, flip_test: bool,
        config_path: Path | None = None, *, config=None):
    """Reconstruct the exact config serialized by the evaluation producer."""
    if flip_test is not True and flip_test is not False:
        raise MetricError('flip_test must be a boolean')
    from mmengine.config import Config

    from mambapose_opt.determinism import deterministic_dataloader_config

    if (config_path is None) == (config is None):
        raise MetricError(
            'evaluation config requires exactly one path or authorized Config')
    config = (
        Config.fromfile(config_path) if config is None
        else copy.deepcopy(config))
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    configured_imports = list(
        config.get('custom_imports', {}).get('imports', ()))
    if 'mambapose_opt.determinism' not in configured_imports:
        configured_imports.append('mambapose_opt.determinism')
    config.custom_imports = dict(
        imports=configured_imports, allow_failed_imports=False)
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        config[name] = deterministic_dataloader_config(
            config[name], seed=candidate.seed, worker_count=2)
    config.model.test_cfg.flip_test = flip_test
    return config


def evaluation_mode_config_sha256(
        candidate: CandidateSpec, *, flip_test: bool,
        config_path: Path) -> str:
    """Hash the canonical mode-specific Python config dump."""
    config = build_deterministic_evaluation_config(
        candidate, flip_test, config_path)
    serialized = config.dump()
    if not isinstance(serialized, str):
        raise MetricError('deterministic evaluation config is not serializable')
    return _bytes_sha256(serialized.encode('utf-8'))


def _official_authority_expectations(value: object) -> dict[str, Any]:
    top_fields = {
        'schema_version', 'dataset', 'split', 'inventory', 'annotation',
        'images', 'detections'}
    inventory_fields = {'path', 'sha256'}
    annotation_fields = {
        'path', 'sha256', 'image_count', 'annotation_count',
        'inventory_asset_id', 'inventory_archive_sha256'}
    image_fields = {
        'prefix', 'image_count', 'corpus_digest_algorithm', 'corpus_sha256',
        'inventory_asset_id', 'inventory_archive_sha256'}
    detection_fields = {'path', 'sha256', 'record_count', 'inventory_asset_id'}
    if (
            not isinstance(value, Mapping) or set(value) != top_fields
            or value.get('schema_version') != 1
            or value.get('dataset') != 'coco'
            or value.get('split') != 'val2017'
            or not isinstance(value.get('inventory'), Mapping)
            or set(value['inventory']) != inventory_fields
            or not isinstance(value.get('annotation'), Mapping)
            or set(value['annotation']) != annotation_fields
            or not isinstance(value.get('images'), Mapping)
            or set(value['images']) != image_fields
            or not isinstance(value.get('detections'), Mapping)
            or set(value['detections']) != detection_fields):
        raise MetricError('recorded COCO authority blob is malformed')
    annotation = value['annotation']
    images = value['images']
    detections = value['detections']
    inventory = value['inventory']
    asset_ids = (
        annotation.get('inventory_asset_id'), images.get('inventory_asset_id'),
        detections.get('inventory_asset_id'))
    if (
            annotation.get('path') != _COCO_ANNOTATION
            or inventory.get('path') != 'data/inventory.json'
            or images.get('prefix') != 'data/coco/val2017'
            or detections.get('path') != _COCO_DETECTIONS
            or annotation.get('image_count') != _COCO_IMAGE_COUNT
            or images.get('image_count') != _COCO_IMAGE_COUNT
            or annotation.get('annotation_count') != _COCO_ANNOTATION_COUNT
            or detections.get('record_count') != _COCO_DETECTION_COUNT
            or any(not isinstance(item, str) or not item for item in asset_ids)
            or len(set(asset_ids)) != 3
            or images.get('corpus_digest_algorithm') != (
                _COCO_IMAGE_DIGEST_ALGORITHM)):
        raise MetricError('recorded COCO authority is not official val2017')
    hashes = {
        'annotation_authority_sha256': annotation.get('sha256'),
        'detection_authority_sha256': detections.get('sha256'),
        'image_corpus_authority_sha256': images.get('corpus_sha256'),
        'inventory_annotation_archive_sha256': annotation.get(
            'inventory_archive_sha256'),
        'inventory_image_archive_sha256': images.get(
            'inventory_archive_sha256'),
        'inventory_authority_sha256': inventory.get('sha256'),
    }
    if any(not isinstance(item, str) or not _SHA256.fullmatch(item)
           for item in hashes.values()):
        raise MetricError('recorded COCO authority hashes are invalid')
    return {
        **hashes,
        'authority_image_count': _COCO_IMAGE_COUNT,
        'authority_annotation_count': _COCO_ANNOTATION_COUNT,
        'authority_detection_count': _COCO_DETECTION_COUNT,
        'image_corpus_digest_algorithm': _COCO_IMAGE_DIGEST_ALGORITHM,
        'annotation_inventory_asset_id': annotation['inventory_asset_id'],
        'image_inventory_asset_id': images['inventory_asset_id'],
        'detection_inventory_asset_id': detections['inventory_asset_id'],
    }


def resolve_artifact_source(
        envelope: Mapping[str, Any], *, repository_root: Path,
        expected_manifest_path: Path | None = None,
        ) -> tuple[Mapping[str, str], CandidateSpec, Mapping[str, Any]]:
    result = envelope.get('result')
    source = result.get('source') if isinstance(result, Mapping) else None
    if not isinstance(source, Mapping) or set(source) != _SOURCE_BINDING_FIELDS:
        raise MetricError('artifact source manifest binding is missing or invalid')
    normalized = dict(source)
    commit = normalized.get('git_commit')
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise MetricError('artifact source git commit is invalid')
    for name in ('manifest_path', 'config_path', 'authority_path'):
        item = normalized.get(name)
        if (
                not isinstance(item, str) or not item
                or Path(item).is_absolute()
                or any(part in {'.', '..'} for part in Path(item).parts)):
            raise MetricError(f'artifact source {name} is invalid')
    for name in ('manifest_sha256', 'config_sha256', 'authority_sha256'):
        item = normalized.get(name)
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise MetricError(f'artifact source {name} is invalid')
    root = Path(repository_root).resolve()
    if expected_manifest_path is not None:
        expected_relative = _safe_repository_relative(
            expected_manifest_path, root, label='candidate manifest')
        if normalized['manifest_path'] != expected_relative:
            raise MetricError('artifact source manifest path is not authoritative')
    blobs: dict[str, bytes] = {}
    for name in ('manifest', 'config', 'authority'):
        relative = normalized[f'{name}_path']
        blob = _git_blob(root, commit, relative)
        if _bytes_sha256(blob) != normalized[f'{name}_sha256']:
            raise MetricError(f'artifact source {name} hash is invalid')
        blobs[name] = blob
    if normalized['authority_path'] != _COCO_AUTHORITY:
        raise MetricError('artifact source authority path is invalid')
    try:
        manifest_value = json.loads(blobs['manifest'])
        authority_value = json.loads(blobs['authority'])
    except json.JSONDecodeError as error:
        raise MetricError(f'artifact source JSON is invalid: {error}') from error
    try:
        candidates = parse_candidate_manifest(manifest_value)
    except CandidateManifestError as error:
        raise MetricError(f'artifact source manifest is invalid: {error}') from error
    candidate_id = envelope.get('candidate_id')
    selected = tuple(row for row in candidates if row.id == candidate_id)
    if len(selected) != 1:
        raise MetricError('artifact candidate is absent from source manifest')
    candidate = selected[0]
    if candidate.config.as_posix() != normalized['config_path']:
        raise MetricError('artifact source config disagrees with manifest')
    expectations = _official_authority_expectations(authority_value)
    return MappingProxyType(normalized), candidate, MappingProxyType(expectations)


def _repository_root_from_artifacts(root: Path) -> Path:
    trusted = _TRUSTED_REPOSITORY_ROOT.resolve(strict=True)
    try:
        repository = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
            text=True, stderr=subprocess.DEVNULL).strip()
        resolved = Path(repository).resolve(strict=True)
        root.resolve().relative_to(resolved)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise MetricError(
            'candidate artifact root must belong to a Git repository') from error
    if resolved != trusted:
        raise MetricError(
            'candidate artifacts must belong to the trusted source repository')
    return resolved


def _reject_artifact_root_symlinks(root: Path) -> None:
    """Reject lexical symlinks before any candidate-root resolution."""
    trusted = _TRUSTED_REPOSITORY_ROOT.resolve(strict=True)
    lexical = Path(root)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = lexical.absolute()
    try:
        relative = lexical.relative_to(trusted)
    except ValueError as error:
        raise MetricError(
            'candidate artifact root must belong to the trusted source '
            'repository') from error
    cursor = trusted
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise MetricError(
                'candidate artifact root and ancestors must not be symlinks')


def _image_corpus_sha256(image_dir: Path, file_names: set[str]) -> str:
    """Hash canonical names, byte lengths, and contents for a whole corpus."""
    digest = hashlib.sha256()
    for name in sorted(file_names):
        path = image_dir / name
        size = path.stat().st_size
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(size).encode('ascii'))
        digest.update(b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        digest.update(b'\0')
    return digest.hexdigest()


def _verify_inventory_archive(
        root: Path, entry: Mapping[str, Any], expected_sha256: str,
        *, label: str) -> str | None:
    if entry.get('sha256') != expected_sha256:
        raise MetricError(f'{label} inventory hash disagrees with COCO authority')
    archive_relative = entry.get('path')
    required = entry.get('required_paths')
    if not isinstance(archive_relative, str) or not archive_relative:
        raise MetricError(f'{label} inventory archive path is invalid')
    archive = root / archive_relative
    if archive.exists():
        if not archive.is_file():
            raise MetricError(f'{label} inventory archive is not a file')
        actual = _file_sha256(archive)
        if actual != expected_sha256:
            raise MetricError(f'{label} inventory archive hash is invalid')
        return actual
    if (
            not isinstance(required, list) or not required
            or any(not isinstance(item, str) or not (root / item).exists()
                   for item in required)):
        raise MetricError(
            f'{label} inventory archive and extracted required paths are missing')
    return None


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

    source_root = Path(repository_root).resolve()
    asset_root = resolve_project_asset_root(source_root)
    annotation_path = asset_root / _COCO_ANNOTATION
    detection_path = asset_root / _COCO_DETECTIONS
    inventory_path = asset_root / 'data/inventory.json'
    authority_path = source_root / _COCO_AUTHORITY
    try:
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        detections = json.loads(detection_path.read_text(encoding='utf-8'))
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
        authority = json.loads(authority_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot verify COCO val assets: {error}') from error
    authority_fields = {
        'schema_version', 'dataset', 'split', 'inventory', 'annotation',
        'images', 'detections'}
    inventory_authority_fields = {'path', 'sha256'}
    annotation_authority_fields = {
        'path', 'sha256', 'image_count', 'annotation_count',
        'inventory_asset_id', 'inventory_archive_sha256'}
    image_authority_fields = {
        'prefix', 'image_count', 'corpus_digest_algorithm', 'corpus_sha256',
        'inventory_asset_id', 'inventory_archive_sha256'}
    detection_authority_fields = {
        'path', 'sha256', 'record_count', 'inventory_asset_id'}
    if (
            not isinstance(authority, Mapping)
            or set(authority) != authority_fields
            or authority.get('schema_version') != 1
            or authority.get('dataset') != 'coco'
            or authority.get('split') != 'val2017'
            or not isinstance(authority.get('inventory'), Mapping)
            or set(authority['inventory']) != inventory_authority_fields
            or not isinstance(authority.get('annotation'), Mapping)
            or set(authority['annotation']) != annotation_authority_fields
            or not isinstance(authority.get('images'), Mapping)
            or set(authority['images']) != image_authority_fields
            or not isinstance(authority.get('detections'), Mapping)
            or set(authority['detections']) != detection_authority_fields):
        raise MetricError('COCO val authority manifest is malformed')
    annotation_authority = authority['annotation']
    image_authority = authority['images']
    detection_authority = authority['detections']
    inventory_authority = authority['inventory']
    if (
            any(
                not isinstance(row['inventory_asset_id'], str)
                or not row['inventory_asset_id']
                for row in (
                    annotation_authority, image_authority,
                    detection_authority))
            or len({
                annotation_authority['inventory_asset_id'],
                image_authority['inventory_asset_id'],
                detection_authority['inventory_asset_id']}) != 3
            or isinstance(annotation_authority['image_count'], bool)
            or inventory_authority.get('path') != 'data/inventory.json'
            or not isinstance(inventory_authority.get('sha256'), str)
            or not _SHA256.fullmatch(inventory_authority['sha256'])
            or not isinstance(annotation_authority['image_count'], int)
            or isinstance(image_authority['image_count'], bool)
            or not isinstance(image_authority['image_count'], int)
            or annotation_authority['path'] != _COCO_ANNOTATION
            or image_authority['prefix'] != 'data/coco/val2017'
            or image_authority['corpus_digest_algorithm'] != (
                _COCO_IMAGE_DIGEST_ALGORITHM)
            or detection_authority['path'] != _COCO_DETECTIONS
            or annotation_authority['image_count'] != expected_image_count
            or image_authority['image_count'] != expected_image_count
            or isinstance(annotation_authority['annotation_count'], bool)
            or not isinstance(annotation_authority['annotation_count'], int)
            or annotation_authority['annotation_count'] <= 0
            or isinstance(detection_authority['record_count'], bool)
            or not isinstance(detection_authority['record_count'], int)
            or detection_authority['record_count'] <= 0
            or (
                expected_image_count == _COCO_IMAGE_COUNT
                and annotation_authority['annotation_count'] != (
                    _COCO_ANNOTATION_COUNT))
            or (
                expected_image_count == _COCO_IMAGE_COUNT
                and detection_authority['record_count'] != (
                    _COCO_DETECTION_COUNT))
            or any(
                not isinstance(authority_row[field], str)
                or not _SHA256.fullmatch(authority_row[field])
                for authority_row, fields in (
                    (annotation_authority,
                     ('sha256', 'inventory_archive_sha256')),
                    (image_authority,
                     ('inventory_archive_sha256', 'corpus_sha256')),
                    (detection_authority, ('sha256',)))
                for field in fields)):
        raise MetricError('COCO val authority manifest values are invalid')
    images = annotation.get('images') if isinstance(annotation, Mapping) else None
    annotations = annotation.get('annotations') if isinstance(annotation, Mapping) else None
    if (
            not isinstance(images, list) or len(images) != expected_image_count
            or not isinstance(annotations, list)
            or len(annotations) != annotation_authority['annotation_count']):
        raise MetricError(
            'COCO val annotation counts disagree with authority')
    annotation_sha256 = _file_sha256(annotation_path)
    if annotation_sha256 != annotation_authority['sha256']:
        raise MetricError('COCO val annotation hash disagrees with authority')
    ids: set[int] = set()
    file_names: set[str] = set()
    file_identities: set[tuple[int, int]] = set()
    image_dir = asset_root / data_root / 'val2017'
    for row in images:
        if (
                not isinstance(row, Mapping)
                or isinstance(row.get('id'), bool)
                or not isinstance(row.get('id'), int)
                or row['id'] in ids
                or not isinstance(row.get('file_name'), str)
                or not row['file_name']
                or row['file_name'] in file_names
                or Path(row['file_name']).name != row['file_name']
                or not (image_dir / row['file_name']).is_file()):
            raise MetricError(
                'COCO val image IDs, names, and files must be unique and valid')
        stat = (image_dir / row['file_name']).stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in file_identities:
            raise MetricError('COCO val image file identities must be unique')
        ids.add(row['id'])
        file_names.add(row['file_name'])
        file_identities.add(identity)
    image_corpus_sha256 = _image_corpus_sha256(image_dir, file_names)
    if image_corpus_sha256 != image_authority['corpus_sha256']:
        raise MetricError('COCO val image corpus hash disagrees with authority')
    if (
            not isinstance(detections, list)
            or len(detections) != detection_authority['record_count']
            or any(
                not isinstance(row, Mapping) or row.get('image_id') not in ids
                for row in detections)):
        raise MetricError('COCO val detections disagree with authority or image IDs')
    assets = inventory.get('assets') if isinstance(inventory, Mapping) else None
    inventory_sha256 = _file_sha256(inventory_path)
    if inventory_sha256 != inventory_authority['sha256']:
        raise MetricError('data inventory hash disagrees with COCO authority')
    if not isinstance(assets, list):
        raise MetricError('data inventory assets are invalid')
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in assets:
        if not isinstance(row, Mapping) or not isinstance(row.get('id'), str):
            raise MetricError('data inventory assets are invalid')
        if row['id'] in by_id:
            raise MetricError('data inventory asset IDs must be unique')
        by_id[row['id']] = row
    required_assets = {
        annotation_authority['inventory_asset_id']:
            annotation_authority['inventory_archive_sha256'],
        image_authority['inventory_asset_id']:
            image_authority['inventory_archive_sha256'],
        detection_authority['inventory_asset_id']:
            detection_authority['sha256'],
    }
    if any(
            asset_id not in by_id
            or by_id[asset_id].get('sha256') != expected_hash
            for asset_id, expected_hash in required_assets.items()):
        raise MetricError('data inventory hashes disagree with COCO authority')
    annotation_archive_sha256 = _verify_inventory_archive(
        asset_root, by_id[annotation_authority['inventory_asset_id']],
        annotation_authority['inventory_archive_sha256'],
        label='annotation')
    image_archive_sha256 = _verify_inventory_archive(
        asset_root, by_id[image_authority['inventory_asset_id']],
        image_authority['inventory_archive_sha256'], label='image')
    detection_entry = by_id[detection_authority['inventory_asset_id']]
    if detection_entry.get('path') != _COCO_DETECTIONS:
        raise MetricError('data inventory lacks the official COCO val detections')
    detection_sha256 = _file_sha256(detection_path)
    if (
            detection_entry.get('sha256') != detection_sha256
            or detection_authority['sha256'] != detection_sha256):
        raise MetricError(
            'detection inventory/authority hash does not match actual asset')
    return {
        'dataset': 'coco', 'split': 'val2017', 'complete_split': True,
        'authority_path': _COCO_AUTHORITY,
        'authority_sha256': _file_sha256(authority_path),
        'authority_image_count': image_authority['image_count'],
        'authority_annotation_count': annotation_authority['annotation_count'],
        'authority_detection_count': detection_authority['record_count'],
        'inventory_authority_sha256': inventory_authority['sha256'],
        'annotation_authority_sha256': annotation_authority['sha256'],
        'detection_authority_sha256': detection_authority['sha256'],
        'image_corpus_digest_algorithm': _COCO_IMAGE_DIGEST_ALGORITHM,
        'image_corpus_authority_sha256': image_authority['corpus_sha256'],
        'image_corpus_sha256': image_corpus_sha256,
        'inventory_annotation_archive_sha256': annotation_archive_sha256,
        'inventory_image_archive_sha256': image_archive_sha256,
        'inventory_projection': {
            'inventory_path': 'data/inventory.json',
            'inventory_sha256': inventory_sha256,
            'annotation_asset_id': annotation_authority[
                'inventory_asset_id'],
            'annotation_declared_sha256': annotation_authority[
                'inventory_archive_sha256'],
            'annotation_observed_archive_sha256': annotation_archive_sha256,
            'image_asset_id': image_authority['inventory_asset_id'],
            'image_declared_sha256': image_authority[
                'inventory_archive_sha256'],
            'image_observed_archive_sha256': image_archive_sha256,
            'detection_asset_id': detection_authority['inventory_asset_id'],
            'detection_declared_sha256': detection_entry['sha256'],
            'detection_observed_sha256': detection_sha256,
        },
        'annotation_sha256': annotation_sha256,
        'detection_sha256': detection_sha256,
        'inventory_detection_sha256': detection_entry['sha256'],
        'annotation_image_count': len(images),
        'annotation_record_count': len(annotations),
        'detection_record_count': len(detections),
        'verified_image_count': len(images),
    }


def validate_live_coco_observation(
        recorded: Mapping[str, Any], *, config: Mapping[str, Any],
        repository_root: Path) -> Mapping[str, Any]:
    """Reobserve external COCO assets and match a formal recorded claim."""
    live = validate_coco_val_protocol(
        config, repository_root=repository_root)
    for name, observed in live.items():
        if recorded.get(name) != observed:
            raise MetricError(
                f'recorded COCO asset observation is stale or invalid: {name}')
    return _freeze(live)


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


def _validate_recorded_coco_protocol(
        value: object, *, expected_authority: Mapping[str, Any] | None = None,
        ) -> Mapping[str, Any]:
    required = {
        'dataset', 'split', 'complete_split', 'batch_size',
        'authority_path', 'authority_sha256',
        'authority_image_count', 'authority_annotation_count',
        'authority_detection_count',
        'inventory_authority_sha256',
        'annotation_authority_sha256', 'detection_authority_sha256',
        'image_corpus_digest_algorithm',
        'image_corpus_authority_sha256', 'image_corpus_sha256',
        'inventory_annotation_archive_sha256',
        'inventory_image_archive_sha256',
        'annotation_sha256', 'detection_sha256',
        'inventory_detection_sha256', 'annotation_image_count',
        'annotation_record_count', 'detection_record_count',
        'verified_image_count', 'source_config', 'checkpoint',
        'data_inventory', 'inventory_projection'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise MetricError(
            'evaluation protocol must contain the exact authority fields')
    if (
            value.get('dataset') != 'coco'
            or value.get('split') != 'val2017'
            or value.get('complete_split') is not True
            or isinstance(value.get('batch_size'), bool)
            or not isinstance(value.get('batch_size'), int)
            or value['batch_size'] <= 0):
        raise MetricError('evaluation protocol is not complete COCO val2017')
    for name in (
            'authority_sha256', 'annotation_authority_sha256',
            'detection_authority_sha256', 'annotation_sha256',
            'detection_sha256', 'inventory_detection_sha256',
            'image_corpus_authority_sha256', 'image_corpus_sha256',
            'inventory_authority_sha256'):
        if not isinstance(value.get(name), str) or not _SHA256.fullmatch(
                value[name]):
            raise MetricError(f'evaluation protocol {name} is invalid')
    if (
            value['annotation_sha256'] != value['annotation_authority_sha256']
            or value['detection_sha256'] != value['detection_authority_sha256']
            or value['detection_sha256'] != value['inventory_detection_sha256']
            or value['image_corpus_sha256'] != (
                value['image_corpus_authority_sha256'])):
        raise MetricError(
            'evaluation recorded hashes disagree with corpus authority')
    if (
            value['authority_image_count'] != value['annotation_image_count']
            or value['authority_image_count'] != value['verified_image_count']
            or value['authority_annotation_count'] != (
                value['annotation_record_count'])
            or value['authority_detection_count'] != (
                value['detection_record_count'])):
        raise MetricError('evaluation recorded counts disagree with authority')
    counts = (
        value.get('authority_image_count'),
        value.get('authority_annotation_count'),
        value.get('authority_detection_count'),
        value.get('annotation_image_count'),
        value.get('annotation_record_count'),
        value.get('detection_record_count'),
        value.get('verified_image_count'))
    if (
            any(isinstance(item, bool) or not isinstance(item, int)
                or item < 0 for item in counts)
            or value['annotation_image_count'] != 5000
            or value['verified_image_count'] != 5000
            or value['authority_image_count'] != 5000
            or value['authority_annotation_count'] != _COCO_ANNOTATION_COUNT
            or value['authority_detection_count'] != _COCO_DETECTION_COUNT
            or value['annotation_record_count'] != _COCO_ANNOTATION_COUNT
            or value['detection_record_count'] != _COCO_DETECTION_COUNT):
        raise MetricError('evaluation protocol COCO asset counts are invalid')
    if value['image_corpus_digest_algorithm'] != _COCO_IMAGE_DIGEST_ALGORITHM:
        raise MetricError('evaluation protocol image corpus algorithm is invalid')
    for name in (
            'inventory_annotation_archive_sha256',
            'inventory_image_archive_sha256'):
        item = value[name]
        if item is not None and (
                not isinstance(item, str) or not _SHA256.fullmatch(item)):
            raise MetricError(f'evaluation protocol {name} is invalid')
    if value['authority_path'] != _COCO_AUTHORITY:
        raise MetricError('evaluation protocol authority path is invalid')
    for name in ('source_config', 'checkpoint', 'data_inventory'):
        item = value.get(name)
        if (
                not isinstance(item, str) or not item
                or Path(item).is_absolute()
                or any(part in {'.', '..'} for part in Path(item).parts)):
            raise MetricError(f'evaluation protocol {name} path is invalid')
    if value['data_inventory'] != 'data/inventory.json':
        raise MetricError('evaluation protocol data inventory path is invalid')
    projection_fields = {
        'inventory_path', 'inventory_sha256', 'annotation_asset_id',
        'annotation_declared_sha256',
        'annotation_observed_archive_sha256', 'image_asset_id',
        'image_declared_sha256', 'image_observed_archive_sha256',
        'detection_asset_id', 'detection_declared_sha256',
        'detection_observed_sha256'}
    projection = value['inventory_projection']
    if (
            not isinstance(projection, Mapping)
            or set(projection) != projection_fields
            or projection.get('inventory_path') != 'data/inventory.json'
            or not isinstance(projection.get('inventory_sha256'), str)
            or not _SHA256.fullmatch(projection['inventory_sha256'])):
        raise MetricError('evaluation inventory projection is invalid')
    if projection['inventory_sha256'] != value['inventory_authority_sha256']:
        raise MetricError(
            'evaluation inventory hash disagrees with corpus authority')
    for name in (
            'annotation_declared_sha256', 'image_declared_sha256',
            'detection_declared_sha256', 'detection_observed_sha256'):
        item = projection[name]
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise MetricError('evaluation inventory projection hash is invalid')
    for name in (
            'annotation_observed_archive_sha256',
            'image_observed_archive_sha256'):
        item = projection[name]
        if item is not None and (
                not isinstance(item, str) or not _SHA256.fullmatch(item)):
            raise MetricError('evaluation inventory observation is invalid')
    if expected_authority is not None:
        for name in (
                'authority_image_count', 'authority_annotation_count',
                'authority_detection_count',
                'inventory_authority_sha256',
                'annotation_authority_sha256',
                'detection_authority_sha256',
                'image_corpus_digest_algorithm',
                'image_corpus_authority_sha256'):
            if value[name] != expected_authority[name]:
                raise MetricError(
                    f'evaluation protocol {name} disagrees with authority blob')
        for name in (
                'inventory_annotation_archive_sha256',
                'inventory_image_archive_sha256'):
            if value[name] is not None and value[name] != expected_authority[name]:
                raise MetricError(
                    f'evaluation protocol {name} disagrees with authority blob')
        expected_projection = {
            'annotation_asset_id': expected_authority[
                'annotation_inventory_asset_id'],
            'annotation_declared_sha256': expected_authority[
                'inventory_annotation_archive_sha256'],
            'image_asset_id': expected_authority['image_inventory_asset_id'],
            'image_declared_sha256': expected_authority[
                'inventory_image_archive_sha256'],
            'detection_asset_id': expected_authority[
                'detection_inventory_asset_id'],
            'detection_declared_sha256': expected_authority[
                'detection_authority_sha256'],
            'detection_observed_sha256': expected_authority[
                'detection_authority_sha256'],
        }
        if any(projection[name] != expected
               for name, expected in expected_projection.items()):
            raise MetricError(
                'evaluation inventory projection disagrees with authority blob')
        observed_pairs = (
            ('annotation_observed_archive_sha256',
             'inventory_annotation_archive_sha256'),
            ('image_observed_archive_sha256',
             'inventory_image_archive_sha256'))
        if any(projection[observed] != value[protocol]
               for observed, protocol in observed_pairs):
            raise MetricError(
                'evaluation inventory observations disagree with protocol')
    return _freeze(value)


def validate_evaluation_envelope(
        value: object, *, expected_candidate_id: str | None = None,
        expected_route: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_data_inventory_sha256: str | None = None,
        expected_authority_sha256: str | None = None,
        expected_source_config: str | None = None,
        expected_checkpoint: str | None = None,
        expected_seed: int | None = None,
        expected_git_commit: str | None = None,
        expected_source_binding: Mapping[str, str] | None = None,
        expected_authority: Mapping[str, Any] | None = None,
        require_source_binding: bool = False,
        expected_pwl_stage_a: Mapping[str, str] | None = None,
        ) -> Mapping[str, Any]:
    """Validate the canonical formal dual-mode evaluation artifact."""
    required = {'schema_version', 'candidate_id', 'stage', 'result'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise MetricError('evaluation artifact must use the stage envelope')
    if value['schema_version'] != 1:
        raise MetricError('evaluation schema_version must be 1')
    if value['stage'] != 'evaluate':
        raise MetricError('evaluation artifact stage must be evaluate')
    candidate_id = value['candidate_id']
    if not isinstance(candidate_id, str) or not candidate_id:
        raise MetricError('evaluation candidate_id must be non-empty')
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise MetricError('evaluation candidate identity mismatch')
    result = value['result']
    expected_result_fields = {'route', 'calibration_split', 'modes', 'source'}
    if expected_pwl_stage_a is not None:
        expected_result_fields.add('pwl_stage_a')
    if (
            not isinstance(result, Mapping)
            or set(result) not in (
                expected_result_fields,
                expected_result_fields - {'source'})):
        raise MetricError('evaluation result has invalid fields')
    source = result.get('source')
    if require_source_binding and source is None:
        raise MetricError('evaluation source manifest binding is required')
    if source is not None:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_BINDING_FIELDS:
            raise MetricError('evaluation source manifest binding is invalid')
        if expected_source_binding is not None and dict(source) != dict(
                expected_source_binding):
            raise MetricError('evaluation source manifest binding mismatch')
    pwl_stage_a = _validate_pwl_stage_a_binding(
        result.get('pwl_stage_a'), expected=expected_pwl_stage_a)
    route = result['route']
    if not isinstance(route, str) or not route:
        raise MetricError('evaluation route must be non-empty')
    if expected_route is not None and route != expected_route:
        raise MetricError('evaluation route disagrees with candidate manifest')
    calibration_split = result['calibration_split']
    if calibration_split not in {None, 'train2017'}:
        raise MetricError(
            'calibration_split must be absent or train2017, never val2017')
    modes = result['modes']
    if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
        raise MetricError('evaluation must contain both flip and no_flip modes')
    normalized_modes: dict[str, Mapping[str, Any]] = {}
    mode_fields = {'metrics', 'provenance', 'determinism', 'protocol'}
    for name, row in modes.items():
        if not isinstance(row, Mapping) or set(row) != mode_fields:
            raise MetricError(f'evaluation {name} mode has invalid fields')
        provenance = validate_provenance(row['provenance'])
        determinism = _validate_determinism(row['determinism'], provenance)
        protocol = _validate_recorded_coco_protocol(
            row['protocol'], expected_authority=expected_authority)
        metrics = CocoMetrics.from_dict(row['metrics'])
        if expected_seed is not None and any(
                determinism[field] != expected_seed
                for field in ('python_seed', 'numpy_seed', 'torch_seed')):
            raise MetricError('evaluation seed disagrees with candidate manifest')
        if (
                expected_git_commit is not None
                and provenance['git_commit'] != expected_git_commit):
            raise MetricError('evaluation commit disagrees with source binding')
        if (
                expected_checkpoint_sha256 is not None
                and provenance['checkpoint_sha256'] != expected_checkpoint_sha256):
            raise MetricError(
                'evaluation checkpoint hash disagrees with candidate manifest')
        if (
                expected_data_inventory_sha256 is not None
                and provenance['data_inventory_sha256'] != (
                    expected_data_inventory_sha256)):
            raise MetricError(
                'evaluation data inventory hash disagrees with repository')
        if provenance['data_inventory_sha256'] != (
                protocol['inventory_projection']['inventory_sha256']):
            raise MetricError(
                'evaluation inventory observation disagrees with provenance')
        if (
                expected_authority_sha256 is not None
                and protocol['authority_sha256'] != expected_authority_sha256):
            raise MetricError(
                'evaluation corpus authority hash disagrees with repository')
        if (
                expected_source_config is not None
                and protocol['source_config'] != expected_source_config):
            raise MetricError(
                'evaluation config path disagrees with candidate manifest')
        if (
                expected_checkpoint is not None
                and protocol['checkpoint'] != expected_checkpoint):
            raise MetricError(
                'evaluation checkpoint path disagrees with candidate manifest')
        normalized_modes[name] = MappingProxyType({
            'metrics': metrics,
            'provenance': MappingProxyType(provenance),
            'determinism': determinism,
            'protocol': protocol,
        })
    for field in ('checkpoint_sha256', 'data_inventory_sha256', 'git_commit'):
        if normalized_modes['flip']['provenance'][field] != (
                normalized_modes['no_flip']['provenance'][field]):
            raise MetricError(f'evaluation mode provenance disagrees on {field}')
    for field in (
            'annotation_sha256', 'detection_sha256',
            'inventory_detection_sha256', 'annotation_image_count',
            'annotation_record_count', 'detection_record_count',
            'verified_image_count', 'source_config', 'checkpoint',
            'data_inventory', 'authority_path', 'authority_sha256',
            'authority_image_count', 'authority_annotation_count',
            'authority_detection_count', 'annotation_authority_sha256',
            'inventory_authority_sha256',
            'detection_authority_sha256', 'image_corpus_digest_algorithm',
            'image_corpus_authority_sha256', 'image_corpus_sha256',
            'inventory_annotation_archive_sha256',
            'inventory_image_archive_sha256', 'inventory_projection'):
        if normalized_modes['flip']['protocol'][field] != (
                normalized_modes['no_flip']['protocol'][field]):
            raise MetricError(f'evaluation mode protocol disagrees on {field}')
    return MappingProxyType({
        'candidate_id': candidate_id,
        'route': route,
        'calibration_split': calibration_split,
        'modes': MappingProxyType(normalized_modes),
        'source': _freeze(source) if source is not None else None,
        **({'pwl_stage_a': _freeze(pwl_stage_a)}
           if pwl_stage_a is not None else {}),
    })


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
    pwl_stage_a: Mapping[str, str] | None
    artifact_paths: Mapping[str, Path]
    evaluation_artifact: Path

    @classmethod
    def from_artifacts(
            cls, root: Path | str, *, mode: str = 'flip') -> 'CandidateResult':
        root = Path(root)
        _reject_artifact_root_symlinks(root)
        if root.is_file() or not root.is_dir():
            raise MetricError('candidate artifact root must be a directory')
        if mode not in {'flip', 'no_flip'}:
            raise MetricError('evaluation mode must be flip or no_flip')
        path = root / 'evaluate/evaluate.json'
        try:
            envelope = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise MetricError(f'cannot load evaluation artifact: {error}') from error
        repository_root = _repository_root_from_artifacts(root)
        source, candidate, authority = resolve_artifact_source(
            envelope, repository_root=repository_root)
        profile_path = root / 'profile/profile.json'
        expected_runtime: Mapping[str, Any] = MappingProxyType({
            'config': MappingProxyType({
                'path': candidate.config.as_posix(),
                'sha256': source['config_sha256'],
            }),
            'checkpoint': MappingProxyType({
                'path': candidate.checkpoint.as_posix(),
                'sha256': candidate.checkpoint_sha256,
            }),
        })
        expected_profile_parent: Mapping[str, str] | None = None
        expected_profile_source: Mapping[str, Any] | None = None
        expected_pwl_stage_a: Mapping[str, str] | None = None
        if candidate.route == 'ssm-quant-pwl':
            try:
                from .numeric_runtime import resolve_numeric_runtime
                from .numeric_source import validate_numeric_source_binding

                raw_profile = json.loads(
                    profile_path.read_text(encoding='utf-8'))
                if not isinstance(raw_profile, Mapping):
                    raise ValueError('numeric profile root must be an object')
                expected_profile_source = validate_numeric_source_binding(
                    raw_profile.get('source'), repository_root=repository_root,
                    candidate=candidate,
                    manifest_path=repository_root / source['manifest_path'])
                numeric_runtime = resolve_numeric_runtime(
                    candidate, repository_root=repository_root,
                    manifest_path=repository_root / source['manifest_path'],
                    downstream_output=profile_path)
                expected_runtime = MappingProxyType({
                    'config': MappingProxyType({
                        'path': numeric_runtime['config_path'].relative_to(
                            repository_root).as_posix(),
                        'sha256': numeric_runtime['config_sha256'],
                    }),
                    'checkpoint': MappingProxyType({
                        'path': numeric_runtime['checkpoint_name'],
                        'sha256': numeric_runtime['checkpoint_sha256'],
                    }),
                })
                expected_profile_parent = MappingProxyType({
                    'config': candidate.config.as_posix(),
                    'checkpoint': candidate.checkpoint.as_posix(),
                    'checkpoint_sha256': candidate.checkpoint_sha256,
                })
                expected_pwl_stage_a = numeric_runtime.get('pwl_stage_a')
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise MetricError(
                    f'numeric profile authority is invalid: {error}') from error
        validated = validate_evaluation_envelope(
            envelope,
            expected_candidate_id=candidate.id,
            expected_route=candidate.route,
            expected_checkpoint_sha256=expected_runtime[
                'checkpoint']['sha256'],
            expected_authority_sha256=source['authority_sha256'],
            expected_source_config=expected_runtime['config']['path'],
            expected_checkpoint=expected_runtime['checkpoint']['path'],
            expected_seed=candidate.seed,
            expected_git_commit=source['git_commit'],
            expected_source_binding=source,
            expected_authority=authority,
            require_source_binding=True,
            expected_pwl_stage_a=expected_pwl_stage_a,
        )
        candidate_id = validated['candidate_id']
        route = validated['route']
        calibration_split = validated['calibration_split']
        selected = validated['modes'][mode]
        metrics = selected['metrics']
        provenance = selected['provenance']
        determinism = selected['determinism']
        protocol = selected['protocol']
        profile: Mapping[str, Any] | None = None
        latency: Mapping[str, Any] | None = None
        gpu_lease: Mapping[str, Any] | None = None
        artifact_paths: dict[str, Path] = {'evaluation': path.resolve()}
        latency_path = root / 'latency/latency.json'
        profile = _load_profile(
            profile_path, candidate_id=candidate_id,
            provenance=provenance,
            expected_runtime=(
                expected_runtime
                if candidate.route == 'ssm-quant-pwl' else None),
            expected_parent=expected_profile_parent,
            expected_source=expected_profile_source,
            expected_pwl_stage_a=expected_pwl_stage_a)
        latency, gpu_lease = _load_latency(
            latency_path, candidate_id=candidate_id, route=route,
            provenance=provenance, source=source,
            expected_runtime=expected_runtime, authority=authority,
            expected_pwl_stage_a=expected_pwl_stage_a)
        if (
                profile['schema_version'] == 2
                and profile['device']['physical_index'] !=
                gpu_lease['device_index']):
            raise MetricError(
                'profile CUDA device disagrees with latency GPU lease')
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
                'authority_path', 'authority_sha256',
                'authority_image_count', 'authority_annotation_count',
                'authority_detection_count',
                'annotation_authority_sha256',
                'detection_authority_sha256', 'annotation_sha256',
                'detection_sha256',
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
        if expected_pwl_stage_a is not None:
            smoke_path = repository_root / expected_pwl_stage_a['path']
            if (_file_sha256(smoke_path) != expected_pwl_stage_a['sha256']
                    or validated.get('pwl_stage_a') != expected_pwl_stage_a
                    or profile.get('pwl_stage_a') != expected_pwl_stage_a
                    or latency.get('pwl_stage_a') != expected_pwl_stage_a):
                raise MetricError(
                    'PWL Stage-A smoke binding differs across artifacts')
            artifact_paths['pwl_stage_a'] = smoke_path.resolve()
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
            pwl_stage_a=(
                _freeze(expected_pwl_stage_a)
                if expected_pwl_stage_a is not None else None),
            artifact_paths=MappingProxyType(artifact_paths),
            evaluation_artifact=path.resolve(),
        )


def _load_profile(
        path: Path, *, candidate_id: str,
        provenance: Mapping[str, str],
        expected_runtime: Mapping[str, Any] | None = None,
        expected_parent: Mapping[str, str] | None = None,
        expected_source: Mapping[str, Any] | None = None,
        expected_pwl_stage_a: Mapping[str, str] | None = None,
        ) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot load profile artifact: {error}') from error
    legacy_fields = {
        'schema_version', 'git_commit', 'candidate', 'config', 'checkpoint',
        'checkpoint_sha256', 'input_shapes', 'output_shapes', 'parameters',
        'modules',
    }
    runtime_fields = legacy_fields | {'device', 'parent', 'runtime'}
    if expected_source is not None:
        runtime_fields.add('source')
    if expected_pwl_stage_a is not None:
        runtime_fields.add('pwl_stage_a')
    if (
            not isinstance(value, Mapping)
            or set(value) not in (legacy_fields, runtime_fields)):
        raise MetricError('profile artifact has invalid fields')
    expected_schema = 2 if set(value) == runtime_fields else 1
    if (
            value['schema_version'] != expected_schema
            or value['candidate'] != candidate_id):
        raise MetricError('profile artifact identity mismatch')
    if expected_runtime is not None:
        if expected_schema != 2 or value['runtime'] != expected_runtime:
            raise MetricError('profile runtime deployment binding mismatch')
    if expected_parent is not None and value.get('parent') != expected_parent:
        raise MetricError('profile parent binding mismatch')
    if expected_source is not None and value.get('source') != expected_source:
        raise MetricError('profile source binding mismatch')
    _validate_pwl_stage_a_binding(
        value.get('pwl_stage_a'), expected=expected_pwl_stage_a)
    if expected_schema == 2:
        device = value['device']
        if (
                not isinstance(device, Mapping)
                or set(device) != {'logical', 'physical_index', 'kind'}
                or device.get('logical') != 'cuda:0'
                or device.get('kind') != 'cuda'
                or isinstance(device.get('physical_index'), bool)
                or not isinstance(device.get('physical_index'), int)
                or device['physical_index'] < 0):
            raise MetricError('profile CUDA device provenance is invalid')
        parent = value['parent']
        if (
                not isinstance(parent, Mapping)
                or set(parent) != {
                    'config', 'checkpoint', 'checkpoint_sha256'}
                or any(not isinstance(parent[name], str) or not parent[name]
                       for name in parent)):
            raise MetricError('profile parent provenance is invalid')
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
        provenance: Mapping[str, str], source: Mapping[str, str],
        expected_runtime: Mapping[str, Any], authority: Mapping[str, Any],
        expected_pwl_stage_a: Mapping[str, str] | None = None,
        ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        envelope = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricError(f'cannot load latency artifact: {error}') from error
    validated = validate_latency_envelope(
        envelope, expected_candidate_id=candidate_id, expected_route=route,
        expected_checkpoint_sha256=expected_runtime[
            'checkpoint']['sha256'],
        expected_data_inventory_sha256=provenance['data_inventory_sha256'],
        expected_authority_sha256=source['authority_sha256'],
        expected_source_config=expected_runtime['config']['path'],
        expected_checkpoint=expected_runtime['checkpoint']['path'],
        expected_git_commit=provenance['git_commit'],
        expected_config_sha256=expected_runtime['config']['sha256'],
        expected_source_binding=source,
        expected_authority=authority,
        require_source_binding=True,
        expected_pwl_stage_a=expected_pwl_stage_a)
    return validated, validated['gpu_lease']


def validate_latency_envelope(
        envelope: object, *, expected_candidate_id: str | None = None,
        expected_route: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_data_inventory_sha256: str | None = None,
        expected_authority_sha256: str | None = None,
        expected_source_config: str | None = None,
        expected_checkpoint: str | None = None,
        expected_git_commit: str | None = None,
        expected_device_index: int | None = None,
        expected_config_sha256: str | None = None,
        expected_source_binding: Mapping[str, str] | None = None,
        expected_authority: Mapping[str, Any] | None = None,
        require_source_binding: bool = False,
        expected_pwl_stage_a: Mapping[str, str] | None = None,
        ) -> Mapping[str, Any]:
    """Validate the one formal dual-mode latency stage envelope."""
    if (
            not isinstance(envelope, Mapping)
            or set(envelope) != {
                'schema_version', 'candidate_id', 'stage', 'result'}
            or envelope['schema_version'] != 1
            or envelope['stage'] != 'latency'):
        raise MetricError('latency artifact envelope identity mismatch')
    candidate_id = envelope['candidate_id']
    if not isinstance(candidate_id, str) or not candidate_id:
        raise MetricError('latency candidate identity mismatch')
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise MetricError('latency candidate identity mismatch')
    result = envelope['result']
    if (
            not isinstance(result, Mapping)
            or set(result) not in ({
                'route', 'provenance', 'protocol', 'modes', 'gpu_lease',
                'source'}, {
                'route', 'provenance', 'protocol', 'modes', 'gpu_lease'}, {
                'route', 'provenance', 'protocol', 'modes', 'gpu_lease',
                'source', 'pwl_stage_a'}, {
                'route', 'provenance', 'protocol', 'modes', 'gpu_lease',
                'pwl_stage_a'})
            or not isinstance(result['route'], str)
            or not result['route']):
        raise MetricError('latency artifact provenance or route mismatch')
    source = result.get('source')
    if require_source_binding and source is None:
        raise MetricError('latency source manifest binding is required')
    if source is not None:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_BINDING_FIELDS:
            raise MetricError('latency source manifest binding is invalid')
        if expected_source_binding is not None and dict(source) != dict(
                expected_source_binding):
            raise MetricError('latency source manifest binding mismatch')
    pwl_stage_a = _validate_pwl_stage_a_binding(
        result.get('pwl_stage_a'), expected=expected_pwl_stage_a)
    if expected_route is not None and result['route'] != expected_route:
        raise MetricError('latency artifact provenance or route mismatch')
    latency_provenance = validate_provenance(result['provenance'])
    expected_provenance = {
        'checkpoint_sha256': expected_checkpoint_sha256,
        'data_inventory_sha256': expected_data_inventory_sha256,
        'git_commit': expected_git_commit,
        'config_sha256': expected_config_sha256,
    }
    if any(expected is not None and latency_provenance[field] != expected
           for field, expected in expected_provenance.items()):
        raise MetricError('latency artifact provenance or route mismatch')
    protocol = result['protocol']
    protocol_fields = {
        'batch_size', 'warmup', 'iterations', 'timer', 'synchronize',
        'scope', 'lease_max_age_seconds',
        'lease_max_future_skew_seconds', 'source_config', 'checkpoint',
        'data_inventory', 'data'}
    if (
            not isinstance(protocol, Mapping)
            or set(protocol) != protocol_fields
            or protocol.get('batch_size') != 1
            or protocol.get('timer') != 'torch.cuda.Event'
            or protocol.get('synchronize') is not True
            or protocol.get('scope') != 'full_topdown_model'
            or isinstance(protocol.get('lease_max_age_seconds'), bool)
            or not isinstance(protocol.get('lease_max_age_seconds'), int)
            or protocol.get('lease_max_age_seconds') != 300
            or isinstance(protocol.get('lease_max_future_skew_seconds'), bool)
            or not isinstance(
                protocol.get('lease_max_future_skew_seconds'), int)
            or protocol.get('lease_max_future_skew_seconds') != 30
            or isinstance(protocol.get('warmup'), bool)
            or not isinstance(protocol.get('warmup'), int)
            or protocol['warmup'] != 50
            or isinstance(protocol.get('iterations'), bool)
            or not isinstance(protocol.get('iterations'), int)
            or protocol['iterations'] != 200):
        raise MetricError('latency protocol is invalid')
    latency_data = protocol['data']
    if not isinstance(latency_data, Mapping):
        raise MetricError('latency data protocol is invalid')
    data_protocol = _validate_recorded_coco_protocol({
        **latency_data,
        'batch_size': protocol['batch_size'],
        'source_config': protocol['source_config'],
        'checkpoint': protocol['checkpoint'],
        'data_inventory': protocol['data_inventory'],
    }, expected_authority=expected_authority)
    if latency_provenance['data_inventory_sha256'] != (
            data_protocol['inventory_projection']['inventory_sha256']):
        raise MetricError(
            'latency inventory observation disagrees with provenance')
    if (
            expected_authority_sha256 is not None
            and data_protocol['authority_sha256'] != expected_authority_sha256):
        raise MetricError('latency corpus authority hash mismatch')
    if (
            expected_source_config is not None
            and protocol['source_config'] != expected_source_config):
        raise MetricError('latency source config mismatch')
    if (
            expected_checkpoint is not None
            and protocol['checkpoint'] != expected_checkpoint):
        raise MetricError('latency checkpoint path mismatch')
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
    if (
            expected_device_index is not None
            and lease['device_index'] != expected_device_index):
        raise MetricError('latency GPU lease device mismatch')
    return _freeze(result)


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
