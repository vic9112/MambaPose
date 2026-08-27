"""Strict runtime-checkpoint handoff for conditional Route 3 candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .numeric_source import (
    file_sha256, validate_numeric_source_binding)
from .schema import CandidateSpec


class NumericRuntimeError(ValueError):
    """Raised when a trained numeric runtime reference is missing or stale."""


def _file(root: Path, record: object, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {'path', 'sha256'}:
        raise NumericRuntimeError(f'{label} reference is invalid')
    relative = Path(str(record['path']))
    if relative.is_absolute() or any(part in {'.', '..'} for part in relative.parts):
        raise NumericRuntimeError(f'{label} path is unsafe')
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise NumericRuntimeError(f'{label} path must not use symlinks')
    if (not cursor.is_file() or not isinstance(record['sha256'], str)
            or file_sha256(cursor) != record['sha256']):
        raise NumericRuntimeError(f'{label} hash changed')
    return cursor


def validate_numeric_train_artifact(
        value: Mapping[str, Any], *, candidate: CandidateSpec,
        repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) != {'schema_version', 'candidate_id', 'stage', 'result'}
            or value.get('schema_version') != 1
            or value.get('candidate_id') != candidate.id
            or value.get('stage') != 'train'
            or not isinstance(value.get('result'), Mapping)):
        raise NumericRuntimeError('numeric train envelope identity is invalid')
    result = value['result']
    if set(result) != {
            'route', 'source', 'parent', 'dependency', 'protocol', 'runtime'}:
        raise NumericRuntimeError('numeric train result fields are invalid')
    if result['route'] != 'ssm-quant-pwl':
        raise NumericRuntimeError('numeric train route is invalid')
    validate_numeric_source_binding(
        result['source'], repository_root=repository_root,
        candidate=candidate, manifest_path=manifest_path)
    parent = result['parent']
    if (not isinstance(parent, Mapping)
            or set(parent) != {'config', 'checkpoint', 'checkpoint_sha256'}
            or parent['config'] != candidate.config.as_posix()
            or parent['checkpoint'] != candidate.checkpoint.as_posix()
            or parent['checkpoint_sha256'] != candidate.checkpoint_sha256):
        raise NumericRuntimeError('numeric train parent identity is invalid')
    protocol = result['protocol']
    if (not isinstance(protocol, Mapping)
            or set(protocol) != {
                'seed', 'operation', 'attributed_error',
                'max_preliminary_ap_drop'}
            or protocol['seed'] != candidate.seed
            or protocol['operation'] != 'one-bounded-numeric-recovery'
            or not isinstance(protocol['attributed_error'], str)
            or not protocol['attributed_error']
            or isinstance(protocol['max_preliminary_ap_drop'], bool)
            or not isinstance(protocol['max_preliminary_ap_drop'], (int, float))
            or not 0 <= protocol['max_preliminary_ap_drop'] <= 0.3):
        raise NumericRuntimeError('numeric train protocol is invalid')
    dependency = result['dependency']
    if (not isinstance(dependency, Mapping)
            or set(dependency) not in (
                {'recovery_admission'},
                {'recovery_admission', 'calibration'})):
        raise NumericRuntimeError('numeric train dependency is invalid')
    for name, reference in dependency.items():
        _file(repository_root, reference, f'numeric train {name}')
    runtime = result['runtime']
    if (not isinstance(runtime, Mapping)
            or set(runtime) != {
                'config', 'checkpoint', 'metadata', 'transform'}
            or runtime['transform'] != 'bounded-numeric-recovery-v1'):
        raise NumericRuntimeError('numeric runtime fields are invalid')
    config = _file(repository_root, runtime['config'], 'runtime config')
    checkpoint = _file(
        repository_root, runtime['checkpoint'], 'runtime checkpoint')
    metadata_path = _file(
        repository_root, runtime['metadata'], 'runtime metadata')
    expected = candidate.features.get('runtime_checkpoint')
    if not isinstance(expected, str) or runtime['checkpoint']['path'] != expected:
        raise NumericRuntimeError('runtime checkpoint path disagrees with manifest')
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericRuntimeError('runtime metadata is invalid JSON') from error
    if (not isinstance(metadata, Mapping)
            or set(metadata) != {
                'schema_version', 'candidate_id', 'route', 'numeric_kind',
                'parent_checkpoint_sha256', 'runtime_checkpoint_sha256',
                'recovery_admission_sha256'}
            or metadata.get('schema_version') != 1
            or metadata.get('candidate_id') != candidate.id
            or metadata.get('route') != candidate.route
            or metadata.get('numeric_kind') != candidate.features.get('numeric_kind')
            or metadata.get('parent_checkpoint_sha256') !=
                candidate.checkpoint_sha256
            or metadata.get('runtime_checkpoint_sha256') !=
                runtime['checkpoint']['sha256']
            or metadata.get('recovery_admission_sha256') !=
                dependency['recovery_admission']['sha256']):
        raise NumericRuntimeError('runtime metadata identity is invalid')
    return {
        'config_path': config,
        'config_sha256': runtime['config']['sha256'],
        'checkpoint_path': checkpoint,
        'checkpoint_sha256': runtime['checkpoint']['sha256'],
        'train': dict(value),
    }


def resolve_numeric_runtime(
        candidate: CandidateSpec, *, repository_root: Path,
        manifest_path: Path, downstream_output: Path) -> dict[str, Any]:
    if candidate.route != 'ssm-quant-pwl':
        config_path = repository_root / candidate.config
        return {
            'config_path': config_path,
            'config_sha256': (
                file_sha256(config_path) if config_path.is_file() else None),
            'checkpoint_path': repository_root / candidate.checkpoint,
            'checkpoint_sha256': candidate.checkpoint_sha256,
            'train': None,
        }
    if candidate.features.get('numeric_kind') == 'w8a8' \
            and candidate.features.get('recovery_candidate') is not True:
        conversion_path = downstream_output.parent.parent / 'convert/convert.json'
        try:
            conversion = json.loads(conversion_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise NumericRuntimeError(
                'W8A8 runtime requires completed convert artifact') from error
        if (not isinstance(conversion, Mapping)
                or set(conversion) != {
                    'schema_version', 'candidate_id', 'stage', 'result'}
                or conversion.get('schema_version') != 1
                or conversion.get('candidate_id') != candidate.id
                or conversion.get('stage') != 'convert'
                or not isinstance(conversion.get('result'), Mapping)):
            raise NumericRuntimeError('W8A8 convert envelope is invalid')
        result = conversion['result']
        validate_numeric_source_binding(
            result.get('source'), repository_root=repository_root,
            candidate=candidate, manifest_path=manifest_path)
        runtime_config = _file(
            repository_root, result.get('runtime_config'),
            'W8A8 runtime config')
        bindings = result.get('runtime_bindings')
        if (not isinstance(bindings, Mapping)
                or set(bindings) != {
                    'config', 'checkpoint', 'policy', 'calibration'}):
            raise NumericRuntimeError('W8A8 runtime bindings are incomplete')
        for name, reference in bindings.items():
            _file(repository_root, reference, f'W8A8 {name}')
        return {
            'config_path': runtime_config,
            'config_sha256': result['runtime_config']['sha256'],
            'checkpoint_path': repository_root / candidate.checkpoint,
            'checkpoint_sha256': candidate.checkpoint_sha256,
            'train': None,
        }
    if candidate.features.get('recovery_candidate') is not True:
        return {
            'config_path': repository_root / candidate.config,
            'config_sha256': file_sha256(repository_root / candidate.config),
            'checkpoint_path': repository_root / candidate.checkpoint,
            'checkpoint_sha256': candidate.checkpoint_sha256,
            'train': None,
        }
    train_path = downstream_output.parent.parent / 'train/train.json'
    if not train_path.is_file():
        raise NumericRuntimeError(
            'conditional numeric runtime requires completed train artifact')
    try:
        value = json.loads(train_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericRuntimeError('numeric train artifact is unreadable') from error
    return validate_numeric_train_artifact(
        value, candidate=candidate, repository_root=repository_root,
        manifest_path=manifest_path)


__all__ = [
    'NumericRuntimeError', 'resolve_numeric_runtime',
    'validate_numeric_train_artifact']
