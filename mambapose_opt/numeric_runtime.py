"""Strict runtime-checkpoint handoff for conditional Route 3 candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .numeric_source import (
    file_sha256, resolve_numeric_file, validate_numeric_source_binding)
from .schema import CandidateSpec


class NumericRuntimeError(ValueError):
    """Raised when a trained numeric runtime reference is missing or stale."""


def validate_numeric_convert_artifact(
        value: Mapping[str, Any], *, candidate: CandidateSpec,
        repository_root: Path, manifest_path: Path,
        artifact_path: Path) -> dict[str, Any]:
    stage = value.get('stage') if isinstance(value, Mapping) else None
    if (stage not in {'convert', 'export'}
            or set(value) != {'schema_version', 'candidate_id', 'stage', 'result'}
            or value.get('schema_version') != 1
            or value.get('candidate_id') != candidate.id
            or not isinstance(value.get('result'), Mapping)):
        raise NumericRuntimeError('numeric convert envelope identity is invalid')
    kind = candidate.features.get('numeric_kind')
    if kind not in {'weight-only', 'w8a8'}:
        raise NumericRuntimeError('numeric convert candidate kind is invalid')
    result = value['result']
    expected_fields = {
        'source', 'runtime_bindings', 'conversion', 'precision_invariants',
        'latency_claim'}
    if kind == 'w8a8':
        expected_fields.add('runtime_config')
    if stage == 'export':
        expected_fields.add('export')
    if set(result) != expected_fields:
        raise NumericRuntimeError('numeric convert result fields are invalid')
    if result.get('latency_claim') != (
            'none-fake-quant-is-not-an-integer-kernel'):
        raise NumericRuntimeError('numeric fake-QDQ latency claim is invalid')
    source = validate_numeric_source_binding(
        result['source'], repository_root=repository_root,
        candidate=candidate, manifest_path=manifest_path)
    bindings = result['runtime_bindings']
    expected_roles = {'config', 'checkpoint', 'policy'}
    if kind == 'w8a8':
        expected_roles.add('calibration')
    if not isinstance(bindings, Mapping) or set(bindings) != expected_roles:
        raise NumericRuntimeError('numeric runtime bindings are incomplete')
    expected_static = {
        'config': (candidate.config.as_posix(), source['config_sha256']),
        'checkpoint': (
            candidate.checkpoint.as_posix(), candidate.checkpoint_sha256),
        'policy': (candidate.config.as_posix(), source['policy_sha256']),
    }
    for role, (path, sha256) in expected_static.items():
        if bindings.get(role) != {'path': path, 'sha256': sha256}:
            raise NumericRuntimeError(
                f'numeric {role} binding is not canonical')
        _file(repository_root, bindings[role], f'numeric {role}')
    conversion = result['conversion']
    conversion_fields = {
        'converted', 'skipped', 'original_weight_bytes',
        'simulated_weight_bytes', 'simulated_coverage'}
    if (not isinstance(conversion, Mapping)
            or set(conversion) != conversion_fields
            or not isinstance(conversion.get('converted'), list)
            or not conversion['converted']
            or len(conversion['converted']) != len(set(conversion['converted']))
            or not isinstance(conversion.get('skipped'), list)
            or any(not isinstance(item, str) or not item
                   for item in conversion['converted'] + conversion['skipped'])
            or any(isinstance(conversion.get(field), bool)
                   or not isinstance(conversion.get(field), int)
                   or conversion[field] < 0 for field in (
                       'original_weight_bytes', 'simulated_weight_bytes'))
            or isinstance(conversion.get('simulated_coverage'), bool)
            or not isinstance(conversion.get('simulated_coverage'), (int, float))
            or not 0 <= conversion['simulated_coverage'] <= 1):
        raise NumericRuntimeError('numeric conversion report is invalid')
    from mmengine.config import Config
    policy_config = Config.fromfile(repository_root / candidate.config)
    numeric = policy_config.get('numeric_optimization')
    if not isinstance(numeric, Mapping):
        raise NumericRuntimeError('numeric policy config is missing')
    policy = numeric.get('quant_policy')
    if (not isinstance(policy, Mapping)
            or conversion['converted'] != list(policy.get('allow', ()))
            or conversion['skipped'] != list(policy.get('deny', ()))
            or result['precision_invariants'] != dict(
                numeric.get('precision_invariants', {}))):
        raise NumericRuntimeError(
            'numeric conversion report disagrees with policy')
    if kind == 'w8a8':
        expected_calibration = (
            artifact_path.parent.parent / 'calibrate/calibrate.json')
        try:
            expected_relative = expected_calibration.relative_to(
                repository_root).as_posix()
        except ValueError as error:
            raise NumericRuntimeError(
                'W8A8 calibration stage escapes repository') from error
        calibration_reference = bindings.get('calibration')
        if (not isinstance(calibration_reference, Mapping)
                or calibration_reference.get('path') != expected_relative):
            raise NumericRuntimeError(
                'W8A8 calibration dependency is not the canonical stage output')
        calibration_path = _file(
            repository_root, calibration_reference, 'W8A8 calibration')
        try:
            calibration = json.loads(
                calibration_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise NumericRuntimeError('W8A8 calibration is invalid JSON') from error
        from .numeric_calibration import (
            CalibrationContractError, validate_calibration_provenance)
        try:
            validate_calibration_provenance(
                calibration, expected_candidate=candidate,
                repository_root=repository_root,
                manifest_path=manifest_path)
        except CalibrationContractError as error:
            raise NumericRuntimeError(
                f'W8A8 calibration contract is invalid: {error}') from error
        runtime_reference = result['runtime_config']
        expected_runtime = artifact_path.parent / 'resolved-runtime.py'
        if (not isinstance(runtime_reference, Mapping)
                or runtime_reference.get('path') != expected_runtime.relative_to(
                    repository_root).as_posix()):
            raise NumericRuntimeError(
                'W8A8 runtime config is not the canonical convert output')
        runtime_path = _file(
            repository_root, runtime_reference, 'W8A8 runtime config')
        policy_config.numeric_optimization.quant_policy.calibration_artifact = {
            'path': expected_relative,
            'sha256': calibration_reference['sha256']}
        if Config.fromfile(runtime_path).to_dict() != policy_config.to_dict():
            raise NumericRuntimeError(
                'W8A8 runtime config was not derived from candidate policy')
    if stage == 'export':
        exported = result['export']
        if (not isinstance(exported, Mapping)
                or set(exported) != {'path', 'sha256', 'bytes', 'format'}
                or exported.get('format') !=
                'symmetric-int8-per-output-channel-v1'):
            raise NumericRuntimeError('numeric export reference is invalid')
        export_path = _file(repository_root, {
            'path': exported.get('path'), 'sha256': exported.get('sha256')},
            'numeric export')
        if (not isinstance(exported.get('bytes'), int)
                or isinstance(exported['bytes'], bool)
                or exported['bytes'] != export_path.stat().st_size):
            raise NumericRuntimeError('numeric export byte count is invalid')
    return dict(value)


def _file(root: Path, record: object, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {'path', 'sha256'}:
        raise NumericRuntimeError(f'{label} reference is invalid')
    relative = Path(str(record['path']))
    if relative.is_absolute() or any(part in {'.', '..'} for part in relative.parts):
        raise NumericRuntimeError(f'{label} path is unsafe')
    try:
        cursor = resolve_numeric_file(root, relative, label)
    except ValueError as error:
        raise NumericRuntimeError(str(error)) from error
    if (not cursor.is_file() or not isinstance(record['sha256'], str)
            or file_sha256(cursor) != record['sha256']):
        raise NumericRuntimeError(f'{label} hash changed')
    return cursor


def _evaluation_ap(
        reference: object, *, expected_path: str,
        candidate: CandidateSpec, repository_root: Path,
        manifest_path: Path) -> float:
    if not isinstance(reference, Mapping) or reference.get('path') != expected_path:
        raise NumericRuntimeError('recovery evaluation reference is not canonical')
    path = _file(repository_root, reference, 'recovery evaluation')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        from .evaluation import (
            resolve_artifact_source, validate_evaluation_envelope)
        source, selected, authority = resolve_artifact_source(
            value, repository_root=repository_root,
            expected_manifest_path=manifest_path)
        if selected != candidate:
            raise NumericRuntimeError(
                'recovery evaluation candidate identity is invalid')
        runtime = resolve_numeric_runtime(
            candidate, repository_root=repository_root,
            manifest_path=manifest_path, downstream_output=path)
        validated = validate_evaluation_envelope(
            value, expected_candidate_id=candidate.id,
            expected_route=candidate.route,
            expected_checkpoint_sha256=runtime['checkpoint_sha256'],
            expected_authority_sha256=source['authority_sha256'],
            expected_source_config=runtime['config_path'].relative_to(
                repository_root).as_posix(),
            expected_checkpoint=runtime['checkpoint_path'].relative_to(
                repository_root).as_posix(),
            expected_seed=candidate.seed,
            expected_git_commit=source['git_commit'],
            expected_source_binding=source, expected_authority=authority,
            require_source_binding=True)
    except NumericRuntimeError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise NumericRuntimeError(
            f'recovery evaluation evidence is invalid: {error}') from error
    ap = validated['modes']['flip']['metrics']['AP']
    if isinstance(ap, bool) or not isinstance(ap, (int, float)):
        raise NumericRuntimeError('recovery evaluation AP is invalid')
    return float(ap)


def validate_recovery_admission(
        path: Path, *, candidate: CandidateSpec, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericRuntimeError(
            'numeric recovery admission is unreadable') from error
    fields = {
        'schema_version', 'candidate_id', 'decision', 'attributed_error',
        'threshold_ap', 'preliminary_ap_drop', 'baseline_evaluation',
        'candidate_evaluation'}
    screen_id = candidate.features.get('recovery_screen_candidate')
    baseline_path = candidate.features.get('recovery_baseline_evaluation')
    screen_path = candidate.features.get('recovery_candidate_evaluation')
    if (not isinstance(value, Mapping) or set(value) != fields
            or value.get('schema_version') != 1
            or value.get('candidate_id') != candidate.id
            or value.get('decision') != 'admit-one-bounded-recovery'
            or not isinstance(value.get('attributed_error'), str)
            or not value['attributed_error']
            or value.get('threshold_ap') != 0.3
            or isinstance(value.get('preliminary_ap_drop'), bool)
            or not isinstance(value.get('preliminary_ap_drop'), (int, float))
            or value['preliminary_ap_drop'] <= value['threshold_ap']
            or not all(isinstance(item, str) and item for item in (
                screen_id, baseline_path, screen_path))):
        raise NumericRuntimeError('numeric recovery admission contract is invalid')
    from .schema import load_candidate_manifest
    candidates = load_candidate_manifest(manifest_path)
    baseline = tuple(item for item in candidates if item.id == 'full-s-v1')
    screen = tuple(item for item in candidates if item.id == screen_id)
    if len(baseline) != 1 or len(screen) != 1:
        raise NumericRuntimeError(
            'numeric recovery screen candidates are not canonical')
    baseline_ap = _evaluation_ap(
        value['baseline_evaluation'], expected_path=baseline_path,
        candidate=baseline[0], repository_root=repository_root,
        manifest_path=manifest_path)
    screen_ap = _evaluation_ap(
        value['candidate_evaluation'], expected_path=screen_path,
        candidate=screen[0], repository_root=repository_root,
        manifest_path=manifest_path)
    observed_drop = baseline_ap - screen_ap
    if (observed_drop <= 0.3
            or abs(observed_drop - value['preliminary_ap_drop']) > 1e-9):
        raise NumericRuntimeError(
            'numeric recovery admission disagrees with preliminary AP evidence')
    return dict(value)


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
                'preliminary_ap_drop'}
            or protocol['seed'] != candidate.seed
            or protocol['operation'] != 'one-bounded-numeric-recovery'
            or not isinstance(protocol['attributed_error'], str)
            or not protocol['attributed_error']
            or isinstance(protocol['preliminary_ap_drop'], bool)
            or not isinstance(protocol['preliminary_ap_drop'], (int, float))
            or protocol['preliminary_ap_drop'] <= 0.3):
        raise NumericRuntimeError('numeric train protocol is invalid')
    dependency = result['dependency']
    if (not isinstance(dependency, Mapping)
            or set(dependency) not in (
                {'recovery_admission'},
                {'recovery_admission', 'calibration'})):
        raise NumericRuntimeError('numeric train dependency is invalid')
    dependency_paths = {
        name: _file(repository_root, reference, f'numeric train {name}')
        for name, reference in dependency.items()}
    admission = validate_recovery_admission(
        dependency_paths['recovery_admission'], candidate=candidate,
        repository_root=repository_root, manifest_path=manifest_path)
    if (protocol['attributed_error'] != admission['attributed_error']
            or protocol['preliminary_ap_drop'] !=
            admission['preliminary_ap_drop']):
        raise NumericRuntimeError(
            'numeric train protocol disagrees with recovery admission')
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
    stage_dir = config.parent
    if (config != stage_dir / 'resolved-train.py'
            or checkpoint != stage_dir / 'best_numeric.pth'
            or metadata_path != stage_dir / 'runtime-metadata.json'
            or dependency_paths['recovery_admission'] !=
            stage_dir / 'recovery-admission.json'):
        raise NumericRuntimeError(
            'numeric train runtime files are not canonical stage outputs')
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
        validate_numeric_convert_artifact(
            conversion, candidate=candidate, repository_root=repository_root,
            manifest_path=manifest_path, artifact_path=conversion_path)
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
    'validate_numeric_convert_artifact', 'validate_numeric_train_artifact',
    'validate_recovery_admission']
