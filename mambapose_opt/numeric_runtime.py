"""Strict runtime-checkpoint handoff for conditional Route 3 candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .numeric_source import (
    file_sha256, resolve_numeric_file, validate_numeric_config_closure,
    validate_numeric_source_binding)
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
        'simulated_weight_bytes', 'simulated_coverage', 'simulation_only',
        'integer_kernel_latency_claimed'}
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
            or not 0 <= conversion['simulated_coverage'] <= 1
            or conversion.get('simulation_only') is not True
            or conversion.get('integer_kernel_latency_claimed') is not False):
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
        reference: object, *, expected_reference: Mapping[str, str],
        candidate: CandidateSpec, repository_root: Path,
        manifest_path: Path) -> float:
    if not isinstance(reference, Mapping) or dict(reference) != dict(
            expected_reference):
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
        if any(
                validated['modes'][mode]['provenance']['config_sha256'] !=
                runtime['config_sha256'] for mode in ('flip', 'no_flip')):
            raise NumericRuntimeError(
                'recovery evaluation runtime config hash is invalid')
    except NumericRuntimeError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise NumericRuntimeError(
            f'recovery evaluation evidence is invalid: {error}') from error
    ap = validated['modes']['flip']['metrics'].ap
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
    baseline_reference = {
        'path': candidate.features.get('recovery_baseline_evaluation'),
        'sha256': candidate.features.get(
            'recovery_baseline_evaluation_sha256')}
    screen_reference = {
        'path': candidate.features.get('recovery_candidate_evaluation'),
        'sha256': candidate.features.get(
            'recovery_candidate_evaluation_sha256')}
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
                screen_id, *baseline_reference.values(),
                *screen_reference.values()))):
        raise NumericRuntimeError('numeric recovery admission contract is invalid')
    from .schema import load_candidate_manifest
    screen_manifest_path = _file(repository_root, {
        'path': candidate.features.get('recovery_screen_manifest'),
        'sha256': candidate.features.get(
            'recovery_screen_manifest_sha256')}, 'numeric screen manifest')
    candidates = load_candidate_manifest(screen_manifest_path)
    baseline = tuple(item for item in candidates if item.id == 'full-s-v1')
    matching_screen = tuple(item for item in candidates if item.id == screen_id)
    if len(baseline) != 1 or len(matching_screen) != 1:
        raise NumericRuntimeError(
            'numeric recovery screen candidates are not canonical')
    screen = matching_screen[0]
    if candidate.features.get('numeric_kind') == 'w8a8':
        calibrated_screen, _calibration_reference, _calibration = (
            validate_recovery_calibration_dependency(
                candidate, repository_root=repository_root,
                recovery_manifest_path=manifest_path,
                recovery_stage_dir=path.parent))
        if calibrated_screen != screen:
            raise NumericRuntimeError(
                'numeric recovery calibration disagrees with screen candidate')
    else:
        try:
            recovery_manifest = resolve_numeric_file(
                repository_root, manifest_path, 'numeric recovery manifest')
        except ValueError as error:
            raise NumericRuntimeError(str(error)) from error
        if recovery_manifest == screen_manifest_path:
            raise NumericRuntimeError(
                'numeric recovery requires a follow-on manifest to avoid '
                'cyclic artifact authority')
        recoveries = tuple(
            item for item in load_candidate_manifest(recovery_manifest)
            if item.id == candidate.id)
        expected_stage = (
            Path(repository_root).resolve() / 'work_dirs/optimization' /
            candidate.route / candidate.id / str(candidate.seed) / 'train')
        if (recoveries != (candidate,)
                or path.parent.absolute() != expected_stage.absolute()
                or screen.id == candidate.id
                or screen.route != candidate.route
                or screen.features.get('recovery_candidate') is True
                or screen.features.get('numeric_kind') !=
                    candidate.features.get('numeric_kind')
                or screen.config != candidate.config
                or screen.checkpoint != candidate.checkpoint
                or screen.checkpoint_sha256 != candidate.checkpoint_sha256
                or screen.seed != candidate.seed):
            raise NumericRuntimeError(
                'numeric recovery runtime lineage disagrees with screen '
                'candidate')
    baseline_ap = _evaluation_ap(
        value['baseline_evaluation'], expected_reference=baseline_reference,
        candidate=baseline[0], repository_root=repository_root,
        manifest_path=screen_manifest_path)
    screen_ap = _evaluation_ap(
        value['candidate_evaluation'], expected_reference=screen_reference,
        candidate=screen, repository_root=repository_root,
        manifest_path=screen_manifest_path)
    observed_drop = baseline_ap - screen_ap
    if (observed_drop <= 0.3
            or abs(observed_drop - value['preliminary_ap_drop']) > 1e-9):
        raise NumericRuntimeError(
            'numeric recovery admission disagrees with preliminary AP evidence')
    return dict(value)


def validate_recovery_calibration_dependency(
        candidate: CandidateSpec, *, repository_root: Path,
        recovery_manifest_path: Path,
        recovery_stage_dir: Path) -> tuple[
            CandidateSpec, dict[str, str], dict[str, Any]]:
    """Authenticate a W8A8 recovery's already-screened calibration.

    The calibration remains bound to the immutable manifest that produced the
    initial screen.  A later commit-addressed recovery manifest can therefore
    record the artifact hash without creating a manifest/artifact hash cycle.
    """
    if (candidate.route != 'ssm-quant-pwl'
            or candidate.features.get('numeric_kind') != 'w8a8'
            or candidate.features.get('recovery_candidate') is not True):
        raise NumericRuntimeError(
            'screen calibration dependency requires a W8A8 recovery candidate')
    screen_id = candidate.features.get('recovery_screen_candidate')
    screen_manifest_value = candidate.features.get('recovery_screen_manifest')
    screen_manifest_sha = candidate.features.get(
        'recovery_screen_manifest_sha256')
    path_value = candidate.features.get('recovery_calibration_artifact')
    sha256 = candidate.features.get('recovery_calibration_sha256')
    if (not isinstance(screen_id, str) or not screen_id
            or not isinstance(screen_manifest_value, str)
            or not screen_manifest_value
            or not isinstance(screen_manifest_sha, str)
            or not isinstance(path_value, str) or not path_value
            or not isinstance(sha256, str)):
        raise NumericRuntimeError(
            'W8A8 recovery calibration manifest features are incomplete')
    reference = {'path': path_value, 'sha256': sha256}
    calibration_path = _file(
        repository_root, reference, 'W8A8 recovery calibration')
    try:
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericRuntimeError(
            'W8A8 recovery calibration is invalid JSON') from error
    source = calibration.get('source') if isinstance(calibration, Mapping) else None
    source_manifest = source.get('manifest_path') \
        if isinstance(source, Mapping) else None
    source_manifest_sha = source.get('manifest_sha256') \
        if isinstance(source, Mapping) else None
    if (not isinstance(source_manifest, str) or not source_manifest
            or not isinstance(source_manifest_sha, str)):
        raise NumericRuntimeError(
            'W8A8 recovery calibration source manifest is missing')
    if (source_manifest != screen_manifest_value
            or source_manifest_sha != screen_manifest_sha):
        raise NumericRuntimeError(
            'W8A8 calibration source disagrees with recovery screen manifest')
    screen_manifest_path = _file(repository_root, {
        'path': screen_manifest_value, 'sha256': screen_manifest_sha},
        'W8A8 screen source manifest')
    if file_sha256(screen_manifest_path) != source_manifest_sha:
        raise NumericRuntimeError(
            'W8A8 screen source manifest hash changed')
    from .schema import load_candidate_manifest
    screens = tuple(
        item for item in load_candidate_manifest(screen_manifest_path)
        if item.id == screen_id)
    if len(screens) != 1:
        raise NumericRuntimeError(
            'W8A8 recovery screen candidate is not canonical')
    screen = screens[0]
    if (screen.id == candidate.id
            or screen.route != 'ssm-quant-pwl'
            or screen.features.get('numeric_kind') != 'w8a8'
            or screen.features.get('recovery_candidate') is True
            or candidate.config != screen.config
            or candidate.checkpoint != screen.checkpoint
            or candidate.checkpoint_sha256 != screen.checkpoint_sha256
            or candidate.seed != screen.seed):
        raise NumericRuntimeError(
            'W8A8 recovery runtime lineage disagrees with screen candidate')
    root = Path(repository_root).resolve()
    expected_stage = (
        root / 'work_dirs/optimization' / candidate.route / candidate.id /
        str(candidate.seed) / 'train')
    if Path(recovery_stage_dir).absolute() != expected_stage.absolute():
        raise NumericRuntimeError(
            'W8A8 recovery train stage path is not canonical')
    expected_calibration = (
        root / 'work_dirs/optimization' / screen.route / screen.id /
        str(screen.seed) / 'calibrate/calibrate.json')
    if calibration_path.absolute() != expected_calibration.absolute():
        raise NumericRuntimeError(
            'W8A8 recovery calibration path is not the screen stage output')
    try:
        from .numeric_calibration import (
            CalibrationContractError, validate_calibration_provenance)
        validate_calibration_provenance(
            calibration, expected_candidate=screen,
            repository_root=root, manifest_path=screen_manifest_path)
    except CalibrationContractError as error:
        raise NumericRuntimeError(
            f'W8A8 recovery calibration contract is invalid: {error}') from error
    if file_sha256(calibration_path) != sha256:
        raise NumericRuntimeError(
            'W8A8 recovery calibration hash changed during validation')
    # The recovery manifest is validated separately by the train source
    # binding.  Resolve it here so a caller cannot silently supply a missing or
    # unsafe alternate manifest path.
    try:
        resolved_recovery_manifest = resolve_numeric_file(
            root, Path(recovery_manifest_path), 'W8A8 recovery manifest')
    except ValueError as error:
        raise NumericRuntimeError(str(error)) from error
    if resolved_recovery_manifest == screen_manifest_path:
        raise NumericRuntimeError(
            'W8A8 recovery requires a follow-on manifest to avoid cyclic '
            'artifact authority')
    recoveries = tuple(
        item for item in load_candidate_manifest(resolved_recovery_manifest)
        if item.id == candidate.id)
    if len(recoveries) != 1 or recoveries[0] != candidate:
        raise NumericRuntimeError(
            'W8A8 recovery candidate differs from recovery manifest')
    return screen, reference, dict(calibration)


def _canonical_w8a8_recovery_config(
        *, candidate: CandidateSpec, repository_root: Path,
        source: Mapping[str, str], screen: CandidateSpec,
        calibration_reference: Mapping[str, str],
        calibration: Mapping[str, Any], stage_dir: Path):
    """Rebuild the only admitted W8A8 recovery config from trusted inputs."""
    from mmengine.config import Config

    from .numeric_conversion import (
        NumericBindingError, quant_policy_from_config)
    from .schema import load_candidate_manifest

    if (source.get('config_path') != candidate.config.as_posix()
            or source.get('policy_path') != candidate.config.as_posix()
            or source.get('config_sha256') != source.get('policy_sha256')):
        raise NumericRuntimeError(
            'W8A8 recovery source policy is not the candidate config')
    screen_manifest_path = _file(repository_root, {
        'path': candidate.features.get('recovery_screen_manifest'),
        'sha256': candidate.features.get(
            'recovery_screen_manifest_sha256')}, 'W8A8 screen manifest')
    candidates = load_candidate_manifest(screen_manifest_path)
    students = tuple(item for item in candidates if item.id == 'full-s-v1')
    teachers = tuple(item for item in candidates if item.id == 'coco-b-teacher')
    if len(students) != 1 or len(teachers) != 1:
        raise NumericRuntimeError(
            'W8A8 recovery teacher/student lineage is incomplete')
    student, teacher = students[0], teachers[0]
    if (screen.config != candidate.config
            or screen.checkpoint != student.checkpoint
            or screen.checkpoint_sha256 != student.checkpoint_sha256
            or candidate.checkpoint != student.checkpoint
            or candidate.checkpoint_sha256 != student.checkpoint_sha256
            or screen.seed != student.seed
            or candidate.seed != student.seed
            or teacher.route != 'baseline'
            or teacher.features.get('role') != 'teacher'):
        raise NumericRuntimeError(
            'W8A8 recovery teacher/student lineage is invalid')
    screen_commit = calibration.get('source', {}).get('git_commit')
    try:
        validate_numeric_config_closure(
            repository_root, student.config, git_commit=screen_commit)
        validate_numeric_config_closure(
            repository_root, teacher.config, git_commit=screen_commit)
    except ValueError as error:
        raise NumericRuntimeError(
            f'W8A8 recovery lineage config is invalid: {error}') from error
    _file(repository_root, {
        'path': teacher.checkpoint.as_posix(),
        'sha256': teacher.checkpoint_sha256}, 'W8A8 recovery teacher checkpoint')
    try:
        expected = Config.fromfile(repository_root / candidate.config)
    except (OSError, TypeError, ValueError) as error:
        raise NumericRuntimeError(
            f'W8A8 recovery candidate config is invalid: {error}') from error
    imports = expected.get('custom_imports')
    hooks = expected.get('custom_hooks')
    numeric = expected.get('numeric_optimization')
    if (not isinstance(imports, Mapping)
            or dict(imports) != {
                'imports': ['mambapose_opt.numeric_conversion'],
                'allow_failed_imports': False}
            or not isinstance(hooks, list)
            or hooks != [
                {'type': 'NumericRuntimeHook', 'priority': 'VERY_HIGH'}]
            or not isinstance(numeric, Mapping)
            or numeric.get('candidate_kind') != 'w8a8'):
        raise NumericRuntimeError(
            'W8A8 recovery config lacks the canonical runtime hook')
    calibration_policy = numeric.get('calibration')
    train_envelope = numeric.get('train_envelope')
    expected_lineage = {
        'recovery': 'one-bounded-qat-or-distillation-run',
        'requires_attributed_error': True,
        'max_preliminary_ap_drop': 0.3,
        'resume_checkpoints': 2,
        'student_candidate': student.id,
        'student_config': student.config.as_posix(),
        'student_checkpoint': student.checkpoint.as_posix(),
        'student_checkpoint_sha256': student.checkpoint_sha256,
        'teacher_candidate': teacher.id,
        'teacher_config': teacher.config.as_posix(),
        'teacher_checkpoint': teacher.checkpoint.as_posix(),
        'teacher_checkpoint_sha256': teacher.checkpoint_sha256,
    }
    if (not isinstance(calibration_policy, Mapping)
            or calibration_policy.get('source_candidate') != student.id
            or not isinstance(train_envelope, Mapping)
            or dict(train_envelope) != expected_lineage):
        raise NumericRuntimeError(
            'W8A8 recovery config teacher/student lineage is invalid')
    quant_policy = numeric.get('quant_policy')
    if not isinstance(quant_policy, Mapping):
        raise NumericRuntimeError('W8A8 recovery quant policy is missing')
    try:
        quant_policy_from_config(
            quant_policy, calibration_artifact=calibration)
    except NumericBindingError as error:
        raise NumericRuntimeError(
            f'W8A8 recovery quant policy is invalid: {error}') from error
    expected.numeric_optimization.quant_policy.calibration_artifact = dict(
        calibration_reference)
    expected.work_dir = str(stage_dir / 'mmpose')
    expected.load_from = str(repository_root / candidate.checkpoint)
    expected.resume = False
    expected.randomness = dict(seed=candidate.seed, deterministic=True)
    return expected


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
    source = validate_numeric_source_binding(
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
    kind = candidate.features.get('numeric_kind')
    expected_dependencies = (
        {'recovery_admission', 'screen_calibration'}
        if kind == 'w8a8' else {'recovery_admission'})
    if (not isinstance(dependency, Mapping)
            or set(dependency) != expected_dependencies):
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
    screen = None
    if kind == 'w8a8':
        screen, expected_calibration, calibration = (
            validate_recovery_calibration_dependency(
                candidate, repository_root=repository_root,
                recovery_manifest_path=manifest_path,
                recovery_stage_dir=stage_dir))
        if (dependency['screen_calibration'] != expected_calibration
                or dependency_paths['screen_calibration'] !=
                repository_root / expected_calibration['path']):
            raise NumericRuntimeError(
                'numeric train screen calibration dependency is not canonical')
        canonical_config = _canonical_w8a8_recovery_config(
            candidate=candidate, repository_root=repository_root,
            source=source, screen=screen,
            calibration_reference=expected_calibration,
            calibration=calibration, stage_dir=stage_dir)
        try:
            runtime_text = config.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as error:
            raise NumericRuntimeError(
                'W8A8 recovery runtime config is unreadable') from error
        if (runtime_text != canonical_config.pretty_text
                or file_sha256(config) != runtime['config']['sha256']):
            raise NumericRuntimeError(
                'W8A8 recovery runtime config is not canonical')
    expected = candidate.features.get('runtime_checkpoint')
    if not isinstance(expected, str) or runtime['checkpoint']['path'] != expected:
        raise NumericRuntimeError('runtime checkpoint path disagrees with manifest')
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericRuntimeError('runtime metadata is invalid JSON') from error
    metadata_fields = {
        'schema_version', 'candidate_id', 'route', 'numeric_kind',
        'parent_checkpoint_sha256', 'runtime_checkpoint_sha256',
        'recovery_admission_sha256'}
    if kind == 'w8a8':
        metadata_fields.update({
            'screen_calibration_candidate_id',
            'screen_calibration_sha256'})
    if (not isinstance(metadata, Mapping)
            or set(metadata) != metadata_fields
            or metadata.get('schema_version') != 1
            or metadata.get('candidate_id') != candidate.id
            or metadata.get('route') != candidate.route
            or metadata.get('numeric_kind') != candidate.features.get('numeric_kind')
            or metadata.get('parent_checkpoint_sha256') !=
                candidate.checkpoint_sha256
            or metadata.get('runtime_checkpoint_sha256') !=
                runtime['checkpoint']['sha256']
            or metadata.get('recovery_admission_sha256') !=
                dependency['recovery_admission']['sha256']
            or (kind == 'w8a8' and (
                metadata.get('screen_calibration_candidate_id') != screen.id
                or metadata.get('screen_calibration_sha256') !=
                dependency['screen_calibration']['sha256']))):
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
    'validate_recovery_admission',
    'validate_recovery_calibration_dependency']
