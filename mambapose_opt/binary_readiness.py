"""Fail-closed readiness contracts for the conditional Binary Q/K route."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .artifacts import lexical_repository_root
from .evaluation import CandidateResult
from .schema import CandidateSpec, load_candidate_manifest


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_GATE_FIELDS = {
    'schema_version', 'artifact_kind', 'decision', 'ap_drop_limit_points',
    'baseline_candidate_id', 'pwl_candidate_id', 'pwl_policy',
    'baseline_authority', 'pwl_authority', 'modes',
}
_MODE_FIELDS = {
    'baseline_root', 'baseline_evaluation_sha256', 'candidate_root',
    'candidate_evaluation_sha256', 'ap_drop_points',
}
_PWL_AUTHORITY_FIELDS = {
    'candidate_id', 'candidate_row_sha256', 'git_commit', 'manifest_path',
    'manifest_sha256', 'config_path', 'config_sha256', 'checkpoint_path',
    'checkpoint_sha256', 'policy_path', 'policy_sha256', 'authority_path',
    'authority_sha256', 'seed', 'calibration', 'selection_policy',
    'installation', 'operation_manifest_sha256', 'config_closure',
}
_SOURCE_FIELDS = {
    'git_commit', 'manifest_path', 'manifest_sha256', 'config_path',
    'config_sha256', 'authority_path', 'authority_sha256',
}
_BASELINE_AUTHORITY_FIELDS = {
    'candidate_id', 'candidate_row_sha256', 'seed', 'source', 'checkpoint',
    'config_closure', 'modes',
}
_BASELINE_MODE_FIELDS = {
    'artifact_root', 'evaluation_sha256', 'provenance_sha256',
    'protocol_sha256', 'determinism_sha256',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty relative path')
    raw = str(value)
    path = Path(raw)
    if (
            path.is_absolute()
            or any(part in {'', '.', '..'} for part in raw.split('/'))):
        raise ValueError(f'{label} must be a safe relative path')
    return path


def _regular_file(root: Path, relative: Path, *, label: str) -> Path:
    lexical = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not contain symlinks')
    try:
        effective = lexical.resolve(strict=True)
        effective.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes the repository') from error
    if not effective.is_file():
        raise ValueError(f'{label} must be a regular file')
    return effective


def _artifact_root(root: Path, value: object, *, label: str) -> Path:
    relative = _relative(value, label=label)
    if relative.parts[:2] != ('work_dirs', 'optimization'):
        raise ValueError(f'{label} must stay under work_dirs/optimization')
    lexical = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not contain symlinks')
    try:
        effective = lexical.resolve(strict=True)
        effective.relative_to(root / 'work_dirs/optimization')
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes artifact authority') from error
    if not effective.is_dir():
        raise ValueError(f'{label} must be a directory')
    return effective


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f'{label} must be a lowercase sha256')
    return value


def _finite(value: object, *, label: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f'{label} must be finite')
    return float(value)


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _plain_json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _plain_json_value(value), sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _result_mapping(result: Any, name: str, *, label: str) -> Mapping[str, Any]:
    value = getattr(result, name, None)
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} CandidateResult {name} is invalid')
    return value


def _paired_projection(
        value: Mapping[str, Any], *, excluded: set[str]) -> dict[str, Any]:
    return {
        name: _plain_json_value(item) for name, item in value.items()
        if name not in excluded
    }


def _manifest_candidate_row_sha256(
        manifest_path: Path, candidate_id: str) -> str:
    try:
        value = json.loads(manifest_path.read_text(encoding='utf-8'))
        rows = value['candidates']
        matches = [
            row for row in rows
            if isinstance(row, Mapping) and row.get('id') == candidate_id]
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise ValueError('current candidate manifest is invalid') from error
    if len(matches) != 1:
        raise ValueError('current candidate row is not unique')
    payload = json.dumps(
        matches[0], sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_binary_qk_admission(
        candidate: CandidateSpec, *, repository_root: Path,
        manifest_path: Path,
        ) -> Mapping[str, Any]:
    """Require a hash-bound, public-valid PWL Stage-B pass before Binary Q/K."""
    if candidate.kind != 'binary-qk':
        raise ValueError('binary admission requires a binary-qk candidate')
    root = lexical_repository_root(Path(repository_root))
    raw_gate = candidate.features.get('pwl_stage_b_artifact')
    gate_relative = _relative(raw_gate, label='PWL Stage-B artifact')
    if gate_relative.parts[:2] != ('work_dirs', 'optimization'):
        raise ValueError(
            'PWL Stage-B artifact must stay under work_dirs/optimization')
    gate_path = _regular_file(
        root, gate_relative, label='PWL Stage-B artifact')
    expected_gate = _digest(
        candidate.features.get('pwl_stage_b_sha256'),
        label='PWL Stage-B artifact hash')
    if _sha256(gate_path) != expected_gate:
        raise ValueError('PWL Stage-B artifact hash mismatch')
    try:
        value = json.loads(gate_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'PWL Stage-B artifact is invalid JSON: {error}') from error
    if not isinstance(value, Mapping) or set(value) != _GATE_FIELDS:
        raise ValueError('PWL Stage-B artifact has invalid fields')
    limit = _finite(value['ap_drop_limit_points'], label='PWL AP-drop limit')
    if (
            value['schema_version'] != 3
            or value['artifact_kind'] != 'pwl-stage-b-pass'
            or value['decision'] != 'passed'
            or limit != 0.3):
        raise ValueError('PWL Stage-B artifact is not a passed 0.3-point screen')

    raw_manifest = str(manifest_path)
    manifest_input = Path(raw_manifest)
    if manifest_input.is_absolute():
        try:
            manifest_relative = manifest_input.absolute().relative_to(root)
        except ValueError as error:
            raise ValueError('PWL current manifest escapes repository') from error
    else:
        manifest_relative = _relative(
            raw_manifest, label='PWL current manifest')
    manifest_file = _regular_file(
        root, manifest_relative, label='PWL current manifest')
    candidates = load_candidate_manifest(manifest_file)
    by_id = {item.id: item for item in candidates}
    baseline = by_id.get(value['baseline_candidate_id'])
    pwl = by_id.get(value['pwl_candidate_id'])
    if baseline is None or baseline.kind != 'float' or baseline.route != 'baseline':
        raise ValueError('PWL Stage-B baseline authority is invalid')
    if pwl is None or pwl.kind != 'pwl' or pwl.route != 'ssm-quant-pwl':
        raise ValueError('PWL Stage-B candidate authority is invalid')
    policy = value['pwl_policy']
    if not isinstance(policy, Mapping) or set(policy) != {
            'path', 'sha256', 'function'}:
        raise ValueError('PWL Stage-B policy binding is invalid')
    if (
            policy['path'] != pwl.config.as_posix()
            or policy['function'] != pwl.features.get('pwl_function')):
        raise ValueError('PWL Stage-B policy differs from candidate authority')
    policy_path = _regular_file(root, pwl.config, label='PWL policy config')
    expected_policy_hash = _digest(
        policy['sha256'], label='PWL policy config hash')
    if _sha256(policy_path) != expected_policy_hash:
        raise ValueError('PWL Stage-B policy config hash mismatch')

    pwl_authority = value['pwl_authority']
    if not isinstance(pwl_authority, Mapping) or set(
            pwl_authority) != _PWL_AUTHORITY_FIELDS:
        raise ValueError('PWL authority fields are invalid')
    calibration = pwl_authority.get('calibration')
    installation = pwl_authority.get('installation')
    config_closure = pwl_authority.get('config_closure')
    if pwl_authority.get('authority_path') != (
            'optimization/coco_train2017_authority.json'):
        raise ValueError(
            'PWL numeric authority must use canonical train2017 authority')
    if (
            pwl_authority.get('candidate_id') != pwl.id
            or pwl_authority.get('seed') != pwl.seed
            or pwl_authority.get('config_path') != pwl.config.as_posix()
            or pwl_authority.get('checkpoint_path') !=
            pwl.checkpoint.as_posix()
            or pwl_authority.get('checkpoint_sha256') !=
            pwl.checkpoint_sha256
            or pwl_authority.get('policy_path') != pwl.config.as_posix()
            or pwl_authority.get('config_sha256') != expected_policy_hash
            or pwl_authority.get('policy_sha256') != expected_policy_hash
            or pwl_authority.get('selection_policy') !=
            'observed-range-max-then-mean-v1'
            or pwl_authority.get('manifest_path') !=
            manifest_relative.as_posix()
            or pwl_authority.get('candidate_row_sha256') !=
            _manifest_candidate_row_sha256(manifest_file, pwl.id)
            or not isinstance(pwl_authority.get('git_commit'), str)
            or not re.fullmatch(r'[0-9a-f]{40}', pwl_authority['git_commit'])
            or any(
                not isinstance(pwl_authority.get(name), str)
                or not _SHA256.fullmatch(pwl_authority[name])
                for name in (
                    'candidate_row_sha256', 'manifest_sha256',
                    'config_sha256', 'checkpoint_sha256', 'policy_sha256',
                    'authority_sha256', 'operation_manifest_sha256'))
            or not isinstance(calibration, Mapping)
            or set(calibration) != {
                'path', 'sha256', 'schema_version', 'sample_order_sha256'}
            or calibration.get('schema_version') != 3
            or not isinstance(calibration.get('sha256'), str)
            or not _SHA256.fullmatch(calibration['sha256'])
            or not isinstance(calibration.get('sample_order_sha256'), str)
            or not _SHA256.fullmatch(calibration['sample_order_sha256'])
            or not isinstance(installation, Mapping)
            or set(installation) != {'path', 'sha256'}
            or not isinstance(installation.get('sha256'), str)
            or not _SHA256.fullmatch(installation['sha256'])
            or not isinstance(config_closure, list)
            or not config_closure):
        raise ValueError('PWL authority identity is invalid')
    try:
        from .numeric_source import validate_numeric_config_closure
        current_closure = list(validate_numeric_config_closure(
            root, pwl.config))
    except ValueError as error:
        raise ValueError(
            f'PWL authority config closure is invalid: {error}') from error
    if config_closure != current_closure:
        raise ValueError('PWL authority config closure differs from current policy')
    for name in (
            'manifest_path', 'config_path', 'checkpoint_path', 'policy_path',
            'authority_path'):
        _relative(pwl_authority[name], label=f'PWL authority {name}')
    calibration_path = _relative(
        calibration['path'], label='PWL authority calibration')
    installation_path = _relative(
        installation['path'], label='PWL authority installation')
    canonical_root = Path('work_dirs/optimization') / pwl.route / pwl.id / str(
        pwl.seed)
    if (
            calibration_path != canonical_root / 'calibrate/calibrate.json'
            or installation_path != canonical_root /
            'convert/pwl-installation.json'):
        raise ValueError('PWL authority artifact paths are not canonical')

    modes = value['modes']
    if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
        raise ValueError('PWL Stage-B artifact must contain both modes')
    baseline_authority = value['baseline_authority']
    if (
            not isinstance(baseline_authority, Mapping)
            or set(baseline_authority) != _BASELINE_AUTHORITY_FIELDS):
        raise ValueError('PWL Stage-B baseline authority fields are invalid')
    baseline_source = baseline_authority.get('source')
    baseline_checkpoint = baseline_authority.get('checkpoint')
    baseline_config_closure = baseline_authority.get('config_closure')
    baseline_mode_authority = baseline_authority.get('modes')
    if (
            baseline_authority.get('candidate_id') != baseline.id
            or baseline_authority.get('candidate_row_sha256') !=
            _manifest_candidate_row_sha256(manifest_file, baseline.id)
            or baseline_authority.get('seed') != baseline.seed
            or not isinstance(baseline_source, Mapping)
            or set(baseline_source) != _SOURCE_FIELDS
            or baseline_source.get('config_path') != baseline.config.as_posix()
            or baseline_source.get('config_sha256') !=
            _sha256(_regular_file(
                root, baseline.config, label='PWL baseline config'))
            or baseline_source.get('git_commit') !=
            pwl_authority.get('git_commit')
            or baseline_source.get('manifest_path') !=
            pwl_authority.get('manifest_path')
            or baseline_source.get('manifest_sha256') !=
            pwl_authority.get('manifest_sha256')
            or baseline_source.get('authority_path') !=
            'optimization/coco_val2017_authority.json'
            or not isinstance(baseline_checkpoint, Mapping)
            or set(baseline_checkpoint) != {'path', 'sha256'}
            or baseline_checkpoint.get('path') != baseline.checkpoint.as_posix()
            or baseline_checkpoint.get('sha256') != baseline.checkpoint_sha256
            or baseline_checkpoint.get('path') !=
            pwl_authority.get('checkpoint_path')
            or baseline_checkpoint.get('sha256') !=
            pwl_authority.get('checkpoint_sha256')
            or not isinstance(baseline_config_closure, list)
            or not baseline_config_closure
            or not isinstance(baseline_mode_authority, Mapping)
            or set(baseline_mode_authority) != {'flip', 'no_flip'}):
        raise ValueError(
            'PWL authority and baseline authority pairing is invalid')
    try:
        from .numeric_source import validate_numeric_config_closure
        current_baseline_closure = list(validate_numeric_config_closure(
            root, baseline.config))
    except ValueError as error:
        raise ValueError(
            f'PWL baseline config closure is invalid: {error}') from error
    if baseline_config_closure != current_baseline_closure:
        raise ValueError('PWL baseline config closure differs from current source')

    observed_baseline_roots = set()
    observed_pwl_roots = set()
    expected_baseline_modes = {}
    for mode, row in modes.items():
        if not isinstance(row, Mapping) or set(row) != _MODE_FIELDS:
            raise ValueError(f'PWL Stage-B {mode} row has invalid fields')
        baseline_root = _artifact_root(
            root, row['baseline_root'], label=f'{mode} baseline root')
        pwl_root = _artifact_root(
            root, row['candidate_root'], label=f'{mode} PWL root')
        for result_root, digest_field, label in (
                (baseline_root, 'baseline_evaluation_sha256', 'baseline'),
                (pwl_root, 'candidate_evaluation_sha256', 'PWL')):
            evaluation = _regular_file(
                root, (result_root / 'evaluate/evaluate.json').relative_to(root),
                label=f'{mode} {label} evaluation')
            expected = _digest(
                row[digest_field], label=f'{mode} {label} evaluation hash')
            if _sha256(evaluation) != expected:
                raise ValueError(f'{mode} {label} evaluation hash mismatch')
        baseline_result = CandidateResult.from_artifacts(
            baseline_root, mode=mode)
        pwl_result = CandidateResult.from_artifacts(pwl_root, mode=mode)
        if (
                baseline_result.candidate_id != baseline.id
                or baseline_result.candidate_kind != baseline.kind
                or baseline_result.route != baseline.route
                or baseline_result.seed != baseline.seed
                or baseline_result.flip_test != (mode == 'flip')):
            raise ValueError(f'{mode} public baseline CandidateResult mismatch')
        if (
                pwl_result.candidate_id != pwl.id
                or pwl_result.candidate_kind != pwl.kind
                or pwl_result.route != pwl.route
                or pwl_result.seed != pwl.seed
                or pwl_result.flip_test != (mode == 'flip')):
            raise ValueError(f'{mode} public PWL CandidateResult mismatch')
        observed_baseline_roots.add(row['baseline_root'])
        observed_pwl_roots.add(row['candidate_root'])
        result_baseline_source = _result_mapping(
            baseline_result, 'source', label=f'{mode} baseline')
        result_pwl_source = _result_mapping(
            pwl_result, 'source', label=f'{mode} PWL')
        baseline_provenance = _result_mapping(
            baseline_result, 'provenance', label=f'{mode} baseline')
        pwl_provenance = _result_mapping(
            pwl_result, 'provenance', label=f'{mode} PWL')
        baseline_protocol = _result_mapping(
            baseline_result, 'protocol', label=f'{mode} baseline')
        pwl_protocol = _result_mapping(
            pwl_result, 'protocol', label=f'{mode} PWL')
        baseline_determinism = _result_mapping(
            baseline_result, 'determinism', label=f'{mode} baseline')
        pwl_determinism = _result_mapping(
            pwl_result, 'determinism', label=f'{mode} PWL')
        expected_pwl_source = {
            name: pwl_authority[name] for name in _SOURCE_FIELDS}
        expected_pwl_source.update({
            'authority_path': baseline_source['authority_path'],
            'authority_sha256': baseline_source['authority_sha256'],
        })
        if (
                _plain_json_value(result_baseline_source) !=
                _plain_json_value(baseline_source)
                or _plain_json_value(result_pwl_source) != expected_pwl_source
                or baseline_provenance.get('checkpoint_sha256') !=
                baseline.checkpoint_sha256
                or baseline_provenance.get('config_sha256') !=
                baseline_source['config_sha256']
                or baseline_provenance.get('git_commit') !=
                baseline_source['git_commit']
                or pwl_provenance.get('checkpoint_sha256') !=
                pwl.checkpoint_sha256
                or pwl_provenance.get('config_sha256') !=
                result_pwl_source['config_sha256']
                or pwl_provenance.get('git_commit') !=
                result_pwl_source['git_commit']
                or baseline_protocol.get('authority_path') !=
                result_baseline_source['authority_path']
                or baseline_protocol.get('authority_sha256') !=
                result_baseline_source['authority_sha256']
                or pwl_protocol.get('authority_path') !=
                result_pwl_source['authority_path']
                or pwl_protocol.get('authority_sha256') !=
                result_pwl_source['authority_sha256']):
            raise ValueError(
                f'{mode} baseline/PWL public source authority is invalid')
        if (
                baseline_provenance.get('data_inventory_sha256') !=
                pwl_provenance.get('data_inventory_sha256')
                or _paired_projection(
                    baseline_protocol,
                    excluded={'source_config', 'checkpoint'}) !=
                _paired_projection(
                    pwl_protocol,
                    excluded={'source_config', 'checkpoint'})
                or _paired_projection(
                    baseline_determinism, excluded={'provenance'}) !=
                _paired_projection(
                    pwl_determinism, excluded={'provenance'})):
            raise ValueError(
                f'{mode} paired baseline/PWL evaluation protocol differs')
        expected_baseline_modes[mode] = {
            'artifact_root': row['baseline_root'],
            'evaluation_sha256': row['baseline_evaluation_sha256'],
            'provenance_sha256': _canonical_json_sha256(baseline_provenance),
            'protocol_sha256': _canonical_json_sha256(baseline_protocol),
            'determinism_sha256': _canonical_json_sha256(
                baseline_determinism),
        }
        recorded_mode_authority = baseline_mode_authority.get(mode)
        if (
                not isinstance(recorded_mode_authority, Mapping)
                or set(recorded_mode_authority) != _BASELINE_MODE_FIELDS):
            raise ValueError(
                f'{mode} PWL baseline mode authority is invalid')
        result_profile = getattr(pwl_result, 'profile', None)
        result_parent = (
            result_profile.get('parent')
            if isinstance(result_profile, Mapping) else None)
        if (
                not isinstance(result_parent, Mapping)
                or result_parent.get('config') != pwl.config.as_posix()):
            raise ValueError(
                f'{mode} CandidateResult comes from an alternate PWL policy')
        if (
                not isinstance(pwl_result.pwl_authority, Mapping)
                or _plain_json_value(pwl_result.pwl_authority) !=
                _plain_json_value(pwl_authority)):
            raise ValueError(
                f'{mode} public CandidateResult PWL authority mismatch')
        observed_drop = float(baseline_result.metrics.ap - pwl_result.metrics.ap)
        recorded_drop = _finite(
            row['ap_drop_points'], label=f'{mode} PWL AP drop')
        if not math.isclose(
                observed_drop, recorded_drop, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f'{mode} PWL AP drop disagrees with CandidateResult')
        if observed_drop > limit:
            raise ValueError(f'{mode} PWL Stage-B AP drop exceeds 0.3 points')
    if len(observed_baseline_roots) != 1 or len(observed_pwl_roots) != 1:
        raise ValueError(
            'PWL Stage-B modes must use the same artifact root')
    expected_baseline_authority = {
        'candidate_id': baseline.id,
        'candidate_row_sha256': _manifest_candidate_row_sha256(
            manifest_file, baseline.id),
        'seed': baseline.seed,
        'source': _plain_json_value(baseline_source),
        'checkpoint': {
            'path': baseline.checkpoint.as_posix(),
            'sha256': baseline.checkpoint_sha256,
        },
        'config_closure': current_baseline_closure,
        'modes': expected_baseline_modes,
    }
    if (
            _plain_json_value(baseline_authority) !=
            expected_baseline_authority):
        raise ValueError(
            'PWL Stage-B baseline authority disagrees with public evidence')
    return value
