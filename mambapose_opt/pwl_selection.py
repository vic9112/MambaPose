"""Source-attested selection for the four canonical PWL screens.

This module deliberately separates the pure, deterministic record builder from
the production validator.  The latter reconstructs every row from a tracked
candidate config and a repository-relative, hash-bound calibration artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .pwl_artifacts import (
    PWLArtifactError, require_pwl_fit_admitted, validate_pwl_fit_report)
from .pwl_paths import canonical_path, canonical_relative_path


class PWLSelectionError(ValueError):
    """Raised when four-candidate evidence is incomplete or incomparable."""


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
CANONICAL_PWL_CANDIDATES = (
    ('pwl-silu-s-v1', 'silu'),
    ('pwl-gelu-s-v1', 'gelu'),
    ('pwl-softplus-s-v1', 'softplus'),
    ('pwl-exp-s-v1', 'exp'),
)
_CANONICAL_SELECTION_PATH = Path(
    'work_dirs/optimization/ssm-quant-pwl/pwl-selection/selection.json')
_COMMON_SOURCE_FIELDS = (
    'git_commit', 'manifest_path', 'manifest_sha256',
    'checkpoint_path', 'checkpoint_sha256',
    'authority_path', 'authority_sha256')
_COMMON_IDENTITY_FIELDS = (
    'config', 'config_sha256', 'checkpoint', 'checkpoint_sha256',
    'dataset', 'git_commit', 'split')
_COMMON_PROTOCOL_FIELDS = (
    'model_mode', 'grad_enabled', 'shuffle', 'worker_count', 'sample_count',
    'sample_order_sha256', 'root_determinism')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _reference(value: object, *, label: str) -> dict[str, str]:
    if (not isinstance(value, Mapping)
            or set(value) != {'path', 'sha256'}
            or not isinstance(value.get('path'), str)
            or not value['path']
            or not isinstance(value.get('sha256'), str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise PWLSelectionError(f'{label} reference is invalid')
    try:
        relative = canonical_relative_path(value['path'], label=label)
    except ValueError as error:
        raise PWLSelectionError(str(error)) from error
    return {'path': relative.as_posix(), 'sha256': value['sha256']}


def _common_rows(calibrations: Mapping[str, Mapping[str, Any]],
                 field: str, names: tuple[str, ...], *, label: str) -> dict:
    values = []
    for candidate_id, _ in CANONICAL_PWL_CANDIDATES:
        envelope = calibrations[candidate_id].get('calibration')
        row = envelope.get(field) if isinstance(envelope, Mapping) else None
        if not isinstance(row, Mapping) or any(name not in row for name in names):
            raise PWLSelectionError(
                f'{label} is incomplete for {candidate_id}')
        values.append({name: row[name] for name in names})
    if any(value != values[0] for value in values[1:]):
        raise PWLSelectionError(
            f'PWL calibration {label} differs across four candidates')
    return values[0]


def build_pwl_selection_record(
        *, manifest_reference: Mapping[str, Any],
        calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build one deterministic decision from exactly four verified envelopes.

    ``calibrations`` is keyed by canonical candidate ID and each value contains
    ``reference``, tracked ``policy``, and the decoded schema-v3
    ``calibration``.  Production callers must use
    :func:`build_pwl_selection_artifact`, which reconstructs those envelopes.
    """
    expected_ids = tuple(item[0] for item in CANONICAL_PWL_CANDIDATES)
    if (not isinstance(calibrations, Mapping)
            or tuple(calibrations) != expected_ids
            or set(calibrations) != set(expected_ids)):
        raise PWLSelectionError(
            'PWL selection requires exactly four canonical candidates in order')
    manifest = _reference(manifest_reference, label='PWL source manifest')
    source_authority = _common_rows(
        calibrations, 'source', _COMMON_SOURCE_FIELDS,
        label='source authority')
    identity_authority = _common_rows(
        calibrations, 'identity', _COMMON_IDENTITY_FIELDS,
        label='source authority')
    protocol = _common_rows(
        calibrations, 'protocol', _COMMON_PROTOCOL_FIELDS,
        label='protocol')
    if (protocol.get('sample_count') != 512
            or protocol.get('model_mode') != 'eval'
            or protocol.get('grad_enabled') is not False
            or protocol.get('shuffle') is not False
            or protocol.get('worker_count') != 0):
        raise PWLSelectionError(
            'PWL calibration protocol is not the canonical 512-sample order')
    if (source_authority['manifest_path'] != manifest['path']
            or source_authority['manifest_sha256'] != manifest['sha256']
            or source_authority['git_commit'] != identity_authority['git_commit']
            or source_authority['checkpoint_sha256'] !=
            identity_authority['checkpoint_sha256']):
        raise PWLSelectionError(
            'PWL calibration source authority disagrees with selection source')

    rows = []
    eligible = []
    for candidate_id, function_name in CANONICAL_PWL_CANDIDATES:
        item = calibrations[candidate_id]
        if not isinstance(item, Mapping) or set(item) != {
                'reference', 'policy', 'calibration'}:
            raise PWLSelectionError(
                f'PWL calibration envelope is invalid for {candidate_id}')
        reference = _reference(
            item['reference'], label=f'{candidate_id} calibration')
        calibration = item['calibration']
        if (not isinstance(calibration, Mapping)
                or calibration.get('schema_version') != 3
                or calibration.get('candidate_id') != candidate_id
                or calibration.get('stage') != 'calibrate'):
            raise PWLSelectionError(
                f'PWL schema-v3 calibration identity is invalid for '
                f'{candidate_id}')
        try:
            fit = validate_pwl_fit_report(
                calibration.get('pwl_fit'),
                expected_candidate_id=candidate_id,
                expected_policy=item['policy'])
        except (PWLArtifactError, TypeError) as error:
            raise PWLSelectionError(
                f'PWL fit is invalid for {candidate_id}: {error}') from error
        if fit['function_name'] != function_name:
            raise PWLSelectionError(
                f'PWL function disagrees with canonical candidate '
                f'{candidate_id}')
        comparator = fit['exact_comparator']
        if function_name == 'exp':
            if comparator != {
                    'kind': 'exact-export-time-constant-folding',
                    'applicable_source': 'static-parameter',
                    'runtime_nonlinear_operations': 0,
                    'max_error': 0.0, 'mean_error': 0.0,
                    'preferred_over_pwl_when_exportable': True}:
                raise PWLSelectionError(
                    'exp exact constant-fold comparator is not executable')
            status = 'excluded-exact-constant-fold'
        elif fit['admission']['decision'] == 'passed':
            require_pwl_fit_admitted(fit)
            status = 'eligible'
            eligible.append(fit)
        else:
            status = 'rejected-domain'
        rows.append({
            'candidate_id': candidate_id,
            'function_name': function_name,
            'calibration_artifact': reference,
            'fit_sha256': _json_sha256(fit),
            'admission': fit['admission'],
            'observed_range_error': fit['observed_range_error'],
            'domain_coverage': fit['domain_coverage'],
            'exact_comparator': comparator,
            'selection_status': status,
        })
    ranking = [item['candidate_id'] for item in sorted(eligible, key=lambda fit: (
        float(fit['observed_range_error']['max']),
        float(fit['observed_range_error']['mean']),
        fit['candidate_id']))]
    selected = ranking[0] if ranking else None
    return {
        'schema_version': 2,
        'artifact_kind': 'pwl-four-candidate-selection',
        'manifest': manifest,
        'source_authority': source_authority,
        'identity_authority': identity_authority,
        'protocol': protocol,
        'candidates': rows,
        'ranking': ranking,
        'selected_candidate_id': selected,
        'decision': 'selected' if selected else 'no-pwl-candidate-admitted',
        'claim_limits': {
            'hardware_latency_claimed': False,
            'fpga_speedup_claimed': False,
            'fastmamba_composite_mechanisms_inherited': False,
            'note': ('PWL-only evidence does not inherit Hadamard, SSM/conv '
                     'quantization, nonlinear-unit, or FPGA claims.'),
        },
    }


def _load_selection_file(
        reference: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    normalized = _reference(reference, label='PWL selection artifact')
    relative = Path(normalized['path'])
    if relative.parts[:2] != ('work_dirs', 'optimization'):
        raise PWLSelectionError(
            'PWL selection artifact path must be under work_dirs/optimization')
    root = Path(repository_root).resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PWLSelectionError(
                'PWL selection artifact path must not use symlink')
    if relative != _CANONICAL_SELECTION_PATH:
        raise PWLSelectionError(
            'PWL selection artifact path is not canonical')
    if not cursor.is_file() or _sha256(cursor) != normalized['sha256']:
        raise PWLSelectionError('PWL selection artifact hash changed')
    try:
        value = json.loads(cursor.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise PWLSelectionError(
            'PWL selection artifact is invalid JSON') from error
    if not isinstance(value, Mapping):
        raise PWLSelectionError('PWL selection artifact root is invalid')
    return dict(value)


def _canonical_envelopes(
        value: Mapping[str, Any], *, repository_root: Path,
        manifest_path: Path) -> dict[str, Mapping[str, Any]]:
    from .checkpoints import authorize_tracked_config
    from .numeric_calibration import validate_calibration_provenance
    from .schema import load_candidate_manifest

    candidates = tuple(
        item for item in load_candidate_manifest(manifest_path)
        if item.features.get('numeric_kind') == 'pwl')
    actual = tuple((item.id, item.features.get('pwl_function'))
                   for item in candidates)
    if actual != CANONICAL_PWL_CANDIDATES:
        raise PWLSelectionError(
            'tracked manifest does not define exactly four canonical PWL '
            'candidates in order')
    rows = value.get('candidates') if isinstance(value, Mapping) else None
    if (not isinstance(rows, list) or len(rows) != 4
            or [row.get('candidate_id') if isinstance(row, Mapping) else None
                for row in rows] != [item[0] for item in CANONICAL_PWL_CANDIDATES]):
        raise PWLSelectionError(
            'PWL selection does not reference exactly four canonical rows')
    result = {}
    for candidate, row in zip(candidates, rows):
        reference = row.get('calibration_artifact')
        normalized = _reference(
            reference, label=f'{candidate.id} calibration')
        relative = Path(normalized['path'])
        expected = (Path('work_dirs/optimization') / candidate.route /
                    candidate.id / str(candidate.seed) /
                    'calibrate/calibrate.json')
        if relative != expected:
            raise PWLSelectionError(
                f'{candidate.id} calibration path is not canonical')
        # Reuse the selection loader's strict path and hash authority, but load
        # the calibration envelope instead of a selection record.
        root = Path(repository_root).resolve(strict=True)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PWLSelectionError(
                    f'{candidate.id} calibration path must not use symlink')
        if not cursor.is_file() or _sha256(cursor) != normalized['sha256']:
            raise PWLSelectionError(
                f'{candidate.id} calibration hash changed')
        try:
            calibration = json.loads(cursor.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise PWLSelectionError(
                f'{candidate.id} calibration is invalid JSON') from error
        try:
            validate_calibration_provenance(
                calibration, expected_candidate=candidate,
                repository_root=root, manifest_path=manifest_path)
        except ValueError as error:
            raise PWLSelectionError(
                f'{candidate.id} calibration provenance is invalid: '
                f'{error}') from error
        config = authorize_tracked_config(
            root, manifest_path, candidate).load_config()
        pwl = config.numeric_optimization.pwl
        policy = {name: pwl[name] for name in (
            'enabled_function', 'source', 'roles', 'domain', 'segments',
            'grid_points', 'saturation', 'qat_form', 'selection_policy')}
        result[candidate.id] = {
            'reference': normalized, 'policy': policy,
            'calibration': calibration}
    return result


def validate_pwl_selection_artifact(
        value: Mapping[str, Any], *, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    """Reconstruct a production decision from tracked source and four fits."""
    if (not isinstance(value, Mapping)
            or value.get('schema_version') != 2):
        raise PWLSelectionError('PWL selection schema version is invalid')
    root = Path(repository_root).resolve(strict=True)
    supplied = canonical_path(
        str(manifest_path), label='PWL source manifest', allow_absolute=True)
    lexical = supplied if supplied.is_absolute() else root / supplied
    lexical = lexical.absolute()
    try:
        relative_manifest = lexical.relative_to(root)
    except ValueError as error:
        raise PWLSelectionError(
            'PWL source manifest must be inside repository') from error
    if relative_manifest != Path('optimization/candidates.json'):
        raise PWLSelectionError('PWL source manifest path is not canonical')
    cursor = root
    for part in relative_manifest.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PWLSelectionError(
                'PWL source manifest path must not use symlink')
    manifest = cursor.resolve(strict=True)
    expected_reference = {
        'path': relative_manifest.as_posix(), 'sha256': _sha256(manifest)}
    envelopes = _canonical_envelopes(
        value, repository_root=root, manifest_path=manifest)
    rebuilt = build_pwl_selection_record(
        manifest_reference=expected_reference, calibrations=envelopes)
    if dict(value) != rebuilt:
        raise PWLSelectionError(
            'PWL selection artifact disagrees with reconstructed decision')
    return rebuilt


def build_pwl_selection_artifact(
        *, repository_root: Path, manifest_path: Path,
        calibration_references: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Production producer using only canonical SHA-bound inputs."""
    placeholder = {
        'candidates': [
            {'candidate_id': candidate_id,
             'calibration_artifact': calibration_references.get(candidate_id)}
            for candidate_id, _ in CANONICAL_PWL_CANDIDATES]
    }
    root = Path(repository_root).resolve(strict=True)
    supplied = canonical_path(
        str(manifest_path), label='PWL source manifest', allow_absolute=True)
    lexical = supplied if supplied.is_absolute() else root / supplied
    lexical = lexical.absolute()
    try:
        relative_manifest = lexical.relative_to(root)
    except ValueError as error:
        raise PWLSelectionError(
            'PWL source manifest must be inside repository') from error
    if relative_manifest != Path('optimization/candidates.json'):
        raise PWLSelectionError('PWL source manifest path is not canonical')
    cursor = root
    for part in relative_manifest.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PWLSelectionError(
                'PWL source manifest path must not use symlink')
    manifest = cursor.resolve(strict=True)
    envelopes = _canonical_envelopes(
        placeholder, repository_root=root, manifest_path=manifest)
    return build_pwl_selection_record(
        manifest_reference={
            'path': relative_manifest.as_posix(),
            'sha256': _sha256(manifest)},
        calibrations=envelopes)


def load_pwl_selection_reference(
        reference: Mapping[str, Any], *, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    """Load and publicly validate one SHA-bound production selection."""
    value = _load_selection_file(reference, repository_root=repository_root)
    return validate_pwl_selection_artifact(
        value, repository_root=repository_root, manifest_path=manifest_path)


__all__ = [
    'CANONICAL_PWL_CANDIDATES', 'PWLSelectionError',
    'build_pwl_selection_artifact', 'build_pwl_selection_record',
    'load_pwl_selection_reference', 'validate_pwl_selection_artifact']
