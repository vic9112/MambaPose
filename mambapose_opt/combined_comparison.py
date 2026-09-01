"""Authenticated full-S-V1 versus no-PIF+PWL accuracy comparison."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APM', 'APL', 'AR')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class CombinedComparisonError(ValueError):
    """Raised when the direct full-versus-combined comparison is invalid."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False
    ) + '\n').encode('utf-8')


def _reference(value: object, *, label: str) -> dict[str, str]:
    if (not isinstance(value, Mapping)
            or set(value) != {'path', 'sha256'}
            or not isinstance(value.get('path'), str)
            or Path(value['path']).is_absolute()
            or any(part in {'', '.', '..'} for part in Path(value['path']).parts)
            or not isinstance(value.get('sha256'), str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise CombinedComparisonError(f'{label} reference is invalid')
    return {'path': value['path'], 'sha256': value['sha256']}


def _metrics(value: object, *, label: str) -> dict[str, float | str]:
    if not isinstance(value, Mapping) or set(value) != {
            *METRIC_NAMES, 'unit'}:
        raise CombinedComparisonError(f'{label} metrics are invalid')
    if value.get('unit') != 'percentage_points':
        raise CombinedComparisonError(
            f'{label} metrics must use percentage points')
    result: dict[str, float | str] = {'unit': 'percentage_points'}
    for name in METRIC_NAMES:
        metric = value[name]
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))):
            raise CombinedComparisonError(f'{label} {name} is not finite')
        result[name] = float(metric)
    return result


def _evaluation_modes(
        value: object, *, candidate_id: str,
        ) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
    if (not isinstance(value, Mapping)
            or value.get('schema_version') != 1
            or value.get('candidate_id') != candidate_id
            or value.get('stage') != 'evaluate'
            or not isinstance(value.get('result'), Mapping)):
        raise CombinedComparisonError('evaluation envelope identity is invalid')
    result = value['result']
    modes = result.get('modes')
    if not isinstance(modes, Mapping) or set(modes) != {'flip', 'no_flip'}:
        raise CombinedComparisonError(
            'evaluation must contain flip and no_flip modes')
    normalized: dict[str, dict[str, Any]] = {}
    for mode in ('flip', 'no_flip'):
        row = modes[mode]
        if (not isinstance(row, Mapping)
                or not isinstance(row.get('protocol'), Mapping)):
            raise CombinedComparisonError(
                f'evaluation {mode} protocol is invalid')
        normalized[mode] = {
            'metrics': _metrics(row.get('metrics'), label=mode),
            'protocol': dict(row['protocol']),
        }
    return result, normalized


def _protocol_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only the model-specific paths from one COCO protocol."""
    return {
        key: value for key, value in protocol.items()
        if key not in {'source_config', 'checkpoint'}
    }


def validate_comparator_evaluation(
        value: object, *, comparator: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Validate the exact frozen full-S-V1 evaluation and its model identity."""
    expected_fields = {
        'candidate_id', 'checkpoint', 'checkpoint_sha256', 'config',
        'config_sha256', 'evaluation_artifact', 'evaluation_candidate_id',
        'source_commit'}
    if not isinstance(comparator, Mapping) or set(comparator) != expected_fields:
        raise CombinedComparisonError('full comparator authority is invalid')
    result, modes = _evaluation_modes(
        value, candidate_id=comparator['evaluation_candidate_id'])
    if (result.get('route') != 'baseline'
            or result.get('calibration_split') is not None
            or result.get('runtime') != {
                'checkpoint': {
                    'path': comparator['checkpoint'],
                    'sha256': comparator['checkpoint_sha256']},
                'config': {
                    'path': comparator['config'],
                    'sha256': comparator['config_sha256']},
                'metadata': None, 'table': None,
                'transform': 'identity-v1'}):
        raise CombinedComparisonError(
            'full comparator runtime identity is invalid')
    source = result.get('source')
    if (not isinstance(source, Mapping)
            or source.get('git_commit') != comparator['source_commit']
            or source.get('config_path') != comparator['config']
            or source.get('config_sha256') != comparator['config_sha256']):
        raise CombinedComparisonError(
            'full comparator source identity is invalid')
    for mode, row in modes.items():
        raw = result['modes'][mode]
        provenance = raw.get('provenance')
        determinism = raw.get('determinism')
        if (not isinstance(provenance, Mapping)
                or provenance.get('checkpoint_sha256') !=
                comparator['checkpoint_sha256']
                or provenance.get('git_commit') != comparator['source_commit']
                or not isinstance(determinism, Mapping)
                or any(determinism.get(name) != 0 for name in (
                    'python_seed', 'numpy_seed', 'torch_seed'))
                or row['protocol'].get('source_config') != comparator['config']
                or row['protocol'].get('checkpoint') !=
                comparator['checkpoint']):
            raise CombinedComparisonError(
                f'full comparator {mode} model identity is invalid')
    if _protocol_identity(modes['flip']['protocol']) != _protocol_identity(
            modes['no_flip']['protocol']):
        raise CombinedComparisonError(
            'full comparator modes use different COCO protocols')
    return {
        'candidate_id': comparator['evaluation_candidate_id'],
        'modes': modes,
    }


def build_combined_comparison(
        full_evaluation: object, combined_evaluation: object, *,
        full_reference: Mapping[str, Any],
        combined_reference: Mapping[str, Any], candidate_id: str,
        ) -> dict[str, Any]:
    """Recompute every direct delta; isolated parent deltas are never inputs."""
    full_ref = _reference(full_reference, label='full evaluation')
    combined_ref = _reference(
        combined_reference, label='combined evaluation')
    if not isinstance(candidate_id, str) or not candidate_id:
        raise CombinedComparisonError('combined candidate identity is invalid')
    full_id = (
        full_evaluation.get('candidate_id')
        if isinstance(full_evaluation, Mapping) else None)
    _, full_modes = _evaluation_modes(full_evaluation, candidate_id=full_id)
    _, combined_modes = _evaluation_modes(
        combined_evaluation, candidate_id=candidate_id)
    protocol_identity = _protocol_identity(full_modes['flip']['protocol'])
    for label, modes in (
            ('full', full_modes), ('combined', combined_modes)):
        for mode in ('flip', 'no_flip'):
            if _protocol_identity(modes[mode]['protocol']) != protocol_identity:
                raise CombinedComparisonError(
                    f'{label} {mode} COCO protocol differs from comparator')
    rows: dict[str, Any] = {}
    for mode in ('flip', 'no_flip'):
        full_metrics = full_modes[mode]['metrics']
        combined_metrics = combined_modes[mode]['metrics']
        candidate_minus_full = {
            name: float(combined_metrics[name]) - float(full_metrics[name])
            for name in METRIC_NAMES
        }
        candidate_minus_full['unit'] = 'percentage_points'
        drop_full_minus_candidate = {
            name: float(full_metrics[name]) - float(combined_metrics[name])
            for name in METRIC_NAMES
        }
        drop_full_minus_candidate['unit'] = 'percentage_points'
        rows[mode] = {
            'full': full_metrics,
            'combined': combined_metrics,
            'candidate_minus_full': candidate_minus_full,
            'drop_full_minus_candidate': drop_full_minus_candidate,
        }
    return {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'stage': 'compare',
        'result': {
            'comparator_candidate_id': full_id,
            'full_evaluation': full_ref,
            'combined_evaluation': combined_ref,
            'protocol_identity_sha256': hashlib.sha256(
                _canonical_json(protocol_identity)).hexdigest(),
            'modes': rows,
            'claim_limits': {
                'direct_full_comparison': True,
                'formal_paired_pass': False,
                'isolated_deltas_additive': False,
                'screen_kind': 'single-checkpoint-preliminary',
            },
        },
    }


def validate_combined_comparison_artifact(
        value: object, *, candidate, artifact_path: Path,
        repository_root: Path, manifest_path: Path,
        ) -> dict[str, Any]:
    """Rebuild a canonical compare artifact from both SHA-bound evaluations."""
    from .combined_candidate import (
        load_combined_parent_authority, read_bound_workspace_artifact)

    root = Path(repository_root).resolve(strict=True)
    path = Path(artifact_path).absolute()
    expected_path = (
        root / 'work_dirs/optimization' / candidate.route / candidate.id /
        str(candidate.seed) / 'compare/compare.json').absolute()
    if path != expected_path or path.is_symlink():
        raise CombinedComparisonError(
            'combined comparison output path is not canonical')
    authority = load_combined_parent_authority(
        candidate, repository_root=root, manifest_path=manifest_path)
    comparator = authority['comparator']
    full = read_bound_workspace_artifact(
        comparator['evaluation_artifact'], checkout_root=root)
    validate_comparator_evaluation(full, comparator=comparator)
    evaluation_path = path.parent.parent / 'evaluate/evaluate.json'
    try:
        evaluation_reference = {
            'path': evaluation_path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        }
    except (OSError, ValueError) as error:
        raise CombinedComparisonError(
            'combined evaluation artifact is unavailable') from error
    combined = read_bound_workspace_artifact(
        evaluation_reference, checkout_root=root)
    try:
        from .evaluation import (
            resolve_artifact_source, validate_evaluation_envelope)
        from .numeric_runtime import resolve_numeric_runtime

        source, source_candidate, coco_authority = resolve_artifact_source(
            combined, repository_root=root,
            expected_manifest_path=manifest_path)
        if source_candidate != candidate:
            raise CombinedComparisonError(
                'combined evaluation source identity is invalid')
        runtime = resolve_numeric_runtime(
            candidate, repository_root=root, manifest_path=manifest_path,
            downstream_output=evaluation_path)
        validate_evaluation_envelope(
            combined, expected_candidate_id=candidate.id,
            expected_route=candidate.route,
            expected_checkpoint_sha256=runtime['checkpoint_sha256'],
            expected_authority_sha256=source['authority_sha256'],
            expected_source_config=runtime['config_path'].relative_to(
                root).as_posix(),
            expected_checkpoint=runtime['checkpoint_name'],
            expected_seed=candidate.seed,
            expected_git_commit=source['git_commit'],
            expected_source_binding=source,
            expected_authority=coco_authority,
            require_source_binding=True,
            expected_pwl_stage_a=runtime['pwl_stage_a'])
    except CombinedComparisonError:
        raise
    except (OSError, ValueError) as error:
        raise CombinedComparisonError(
            f'combined evaluation model identity is invalid: {error}') \
            from error
    rebuilt = build_combined_comparison(
        full, combined,
        full_reference=comparator['evaluation_artifact'],
        combined_reference=evaluation_reference,
        candidate_id=candidate.id)
    if value != rebuilt:
        raise CombinedComparisonError(
            'combined comparison differs from authenticated evaluations')
    return rebuilt


def load_combined_comparison_binding(
        candidate, *, repository_root: Path, manifest_path: Path,
        downstream_output: Path) -> dict[str, str]:
    """Validate and bind the one compare artifact consumed by latency."""
    root = Path(repository_root).resolve(strict=True)
    compare_path = Path(downstream_output).parent.parent / 'compare/compare.json'
    try:
        payload = compare_path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise CombinedComparisonError(
            'combined comparison artifact is unavailable') from error
    validate_combined_comparison_artifact(
        value, candidate=candidate, artifact_path=compare_path,
        repository_root=root, manifest_path=manifest_path)
    return {
        'path': compare_path.relative_to(root).as_posix(),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }
