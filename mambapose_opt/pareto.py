"""Artifact-derived paired accuracy and Binary Q/K Pareto rules."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from .binary_operation import (
    canonical_json_sha256, validate_binary_operation_manifest)
from .evaluation import CandidateResult


_T_CRITICAL_95 = {3: 4.302652729911275, 5: 2.7764451051977987}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact_root(
        repository_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty relative path')
    raw = value
    relative = Path(raw)
    if (
            relative.is_absolute()
            or any(part in {'', '.', '..'} for part in raw.split('/'))
            or relative.parts[:2] != ('work_dirs', 'optimization')):
        raise ValueError(f'{label} must be a safe optimization artifact root')
    root = Path(repository_root).resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} must not contain symlinks')
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root / 'work_dirs/optimization')
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes authority') from error
    if not resolved.is_dir():
        raise ValueError(f'{label} must be a directory')
    return resolved


def _roots(value: Mapping[int, str], *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} roots must be a mapping')
    seeds = tuple(sorted(value))
    if seeds not in ((0, 1, 2), (0, 1, 2, 3, 4)):
        raise ValueError(f'{label} roots require exact seeds 0..2 or 0..4')
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f'{label} root seed keys are invalid')
    return seeds


def _artifact_hashes(result: Any, artifact_root: Path) -> dict[str, str]:
    paths = getattr(result, 'artifact_paths', None)
    if not isinstance(paths, Mapping) or set(paths) < {
            'evaluation', 'profile', 'latency'}:
        raise ValueError('public CandidateResult artifact paths are incomplete')
    hashes = {}
    for role, value in sorted(paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError('public CandidateResult artifact role is invalid')
        path = Path(value).resolve(strict=True)
        try:
            relative = path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                'public CandidateResult artifact escapes its root') from error
        if not path.is_file() or path.is_symlink():
            raise ValueError('public CandidateResult artifact is not regular')
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _finite_ap(value: object, *, label: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0):
        raise ValueError(f'{label} AP is invalid')
    return float(value)


def _load_seed_rows(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> tuple[list[dict[str, Any]], dict]:
    seeds = _roots(baseline_roots, label='baseline')
    if _roots(candidate_roots, label='candidate') != seeds:
        raise ValueError('baseline and candidate seed roots disagree')
    rows = []
    common_operation = None
    for seed in seeds:
        baseline_root = _artifact_root(
            repository_root, baseline_roots[seed],
            label=f'seed {seed} baseline root')
        candidate_root = _artifact_root(
            repository_root, candidate_roots[seed],
            label=f'seed {seed} candidate root')
        baseline = CandidateResult.from_artifacts(baseline_root, mode='flip')
        candidate = CandidateResult.from_artifacts(candidate_root, mode='flip')
        if (
                baseline.candidate_id != 'full-s-v1'
                or baseline.candidate_kind != 'float'
                or baseline.route != 'baseline'
                or baseline.seed != seed
                or baseline.flip_test is not True):
            raise ValueError('public baseline CandidateResult identity is invalid')
        if (
                candidate.candidate_id != candidate_id
                or candidate.candidate_kind != 'binary-qk'
                or candidate.route != 'ssm-quant-pwl'
                or candidate.seed != seed
                or candidate.flip_test is not True):
            raise ValueError('public Binary CandidateResult identity is invalid')
        baseline_source = getattr(baseline, 'source', None)
        candidate_source = getattr(candidate, 'source', None)
        if (
                not isinstance(baseline_source, Mapping)
                or not isinstance(candidate_source, Mapping)
                or baseline_source.get('git_commit') !=
                candidate_source.get('git_commit')
                or baseline_source.get('authority_sha256') !=
                candidate_source.get('authority_sha256')):
            raise ValueError('paired CandidateResult source authority differs')
        operation = validate_binary_operation_manifest(
            candidate.binary_operation)
        if common_operation is None:
            common_operation = dict(operation)
        elif dict(operation) != common_operation:
            raise ValueError('Binary operation manifest differs across seeds')
        baseline_ap = _finite_ap(
            baseline.metrics.ap, label=f'seed {seed} baseline')
        candidate_ap = _finite_ap(
            candidate.metrics.ap, label=f'seed {seed} candidate')
        rows.append({
            'seed': seed,
            'baseline_root': baseline_roots[seed],
            'candidate_root': candidate_roots[seed],
            'baseline_artifact_sha256': _artifact_hashes(
                baseline, baseline_root),
            'candidate_artifact_sha256': _artifact_hashes(
                candidate, candidate_root),
            'baseline_source_sha256': canonical_json_sha256(baseline_source),
            'candidate_source_sha256': canonical_json_sha256(candidate_source),
            'baseline_ap_points': baseline_ap,
            'candidate_ap_points': candidate_ap,
            'ap_drop_points': baseline_ap - candidate_ap,
            'operation_manifest_sha256': canonical_json_sha256(operation),
        })
    assert common_operation is not None
    return rows, common_operation


def _statistics(drops: list[float]) -> dict[str, Any]:
    count = len(drops)
    mean = statistics.fmean(drops)
    stddev = statistics.stdev(drops)
    margin = _T_CRITICAL_95[count] * stddev / math.sqrt(count)
    return {
        'seed_count': count,
        'paired_sample_stddev_points': stddev,
        'paired_95_ci_points': [mean - margin, mean + margin],
        'confidence_method': f'paired-student-t-df-{count - 1}',
    }


def _build_record(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> dict[str, Any]:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('Pareto candidate id must be non-empty')
    rows, operation = _load_seed_rows(
        candidate_id=candidate_id, repository_root=repository_root,
        baseline_roots=baseline_roots, candidate_roots=candidate_roots)
    drops = [float(row['ap_drop_points']) for row in rows]
    statistics_record = _statistics(drops)
    initial_statistics = _statistics(drops[:3])
    initial_ci = initial_statistics['paired_95_ci_points']
    initial_intersects = initial_ci[0] <= 0.1 <= initial_ci[1]
    if len(rows) == 5 and not initial_intersects:
        raise ValueError('seeds 3-4 are only admitted after CI intersection')
    mean_drop = statistics.fmean(drops)
    max_drop = max(drops)
    point_rules = mean_drop < 0.1 and max_drop < 0.3
    requires_more = len(rows) == 3 and initial_intersects
    accuracy_pass = point_rules and not requires_more
    operation_sha = canonical_json_sha256(operation)
    hardware_pass = (
        operation['theoretical_qk_multiplications_replaced'] == 6_489_600
        and operation['bitwise_kernel_present'] is False
        and operation['measured_integer_latency'] is False
        and operation['hardware_claim'] == 'none-software-proxy'
        and operation['binaryattention_reproduction'] is False
        and operation['learnable_attention_bias'] is False)
    if requires_more:
        decision = 'requires-seeds-3-4'
    elif accuracy_pass and hardware_pass:
        decision = 'pareto-eligible'
    else:
        decision = 'rejected'
    return {
        'schema_version': 2,
        'artifact_kind': 'mambapose-final-pareto-record',
        'candidate_id': candidate_id,
        'candidate_kind': 'binary-qk',
        'seeds': rows,
        'statistics': statistics_record,
        'accuracy_gate': {
            'mean_ap_drop_points': mean_drop,
            'max_ap_drop_points': max_drop,
            'mean_limit_exclusive': 0.1,
            'max_limit_exclusive': 0.3,
            'point_rules_passed': point_rules,
            'three_seed_ci_intersects_mean_limit': initial_intersects,
            'passed': accuracy_pass,
        },
        'hardware_evidence': {
            'kind': 'binary-qk-theoretical-operation-replacement',
            'theoretical_qk_multiplications_replaced': 6_489_600,
            'operation_manifest_sha256': operation_sha,
            'reference_scope': operation['reference_scope'],
            'learnable_attention_bias': False,
            'binaryattention_reproduction': False,
            'bitwise_kernel_present': False,
            'measured_integer_latency': False,
            'speedup_claim': 'none-software-proxy',
            'passed': hardware_pass,
        },
        'decision': decision,
    }


def build_final_pareto_record(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> dict[str, Any]:
    """Build Pareto evidence solely from public-valid artifact roots."""
    record = _build_record(
        candidate_id=candidate_id, repository_root=repository_root,
        baseline_roots=baseline_roots, candidate_roots=candidate_roots)
    validate_final_pareto_record(record, repository_root=repository_root)
    return record


def validate_final_pareto_record(
        value: object, *, repository_root: Path) -> Mapping[str, Any]:
    """Reload every artifact and reject any caller-reported derived value."""
    required = {
        'schema_version', 'artifact_kind', 'candidate_id', 'candidate_kind',
        'seeds', 'statistics', 'accuracy_gate', 'hardware_evidence',
        'decision'}
    if (
            not isinstance(value, Mapping) or set(value) != required
            or value.get('schema_version') != 2
            or value.get('artifact_kind') !=
            'mambapose-final-pareto-record'
            or value.get('candidate_kind') != 'binary-qk'
            or value.get('decision') not in {
                'pareto-eligible', 'requires-seeds-3-4', 'rejected'}
            or not isinstance(value.get('seeds'), list)):
        raise ValueError('final Pareto record identity is invalid')
    try:
        baseline_roots = {
            int(row['seed']): row['baseline_root'] for row in value['seeds']}
        candidate_roots = {
            int(row['seed']): row['candidate_root'] for row in value['seeds']}
        expected = _build_record(
            candidate_id=value['candidate_id'],
            repository_root=repository_root,
            baseline_roots=baseline_roots,
            candidate_roots=candidate_roots)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f'final Pareto public evidence is invalid: {error}') from error
    if dict(value) != expected:
        raise ValueError(
            'final Pareto record disagrees with recomputed public evidence')
    return value
