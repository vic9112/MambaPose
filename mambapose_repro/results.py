"""Fail-closed metric and COCO test-dev submission validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    value: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_testdev(
        path: Path | str,
        *,
        expected_image_ids: set[int]) -> ValidationReport:
    path = Path(path)
    errors: list[str] = []
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return ValidationReport(False, [f'cannot load submission: {error}'])
    if not isinstance(raw, list) or not raw:
        return ValidationReport(False, ['submission must be a non-empty list'])
    required = {'image_id', 'category_id', 'keypoints', 'score'}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or not required <= set(row):
            errors.append(f'row {index} lacks required COCO fields')
            continue
        if row['image_id'] not in expected_image_ids:
            errors.append(f'row {index} has unknown image_id {row["image_id"]}')
        if row['category_id'] != 1:
            errors.append(f'row {index} category_id must be 1')
        keypoints = row['keypoints']
        if (not isinstance(keypoints, list) or len(keypoints) != 51
                or not all(_finite_number(value) for value in keypoints)):
            errors.append(f'row {index} must have 51 finite keypoint values')
        if not _finite_number(row['score']):
            errors.append(f'row {index} has non-finite score')
        rows.append(row)
    rows.sort(key=lambda row: (row.get('image_id', -1), -row.get('score', 0)))
    return ValidationReport(not errors, errors, rows)


def normalize_testdev(
        path: Path | str,
        *,
        expected_image_ids: set[int]) -> dict[str, Any]:
    path = Path(path)
    report = validate_testdev(path, expected_image_ids=expected_image_ids)
    if not report.valid:
        raise ValueError(f'invalid test-dev submission: {report.errors}')
    return {
        'evaluation': 'submission_only',
        'prediction_rows': len(report.rows),
        'image_ids_with_predictions': len({
            row['image_id'] for row in report.rows
        }),
        'sha256': _sha256(path),
    }


def validate_metrics(
        path: Path | str,
        allowed_metrics: Iterable[str],
        *,
        provenance: Mapping[str, str]) -> ValidationReport:
    path = Path(path)
    errors: list[str] = []
    try:
        metrics = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return ValidationReport(False, [f'cannot load metrics: {error}'])
    if not isinstance(metrics, dict) or not metrics:
        errors.append('metrics must be a non-empty object')
        metrics = {}
    allowed = set(allowed_metrics)
    unknown = set(metrics) - allowed
    if unknown:
        errors.append(f'unknown metrics: {sorted(unknown)}')
    if any(not _finite_number(value) for value in metrics.values()):
        errors.append('all metric values must be finite numbers')
    required_provenance = {
        'checkpoint_sha256', 'config_sha256', 'data_inventory_sha256'
    }
    if not required_provenance <= set(provenance):
        errors.append('metric provenance is missing required hashes')
    else:
        for key in required_provenance:
            value = provenance[key]
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f'invalid provenance hash: {key}')
    return ValidationReport(not errors, errors, value=metrics)


def validate_ablation_directions(
        measured_ap: Mapping[str, float]) -> ValidationReport:
    """Require each paper ablation to underperform its matched full model."""
    comparisons = (
        ('coco-pif', 'coco-s-v1', 'coco-s-v1-no-pif'),
        ('crowdpose-pif', 'crowdpose-s-v1', 'crowdpose-s-v1-no-pif'),
        ('crowdpose-prior', 'crowdpose-s-v1',
         'crowdpose-s-v1-no-prior'),
        ('crowdpose-cycling', 'crowdpose-s-v1',
         'crowdpose-s-v1-no-cycling'),
    )
    errors = []
    rows = []
    for name, full_id, ablation_id in comparisons:
        full = measured_ap.get(full_id)
        ablation = measured_ap.get(ablation_id)
        if not _finite_number(full) or not _finite_number(ablation):
            errors.append(
                f'{name} lacks finite full/ablation AP measurements')
            continue
        delta = full - ablation
        row = {
            'comparison': name,
            'full_run': full_id,
            'ablation_run': ablation_id,
            'full_ap': full,
            'ablation_ap': ablation,
            'delta_ap': delta,
            'direction_reproduced': delta > 0,
        }
        rows.append(row)
        if delta <= 0:
            errors.append(
                f'{ablation_id} does not reproduce the paper direction: '
                f'full={full}, ablation={ablation}')
    return ValidationReport(not errors, errors, rows)
