#!/usr/bin/env python3
"""Validate campaign artifacts and render target/measured/delta evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_repro.manifest import load_manifest
from mambapose_repro.results import (
    normalize_testdev, validate_ablation_directions, validate_metrics)


ALLOWED_METRICS = {
    'coco/AP', 'coco/AP .5', 'coco/AP .75', 'coco/AP (M)', 'coco/AP (L)',
    'coco/AR', 'coco/AR .5', 'coco/AR .75', 'coco/AR (M)', 'coco/AR (L)',
    'crowdpose/AP', 'crowdpose/AP .5', 'crowdpose/AP .75',
    'crowdpose/AP(E)', 'crowdpose/AP(M)', 'crowdpose/AP(H)',
    'crowdpose/AR', 'crowdpose/AR .5', 'crowdpose/AR .75',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def collect() -> dict:
    manifest = load_manifest(REPO_ROOT / 'reproduction/manifest.json')
    rows = []
    all_valid = True
    for run in manifest.runs:
        source_config_path = REPO_ROOT / run.config
        resolved_config_path = (
            REPO_ROOT / 'work_dirs/reproduction/resolved_configs'
            / f'{run.id}.py')
        config_path = (
            resolved_config_path
            if resolved_config_path.is_file() else source_config_path)
        cfg = Config.fromfile(config_path)
        target = dict(cfg.paper_target)
        work_dir = REPO_ROOT / run.work_dir
        completion_path = work_dir / 'completion.json'
        row = {
            'id': run.id,
            'kind': run.kind,
            'target': target,
            'config_sha256': _sha256(config_path),
            'source_config_sha256': _sha256(source_config_path),
            'status': 'missing',
        }
        if not completion_path.is_file():
            all_valid = False
            rows.append(row)
            continue
        try:
            completion = json.loads(
                completion_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            all_valid = False
            row['status'] = 'invalid'
            row['errors'] = [f'cannot load completion: {error}']
            rows.append(row)
            continue
        artifacts = completion.get('artifacts', [])
        provenance = completion.get('provenance', {})
        provenance_errors = []
        if provenance.get('config_sha256') != row['config_sha256']:
            provenance_errors.append('completion config hash mismatch')
        for artifact in artifacts:
            path = (REPO_ROOT / artifact.get('path', '')).resolve()
            if (not path.is_relative_to(REPO_ROOT) or not path.is_file()
                    or _sha256(path) != artifact.get('sha256')):
                provenance_errors.append(
                    f'invalid completion artifact: {artifact.get("path")}')
        if provenance_errors:
            all_valid = False
            row['status'] = 'invalid'
            row['errors'] = provenance_errors
            rows.append(row)
            continue
        if run.kind == 'train':
            metrics_artifact = next((
                item for item in artifacts if item['path'].endswith('metrics.json')
            ), None)
            checkpoint_artifact = next((
                item for item in artifacts if item['path'].endswith('.pth')
            ), None)
            if metrics_artifact is None or checkpoint_artifact is None:
                all_valid = False
                row['status'] = 'invalid'
                row['errors'] = ['completion lacks checkpoint or metrics']
            else:
                metric_provenance = {
                    'checkpoint_sha256': checkpoint_artifact['sha256'],
                    'config_sha256': row['config_sha256'],
                    'data_inventory_sha256': provenance[
                        'data_inventory_sha256'],
                }
                report = validate_metrics(
                    REPO_ROOT / metrics_artifact['path'],
                    ALLOWED_METRICS,
                    provenance=metric_provenance)
                row['status'] = 'valid' if report.valid else 'invalid'
                row['metrics'] = report.value
                row['errors'] = report.errors
                row['checkpoint_sha256'] = checkpoint_artifact['sha256']
                metric_name = (
                    'coco/AP' if target['dataset'] == 'coco'
                    else 'crowdpose/AP')
                measured = (report.value or {}).get(metric_name)
                measured_ap = (
                    measured * 100
                    if isinstance(measured, (int, float)) and measured <= 1
                    else measured)
                if isinstance(measured_ap, (int, float)):
                    row['measured_ap'] = measured_ap
                    row['delta_ap'] = measured_ap - target['value']
                    row['reproduction_class'] = (
                        'direct_reproduction'
                        if abs(row['delta_ap']) <= 0.5 else 'deviation')
                all_valid &= report.valid
        else:
            submission_artifact = artifacts[0] if artifacts else None
            if submission_artifact is None:
                all_valid = False
                row['status'] = 'invalid'
                row['errors'] = ['completion lacks submission']
            else:
                annotation = json.loads((
                    REPO_ROOT / 'data/coco/annotations/'
                    'image_info_test-dev2017.json').read_text())
                expected_ids = {item['id'] for item in annotation['images']}
                try:
                    row['submission'] = normalize_testdev(
                        REPO_ROOT / submission_artifact['path'],
                        expected_image_ids=expected_ids)
                    row['status'] = 'valid'
                except ValueError as error:
                    all_valid = False
                    row['status'] = 'invalid'
                    row['errors'] = [str(error)]
        rows.append(row)
    measured_ap = {
        row['id']: row['measured_ap'] for row in rows
        if row.get('status') == 'valid' and 'measured_ap' in row
    }
    ablations = validate_ablation_directions(measured_ap)
    all_valid &= ablations.valid
    return {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'valid': all_valid,
        'runs': rows,
        'ablation_directions': {
            'valid': ablations.valid,
            'errors': ablations.errors,
            'comparisons': ablations.rows,
        },
    }


def render_markdown(result: dict) -> str:
    lines = [
        '# MambaPose ICME 2025 Reproduction Results',
        '',
        '| Run | Status | Paper AP | Measured AP | Delta |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in result['runs']:
        target = row['target'].get('value')
        prefix = 'coco/AP' if row['target']['dataset'] == 'coco' else 'crowdpose/AP'
        measured_raw = (row.get('metrics') or {}).get(prefix)
        measured = (
            measured_raw * 100 if isinstance(measured_raw, (int, float))
            and measured_raw <= 1.0 else measured_raw)
        delta = measured - target if measured is not None and target is not None else None
        lines.append(
            f'| {row["id"]} | {row["status"]} | '
            f'{target if target is not None else "-"} | '
            f'{measured if measured is not None else "-"} | '
            f'{delta if delta is not None else "-"} |')
    lines.extend([
        '',
        'COCO test-dev rows are submission-only until authenticated CodaLab '
        'evaluation is supplied; no local AP is fabricated.',
        '',
    ])
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--render', type=Path)
    args = parser.parse_args()
    result = collect()
    if args.output:
        _atomic_json(args.output, result)
    if args.render:
        args.render.parent.mkdir(parents=True, exist_ok=True)
        args.render.write_text(render_markdown(result), encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
