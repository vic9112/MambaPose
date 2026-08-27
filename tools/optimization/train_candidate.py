#!/usr/bin/env python3
"""Run one bounded Route-3 recovery only after explicit error attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.numeric_source import (
    build_numeric_source_binding, file_sha256)
from mambapose_opt.numeric_runtime import validate_recovery_admission
from mambapose_opt.schema import load_candidate_manifest
from mambapose_opt.source import clean_git_commit


def _candidate(path: Path, identifier: str):
    selected = tuple(
        item for item in load_candidate_manifest(path) if item.id == identifier)
    if len(selected) != 1:
        raise ValueError(f'candidate must resolve exactly once: {identifier}')
    return selected[0]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _safe_reference(record: object, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError(f'{label} reference is invalid')
    relative = Path(record['path'])
    if (relative.is_absolute() or any(part in {'.', '..'} for part in relative.parts)
            or not str(record['sha256']).isalnum()
            or len(str(record['sha256'])) != 64):
        raise ValueError(f'{label} reference is unsafe')
    path = REPO_ROOT / relative
    cursor = REPO_ROOT
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not use symlinks')
    if not path.is_file() or file_sha256(path) != record['sha256']:
        raise ValueError(f'{label} reference hash changed')
    return path


def _recovery_admission(
        path: Path, candidate, manifest_path: Path) -> dict:
    if not path.is_file():
        raise ValueError(
            'conditional numeric recovery is non-runnable until a Task-7 '
            'attributed-error admission artifact is supplied')
    return validate_recovery_admission(
        path, candidate=candidate, repository_root=REPO_ROOT,
        manifest_path=manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--manifest', type=Path,
                        default=REPO_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = optimization_output_path(args.output, repository_root=REPO_ROOT)
    candidate = _candidate(args.manifest, args.candidate_id)
    kind = candidate.features.get('numeric_kind')
    if (candidate.route != 'ssm-quant-pwl' or kind not in {
            'w8a8', 'pwl', 'binary-qk'}
            or candidate.features.get('recovery_candidate') is not True):
        parser.error(
            'train dispatcher accepts explicit post-evaluation recovery '
            'candidates only')
    stage_dir = output.parent
    try:
        admission_path = stage_dir / 'recovery-admission.json'
        admission = _recovery_admission(
            admission_path, candidate, args.manifest)
        commit = clean_git_commit(REPO_ROOT)
        config = Config.fromfile(REPO_ROOT / candidate.config)
        dependency = {'recovery_admission': {
            'path': admission_path.relative_to(REPO_ROOT).as_posix(),
            'sha256': file_sha256(admission_path)}}
        if kind == 'w8a8':
            calibration_path = stage_dir.parent / 'calibrate/calibrate.json'
            _safe_reference({
                'path': calibration_path.relative_to(REPO_ROOT).as_posix(),
                'sha256': file_sha256(calibration_path)},
                label='W8A8 calibration')
            config.numeric_optimization.quant_policy.calibration_artifact = {
                'path': calibration_path.relative_to(REPO_ROOT).as_posix(),
                'sha256': file_sha256(calibration_path)}
            dependency['calibration'] = dict(
                config.numeric_optimization.quant_policy.calibration_artifact)
        work_dir = stage_dir / 'mmpose'
        resolved = stage_dir / 'resolved-train.py'
        config.work_dir = str(work_dir)
        config.load_from = str(REPO_ROOT / candidate.checkpoint)
        config.resume = False
        config.randomness = dict(seed=candidate.seed, deterministic=True)
        config.dump(resolved)
        environment = os.environ.copy()
        environment.update({
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        subprocess.run([
            sys.executable, str(REPO_ROOT / 'tools/train.py'), str(resolved),
            '--work-dir', str(work_dir)], cwd=REPO_ROOT, env=environment,
            check=True, shell=False)
        best = sorted(work_dir.glob('best_*.pth'))
        resumes = sorted(work_dir.glob('epoch_*.pth'))[-2:]
        if len(best) != 1 or len(resumes) != 2:
            raise ValueError(
                'bounded recovery must produce one best and two resume checkpoints')
        runtime_checkpoint = stage_dir / 'best_numeric.pth'
        for source, destination in zip(
                (best[0], *resumes),
                (runtime_checkpoint,
                 stage_dir / resumes[0].name, stage_dir / resumes[1].name)):
            if source.is_symlink() or destination.is_symlink():
                raise ValueError('numeric checkpoint paths must not use symlinks')
            shutil.copyfile(source, destination)
        expected_runtime = candidate.features.get('runtime_checkpoint')
        if (not isinstance(expected_runtime, str)
                or (REPO_ROOT / expected_runtime).resolve()
                != runtime_checkpoint.resolve()):
            raise ValueError('runtime_checkpoint feature disagrees with stage layout')
        metadata_path = stage_dir / 'runtime-metadata.json'
        runtime_sha = file_sha256(runtime_checkpoint)
        _atomic_json(metadata_path, {
            'schema_version': 1, 'candidate_id': candidate.id,
            'route': candidate.route, 'numeric_kind': kind,
            'parent_checkpoint_sha256': candidate.checkpoint_sha256,
            'runtime_checkpoint_sha256': runtime_sha,
            'recovery_admission_sha256': file_sha256(admission_path),
        })
        source = build_numeric_source_binding(
            repository_root=REPO_ROOT, candidate=candidate,
            manifest_path=args.manifest, policy_path=REPO_ROOT / candidate.config,
            git_commit=commit)
        _atomic_json(output, {
            'schema_version': 1, 'candidate_id': candidate.id, 'stage': 'train',
            'result': {
                'route': candidate.route, 'source': source,
                'parent': {
                    'config': candidate.config.as_posix(),
                    'checkpoint': candidate.checkpoint.as_posix(),
                    'checkpoint_sha256': candidate.checkpoint_sha256},
                'dependency': dependency,
                'protocol': {
                    'seed': candidate.seed,
                    'operation': 'one-bounded-numeric-recovery',
                    'attributed_error': admission['attributed_error'],
                    'preliminary_ap_drop':
                        admission['preliminary_ap_drop']},
                'runtime': {
                    'config': {'path': resolved.relative_to(REPO_ROOT).as_posix(),
                               'sha256': file_sha256(resolved)},
                    'checkpoint': {'path': expected_runtime,
                                   'sha256': runtime_sha},
                    'metadata': {
                        'path': metadata_path.relative_to(REPO_ROOT).as_posix(),
                        'sha256': file_sha256(metadata_path)},
                    'transform': 'bounded-numeric-recovery-v1'},
            }})
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
