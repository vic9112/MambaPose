#!/usr/bin/env python3
"""Evaluate one frozen candidate on complete COCO val2017 deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / 'work_dirs/optimization'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_opt.determinism import (
    build_determinism_record, deterministic_dataloader_config,
    repeated_order_hash)
from mambapose_opt.evaluation import load_coco_metrics, stage_envelope
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in {'.', '..'} for part in path.parts):
        raise argparse.ArgumentTypeError(
            'output must be repository-relative under work_dirs/optimization')
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'output must be repository-relative under work_dirs/optimization') from error
    return resolved


def _candidate(path: Path, identifier: str) -> CandidateSpec:
    for candidate in load_candidate_manifest(path):
        if candidate.id == identifier:
            return candidate
    raise ValueError(f'candidate not found: {identifier}')


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


def _dump_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.stem}.', suffix='.py', dir=path.parent, text=True)
    os.close(descriptor)
    try:
        config.dump(temporary)
        with Path(temporary).open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _git_commit() -> str:
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    if dirty.stdout.strip():
        raise RuntimeError('evaluation requires a clean tracked Git worktree')
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()


def _deterministic_config(candidate: CandidateSpec, flip_test: bool) -> Config:
    config = Config.fromfile(REPO_ROOT / candidate.config)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    config.custom_imports = dict(
        imports=['mambapose_opt.determinism'], allow_failed_imports=False)
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        config[name] = deterministic_dataloader_config(
            config[name], seed=candidate.seed, worker_count=2)
    config.model.test_cfg.flip_test = flip_test
    return config


def evaluate(
        candidate: CandidateSpec, output: Path, *, flip_test: bool) -> dict:
    checkpoint = REPO_ROOT / candidate.checkpoint
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != candidate.checkpoint_sha256:
        raise ValueError(
            f'checkpoint sha256 mismatch for {candidate.id}: '
            f'{checkpoint_sha256}')
    config = _deterministic_config(candidate, flip_test)
    mode = 'flip' if flip_test else 'no-flip'
    resolved = output.parent / f'resolved-{mode}.py'
    raw_metrics = output.parent / f'raw-{mode}-mmpose-metrics.json'
    work_dir = output.parent / f'mmpose-{mode}'
    _dump_config(config, resolved)
    provenance = {
        'checkpoint_sha256': checkpoint_sha256,
        'config_sha256': _sha256(resolved),
        'data_inventory_sha256': _sha256(REPO_ROOT / 'data/inventory.json'),
        'git_commit': _git_commit(),
    }
    order_hash = repeated_order_hash(
        config.test_dataloader, seed=candidate.seed, epoch=0)
    environment = os.environ.copy()
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'MAMBAPOSE_OPTIMIZATION_SEED': str(candidate.seed),
    })
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / 'tools/test.py'),
        str(resolved),
        str(checkpoint),
        '--work-dir', str(work_dir),
        '--out', str(raw_metrics),
    ], cwd=REPO_ROOT, env=environment, check=True, shell=False)
    metrics = load_coco_metrics(raw_metrics, provenance=provenance)
    determinism = build_determinism_record(
        seed=candidate.seed,
        worker_count=int(config.test_dataloader.num_workers),
        persistent_workers=bool(config.test_dataloader.persistent_workers),
        order_hashes={0: order_hash},
        config_sha256=provenance['config_sha256'],
        data_inventory_sha256=provenance['data_inventory_sha256'],
        checkpoint_sha256=provenance['checkpoint_sha256'],
        git_commit=provenance['git_commit'],
    )
    return stage_envelope(candidate.id, 'evaluate', {
        'route': candidate.route,
        'flip_test': flip_test,
        'metrics': metrics.to_dict(),
        'provenance': provenance,
        'determinism': determinism,
        'calibration_split': None,
        'protocol': {
            'dataset': 'coco',
            'split': 'val2017',
            'batch_size': int(config.test_dataloader.batch_size),
            'complete_split': True,
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--manifest', type=Path,
                        default=REPO_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', type=_output_path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--flip', dest='flip_test', action='store_true')
    mode.add_argument('--no-flip', dest='flip_test', action='store_false')
    parser.set_defaults(flip_test=True)
    args = parser.parse_args()
    candidate = _candidate(args.manifest, args.candidate_id)
    _atomic_json(
        args.output, evaluate(candidate, args.output, flip_test=args.flip_test))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
