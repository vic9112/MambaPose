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

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / 'work_dirs/optimization'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_opt.determinism import (
    build_determinism_record, deterministic_dataloader_config,
    repeated_order_hash)
from mambapose_opt.evaluation import (
    build_source_binding, load_coco_metrics, stage_envelope,
    validate_coco_val_protocol)
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest
from mambapose_opt.source import clean_git_commit
from mambapose_opt.numeric_runtime import resolve_numeric_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _output_path(value: str) -> Path:
    return optimization_output_path(value, repository_root=REPO_ROOT)


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
    return clean_git_commit(REPO_ROOT)


def _deterministic_config(
        candidate: CandidateSpec, flip_test: bool, config_path: Path) -> Config:
    config = Config.fromfile(config_path)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    configured_imports = list(
        config.get('custom_imports', {}).get('imports', ()))
    if 'mambapose_opt.determinism' not in configured_imports:
        configured_imports.append('mambapose_opt.determinism')
    config.custom_imports = dict(
        imports=configured_imports, allow_failed_imports=False)
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        config[name] = deterministic_dataloader_config(
            config[name], seed=candidate.seed, worker_count=2)
    config.model.test_cfg.flip_test = flip_test
    return config


def _evaluate_mode(
        candidate: CandidateSpec, output: Path, *, flip_test: bool,
        checkpoint_sha256: str, git_commit: str,
        checkpoint: Path | None = None,
        config_path: Path | None = None) -> dict:
    checkpoint = checkpoint or REPO_ROOT / candidate.checkpoint
    config_path = config_path or REPO_ROOT / candidate.config
    config = _deterministic_config(candidate, flip_test, config_path)
    protocol = validate_coco_val_protocol(config, repository_root=REPO_ROOT)
    mode = 'flip' if flip_test else 'no-flip'
    resolved = output.parent / f'resolved-{mode}.py'
    raw_metrics = output.parent / f'raw-{mode}-mmpose-metrics.json'
    work_dir = output.parent / f'mmpose-{mode}'
    _dump_config(config, resolved)
    provenance = {
        'checkpoint_sha256': checkpoint_sha256,
        'config_sha256': _sha256(resolved),
        'data_inventory_sha256': protocol['inventory_projection'][
            'inventory_sha256'],
        'git_commit': git_commit,
    }
    order_hash = repeated_order_hash(
        config.test_dataloader, seed=candidate.seed, epoch=0)
    environment = os.environ.copy()
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
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
    return {
        'metrics': metrics.to_dict(),
        'provenance': provenance,
        'determinism': determinism,
        'protocol': {
            **protocol,
            'batch_size': int(config.test_dataloader.batch_size),
            'source_config': config_path.relative_to(REPO_ROOT).as_posix(),
            'checkpoint': checkpoint.relative_to(REPO_ROOT).as_posix(),
            'data_inventory': 'data/inventory.json',
        },
    }


def evaluate(
        candidate: CandidateSpec, output: Path, *,
        modes: tuple[str, ...] = ('flip', 'no_flip'),
        manifest_path: Path | None = None) -> dict:
    manifest = manifest_path or REPO_ROOT / 'optimization/candidates.json'
    runtime = resolve_numeric_runtime(
        candidate, repository_root=REPO_ROOT, manifest_path=manifest,
        downstream_output=output)
    checkpoint = runtime['checkpoint_path']
    config_path = runtime['config_path']
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != runtime['checkpoint_sha256']:
        raise ValueError(
            f'checkpoint sha256 mismatch for {candidate.id}: '
            f'{checkpoint_sha256}')
    git_commit = _git_commit()
    source = build_source_binding(
        repository_root=REPO_ROOT, candidate=candidate,
        manifest_path=manifest,
        git_commit=git_commit)
    rows = {
        mode: _evaluate_mode(
            candidate, output, flip_test=mode == 'flip',
            checkpoint_sha256=checkpoint_sha256, git_commit=git_commit,
            checkpoint=checkpoint, config_path=config_path)
        for mode in modes
    }
    if (candidate.route == 'ssm-quant-pwl'
            and (_sha256(config_path) != runtime['config_sha256']
                 or _sha256(checkpoint) != runtime['checkpoint_sha256'])):
        raise ValueError('evaluation runtime inputs changed during execution')
    return stage_envelope(candidate.id, 'evaluate', {
        'route': candidate.route,
        'calibration_split': None,
        'modes': rows,
        'source': source,
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
    parser.set_defaults(flip_test=None)
    args = parser.parse_args()
    candidate = _candidate(args.manifest, args.candidate_id)
    modes = ('flip', 'no_flip') if args.flip_test is None else (
        ('flip',) if args.flip_test else ('no_flip',))
    _atomic_json(args.output, evaluate(
        candidate, args.output, modes=modes, manifest_path=args.manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
