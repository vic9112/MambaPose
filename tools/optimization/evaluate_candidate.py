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
    build_determinism_record, repeated_order_hash)
from mambapose_opt.evaluation import (
    build_deterministic_evaluation_config, build_source_binding,
    load_coco_metrics, stage_envelope,
    validate_coco_val_protocol)
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest
from mambapose_opt.source import clean_git_commit
from mambapose_opt.numeric_runtime import resolve_numeric_runtime
from mambapose_opt.process_environment import deterministic_child_environment
from mambapose_opt.checkpoints import (
    authorize_pwl_runtime_config,
    materialize_evaluation_config_authority)


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


def _child_environment() -> dict[str, str]:
    return deterministic_child_environment(REPO_ROOT)


_deterministic_config = build_deterministic_evaluation_config


def _evaluate_mode(
        candidate: CandidateSpec, output: Path, *, flip_test: bool,
        checkpoint_sha256: str, git_commit: str,
        checkpoint: Path,
        config_path: Path | None = None,
        checkpoint_name: str | None = None,
        manifest_path: Path | None = None,
        config_authority=None) -> dict:
    config_path = config_path or REPO_ROOT / candidate.config
    mode = 'flip' if flip_test else 'no-flip'
    resolved = output.parent / f'resolved-{mode}.py'
    materialized_authority_path = (
        output.parent / f'resolved-{mode}.config-authority.json')
    materialized = None
    if candidate.features.get('numeric_kind') in {'pwl', 'pwl-combined'}:
        if manifest_path is None or config_authority is None:
            raise ValueError(
                'PWL evaluation requires manifest and runtime ConfigAuthority')
        materialized = materialize_evaluation_config_authority(
            config_authority, flip_test=flip_test, config_path=resolved,
            authority_path=materialized_authority_path)
        config = materialized.load_config()
        resolved = materialized.path
    else:
        config = _deterministic_config(candidate, flip_test, config_path)
        _dump_config(config, resolved)
    protocol = validate_coco_val_protocol(config, repository_root=REPO_ROOT)
    raw_metrics = output.parent / f'raw-{mode}-mmpose-metrics.json'
    work_dir = output.parent / f'mmpose-{mode}'
    provenance = {
        'checkpoint_sha256': checkpoint_sha256,
        'config_sha256': (
            materialized.sha256 if materialized is not None
            else _sha256(resolved)),
        'data_inventory_sha256': protocol['inventory_projection'][
            'inventory_sha256'],
        'git_commit': git_commit,
    }
    order_hash = repeated_order_hash(
        config.test_dataloader, seed=candidate.seed, epoch=0)
    environment = _child_environment()
    environment['MAMBAPOSE_OPTIMIZATION_SEED'] = str(candidate.seed)
    command = [
        sys.executable,
        str(REPO_ROOT / 'tools/test.py'),
        str(resolved),
        str(checkpoint),
        '--work-dir', str(work_dir),
        '--out', str(raw_metrics),
    ]
    if candidate.features.get('numeric_kind') in {'pwl', 'pwl-combined'}:
        if manifest_path is None:
            raise ValueError('PWL evaluation requires its candidate manifest')
        command.extend([
            '--safe-manifest', str(manifest_path),
            '--safe-candidate', candidate.id,
            '--safe-config-authority', str(materialized_authority_path),
        ])
    try:
        subprocess.run(
            command, cwd=REPO_ROOT, env=environment, check=True, shell=False)
    finally:
        if materialized is not None:
            materialized.verify()
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
            'checkpoint': (
                checkpoint_name or checkpoint.relative_to(REPO_ROOT).as_posix()),
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
    config_authority = (
        authorize_pwl_runtime_config(
            REPO_ROOT, manifest, candidate,
            conversion_path=config_path.parent / 'convert.json')
        if candidate.features.get('numeric_kind') in {
            'pwl', 'pwl-combined'} else None)
    rows = {
        mode: _evaluate_mode(
            candidate, output, flip_test=mode == 'flip',
            checkpoint_sha256=checkpoint_sha256, git_commit=git_commit,
            checkpoint=checkpoint, config_path=config_path,
            checkpoint_name=runtime['checkpoint_name'],
            manifest_path=manifest,
            config_authority=config_authority)
        for mode in modes
    }
    if (candidate.route == 'ssm-quant-pwl'
            and (_sha256(config_path) != runtime['config_sha256']
                 or _sha256(checkpoint) != runtime['checkpoint_sha256'])):
        raise ValueError('evaluation runtime inputs changed during execution')
    result = {
        'route': candidate.route,
        'calibration_split': None,
        'modes': rows,
        'source': source,
    }
    if candidate.features.get('numeric_kind') in {'pwl', 'pwl-combined'}:
        result['pwl_stage_a'] = dict(runtime['pwl_stage_a'])
    if candidate.kind == 'binary-qk':
        from mambapose_opt.binary_operation import (
            binary_profile_binding_for_stage)
        result['binary_qk_profile'] = binary_profile_binding_for_stage(
            output.relative_to(REPO_ROOT), repository_root=REPO_ROOT)
    return stage_envelope(candidate.id, 'evaluate', result)


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
