#!/usr/bin/env python3
"""Measure synchronized batch-one flip and no-flip candidate GPU latency."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / 'work_dirs/optimization'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_opt.evaluation import stage_envelope, validate_coco_val_protocol
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.gpu_guard import controller_process_tree
from mambapose_opt.latency import (
    build_latency_result, measure_latency_samples, validate_gpu_lease)
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest


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


def _canonical_gpu_lock() -> Path:
    common = subprocess.check_output(
        ['git', 'rev-parse', '--git-common-dir'], cwd=REPO_ROOT,
        text=True).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = REPO_ROOT / common_path
    common_path = common_path.resolve()
    if common_path.name != '.git':
        raise ValueError('Git common directory is not a checkout .git')
    return common_path.parent / 'work_dirs/optimization/gpu.lock'


def _active_gpu_lease(candidate_id: str, device_index: int) -> dict[str, Any]:
    lock_path = _canonical_gpu_lock()
    try:
        stream = lock_path.open('r+', encoding='utf-8')
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            value = json.load(stream)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            raise ValueError('canonical GPU lease is not actively held')
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read active GPU lease: {error}') from error
    finally:
        if 'stream' in locals():
            stream.close()
    value = validate_gpu_lease(value)
    if value['stage_id'] != f'{candidate_id}:latency':
        raise ValueError('active GPU lease does not match latency stage')
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if value['boot_id'] != boot_id:
        raise ValueError('active GPU lease belongs to a different boot')
    if value['device_index'] != device_index:
        raise ValueError('active GPU lease device does not match latency device')
    if os.getpid() not in controller_process_tree({value['pid']}):
        raise ValueError('latency process is not a live controller descendant')
    return value


def _git_commit() -> str:
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    if dirty.stdout.strip():
        raise RuntimeError('latency measurement requires a clean tracked worktree')
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()


def _latency_admission(
        candidate: CandidateSpec, *, device_index: int) -> dict[str, Any]:
    commit = _git_commit()
    checkpoint = REPO_ROOT / candidate.checkpoint
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != candidate.checkpoint_sha256:
        raise ValueError(f'checkpoint sha256 mismatch for {candidate.id}')
    config_path = REPO_ROOT / candidate.config
    inventory_path = REPO_ROOT / 'data/inventory.json'
    config_hash = _sha256(config_path)
    inventory_hash = _sha256(inventory_path)
    lease = _active_gpu_lease(candidate.id, device_index)
    return {
        'checkpoint_sha256': checkpoint_hash,
        'config_sha256': config_hash,
        'data_inventory_sha256': inventory_hash,
        'git_commit': commit,
        'gpu_lease': lease,
    }


def measure_candidate(
        candidate: CandidateSpec, *, warmup: int, repeats: int,
        device_index: int | None = None) -> dict:
    if device_index is None:
        try:
            device_index = int(os.environ['MAMBAPOSE_PHYSICAL_DEVICE_INDEX'])
        except (KeyError, ValueError) as error:
            raise ValueError('physical GPU device index is required') from error
    admission = _latency_admission(candidate, device_index=device_index)
    checkpoint = REPO_ROOT / candidate.checkpoint
    config_path = REPO_ROOT / candidate.config
    config = Config.fromfile(config_path)
    data_protocol = validate_coco_val_protocol(
        config, repository_root=REPO_ROOT)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    from mmpose.apis import inference_topdown, init_model
    import torch

    random.seed(candidate.seed)
    np.random.seed(candidate.seed)
    torch.manual_seed(candidate.seed)
    torch.cuda.manual_seed_all(candidate.seed)
    torch.use_deterministic_algorithms(True)

    model = init_model(config, str(checkpoint), device='cuda:0')
    frame = np.zeros((256, 192, 3), dtype=np.uint8)
    box = np.array([[0., 0., 192., 256.]], dtype=np.float32)
    samples: dict[str, tuple[float, ...]] = {}
    for name, flip_test in (('flip', True), ('no_flip', False)):
        model.cfg.model.test_cfg.flip_test = flip_test
        model.test_cfg['flip_test'] = flip_test
        samples[name] = measure_latency_samples(
            lambda: inference_topdown(
                model, frame, box, bbox_format='xyxy'),
            warmup=warmup,
            repeats=repeats,
        )
    result = build_latency_result(
        flip=samples['flip'], no_flip=samples['no_flip'],
        warmup=warmup, repeats=repeats,
        gpu_lease=admission['gpu_lease'])
    result.update({
        'route': candidate.route,
        'provenance': {
            key: admission[key] for key in (
                'checkpoint_sha256', 'config_sha256',
                'data_inventory_sha256', 'git_commit')
        },
    })
    result['protocol']['data'] = data_protocol
    result['protocol'].update({
        'source_config': candidate.config.as_posix(),
        'checkpoint': candidate.checkpoint.as_posix(),
        'data_inventory': 'data/inventory.json',
    })
    return stage_envelope(candidate.id, 'latency', result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--manifest', type=Path,
                        default=REPO_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', type=_output_path, required=True)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    candidate = _candidate(args.manifest, args.candidate_id)
    _atomic_json(args.output, measure_candidate(
        candidate, warmup=args.warmup, repeats=args.iterations))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
