#!/usr/bin/env python3
"""Measure synchronized batch-one flip and no-flip candidate GPU latency."""

from __future__ import annotations

import argparse
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

from mambapose_opt.evaluation import stage_envelope
from mambapose_opt.latency import build_latency_result, measure_latency_samples
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


def _gpu_lease(candidate_id: str) -> dict[str, Any]:
    try:
        value = json.loads(_canonical_gpu_lock().read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read active GPU lease: {error}') from error
    if value.get('stage_id') != f'{candidate_id}:latency':
        raise ValueError('active GPU lease does not match latency stage')
    return value


def _git_commit() -> str:
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    if dirty.stdout.strip():
        raise RuntimeError('latency measurement requires a clean tracked worktree')
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()


def measure_candidate(
        candidate: CandidateSpec, *, warmup: int, repeats: int) -> dict:
    checkpoint = REPO_ROOT / candidate.checkpoint
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != candidate.checkpoint_sha256:
        raise ValueError(f'checkpoint sha256 mismatch for {candidate.id}')
    config_path = REPO_ROOT / candidate.config
    config = Config.fromfile(config_path)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    config_sha256 = hashlib.sha256(
        config.pretty_text.encode('utf-8')).hexdigest()
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
        gpu_lease=_gpu_lease(candidate.id))
    result.update({
        'route': candidate.route,
        'provenance': {
            'checkpoint_sha256': checkpoint_hash,
            'config_sha256': config_sha256,
            'data_inventory_sha256': _sha256(
                REPO_ROOT / 'data/inventory.json'),
            'git_commit': _git_commit(),
        },
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
