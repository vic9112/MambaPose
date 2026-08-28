#!/usr/bin/env python3
"""Measure synchronized batch-one flip and no-flip candidate GPU latency."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Callable


_REQUIRED_ENVIRONMENT = {
    'PYTHONDONTWRITEBYTECODE': '1',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
}
if __name__ == '__main__' and any(
        os.environ.get(name) != value
        for name, value in _REQUIRED_ENVIRONMENT.items()):
    environment = os.environ.copy()
    environment.update(_REQUIRED_ENVIRONMENT)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

sys.dont_write_bytecode = True

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / 'work_dirs/optimization'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_opt.evaluation import (
    build_source_binding, resolve_project_asset_root, stage_envelope,
    validate_coco_val_protocol)
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.gpu_guard import controller_process_tree
from mambapose_opt.latency import (
    LEASE_MAX_AGE_SECONDS, LEASE_MAX_FUTURE_SKEW_SECONDS,
    build_latency_result, measure_latency_samples, validate_gpu_lease)
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest
from mambapose_opt.numeric_conversion import NumericRuntimeHook
from mambapose_opt.source import clean_git_commit
from mambapose_opt.numeric_runtime import resolve_numeric_runtime
from mambapose_opt.checkpoints import (
    authorize_pwl_runtime_config, build_manifest_authorized_model)


LEASE_MAX_AGE = timedelta(seconds=LEASE_MAX_AGE_SECONDS)
LEASE_MAX_FUTURE_SKEW = timedelta(seconds=LEASE_MAX_FUTURE_SKEW_SECONDS)


def _require_deterministic_environment() -> None:
    if (
            os.environ.get('PYTHONDONTWRITEBYTECODE') != '1'
            or not sys.dont_write_bytecode):
        raise RuntimeError(
            'latency requires PYTHONDONTWRITEBYTECODE=1 before source/GPU '
            'admission')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError(
            'latency requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before '
            'source/GPU admission')


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


def _active_gpu_lease(
        candidate_id: str, device_index: int, *,
        now: Callable[[], datetime] | None = None) -> dict[str, Any]:
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
    timestamp = datetime.fromisoformat(value['timestamp'])
    current = now() if now is not None else datetime.now(timezone.utc)
    if current - timestamp > LEASE_MAX_AGE:
        raise ValueError('active GPU lease timestamp is stale')
    if timestamp - current > LEASE_MAX_FUTURE_SKEW:
        raise ValueError('active GPU lease timestamp is in the future')
    if os.getpid() not in controller_process_tree({value['pid']}):
        raise ValueError('latency process is not a live controller descendant')
    return value


def _git_commit() -> str:
    return clean_git_commit(REPO_ROOT)


def _latency_admission(
        candidate: CandidateSpec, *, device_index: int,
        checkpoint: Path, checkpoint_sha256: str,
        config_path: Path) -> dict[str, Any]:
    commit = _git_commit()
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != checkpoint_sha256:
        raise ValueError(f'checkpoint sha256 mismatch for {candidate.id}')
    inventory_path = resolve_project_asset_root(REPO_ROOT) / 'data/inventory.json'
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
        device_index: int | None = None,
        manifest_path: Path | None = None,
        output: Path | None = None) -> dict:
    _require_deterministic_environment()
    if device_index is None:
        try:
            device_index = int(os.environ['MAMBAPOSE_PHYSICAL_DEVICE_INDEX'])
        except (KeyError, ValueError) as error:
            raise ValueError('physical GPU device index is required') from error
    manifest = manifest_path or REPO_ROOT / 'optimization/candidates.json'
    runtime = resolve_numeric_runtime(
        candidate, repository_root=REPO_ROOT, manifest_path=manifest,
        downstream_output=(output or REPO_ROOT / 'work_dirs/optimization/'
                           'latency/latency.json'))
    checkpoint = runtime['checkpoint_path']
    config_path = runtime['config_path']
    admission = _latency_admission(
        candidate, device_index=device_index, checkpoint=checkpoint,
        checkpoint_sha256=runtime['checkpoint_sha256'],
        config_path=config_path)
    source = build_source_binding(
        repository_root=REPO_ROOT, candidate=candidate,
        manifest_path=manifest,
        git_commit=admission['git_commit'])
    config_authority = None
    if candidate.features.get('numeric_kind') == 'pwl':
        config_authority = authorize_pwl_runtime_config(
            REPO_ROOT, manifest, candidate,
            conversion_path=config_path.parent / 'convert.json')
        config = config_authority.load_config()
    else:
        config = Config.fromfile(config_path)
    data_protocol = validate_coco_val_protocol(
        config, repository_root=REPO_ROOT)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    from mmpose.apis import inference_topdown
    import torch

    random.seed(candidate.seed)
    np.random.seed(candidate.seed)
    torch.manual_seed(candidate.seed)
    torch.cuda.manual_seed_all(candidate.seed)
    torch.use_deterministic_algorithms(True)

    if candidate.features.get('numeric_kind') == 'pwl':
        model = build_manifest_authorized_model(
            REPO_ROOT, manifest, candidate,
            config_authority=config_authority, device='cuda:0')
    else:
        from mmpose.apis import init_model
        model = init_model(config, str(checkpoint), device='cuda:0')
    numeric = config.get('numeric_optimization')
    if numeric is not None:
        NumericRuntimeHook.apply_to_model(model, numeric)
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
        'source': source,
    })
    if candidate.features.get('numeric_kind') == 'pwl':
        result['pwl_stage_a'] = dict(runtime['pwl_stage_a'])
    result['protocol']['data'] = data_protocol
    result['protocol'].update({
        'source_config': config_path.relative_to(REPO_ROOT).as_posix(),
        'checkpoint': runtime['checkpoint_name'],
        'data_inventory': 'data/inventory.json',
    })
    if (_sha256(config_path) != runtime['config_sha256']
            or _sha256(checkpoint) != runtime['checkpoint_sha256']):
        raise ValueError('latency runtime inputs changed during execution')
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
        candidate, warmup=args.warmup, repeats=args.iterations,
        manifest_path=args.manifest, output=args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
