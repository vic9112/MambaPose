"""Deterministic worker seeding, order tracing, and provenance records."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')


class DeterminismError(ValueError):
    """Raised when a repeatability record is incomplete or inconsistent."""


def seed_deterministic_root(seed: int) -> dict[str, int | bool]:
    """Reset every calibration RNG and enforce deterministic Torch backends."""
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed < 2**32):
        raise DeterminismError('seed must be an unsigned 32-bit integer')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        'seed': seed,
        'python_seed': seed,
        'numpy_seed': seed,
        'torch_seed': seed,
        'torch_cuda_seed': seed,
        'torch_deterministic_algorithms': True,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
    }


def seed_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch for one data-loader worker.

    PyTorch owns the per-worker seed. Python and NumPy receive its low 32 bits
    while Torch retains the complete worker seed.
    """
    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError('worker_id must be an integer')
    if worker_id < 0:
        raise ValueError('worker_id must be non-negative')
    torch_seed = int(torch.initial_seed())
    library_seed = torch_seed % (2**32)
    random.seed(library_seed)
    np.random.seed(library_seed)
    torch.manual_seed(torch_seed)


try:
    from mmengine.registry import FUNCTIONS

    if FUNCTIONS.get('mambapose_seed_worker') is None:
        FUNCTIONS.register_module(
            name='mambapose_seed_worker', module=seed_worker)
except ImportError:  # pragma: no cover - package remains usable without MMEngine
    pass


class OrderHashRecorder:
    """Record an immutable SHA-256 for each epoch's exact sample-ID order."""

    def __init__(self) -> None:
        self._digests: dict[int, str] = {}

    def update(self, epoch: int, sample_ids: Sequence[int]) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a non-negative integer')
        if epoch in self._digests:
            raise ValueError(f'epoch {epoch} is already recorded')
        if not isinstance(sample_ids, Sequence) or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in sample_ids):
            raise ValueError('sample_ids must be a sequence of integers')
        payload = json.dumps(
            {'epoch': epoch, 'sample_ids': list(sample_ids)},
            separators=(',', ':'), sort_keys=True).encode('utf-8')
        self._digests[epoch] = hashlib.sha256(payload).hexdigest()

    def hexdigest(self, epoch: int) -> str:
        try:
            return self._digests[epoch]
        except KeyError as error:
            raise ValueError(f'epoch {epoch} has not been recorded') from error

    def as_dict(self) -> dict[int, str]:
        return dict(sorted(self._digests.items()))


def deterministic_dataloader_config(
        value: Mapping[str, Any], *, seed: int,
        worker_count: int = 2) -> Mapping[str, Any]:
    """Copy a loader config and enforce the campaign worker policy."""
    if not isinstance(value, Mapping):
        raise DeterminismError('data-loader config must be a mapping')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DeterminismError('seed must be a non-negative integer')
    if (
            isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count < 0):
        raise DeterminismError('worker_count must be a non-negative integer')
    config = copy.deepcopy(value)
    config['num_workers'] = worker_count
    config['persistent_workers'] = False
    config['worker_init_fn'] = {'type': 'mambapose_seed_worker'}
    return config


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DeterminismError(f'{field} must be a lowercase SHA-256')
    return value


def _worker_records(worker_count: int) -> list[dict[str, int | str]]:
    return [{
        'worker_id': worker_id,
        'torch_seed_source': 'torch.initial_seed()',
        'python_seed_derivation': 'torch_seed % 2**32',
        'numpy_seed_derivation': 'torch_seed % 2**32',
    } for worker_id in range(worker_count)]


def build_determinism_record(
        *, seed: int, worker_count: int, persistent_workers: bool,
        order_hashes: Mapping[int, str], config_sha256: str,
        data_inventory_sha256: str, checkpoint_sha256: str,
        git_commit: str) -> dict[str, Any]:
    """Build the complete deterministic execution record used by artifacts."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DeterminismError('seed must be a non-negative integer')
    if (
            isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count < 0):
        raise DeterminismError('worker_count must be a non-negative integer')
    if persistent_workers is not False:
        raise DeterminismError('persistent workers must be disabled')
    normalized_hashes: dict[str, str] = {}
    for epoch, digest in order_hashes.items():
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DeterminismError('order-hash epoch must be non-negative')
        normalized_hashes[str(epoch)] = _require_sha256(
            digest, f'order_hashes[{epoch}]')
    if not normalized_hashes:
        raise DeterminismError('at least one order hash is required')
    provenance = {
        'config_sha256': _require_sha256(config_sha256, 'config_sha256'),
        'data_inventory_sha256': _require_sha256(
            data_inventory_sha256, 'data_inventory_sha256'),
        'checkpoint_sha256': _require_sha256(
            checkpoint_sha256, 'checkpoint_sha256'),
    }
    if not isinstance(git_commit, str) or not _COMMIT.fullmatch(git_commit):
        raise DeterminismError('git_commit must be a full lowercase commit hash')
    provenance['git_commit'] = git_commit
    root_seed = seed % (2**32)
    return {
        'python_seed': root_seed,
        'numpy_seed': root_seed,
        'torch_seed': root_seed,
        'worker_count': worker_count,
        'workers': _worker_records(worker_count),
        'persistent_workers': False,
        'order_hashes': dict(sorted(normalized_hashes.items())),
        'provenance': provenance,
    }


def trace_sample_order(
        dataloader_cfg: Mapping[str, Any], *, seed: int,
        epoch: int = 0) -> str:
    """Hash dataset IDs in the exact order produced by an MMEngine sampler."""
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    dataloader = Runner.build_dataloader(dataloader_cfg, seed=seed)
    sampler = dataloader.sampler
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    dataset = dataloader.dataset
    sample_ids: list[int] = []
    for index in sampler:
        data_info = dataset.get_data_info(int(index))
        identifier = data_info.get('img_id', data_info.get('id', index))
        if isinstance(identifier, np.integer):
            identifier = int(identifier)
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise DeterminismError(
                f'dataset sample {index} has no integer image ID')
        sample_ids.append(identifier)
    recorder = OrderHashRecorder()
    recorder.update(epoch, sample_ids)
    return recorder.hexdigest(epoch)


def repeated_order_hash(
        dataloader_cfg: Mapping[str, Any], *, seed: int,
        epoch: int = 0) -> str:
    """Repeat a preflight trace and fail closed if the hashes differ."""
    first = trace_sample_order(dataloader_cfg, seed=seed, epoch=epoch)
    second = trace_sample_order(dataloader_cfg, seed=seed, epoch=epoch)
    if first != second:
        raise DeterminismError(
            f'data-loader order hash mismatch for epoch {epoch}: '
            f'{first} != {second}')
    return first
