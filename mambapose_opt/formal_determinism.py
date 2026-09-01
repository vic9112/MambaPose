"""Deterministic seed and complete-order primitives for formal training."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
import random
import struct
from typing import Iterable, Iterator, Sized

import numpy as np
import torch
from torch.utils.data import Sampler


class FormalDeterminismError(ValueError):
    """A deterministic training input is invalid."""


@dataclass(frozen=True)
class RootDeterminism:
    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < 0 or value > 2**32 - 1:
        raise FormalDeterminismError('seed must be an unsigned 32-bit integer')
    return value


def configure_root_determinism(seed: int) -> RootDeterminism:
    value = _seed(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    # manual_seed_all does not initialize CUDA and is safe before device use.
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return RootDeterminism(value, True, True, False)


def seed_worker(worker_id: int) -> None:
    if isinstance(worker_id, bool) or not isinstance(worker_id, int) \
            or worker_id < 0:
        raise FormalDeterminismError('worker_id must be a non-negative integer')
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _EpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, epoch: int) -> None:
        self._size = len(dataset)
        self._seed = _seed(seed)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise FormalDeterminismError('epoch must be a non-negative integer')
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        # Keep the candidate seed visible while deriving an epoch-specific stream.
        generator.manual_seed((self._seed + self._epoch) % 2**32)
        yield from torch.randperm(self._size, generator=generator).tolist()

    def __len__(self) -> int:
        return self._size


def build_seeded_sampler(
        dataset: Sized, seed: int, epoch: int) -> Sampler[int]:
    return _EpochSampler(dataset, seed, epoch)


def hash_sample_order(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    count = 0
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise FormalDeterminismError(
                'sample index must be a non-negative integer')
        digest.update(struct.pack('>Q', index))
        count += 1
    digest.update(struct.pack('>Q', count))
    return digest.hexdigest()


def trace_epoch_orders(
        spec: object, epochs: int, *, dataset_size: int = 118287
        ) -> tuple[str, ...]:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise FormalDeterminismError('epochs must be a positive integer')
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) \
            or dataset_size < 1:
        raise FormalDeterminismError('dataset_size must be positive')
    if not hasattr(spec, 'seed'):
        raise FormalDeterminismError('formal run spec is missing seed')
    seed = _seed(getattr(spec, 'seed'))
    dataset = range(dataset_size)
    return tuple(hash_sample_order(build_seeded_sampler(
        dataset, seed, epoch)) for epoch in range(epochs))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dataset-size', type=int, default=118287)
    arguments = parser.parse_args()

    class _Spec:
        seed = arguments.seed

    configure_root_determinism(arguments.seed)
    result = {
        'schema_version': 1,
        'pid': os.getpid(),
        'seed': arguments.seed,
        'order_hashes': list(trace_epoch_orders(
            _Spec(), arguments.epochs, dataset_size=arguments.dataset_size)),
        'cuda_initialized': torch.cuda.is_initialized(),
    }
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    _main()
