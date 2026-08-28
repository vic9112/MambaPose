from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mambapose_opt.formal_determinism import (
    FormalDeterminismError,
    build_seeded_sampler,
    configure_root_determinism,
    hash_sample_order,
    trace_epoch_orders,
    seed_worker,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('seed', [0, 1, 2, 2**32 - 1])
def test_root_determinism_accepts_legal_seed(seed):
    authority = configure_root_determinism(seed)
    assert authority.seed == seed
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize('seed', [True, False, -1, 2**32, 1.5, '0'])
def test_root_determinism_rejects_invalid_seed(seed):
    with pytest.raises(FormalDeterminismError, match='seed'):
        configure_root_determinism(seed)


def test_root_and_worker_seed_every_rng_without_cuda_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr('random.seed', lambda value: calls.append(('python', value)))
    monkeypatch.setattr('numpy.random.seed', lambda value: calls.append(('numpy', value)))
    monkeypatch.setattr(torch, 'manual_seed', lambda value: calls.append(('torch', value)))
    monkeypatch.setattr(
        torch.cuda, 'manual_seed_all',
        lambda value: calls.append(('cuda', value)))
    configure_root_determinism(2)
    assert calls == [
        ('python', 2), ('numpy', 2), ('torch', 2), ('cuda', 2)]

    calls.clear()
    monkeypatch.setattr(torch, 'initial_seed', lambda: 2**32 + 17)
    seed_worker(3)
    assert calls == [('numpy', 17), ('python', 17)]


def test_sampler_repeats_for_pair_and_changes_with_seed_or_epoch():
    dataset = tuple(range(127))
    a = tuple(build_seeded_sampler(dataset, seed=2, epoch=17))
    b = tuple(build_seeded_sampler(dataset, seed=2, epoch=17))
    c = tuple(build_seeded_sampler(dataset, seed=1, epoch=17))
    d = tuple(build_seeded_sampler(dataset, seed=2, epoch=18))
    assert a == b
    assert a != c
    assert a != d
    assert sorted(a) == list(range(len(dataset)))


def test_order_hash_has_typed_unambiguous_encoding():
    assert hash_sample_order([0, 1, 23]) == hash_sample_order(iter([0, 1, 23]))
    assert hash_sample_order([0, 1, 23]) != hash_sample_order([0, 12, 3])
    with pytest.raises(FormalDeterminismError, match='index'):
        hash_sample_order([0, True])


def test_trace_epoch_orders_uses_candidate_seed_not_zero():
    class Spec:
        seed = 2

    class Other:
        seed = 1

    assert trace_epoch_orders(Spec(), 2, dataset_size=257) \
        != trace_epoch_orders(Other(), 2, dataset_size=257)


def test_two_fresh_trace_processes_match_but_have_distinct_pids():
    command = [
        sys.executable, '-B', '-m', 'mambapose_opt.formal_determinism',
        '--seed', '2', '--epochs', '2', '--dataset-size', '257',
    ]
    environment = {
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONNOUSERSITE': '1',
        'PATH': str(Path(sys.executable).parent),
    }
    first = json.loads(subprocess.check_output(
        command, cwd=ROOT, env=environment, text=True))
    second = json.loads(subprocess.check_output(
        command, cwd=ROOT, env=environment, text=True))
    assert first['order_hashes'] == second['order_hashes']
    assert first['pid'] != second['pid']
    assert first['cuda_initialized'] is False
