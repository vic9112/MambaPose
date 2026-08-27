import json
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from mmengine.config import Config


def test_seed_worker_repeats_python_numpy_and_torch_streams():
    from mambapose_opt.determinism import seed_worker

    seed_worker(3, base_seed=41)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_worker(3, base_seed=41)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second


def test_order_hash_is_repeatable_epoch_sensitive_and_rejects_rewrite():
    from mambapose_opt.determinism import OrderHashRecorder

    a = OrderHashRecorder()
    b = OrderHashRecorder()
    a.update(0, [4, 1, 9])
    b.update(0, [4, 1, 9])
    assert a.hexdigest(0) == b.hexdigest(0)
    b.update(1, [4, 1, 9])
    assert b.hexdigest(1) != b.hexdigest(0)
    with pytest.raises(ValueError, match='already recorded'):
        a.update(0, [4, 1, 9])


def test_determinism_record_contains_exact_worker_and_provenance_contract():
    from mambapose_opt.determinism import build_determinism_record

    record = build_determinism_record(
        seed=7,
        worker_count=2,
        persistent_workers=False,
        order_hashes={0: 'a' * 64},
        config_sha256='b' * 64,
        data_inventory_sha256='c' * 64,
        checkpoint_sha256='d' * 64,
        git_commit='e' * 40,
    )
    assert record['python_seed'] == 7
    assert record['numpy_seed'] == 7
    assert record['torch_seed'] == 7
    assert record['workers'] == [
        {'worker_id': 0, 'python_seed': 7, 'numpy_seed': 7,
         'torch_seed': 7},
        {'worker_id': 1, 'python_seed': 8, 'numpy_seed': 8,
         'torch_seed': 8},
    ]
    assert record['worker_count'] == 2
    assert record['persistent_workers'] is False
    assert record['order_hashes'] == {'0': 'a' * 64}
    assert record['provenance'] == {
        'config_sha256': 'b' * 64,
        'data_inventory_sha256': 'c' * 64,
        'checkpoint_sha256': 'd' * 64,
        'git_commit': 'e' * 40,
    }


def test_determinism_record_rejects_bad_hash_and_persistent_workers():
    from mambapose_opt.determinism import (
        DeterminismError, build_determinism_record)

    common = dict(
        seed=0, worker_count=2, order_hashes={0: 'a' * 64},
        config_sha256='b' * 64, data_inventory_sha256='c' * 64,
        checkpoint_sha256='d' * 64, git_commit='e' * 40)
    with pytest.raises(DeterminismError, match='persistent workers'):
        build_determinism_record(persistent_workers=True, **common)
    with pytest.raises(DeterminismError, match='config_sha256'):
        build_determinism_record(
            persistent_workers=False, **{**common, 'config_sha256': 'bad'})


def test_optimization_configs_enable_fixed_determinism():
    deterministic = Config.fromfile(
        'configs/optimization/coco_s_v1_deterministic.py')
    no_pif = Config.fromfile(
        'configs/optimization/coco_s_v1_no_pif_seed0.py')

    for cfg in (deterministic, no_pif):
        assert cfg.randomness == dict(seed=0, deterministic=True)
        for loader_name in (
                'train_dataloader', 'val_dataloader', 'test_dataloader'):
            loader = cfg[loader_name]
            assert loader.num_workers == 2
            assert loader.persistent_workers is False
            assert loader.worker_init_fn == dict(
                type='mambapose_seed_worker', base_seed=0)
    assert no_pif.model.head.tokenpose_cfg.pif_mode == 'disabled'


def test_source_loader_is_copied_into_fixed_nonpersistent_worker_policy():
    from mambapose_opt.determinism import deterministic_dataloader_config

    source = Config.fromfile(
        'configs/reproduction/coco_s_v1.py').train_dataloader
    normalized = deterministic_dataloader_config(
        source, seed=0, worker_count=2)

    assert source.persistent_workers is True
    assert normalized.persistent_workers is False
    assert normalized.num_workers == 2
    assert normalized.worker_init_fn == dict(
        type='mambapose_seed_worker', base_seed=0)


def test_trace_dataloader_help_is_directly_executable_without_user_site():
    completed = subprocess.run(
        [sys.executable, 'tools/optimization/trace_dataloader.py', '--help'],
        env={'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'},
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'candidate_id' in completed.stdout
