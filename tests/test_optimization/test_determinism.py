import json
import hashlib
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from mmengine.config import Config


def test_calibration_root_seed_replays_all_rngs_and_backend_policy(
        monkeypatch):
    from mambapose_opt.determinism import seed_deterministic_root

    cuda_seeds = []
    monkeypatch.setattr(
        torch.cuda, 'manual_seed_all', lambda seed: cuda_seeds.append(seed))
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        random.random()
        np.random.random()
        torch.rand(3)
        cuda_seeds.clear()
        first_contract = seed_deterministic_root(7)
        first_cuda_seeds = tuple(cuda_seeds)
        first = (
            random.random(), np.random.random(), torch.rand(3),
            torch.nn.Linear(3, 2).weight.detach().clone())

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        random.random()
        np.random.random()
        torch.rand(3)
        cuda_seeds.clear()
        second_contract = seed_deterministic_root(7)
        second_cuda_seeds = tuple(cuda_seeds)
        second = (
            random.random(), np.random.random(), torch.rand(3),
            torch.nn.Linear(3, 2).weight.detach().clone())

        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
        assert first_contract == second_contract == {
            'seed': 7,
            'python_seed': 7,
            'numpy_seed': 7,
            'torch_seed': 7,
            'torch_cuda_seed': 7,
            'torch_deterministic_algorithms': True,
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
        }
        assert first_cuda_seeds == second_cuda_seeds == (7, 7)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.use_deterministic_algorithms(original_algorithms)
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic


@pytest.mark.parametrize('seed', [True, -1, 1.5, 2**32])
def test_calibration_root_seed_rejects_values_not_shared_by_all_rngs(seed):
    from mambapose_opt.determinism import (
        DeterminismError, seed_deterministic_root)

    with pytest.raises(DeterminismError, match='32-bit integer'):
        seed_deterministic_root(seed)


def test_seed_worker_uses_full_torch_initial_seed_and_32bit_library_seeds(
        monkeypatch):
    from mambapose_opt.determinism import seed_worker

    calls = []
    full_seed = 2**32 + 41
    monkeypatch.setattr(torch, 'initial_seed', lambda: full_seed)
    monkeypatch.setattr(random, 'seed', lambda value: calls.append(('python', value)))
    monkeypatch.setattr(np.random, 'seed', lambda value: calls.append(('numpy', value)))
    monkeypatch.setattr(torch, 'manual_seed', lambda value: calls.append(('torch', value)))

    seed_worker(3)

    assert calls == [('python', 41), ('numpy', 41), ('torch', full_seed)]


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
        {'worker_id': 0, 'torch_seed_source': 'torch.initial_seed()',
         'python_seed_derivation': 'torch_seed % 2**32',
         'numpy_seed_derivation': 'torch_seed % 2**32'},
        {'worker_id': 1, 'torch_seed_source': 'torch.initial_seed()',
         'python_seed_derivation': 'torch_seed % 2**32',
         'numpy_seed_derivation': 'torch_seed % 2**32'},
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
            assert loader.worker_init_fn == dict(type='mambapose_seed_worker')
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
    assert normalized.worker_init_fn == dict(type='mambapose_seed_worker')


def test_trace_dataloader_help_is_directly_executable_without_user_site():
    completed = subprocess.run(
        [sys.executable, 'tools/optimization/trace_dataloader.py', '--help'],
        env={'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'},
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'candidate_id' in completed.stdout


@pytest.mark.parametrize('failure', ['checkpoint', 'dirty'])
def test_trace_refuses_bad_checkpoint_or_dirty_source_before_dataset_and_output(
        tmp_path, monkeypatch, failure):
    import tools.optimization.trace_dataloader as tool
    from mambapose_opt.checkpoints import AuthorizedCandidate
    from mambapose_opt.schema import CandidateSpec

    (tmp_path / 'config.py').write_text('train_dataloader = dict()')
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'actual')
    expected = hashlib.sha256(b'actual').hexdigest()
    if failure == 'checkpoint':
        expected = 'a' * 64
    candidate = CandidateSpec.from_dict({
        'id': 'fixture', 'route': 'accuracy-first', 'kind': 'float',
        'config': 'config.py', 'checkpoint': 'model.pth',
        'checkpoint_sha256': expected, 'seed': 0, 'features': {},
    })
    output = tmp_path / 'work_dirs/optimization/trace.json'
    monkeypatch.setattr(tool, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        tool, 'authorize_manifest_candidate',
        lambda *args: (
            (_ for _ in ()).throw(RuntimeError('dirty source'))
            if failure == 'dirty' else AuthorizedCandidate(
                candidate, tmp_path / candidate.config, checkpoint,
                {'git_commit': 'd' * 40})))
    monkeypatch.setattr(
        tool.Config, 'fromfile',
        lambda *args: (_ for _ in ()).throw(
            AssertionError('dataset/config loading happened too early')))

    with pytest.raises((ValueError, RuntimeError)):
        tool.trace_candidate(candidate, output, epochs=1)
    assert not output.exists()
