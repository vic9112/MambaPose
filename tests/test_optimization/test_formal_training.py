from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import stat
import time

import numpy as np
import pytest
import torch

from mambapose_opt.formal_schema import (
    AssetBinding,
    FormalRunInit,
    FormalRunSpec,
    InitializationAuthority,
    FormalStageCManifest,
    config_closure_sha256,
)
from mambapose_opt.formal_environment import EnvironmentAuthority
import mambapose_opt.formal_training as formal_training
from mambapose_opt.formal_determinism import trace_epoch_orders
from mambapose_opt.formal_training import (
    CooperativeStopRequest,
    FormalTrainingError,
    StageSafeBoundary,
    StatelessStageInputAuthority,
    TrainingResumeAuthority,
    poll_cooperative_stop,
    validate_resume_checkpoint,
    write_formal_run_init,
    write_resume_checkpoint,
    write_stop_acknowledgement,
    build_all_formal_run_inits,
    recover_training_lineage,
    run_synthetic_training_smoke,
    load_authenticated_formal_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _config_init(root: Path) -> FormalRunInit:
    config = Path('configs/leaf.py')
    return replace(
        _run_init(output_root='run'),
        config=config,
        config_closure_sha256=config_closure_sha256(root, config))


def test_authenticated_config_parses_captured_closure_during_transient_swap(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'base.py').write_text('value = 7\n')
    leaf = configs / 'leaf.py'
    leaf.write_text("_base_ = ['./base.py']\nanswer = 11\n")
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    from mmengine.config import Config
    original = Config.fromfile
    parsed_paths = []

    def transient_swap(path, *args, **kwargs):
        parsed_paths.append(Path(path))
        original_bytes = leaf.read_bytes()
        leaf.write_text("_base_ = ['./base.py']\nanswer = 999\n")
        try:
            return original(path, *args, **kwargs)
        finally:
            leaf.write_bytes(original_bytes)

    monkeypatch.setattr(Config, 'fromfile', transient_swap)
    config = load_authenticated_formal_config(init, tmp_path)
    assert config.answer == 11 and config.value == 7
    assert parsed_paths and parsed_paths[0] != leaf
    assert not parsed_paths[0].exists()


def test_authenticated_config_never_follows_repository_snapshot_parent_symlink(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text('answer = 11\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    outside = tmp_path / 'outside'
    outside.mkdir()
    parent = tmp_path / 'work_dirs/optimization/formal-stage-c'
    parent.mkdir(parents=True)
    (parent / '.formal-config-snapshots').symlink_to(outside)
    from mmengine.config import Config
    original = Config.fromfile
    parsed = []

    def observe(path, *args, **kwargs):
        parsed.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Config, 'fromfile', observe)
    assert load_authenticated_formal_config(init, tmp_path).answer == 11
    assert parsed and outside not in parsed[0].resolve().parents
    assert not tuple(outside.iterdir())


def test_authenticated_config_private_tree_is_readonly_and_swap_fails_closed(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text('answer = 11\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    from mmengine.config import Config

    def attack(path, *args, **kwargs):
        source = Path(path)
        assert stat.S_IMODE(source.stat().st_mode) == 0o400
        assert stat.S_IMODE(source.parent.stat().st_mode) == 0o500
        source.chmod(0o600)
        source.write_text('answer = 999\n')
        return Config({'answer': 999})

    monkeypatch.setattr(Config, 'fromfile', attack)
    with pytest.raises(FormalTrainingError, match='private config snapshot'):
        load_authenticated_formal_config(init, tmp_path)


def test_authenticated_config_is_detached_and_fingerprinted_after_cleanup(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text('answer = 11\nnested = dict(value=7)\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    config = load_authenticated_formal_config(init, tmp_path)
    first = formal_training._deterministic_config_fingerprint(config.to_dict())
    assert config.filename is None
    assert config.answer == 11 and config.nested.value == 7
    assert formal_training._deterministic_config_fingerprint(
        config.to_dict()) == first


def test_authenticated_config_materializes_real_formal_closure(monkeypatch):
    relative = Path('configs/optimization/formal_stage_c/full_seed0.py')
    init = replace(
        _run_init(), config=relative,
        config_closure_sha256=config_closure_sha256(ROOT, relative))
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    config = load_authenticated_formal_config(init, ROOT)
    assert config.filename is None
    assert config.formal_role == 'baseline' and config.formal_seed == 0
    assert config.train_cfg.max_epochs == 300
    assert config.train_dataloader.num_workers == 2


def test_authenticated_config_rejects_live_closure_drift_before_capture(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    leaf = configs / 'leaf.py'
    leaf.write_text('answer = 11\n')
    init = _config_init(tmp_path)
    leaf.write_text('answer = 999\n')
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    with pytest.raises(FormalTrainingError, match='config closure'):
        load_authenticated_formal_config(init, tmp_path)


def test_trace_config_uses_captured_manifest_closure_during_transient_swap(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'base.py').write_text('value = 7\n')
    leaf = configs / 'leaf.py'
    leaf.write_text("_base_ = ['./base.py']\nanswer = 11\n")
    closure = config_closure_sha256(tmp_path, Path('configs/leaf.py'))
    spec = FormalRunSpec(
        run_id='full-seed0', role='baseline', seed=0, conditional=False,
        config=Path('configs/leaf.py'), config_sha256=closure,
        initialization_id='vmamba-t-imagenet-262',
        output_root=Path('work_dirs/optimization/formal-stage-c/full-seed0'))
    manifest = FormalStageCManifest(
        schema_version=2, experiment_id='mambapose-formal-stage-c',
        initialization=None, protocol=None, data_authority={},
        prior_artifact=None, runs=(spec,), repository_root=tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_trace_source', lambda _root: '2' * 40,
        raising=False)
    from mmengine.config import Config
    original = Config.fromfile

    def transient_swap(path, *args, **kwargs):
        original_bytes = leaf.read_bytes()
        leaf.write_text("_base_ = ['./base.py']\nanswer = 999\n")
        try:
            return original(path, *args, **kwargs)
        finally:
            leaf.write_bytes(original_bytes)

    monkeypatch.setattr(Config, 'fromfile', transient_swap)
    config = formal_training.load_authenticated_trace_config(
        manifest, 'full-seed0', tmp_path)
    assert config.filename is None
    assert config.answer == 11 and config.value == 7


def test_production_model_callers_never_parse_live_config_path_directly():
    import inspect

    for function in (
            formal_training.run_formal_model_preflight,
            formal_training.train_formal_candidate):
        source = inspect.getsource(function)
        assert 'Config.fromfile' not in source
        assert 'load_authenticated_formal_config' in source


def test_training_never_reinitializes_after_safe_load_or_uses_generic_resume():
    import inspect

    source = inspect.getsource(formal_training.train_formal_candidate)
    assert 'runner.train()' not in source
    assert 'load_or_resume' not in source
    assert '_run_authenticated_training_loop' in source


def test_production_training_recovers_before_resume_choice_and_has_no_fixed_best():
    import inspect

    source = inspect.getsource(formal_training.train_formal_candidate)
    assert source.index('recover_training_lineage(') < source.index(
        'if resume_path is None')
    assert 'best_coco_AP.pth' not in source
    hook_source = inspect.getsource(formal_training._build_training_hook)
    assert 'best_coco_AP.pth' not in hook_source
    assert '_write_training_log' not in hook_source
    first_authority = source.index('_revalidate_resume_state_authority(')
    injection = source.index('model.load_state_dict(')
    restore = source.index('_restore_rng_state(')
    second_authority = source.index(
        '_revalidate_resume_state_authority(', first_authority + 1)
    assert first_authority < injection < restore < second_authority


def test_controlled_runner_lifecycle_preserves_authenticated_state_exactly_once():
    events = []

    class Model:
        def train_step(self):
            pass

        def val_step(self):
            pass

    class OptimWrapper:
        state = 'built'

        def initialize_count_status(self, model, iteration, max_iterations):
            assert model.state == 'resume-restored'
            assert self.state == 'resume-restored'
            assert (iteration, max_iterations) == (17, 99)
            events.append('initialize_count')

    class Scheduler:
        state = 'built'

    class TrainLoop:
        iter = 17
        max_iters = 99

        def __init__(self, runner):
            self.runner = runner

        def run(self):
            assert self.runner.model.state == 'resume-restored'
            assert self.runner.optim_wrapper.state == 'resume-restored'
            assert self.runner.param_schedulers[0].state == 'resume-restored'
            events.append('loop.run')
            return self.runner.model

    class Runner:
        def __init__(self):
            self.model = Model()
            self.model.state = 'uninitialized'
            self._train_loop = object()
            self._val_loop = object()
            self.optim_wrapper = object()
            self.param_schedulers = object()
            self.auto_scale_lr = object()
            self.init_calls = 0

        @property
        def train_loop(self):
            return self._train_loop

        def build_train_loop(self, value):
            events.append('build_train')
            return TrainLoop(self)

        def build_optim_wrapper(self, value):
            events.append('build_optim')
            return OptimWrapper()

        def scale_lr(self, optim_wrapper, auto_scale_lr):
            events.append('scale_lr')

        def build_param_scheduler(self, value):
            events.append('build_scheduler')
            return [Scheduler()]

        def build_val_loop(self, value):
            events.append('build_val')
            return object()

        def call_hook(self, name):
            events.append(name)

        def _init_model_weights(self):
            self.init_calls += 1
            self.model.state = 'initialized'
            events.append('init_weights')

        def _maybe_compile(self, name):
            assert name == 'train_step'
            assert self.model.state == 'resume-restored'
            events.append('maybe_compile')

        def load_or_resume(self):
            raise AssertionError('generic load/resume must never be called')

    runner = Runner()
    model = formal_training._prepare_runner_for_authenticated_load(runner)
    assert events == [
        'build_train', 'build_optim', 'scale_lr', 'build_scheduler',
        'build_val', 'before_run', 'init_weights']
    assert runner.init_calls == 1 and model.state == 'initialized'

    events.append('safe_state_injection')
    model.state = 'resume-restored'
    runner.optim_wrapper.state = 'resume-restored'
    runner.param_schedulers[0].state = 'resume-restored'
    formal_training._mark_authenticated_state_loaded(runner)
    assert formal_training._run_authenticated_training_loop(runner) is model
    assert events == [
        'build_train', 'build_optim', 'scale_lr', 'build_scheduler',
        'build_val', 'before_run', 'init_weights', 'safe_state_injection',
        'initialize_count', 'maybe_compile', 'loop.run', 'after_run']
    assert runner.init_calls == 1
    assert model.state == 'resume-restored'
    assert runner.optim_wrapper.state == 'resume-restored'
    assert runner.param_schedulers[0].state == 'resume-restored'
    with pytest.raises(FormalTrainingError, match='authenticated state'):
        formal_training._run_authenticated_training_loop(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_init(output_root='work_dirs/optimization/formal-stage-c/full-seed0'):
    return FormalRunInit(
        manifest_sha256='1' * 64,
        git_commit='2' * 40,
        config=Path('configs/optimization/formal_stage_c/full_seed0.py'),
        config_closure_sha256='3' * 64,
        resolved_config_sha256='4' * 64,
        environment_inventory_sha256='5' * 64,
        data_authority={f'role_{index}_sha256': f'{index:x}' * 64
                        for index in range(9)},
        run_id='full-seed0', role='baseline', seed=0, epochs=300,
        effective_batch_size=128, worker_count=2,
        persistent_workers=False, output_root=Path(output_root),
        initialization=InitializationAuthority(
            id='vmamba-t-imagenet-262', kind='vmamba-backbone',
            asset=AssetBinding(
                authority_root='pretrained',
                target_root='canonical-main/pretrained',
                asset_relative_path=Path(
                    'vssm_tiny_0230_ckpt_epoch_262.pth'),
                sha256='6' * 64)))


def test_run_init_write_is_atomic_idempotent_and_mismatch_closed(tmp_path):
    init = _run_init()
    destination = tmp_path / init.output_root / 'run-init.json'
    first = write_formal_run_init(init, destination)
    second = write_formal_run_init(init, destination)
    assert first == second
    assert first.sha256 == _sha(destination)
    assert not tuple(tmp_path.glob('*.tmp'))
    with pytest.raises(FormalTrainingError, match='immutable'):
        write_formal_run_init(replace(init, seed=1), destination)


def test_all_ten_run_inits_are_built_before_publication(monkeypatch):
    document = json.loads(
        (ROOT / 'optimization/formal_stage_c.json').read_text(encoding='utf-8'))
    manifest = FormalStageCManifest.from_dict(document, repository_root=ROOT)
    environment = EnvironmentAuthority.capture(ROOT)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: '2' * 40)
    monkeypatch.setattr(
        formal_training, 'validate_environment_authority',
        lambda _authority, _root: None)
    monkeypatch.setattr(
        formal_training, 'load_formal_manifest',
        lambda _path, repository_root: manifest)
    records = build_all_formal_run_inits(manifest, environment, ROOT)
    assert len(records) == 10
    assert tuple((item.role, item.seed) for item in records) == tuple(
        (role, seed) for seed in range(5)
        for role in ('baseline', 'no_pif'))
    assert all(item.epochs == 300 and item.worker_count == 2
               and item.effective_batch_size == 128
               and item.persistent_workers is False for item in records)


def test_resume_checkpoint_roundtrip_binds_complete_training_state(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(2)},
        optimizer_state={'step': torch.tensor(1.), 'momentum': torch.ones(2)},
        scheduler_state={'last_epoch': 1}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11)
    state = validate_resume_checkpoint(
        path, init, repository_root=tmp_path, dataset_size=11)
    assert state.completed_epoch == 1
    assert state.run_init_sha256 == _sha(init_path)
    assert set(state.rng_state) == {'python', 'numpy', 'torch', 'cuda'}


def test_resume_epoch_commit_binds_checkpoint_and_complete_structured_log(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11)
    commit = output / 'epoch-commits/epoch_1.json'
    log = output / 'training.jsonl'
    assert commit.is_file() and log.is_file()
    document = json.loads(commit.read_text(encoding='utf-8'))
    assert document['checkpoint'] == {
        'path': 'epoch_1.pth', 'sha256': _sha(path)}
    assert document['structured_log'] == {
        'path': 'training.jsonl', 'sha256': _sha(log)}
    state = validate_resume_checkpoint(
        path, init, repository_root=tmp_path, dataset_size=11)
    assert state.commit_sha256 == _sha(commit)
    assert state.structured_log_sha256 == _sha(log)


@pytest.mark.parametrize('failure_target', [
    'training.jsonl', 'epoch-commits/epoch_2.json'])
def test_epoch_transaction_failure_recovers_previous_committed_boundary(
        tmp_path, monkeypatch, failure_target):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders[:1],
        dataset_size=11)
    original = formal_training._atomic_write_bytes
    failed = False

    def fail_once(destination, payload):
        nonlocal failed
        if not failed and Path(destination).relative_to(output).as_posix() \
                == failure_target:
            failed = True
            raise OSError('injected durable boundary failure')
        return original(destination, payload)

    monkeypatch.setattr(formal_training, '_atomic_write_bytes', fail_once)
    with pytest.raises(OSError, match='injected'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=2,
            model_state={'weight': torch.full((1,), 2.0)},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders, dataset_size=11)
    assert (output / 'epoch_1.pth').is_file()
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 1
    assert not (output / 'epoch_2.pth').exists()
    assert len((output / 'training.jsonl').read_text().splitlines()) == 1


def test_durable_epoch_commit_precedes_prune_and_survives_prune_failure(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 3, dataset_size=11)
    for epoch in (1, 2):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=epoch,
            model_state={'weight': torch.full((1,), float(epoch))},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders[:epoch], dataset_size=11)
    original = formal_training._durable_unlink
    failed = False

    def fail_once(path):
        nonlocal failed
        if not failed and Path(path).name == 'epoch_1.pth':
            failed = True
            raise OSError('injected prune failure')
        return original(path)

    monkeypatch.setattr(formal_training, '_durable_unlink', fail_once)
    with pytest.raises(OSError, match='prune'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=3,
            model_state={'weight': torch.full((1,), 3.0)},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders, dataset_size=11)
    assert (output / 'epoch_2.pth').is_file()
    assert (output / 'epoch-commits/epoch_3.json').is_file()
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 3
    assert tuple(sorted(path.name for path in output.glob('epoch_*.pth'))) \
        == ('epoch_2.pth', 'epoch_3.pth')


@pytest.mark.parametrize('boundary', ['replace', 'parent_fsync'])
def test_checkpoint_publication_failure_retains_previous_committed_boundary(
        tmp_path, monkeypatch, boundary):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders[:1],
        dataset_size=11)
    failed = False
    if boundary == 'replace':
        original_replace = formal_training.os.replace

        def fail_once(source, destination):
            nonlocal failed
            if not failed and Path(destination) == output / 'epoch_2.pth':
                failed = True
                raise OSError('injected checkpoint replace failure')
            return original_replace(source, destination)

        monkeypatch.setattr(formal_training.os, 'replace', fail_once)
    else:
        original_fsync = formal_training._fsync_directory

        def fail_once(directory):
            nonlocal failed
            if not failed and Path(directory) == output \
                    and (output / 'epoch_2.pth').exists():
                failed = True
                raise OSError('injected checkpoint parent fsync failure')
            return original_fsync(directory)

        monkeypatch.setattr(formal_training, '_fsync_directory', fail_once)
    with pytest.raises(OSError, match='checkpoint'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=2,
            model_state={'weight': torch.full((1,), 2.0)},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders, dataset_size=11)
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 1
    assert tuple(path.name for path in output.glob('epoch_*.pth')) \
        == ('epoch_1.pth',)


def test_epoch_recovery_rejects_non_next_or_multiple_uncommitted_artifacts(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11)
    (output / 'epoch_3.pth').write_bytes(b'unknown')
    with pytest.raises(FormalTrainingError, match='uncommitted'):
        recover_training_lineage(tmp_path, init, dataset_size=11)


def _publish_test_best(tmp_path, init, epoch, metric):
    return formal_training.publish_best_checkpoint(
        tmp_path, init, completed_epoch=epoch, metric=metric,
        tensors={'weight': torch.full((1,), float(epoch))})


def test_epoch_commit_binds_versioned_best_and_inherited_best_authority(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    first_best = _publish_test_best(tmp_path, init, 1, 0.5)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders[:1],
        dataset_size=11, best_decision_path=first_best)
    inherited = formal_training.inherit_best_checkpoint(
        tmp_path, init, completed_epoch=2)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=2,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders,
        dataset_size=11, best_decision_path=inherited)
    first = json.loads(
        (output / 'epoch-commits/epoch_1.json').read_text())['best']
    second = json.loads(
        (output / 'epoch-commits/epoch_2.json').read_text())['best']
    assert first['checkpoint'] == second['checkpoint']
    assert first['checkpoint']['path'] == 'best_coco_AP_epoch_1.pth'
    assert second['decision']['path'] == 'best-lineages/epoch_1.json'
    assert formal_training._load_best_lineage(
        tmp_path, init, completed_epoch=2)[1].name \
        == 'best_coco_AP_epoch_1.pth'


def test_best_pointer_failure_retains_previous_committed_best(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 1, dataset_size=11)
    first_decision = _publish_test_best(tmp_path, init, 1, 0.5)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders,
        dataset_size=11, best_decision_path=first_decision)
    first_best = output / 'best_coco_AP_epoch_1.pth'
    original = formal_training._write_atomic_file_at
    failed = False

    def fail_pointer(directory_fd, destination_name, payload, *,
                     temporary_name):
        nonlocal failed
        if not failed and destination_name == 'best-lineage.json':
            failed = True
            raise OSError('injected best pointer failure')
        return original(
            directory_fd, destination_name, payload,
            temporary_name=temporary_name)

    monkeypatch.setattr(
        formal_training, '_write_atomic_file_at', fail_pointer)
    with pytest.raises(OSError, match='pointer'):
        _publish_test_best(tmp_path, init, 2, 0.6)
    assert first_best.is_file()
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 1
    metric, checkpoint = formal_training._load_best_lineage(
        tmp_path, init, completed_epoch=1)
    assert metric == 0.5 and checkpoint == first_best
    assert not (output / 'best_coco_AP_epoch_2.pth').exists()


def test_committed_best_survives_old_best_prune_failure(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    first_decision = _publish_test_best(tmp_path, init, 1, 0.5)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders[:1],
        dataset_size=11, best_decision_path=first_decision)
    second_decision = _publish_test_best(tmp_path, init, 2, 0.6)
    first_best = output / 'best_coco_AP_epoch_1.pth'
    original = formal_training._durable_unlink
    failed = False

    def fail_old_best(path):
        nonlocal failed
        if not failed and Path(path) == first_best:
            failed = True
            raise OSError('injected old best prune failure')
        return original(path)

    monkeypatch.setattr(formal_training, '_durable_unlink', fail_old_best)
    with pytest.raises(OSError, match='best prune'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=2,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=orders,
            dataset_size=11, best_decision_path=second_decision)
    assert first_best.is_file()
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 2
    metric, checkpoint = formal_training._load_best_lineage(
        tmp_path, init, completed_epoch=2)
    assert metric == 0.6
    assert checkpoint.name == 'best_coco_AP_epoch_2.pth'
    assert not first_best.exists()


@pytest.mark.parametrize('authority', ['previous_commit', 'best_decision',
                                        'structured_log'])
def test_recovery_rejects_committed_chain_authority_tamper(
        tmp_path, authority):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    first_best = _publish_test_best(tmp_path, init, 1, 0.5)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders[:1],
        dataset_size=11, best_decision_path=first_best)
    second_best = formal_training.inherit_best_checkpoint(
        tmp_path, init, completed_epoch=2)
    write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=2,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders,
        dataset_size=11, best_decision_path=second_best)
    if authority == 'previous_commit':
        path = output / 'epoch-commits/epoch_1.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif authority == 'best_decision':
        path = output / 'best-lineages/epoch_1.json'
        path.write_bytes(path.read_bytes() + b' ')
    else:
        path = output / 'training.jsonl'
        payload = bytearray(path.read_bytes())
        payload[0] = ord('[')
        path.write_bytes(payload)
    with pytest.raises(FormalTrainingError, match=(
            'chain|authority|log|commit|best')):
        recover_training_lineage(tmp_path, init, dataset_size=11)


def test_training_hook_commits_best_before_epoch_and_inherits_non_val_epoch(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)

    class OptimWrapper:
        loss_scaler = None

        @staticmethod
        def state_dict():
            return {}

    class TrainLoop:
        val_begin = 1
        val_interval = 5
        max_epochs = 300

    class Runner:
        model = torch.nn.Linear(2, 1)
        optim_wrapper = OptimWrapper()
        param_schedulers = []
        train_loop = TrainLoop()
        val_loop = object()
        train_dataloader = type(
            'Loader', (), {'dataset': tuple(range(11))})()
        epoch = 0

    runner = Runner()
    hook = formal_training._build_training_hook(
        root=tmp_path, init=init, run_init_path=init_path, resume=None)
    orders = trace_epoch_orders(init, 6, dataset_size=11)
    for zero_based_epoch in range(4):
        runner.epoch = zero_based_epoch
        hook.pending_order = orders[zero_based_epoch]
        hook.after_train_epoch(runner)
    assert not (output / 'best-lineage.json').exists()
    runner.epoch = 4
    hook.pending_order = orders[4]
    hook.after_train_epoch(runner)
    assert not (output / 'epoch-commits/epoch_5.json').exists()
    runner.epoch = 5
    hook.after_val_epoch(runner, metrics={'coco/AP': 0.5})
    fifth = json.loads(
        (output / 'epoch-commits/epoch_5.json').read_text())
    assert fifth['best']['decision']['path'] \
        == 'best-lineages/epoch_5.json'
    runner.epoch = 5
    hook.pending_order = orders[5]
    hook.after_train_epoch(runner)
    sixth = json.loads(
        (output / 'epoch-commits/epoch_6.json').read_text())
    assert sixth['best']['decision']['path'] \
        == 'best-lineages/epoch_5.json'
    assert sixth['best']['checkpoint'] == fifth['best']['checkpoint']


@pytest.mark.parametrize('improves,expected_decisions,expected_tenth', [
    (True, ('epoch_5.json', 'epoch_10.json'), 'epoch_10.json'),
    (False, ('epoch_5.json',), 'epoch_5.json'),
])
def test_training_hook_uses_sparse_five_epoch_best_cadence(
        tmp_path, improves, expected_decisions, expected_tenth):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)

    class OptimWrapper:
        loss_scaler = None

        @staticmethod
        def state_dict():
            return {}

    class TrainLoop:
        val_begin = 1
        val_interval = 5
        max_epochs = 300

    class Runner:
        model = torch.nn.Linear(2, 1)
        optim_wrapper = OptimWrapper()
        param_schedulers = []
        train_loop = TrainLoop()
        val_loop = object()
        train_dataloader = type(
            'Loader', (), {'dataset': tuple(range(11))})()
        epoch = 0

    runner = Runner()
    hook = formal_training._build_training_hook(
        root=tmp_path, init=init, run_init_path=init_path, resume=None)
    orders = trace_epoch_orders(init, 10, dataset_size=11)
    for completed in range(1, 11):
        runner.epoch = completed - 1
        hook.pending_order = orders[completed - 1]
        hook.after_train_epoch(runner)
        if completed in (5, 10):
            runner.epoch = completed
            metric = 0.5 if completed == 5 else (0.6 if improves else 0.4)
            hook.after_val_epoch(runner, metrics={'coco/AP': metric})
    observed = tuple(path.name for path in sorted(
        (output / 'best-lineages').iterdir(),
        key=lambda path: int(path.stem.removeprefix('epoch_'))))
    assert observed == expected_decisions
    for epoch in range(6, 10):
        commit = json.loads(
            (output / f'epoch-commits/epoch_{epoch}.json').read_text())
        assert commit['best']['decision']['path'] \
            == 'best-lineages/epoch_5.json'
    tenth = json.loads(
        (output / 'epoch-commits/epoch_10.json').read_text())
    assert tenth['best']['decision']['path'] \
        == f'best-lineages/{expected_tenth}'


@pytest.mark.parametrize('failed_epoch,new_best', [(6, False), (10, True)])
def test_sparse_best_transaction_failure_recovers_prior_effective_authority(
        tmp_path, monkeypatch, failed_epoch, new_best):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, failed_epoch, dataset_size=11)
    decision = None
    for epoch in range(1, failed_epoch):
        if epoch == 5:
            decision = _publish_test_best(tmp_path, init, 5, 0.5)
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=epoch,
            model_state={'weight': torch.full((1,), float(epoch))},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders[:epoch], dataset_size=11,
            best_decision_path=decision)
    if new_best:
        attempted_decision = _publish_test_best(
            tmp_path, init, failed_epoch, 0.6)
    else:
        attempted_decision = formal_training.inherit_best_checkpoint(
            tmp_path, init, completed_epoch=failed_epoch)
    original = formal_training._atomic_write_bytes
    failed = False

    def fail_commit(destination, payload):
        nonlocal failed
        if not failed and Path(destination) == (
                output / f'epoch-commits/epoch_{failed_epoch}.json'):
            failed = True
            raise OSError('injected sparse best commit failure')
        return original(destination, payload)

    monkeypatch.setattr(formal_training, '_atomic_write_bytes', fail_commit)
    with pytest.raises(OSError, match='sparse best'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=failed_epoch,
            model_state={'weight': torch.full((1,), float(failed_epoch))},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders, dataset_size=11,
            best_decision_path=attempted_decision)
    recovered = recover_training_lineage(
        tmp_path, init, dataset_size=11)
    assert recovered is not None
    assert recovered.completed_epoch == failed_epoch - 1
    pointer = json.loads((output / 'best-lineage.json').read_text())
    assert pointer['decision']['path'] == 'best-lineages/epoch_5.json'
    if failed_epoch == 10:
        assert not (output / 'best-lineages/epoch_10.json').exists()
        assert not (output / 'best_coco_AP_epoch_10.pth').exists()


def test_inherited_best_commit_rejects_alternate_same_byte_decision_path(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 6, dataset_size=11)
    decision = None
    for epoch in range(1, 7):
        if epoch == 5:
            decision = _publish_test_best(tmp_path, init, 5, 0.5)
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=epoch,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={},
            order_hashes=orders[:epoch], dataset_size=11,
            best_decision_path=decision)
    alternate = output / 'best-lineages/alternate.json'
    alternate.write_bytes(decision.read_bytes())
    commit = output / 'epoch-commits/epoch_6.json'
    document = json.loads(commit.read_text())
    document['best']['decision']['path'] = 'best-lineages/alternate.json'
    commit.write_bytes(formal_training._canonical_json_bytes(document))
    with pytest.raises(FormalTrainingError, match='best decision'):
        recover_training_lineage(tmp_path, init, dataset_size=11)


def test_best_loader_rejects_symlinked_decision_parent(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    _publish_test_best(tmp_path, init, 1, 0.5)
    outside = tmp_path / 'outside-decisions'
    (output / 'best-lineages').rename(outside)
    (output / 'best-lineages').symlink_to(outside)
    with pytest.raises(FormalTrainingError, match='parent|unsafe'):
        formal_training._load_best_lineage(
            tmp_path, init, completed_epoch=1)


def test_best_writer_rejects_pointer_replacement_after_prior_load(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    first = _publish_test_best(tmp_path, init, 1, 0.5)
    alternate = output / 'best-lineages/alternate.json'
    alternate.write_bytes(first.read_bytes())
    original = formal_training._load_best_lineage
    replaced = False

    def replace_after_load(root, expected, *, completed_epoch):
        nonlocal replaced
        result = original(root, expected, completed_epoch=completed_epoch)
        if completed_epoch == 1 and not replaced:
            replaced = True
            (output / 'best-lineage.json').write_bytes(
                formal_training._canonical_json_bytes({
                    'schema_version': 1,
                    'decision': {
                        'path': 'best-lineages/alternate.json',
                        'sha256': _sha(alternate),
                    },
                }))
        return result

    monkeypatch.setattr(
        formal_training, '_load_best_lineage', replace_after_load)
    with pytest.raises(FormalTrainingError, match='pointer|lineage|authority'):
        _publish_test_best(tmp_path, init, 2, 0.6)
    assert not (output / 'best-lineages/epoch_2.json').exists()


def test_cpu_synthetic_two_epoch_smoke_repeats_and_keeps_exactly_two(tmp_path):
    init = _run_init(output_root='run')
    first = run_synthetic_training_smoke(init, tmp_path / 'first')
    second = run_synthetic_training_smoke(init, tmp_path / 'second')
    assert first.order_hashes == second.order_hashes
    assert first.final_model_sha256 == second.final_model_sha256
    for root, result in ((tmp_path / 'first', first),
                         (tmp_path / 'second', second)):
        assert tuple(path.name for path in result.resume_checkpoints) == (
            'epoch_1.pth', 'epoch_2.pth')
        assert tuple(sorted(path.name for path in (root / 'run').glob(
            'epoch_*.pth'))) == ('epoch_1.pth', 'epoch_2.pth')


def test_resume_retention_never_publishes_a_fourth_checkpoint(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 4, dataset_size=11)
    for epoch in range(1, 5):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=epoch, model_state={'weight': torch.ones(1)},
            optimizer_state={}, scheduler_state={}, scaler_state={},
            order_hashes=orders[:epoch], dataset_size=11)
    assert tuple(sorted(path.name for path in output.glob('epoch_*.pth'))) == (
        'epoch_3.pth', 'epoch_4.pth')


@pytest.mark.parametrize('field', [
    'manifest_sha256', 'config_closure_sha256', 'resolved_config_sha256',
    'environment_inventory_sha256', 'run_id', 'role', 'seed',
])
def test_resume_rejects_every_identity_mismatch(tmp_path, field):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11)
    replacement = 'no_pif' if field == 'role' else (
        1 if field == 'seed' else '8' * (40 if field == 'git_commit' else 64))
    if field == 'run_id':
        replacement = 'no-pif-seed0'
    with pytest.raises(FormalTrainingError, match=field.replace('_', ' ')):
        validate_resume_checkpoint(
            path, replace(init, **{field: replacement}),
            repository_root=tmp_path, dataset_size=11)


def test_resume_rejects_nonfinite_optimizer_and_wrong_order_prefix(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    with pytest.raises(FormalTrainingError, match='finite'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=1,
            model_state={'weight': torch.ones(1)},
            optimizer_state={
                'momentum': torch.tensor(float('inf'))},
            scheduler_state={}, scaler_state={},
            order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
            dataset_size=11)
    with pytest.raises(FormalTrainingError, match='order'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=1,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=('8' * 64,),
            dataset_size=11)
    assert not tuple(output.glob('epoch_*.pth'))


def test_resume_weights_only_loader_rejects_hostile_pickle(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    marker = tmp_path / 'executed'

    class Hostile:
        def __reduce__(self):
            return (Path.write_text, (marker, 'unsafe'))

    path = output / 'epoch_1.pth'
    torch.save({'payload': Hostile()}, path)
    with pytest.raises(FormalTrainingError, match='weights-only'):
        formal_training._load_resume_document(
            path, expected_sha256=_sha(path))
    assert not marker.exists()


def test_resume_loader_uses_captured_checkpoint_bytes_during_transient_swap(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11)
    alternate = tmp_path / 'alternate.pth'
    document = torch.load(path, map_location='cpu', weights_only=True)
    document['model']['weight'] = torch.tensor([999.0])
    torch.save(document, alternate)
    approved = path.read_bytes()
    hostile = alternate.read_bytes()
    original_load = torch.load

    def transient_swap(source, *args, **kwargs):
        if Path(source).absolute() == path.absolute():
            path.write_bytes(hostile)
            try:
                return original_load(source, *args, **kwargs)
            finally:
                path.write_bytes(approved)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', transient_swap)
    resume = validate_resume_checkpoint(
        path, init, repository_root=tmp_path, dataset_size=11)
    assert torch.equal(resume.model_state['weight'], torch.ones(1))


def _stop_request(tmp_path: Path, stage='training'):
    input_path = tmp_path / 'input.json'
    input_path.write_text('{}\n')
    staging = tmp_path / 'partial-staging'
    staging.mkdir()
    (staging / 'partial.bin').write_bytes(b'partial')
    control = tmp_path / 'stop-request.json'
    document = {
        'schema_version': 1,
        'request_id': 'request-001',
        'stage': stage,
        'input_path': str(input_path),
        'input_sha256': _sha(input_path),
        'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(tmp_path / 'stop-ack.json'),
        'partial_staging_path': str(staging),
    }
    control.write_text(json.dumps(document, sort_keys=True) + '\n')
    return control, input_path, staging


@pytest.mark.parametrize('stage,kind', [
    ('training', 'optimizer'), ('profile', 'cuda'), ('evaluate', 'cuda'),
    ('latency', 'cuda'), ('export', 'cuda'), ('stage_a', 'cuda'),
])
def test_generic_stop_acknowledges_only_safe_boundary_and_discards_partial(
        tmp_path, stage, kind):
    control, input_path, staging = _stop_request(tmp_path, stage)
    assert poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary(stage, kind, 3, False)) is None
    request = poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary(stage, kind, 3, True))
    assert isinstance(request, CooperativeStopRequest)
    if stage == 'training':
        resume = TrainingResumeAuthority(
            run_id='full-seed0', completed_epoch=0,
            path=str(input_path), sha256=_sha(input_path))
    else:
        resume = StatelessStageInputAuthority(
            stage=stage, path=str(input_path), sha256=_sha(input_path))
    acknowledgement = write_stop_acknowledgement(request, resume)
    assert acknowledgement.exit_code == 75
    assert not staging.exists()
    assert Path(request.acknowledgement_path).is_file()


def test_stop_ack_rejects_changed_input_and_unsafe_or_stale_request(tmp_path):
    control, input_path, _staging = _stop_request(tmp_path)
    request = poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary('training', 'optimizer', 0, True))
    assert request is not None
    input_path.write_text('{"changed":true}\n')
    with pytest.raises(FormalTrainingError, match='input'):
        write_stop_acknowledgement(
            request, TrainingResumeAuthority(
                'full-seed0', 0, str(input_path), _sha(input_path)))


def test_stop_without_stage_root_authority_cannot_delete_parent_symlink_target(
        tmp_path):
    stage_root = tmp_path / 'stage'
    stage_root.mkdir()
    input_path = stage_root / 'input.json'
    input_path.write_text('{}\n')
    outside = tmp_path / 'outside'
    victim = outside / 'victim'
    victim.mkdir(parents=True)
    (victim / 'valuable.bin').write_bytes(b'valuable')
    (stage_root / 'alias').symlink_to(outside, target_is_directory=True)
    control = stage_root / 'stop-request.json'
    control.write_text(json.dumps({
        'schema_version': 1, 'request_id': 'parent-symlink',
        'stage': 'profile', 'input_path': str(input_path),
        'input_sha256': _sha(input_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(stage_root / 'stop-ack.json'),
        'partial_staging_path': str(stage_root / 'alias' / 'victim'),
    }, sort_keys=True) + '\n')

    with pytest.raises(FormalTrainingError, match='stage output root'):
        request = poll_cooperative_stop(
            control,
            safe_boundary=StageSafeBoundary('profile', 'cuda', 1, True))
        assert request is not None
        write_stop_acknowledgement(
            request, StatelessStageInputAuthority(
                'profile', str(input_path), _sha(input_path)))
    assert victim.is_dir() and (victim / 'valuable.bin').is_file()


def test_stop_rejects_alternate_fixed_paths_under_authorized_stage_root(
        tmp_path):
    stage_root = tmp_path / 'stage'
    stage_root.mkdir()
    input_path = stage_root / 'input.json'
    input_path.write_text('{}\n')
    staging = stage_root / 'unrelated-real-directory'
    staging.mkdir()
    control = stage_root / 'stop-request.json'
    control.write_text(json.dumps({
        'schema_version': 1, 'request_id': 'alternate-paths',
        'stage': 'profile', 'input_path': str(input_path),
        'input_sha256': _sha(input_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(stage_root / 'alternate-ack.json'),
        'partial_staging_path': str(staging),
    }, sort_keys=True) + '\n')
    with pytest.raises(FormalTrainingError, match='canonical'):
        poll_cooperative_stop(
            control, stage_output_root=stage_root,
            safe_boundary=StageSafeBoundary('profile', 'cuda', 1, True))
    assert staging.is_dir()


def test_stop_ack_rejects_control_path_replacement_after_poll(tmp_path):
    control, input_path, staging = _stop_request(tmp_path, 'profile')
    request = poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary('profile', 'cuda', 1, True))
    assert request is not None
    original = control.read_bytes()
    replacement = tmp_path / 'replacement.json'
    replacement.write_bytes(original)
    control.unlink()
    replacement.rename(control)
    with pytest.raises(FormalTrainingError, match='control'):
        write_stop_acknowledgement(
            request, StatelessStageInputAuthority(
                'profile', str(input_path), _sha(input_path)))
    assert staging.is_dir()


def test_stop_ack_rejects_stage_root_replacement_without_touching_victim(
        tmp_path):
    stage_root = tmp_path / 'stage'
    stage_root.mkdir()
    control, input_path, staging = _stop_request(stage_root, 'profile')
    request = poll_cooperative_stop(
        control, stage_output_root=stage_root,
        safe_boundary=StageSafeBoundary('profile', 'cuda', 1, True))
    assert request is not None
    input_bytes = input_path.read_bytes()
    held_original = tmp_path / 'held-original'
    stage_root.rename(held_original)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / input_path.name).write_bytes(input_bytes)
    victim = outside / 'partial-staging'
    victim.mkdir()
    (victim / 'valuable.bin').write_bytes(b'valuable')
    stage_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(FormalTrainingError, match='stage output root'):
        write_stop_acknowledgement(
            request, StatelessStageInputAuthority(
                'profile', str(input_path), _sha(outside / input_path.name)))
    assert victim.is_dir() and (victim / 'valuable.bin').is_file()
    assert (held_original / staging.name / 'partial.bin').is_file()


def test_stop_poll_rejects_stale_request(tmp_path, monkeypatch):
    stage_root = tmp_path / 'stage'
    stage_root.mkdir()
    control, _input_path, staging = _stop_request(stage_root, 'profile')
    document = json.loads(control.read_text(encoding='utf-8'))
    document['issued_at_ns'] = 1
    control.write_text(json.dumps(document, sort_keys=True) + '\n')
    monkeypatch.setattr(formal_training.time, 'time_ns', lambda: 10**20)
    with pytest.raises(FormalTrainingError, match='stale'):
        poll_cooperative_stop(
            control, stage_output_root=stage_root,
            safe_boundary=StageSafeBoundary('profile', 'cuda', 1, True))
    assert staging.is_dir()


def test_training_stop_resume_must_equal_request_input_authority(tmp_path):
    control, input_path, staging = _stop_request(tmp_path, 'training')
    other = tmp_path / 'other-resume.pth'
    other.write_bytes(b'other')
    request = poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary('training', 'optimizer', 1, True))
    assert request is not None
    with pytest.raises(FormalTrainingError, match='resume input mismatch'):
        write_stop_acknowledgement(
            request, TrainingResumeAuthority(
                'full-seed0', 1, str(other), _sha(other)))
    assert input_path.is_file() and staging.is_dir()


def test_stateless_resume_cannot_cross_stage(tmp_path):
    control, input_path, _staging = _stop_request(tmp_path, 'latency')
    request = poll_cooperative_stop(
        control, stage_output_root=tmp_path,
        safe_boundary=StageSafeBoundary('latency', 'cuda', 1, True))
    assert request is not None
    with pytest.raises(FormalTrainingError, match='stage'):
        write_stop_acknowledgement(
            request, StatelessStageInputAuthority(
                'profile', str(input_path), _sha(input_path)))


def test_training_stop_publishes_no_partial_epoch_and_acks_last_init(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    staging = output / 'partial-staging'
    staging.mkdir()
    (staging / 'uncommitted.bin').write_bytes(b'uncommitted')
    control = output / 'stop-request.json'
    control.write_text(json.dumps({
        'schema_version': 1, 'request_id': 'partial-epoch-stop',
        'stage': 'training', 'input_path': str(init_path),
        'input_sha256': _sha(init_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(output / 'stop-ack.json'),
        'partial_staging_path': str(staging),
    }, sort_keys=True) + '\n')
    hook = formal_training._build_training_hook(
        root=tmp_path, init=init, run_init_path=init_path, resume=None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    with pytest.raises(SystemExit) as stopped:
        hook.after_train_iter(object(), 7)
    assert stopped.value.code == 75
    assert not tuple(output.glob('epoch_*.pth'))
    assert not (output / 'training.jsonl').exists()
    assert not staging.exists()
    acknowledgement = json.loads(
        (output / 'stop-ack.json').read_text(encoding='utf-8'))
    assert acknowledgement['resume']['completed_epoch'] == 0
    assert acknowledgement['resume']['sha256'] == _sha(init_path)
