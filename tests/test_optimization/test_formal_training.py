from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import stat

import numpy as np
import pytest
import torch

from mambapose_opt.formal_schema import (
    AssetBinding,
    FormalRunInit,
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
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)},
        optimizer_state={'momentum': torch.ones(1)},
        scheduler_state={}, scaler_state={},
        order_hashes=trace_epoch_orders(init, 1, dataset_size=11),
        dataset_size=11, validate_after_write=False)
    hostile = torch.load(path, map_location='cpu', weights_only=True)
    hostile['optimizer']['momentum'] = torch.tensor(float('inf'))
    torch.save(hostile, path)
    with pytest.raises(FormalTrainingError, match='finite'):
        validate_resume_checkpoint(
            path, init, repository_root=tmp_path, dataset_size=11)

    path.unlink()
    path = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=('8' * 64,),
        dataset_size=11, validate_after_write=False)
    with pytest.raises(FormalTrainingError, match='order'):
        validate_resume_checkpoint(
            path, init, repository_root=tmp_path, dataset_size=11)


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
        validate_resume_checkpoint(
            path, init, repository_root=tmp_path, dataset_size=11)
    assert not marker.exists()


def _stop_request(tmp_path: Path, stage='training'):
    input_path = tmp_path / 'input.json'
    input_path.write_text('{}\n')
    staging = tmp_path / 'partial'
    staging.mkdir()
    (staging / 'partial.bin').write_bytes(b'partial')
    control = tmp_path / 'stop-request.json'
    document = {
        'schema_version': 1,
        'request_id': 'request-001',
        'stage': stage,
        'input_path': str(input_path),
        'input_sha256': _sha(input_path),
        'issued_at_ns': 1,
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
        control,
        safe_boundary=StageSafeBoundary(stage, kind, 3, False)) is None
    request = poll_cooperative_stop(
        control,
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
        control,
        safe_boundary=StageSafeBoundary('training', 'optimizer', 0, True))
    assert request is not None
    input_path.write_text('{"changed":true}\n')
    with pytest.raises(FormalTrainingError, match='input'):
        write_stop_acknowledgement(
            request, TrainingResumeAuthority(
                'full-seed0', 0, str(input_path), _sha(input_path)))


def test_stateless_resume_cannot_cross_stage(tmp_path):
    control, input_path, _staging = _stop_request(tmp_path, 'latency')
    request = poll_cooperative_stop(
        control,
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
    staging = output / 'partial-epoch'
    staging.mkdir()
    (staging / 'uncommitted.bin').write_bytes(b'uncommitted')
    control = output / 'stop-request.json'
    control.write_text(json.dumps({
        'schema_version': 1, 'request_id': 'partial-epoch-stop',
        'stage': 'training', 'input_path': str(init_path),
        'input_sha256': _sha(init_path), 'issued_at_ns': 1,
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
