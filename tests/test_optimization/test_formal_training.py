from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import random
import stat
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mambapose_opt.formal_schema import (
    AssetBinding,
    FileBinding,
    FormalRunInit,
    FormalRunSpec,
    InitializationAuthority,
    FormalStageCManifest,
    INITIALIZATION_SHA256,
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


def test_authenticated_config_parses_captured_bytes_without_named_parser(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'base.py').write_text(
        "value = 7\n"
        "nested = dict(keep=1, replace=2, erased=dict(old=1))\n")
    leaf = configs / 'leaf.py'
    leaf.write_text(
        "_base_ = ['./base.py']\n"
        "answer = value = 11\n"
        "nested = dict(replace=3, erased=dict(_delete_=True, new=4))\n")
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    from mmengine.config import Config
    monkeypatch.setattr(
        Config, 'fromfile',
        lambda *_args, **_kwargs: pytest.fail('named Config.fromfile used'))
    monkeypatch.setattr(
        Config, 'fromstring',
        lambda *_args, **_kwargs: pytest.fail('named Config.fromstring used'))
    config = load_authenticated_formal_config(init, tmp_path)
    assert config.answer == 11 and config.value == 11
    assert config.nested == {
        'keep': 1, 'replace': 3, 'erased': {'new': 4}}
    assert config.filename == 'configs/leaf.py'
    assert "_base_ = ['./base.py']" in config.text
    assert 'nested = dict(keep=1' in config.text


def test_authenticated_config_never_creates_a_named_private_snapshot(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text('answer = 11\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    monkeypatch.setattr(
        formal_training.tempfile, 'mkdtemp',
        lambda *_args, **_kwargs: pytest.fail('named snapshot created'))
    assert load_authenticated_formal_config(init, tmp_path).answer == 11


def test_authenticated_config_rejects_unsupported_import_syntax(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text(
        'import os\nanswer = os.environ.get("HOME")\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    with pytest.raises(FormalTrainingError, match='unsupported.*import'):
        load_authenticated_formal_config(init, tmp_path)


def test_authenticated_config_is_stable_and_fingerprinted_after_parse(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'leaf.py').write_text('answer = 11\nnested = dict(value=7)\n')
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    config = load_authenticated_formal_config(init, tmp_path)
    first = formal_training._deterministic_config_fingerprint(config.to_dict())
    assert config.filename == 'configs/leaf.py'
    assert config.text == 'answer = 11\nnested = dict(value=7)\n'
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
    assert config.filename == relative.as_posix()
    assert '_base_' in config.text
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


def test_trace_config_uses_in_memory_manifest_closure(
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
    monkeypatch.setattr(
        Config, 'fromfile',
        lambda *_args, **_kwargs: pytest.fail('named Config.fromfile used'))
    monkeypatch.setattr(
        Config, 'fromstring',
        lambda *_args, **_kwargs: pytest.fail('named Config.fromstring used'))
    config = formal_training.load_authenticated_trace_config(
        manifest, 'full-seed0', tmp_path)
    assert config.filename == 'configs/leaf.py'
    assert config.answer == 11 and config.value == 7


def test_authenticated_config_rejects_same_byte_parent_replacement(
        tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'base.py').write_text('value = 7\n')
    (configs / 'leaf.py').write_text(
        "_base_ = ['./base.py']\nanswer = 11\n")
    init = _config_init(tmp_path)
    monkeypatch.setattr(
        formal_training, '_validate_frozen_source', lambda _root: init.git_commit)
    original_capture = formal_training._capture_config_closure
    attacked = False

    def replace_after_capture(*args, **kwargs):
        nonlocal attacked
        records = original_capture(*args, **kwargs)
        if attacked:
            return records
        attacked = True
        original = tmp_path / 'configs-original'
        configs.rename(original)
        configs.mkdir()
        for source in original.iterdir():
            (configs / source.name).write_bytes(source.read_bytes())
        return records

    monkeypatch.setattr(
        formal_training, '_capture_config_closure', replace_after_capture)
    with pytest.raises(FormalTrainingError, match='directory authority|changed'):
        load_authenticated_formal_config(init, tmp_path)


def test_formal_config_source_has_one_in_memory_parser_boundary():
    source = inspect.getsource(formal_training)
    assert 'Config.fromfile' not in source
    assert 'Config.fromstring' not in source
    assert 'mambapose-formal-config-' not in source


def test_production_model_callers_never_parse_live_config_path_directly():
    import inspect

    call_paths = (
        (formal_training.run_formal_model_preflight,),
        (formal_training.train_formal_candidate,
         formal_training._prepare_and_train_formal_candidate_held),
    )
    for functions in call_paths:
        source = '\n'.join(inspect.getsource(function)
                           for function in functions)
        assert 'Config.fromfile' not in source
        assert 'load_authenticated_formal_config' in source


def test_training_never_reinitializes_after_safe_load_or_uses_generic_resume():
    import inspect

    source = '\n'.join((
        inspect.getsource(formal_training.train_formal_candidate),
        inspect.getsource(formal_training._prepare_and_train_formal_candidate_held),
        inspect.getsource(formal_training._train_formal_candidate_held),
    ))
    assert 'runner.train()' not in source
    assert 'load_or_resume' not in source
    assert '_run_authenticated_training_loop' in source


def test_production_training_recovers_before_resume_choice_and_has_no_fixed_best():
    import inspect

    source = inspect.getsource(formal_training._train_formal_candidate_held)
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


def test_production_result_publish_runs_full_held_recovery_before_return():
    source = inspect.getsource(formal_training._train_formal_candidate_held)
    publish = source.index(
        "authority.write_immutable(Path('train-result.json'), payload)")
    final_recovery = source.rindex('_recover_training_lineage_held(')
    final_validation = source.rindex('_load_completed_training_result_held(')
    assert publish < final_recovery < final_validation


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


def _inject_authority_failure_once(
        monkeypatch, method_name: str, relative: str, message: str, *,
        after: bool = False):
    original = getattr(formal_training._FormalOutputAuthority, method_name)
    failed = False

    def fail_once(self, target, *args, **kwargs):
        nonlocal failed
        matches = Path(target).as_posix() == relative
        if matches and not failed and not after:
            failed = True
            raise OSError(message)
        result = original(self, target, *args, **kwargs)
        if matches and not failed and after:
            failed = True
            raise OSError(message)
        return result

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, method_name, fail_once)


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


def test_run_init_publish_rejects_output_root_replacement_without_victim_write(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    destination = output / 'run-init.json'
    held_output = tmp_path / 'run-held'
    original = formal_training._canonical_json_bytes
    attacked = False

    def replace_after_output_stat(value):
        nonlocal attacked
        payload = original(value)
        if not attacked and output.is_dir():
            attacked = True
            output.rename(held_output)
            output.mkdir()
        return payload

    monkeypatch.setattr(
        formal_training, '_canonical_json_bytes', replace_after_output_stat)
    with pytest.raises(FormalTrainingError, match='authority|changed'):
        write_formal_run_init(init, destination)
    assert not tuple(output.iterdir())
    assert held_output.is_dir()


def test_run_init_immutable_link_race_never_overwrites_competitor(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    destination = tmp_path / 'run/run-init.json'
    original = formal_training.os.link
    hostile = b'competitor\n'
    attacked = False

    def race_link(source, target, *args, **kwargs):
        nonlocal attacked
        if not attacked and target == 'run-init.json':
            attacked = True
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=kwargs['dst_dir_fd'])
            os.write(descriptor, hostile)
            os.close(descriptor)
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(formal_training.os, 'link', race_link)
    with pytest.raises(FormalTrainingError, match='immutable'):
        write_formal_run_init(init, destination)
    assert destination.read_bytes() == hostile


def test_run_init_publish_never_unlinks_replaced_foreign_pending(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    destination = output / 'run-init.json'
    original = formal_training.os.link
    foreign = b'foreign-pending'

    def replace_pending_before_link(source, target, *args, **kwargs):
        if source == '.run-init.json.pending':
            os.unlink(source, dir_fd=kwargs['src_dir_fd'])
            descriptor = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=kwargs['src_dir_fd'])
            os.write(descriptor, foreign)
            os.close(descriptor)
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(formal_training.os, 'link', replace_pending_before_link)
    with pytest.raises(FormalTrainingError, match='publication|pending'):
        write_formal_run_init(init, destination)
    assert (output / '.run-init.json.pending').read_bytes() == foreign


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


def _load_run_init_builder_module():
    path = ROOT / 'tools/optimization/build_formal_run_init.py'
    spec = importlib.util.spec_from_file_location('formal_run_init_builder', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('check', [False, True])
def test_run_init_builder_publicly_reloads_all_ten_only_after_all_writes(
        tmp_path, monkeypatch, check):
    module = _load_run_init_builder_module()
    environment_path = tmp_path / 'environment.json'
    environment_path.write_text('{}')
    module.ROOT = tmp_path
    module.ENVIRONMENT = environment_path
    module.MANIFEST = tmp_path / 'manifest.json'
    inits = tuple(
        replace(
            _run_init(output_root=f'runs/{role}-seed{seed}'),
            run_id=f'{role}-seed{seed}', role='baseline', seed=seed)
        for seed in range(5) for role in ('full', 'paired'))
    if check:
        for init in inits:
            destination = tmp_path / init.output_root / 'run-init.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{}')
    monkeypatch.setattr(
        module.EnvironmentAuthority, 'from_dict',
        classmethod(lambda _cls, _value: object()))
    monkeypatch.setattr(module, 'load_formal_manifest', lambda *_a, **_kw: object())
    monkeypatch.setattr(
        module, 'build_all_formal_run_inits',
        lambda *_args, **_kwargs: inits)
    writes = []
    loads = []

    def write(init, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{}')
        writes.append(init.run_id)
        observed = destination.parent.stat()
        enriched = replace(
            init, output_root_device=observed.st_dev,
            output_root_inode=observed.st_ino)
        payload = module._canonical_json_bytes(module._init_document(
            enriched, output_identity=(observed.st_dev, observed.st_ino)))
        return SimpleNamespace(
            path=destination.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest())

    def load(path, *, repository_root):
        assert len(writes) == 10
        expected = inits[len(loads)]
        observed = path.parent.stat()
        loaded = replace(
            expected, output_root_device=observed.st_dev,
            output_root_inode=observed.st_ino)
        loads.append(loaded.run_id)
        return loaded

    monkeypatch.setattr(module, 'write_formal_run_init', write)
    monkeypatch.setattr(module, 'load_formal_run_init', load, raising=False)
    monkeypatch.setattr(
        module.sys, 'argv', ['build_formal_run_init.py']
        + (['--check'] if check else []))
    assert module.main() == 0
    assert writes == [init.run_id for init in inits]
    assert loads == writes


def test_run_init_builder_rejects_one_public_reload_mismatch(
        tmp_path, monkeypatch):
    module = _load_run_init_builder_module()
    environment_path = tmp_path / 'environment.json'
    environment_path.write_text('{}')
    module.ROOT = tmp_path
    module.ENVIRONMENT = environment_path
    module.MANIFEST = tmp_path / 'manifest.json'
    inits = tuple(
        replace(
            _run_init(output_root=f'runs/full-seed{seed}'),
            run_id=f'full-seed{seed}', seed=seed)
        for seed in range(10))
    monkeypatch.setattr(
        module.EnvironmentAuthority, 'from_dict',
        classmethod(lambda _cls, _value: object()))
    monkeypatch.setattr(module, 'load_formal_manifest', lambda *_a, **_kw: object())
    monkeypatch.setattr(
        module, 'build_all_formal_run_inits',
        lambda *_args, **_kwargs: inits)

    def write(init, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{}')
        observed = destination.parent.stat()
        enriched = replace(
            init, output_root_device=observed.st_dev,
            output_root_inode=observed.st_ino)
        payload = module._canonical_json_bytes(module._init_document(
            enriched, output_identity=(observed.st_dev, observed.st_ino)))
        return SimpleNamespace(
            path=destination.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest())

    calls = 0

    def load(path, *, repository_root):
        nonlocal calls
        expected = inits[calls]
        calls += 1
        observed = path.parent.stat()
        loaded = replace(
            expected, output_root_device=observed.st_dev,
            output_root_inode=observed.st_ino)
        return replace(loaded, worker_count=3) if calls == 7 else loaded

    monkeypatch.setattr(module, 'write_formal_run_init', write)
    monkeypatch.setattr(module, 'load_formal_run_init', load, raising=False)
    monkeypatch.setattr(module.sys, 'argv', ['build_formal_run_init.py'])
    with pytest.raises((RuntimeError, SystemExit, ValueError), match='reload|mismatch'):
        module.main()


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


def test_formal_output_authority_holds_original_root_across_replacement(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    held_output = tmp_path / 'run-held'
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        output.rename(held_output)
        output.mkdir()
        authority.write_immutable(Path('epoch_1.pth'), b'partial')
        with pytest.raises(FormalTrainingError, match='output.*authority'):
            authority.revalidate()
        assert not tuple(output.iterdir())
        assert (held_output / 'epoch_1.pth').read_bytes() == b'partial'
    assert authority.closed


def test_formal_output_authority_immutable_publish_never_overwrites(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        authority.write_immutable(Path('epoch_1.pth'), b'first')
        with pytest.raises(FormalTrainingError, match='immutable'):
            authority.write_immutable(Path('epoch_1.pth'), b'second')
        assert authority.read_regular(Path('epoch_1.pth')) == b'first'


def test_immutable_publish_never_unlinks_replaced_foreign_pending(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    original = formal_training.os.link
    foreign = b'foreign-pending'

    def replace_pending_before_link(source, target, *args, **kwargs):
        if source == '.epoch_1.pth.pending':
            os.unlink(source, dir_fd=kwargs['src_dir_fd'])
            descriptor = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=kwargs['src_dir_fd'])
            os.write(descriptor, foreign)
            os.close(descriptor)
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(formal_training.os, 'link', replace_pending_before_link)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='pending|publication'):
            authority.write_immutable(Path('epoch_1.pth'), b'ours')
    assert (output / '.epoch_1.pth.pending').read_bytes() == foreign


def test_immutable_publish_preserves_original_pending_after_final_replacement(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    original_link = formal_training.os.link

    def replace_final_after_link(source, target, *args, **kwargs):
        result = original_link(source, target, *args, **kwargs)
        if target == 'epoch_1.pth':
            os.unlink(target, dir_fd=kwargs['dst_dir_fd'])
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600, dir_fd=kwargs['dst_dir_fd'])
            try:
                os.write(descriptor, b'alternate-final')
            finally:
                os.close(descriptor)
        return result

    monkeypatch.setattr(formal_training.os, 'link', replace_final_after_link)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='publication changed'):
            authority.write_immutable(Path('epoch_1.pth'), b'ours')
    assert (output / 'epoch_1.pth').read_bytes() == b'alternate-final'
    assert (output / '.epoch_1.pth.pending').read_bytes() == b'ours'


def test_mutable_publish_detects_pending_replacement_before_replace(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    original = formal_training.os.replace
    foreign = b'foreign-pending'

    def replace_pending(source, target, *args, **kwargs):
        if source == '.training.jsonl.pending':
            os.unlink(source, dir_fd=kwargs['src_dir_fd'])
            descriptor = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=kwargs['src_dir_fd'])
            os.write(descriptor, foreign)
            os.close(descriptor)
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(formal_training.os, 'replace', replace_pending)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='pending|publication'):
            authority.write_mutable(Path('training.jsonl'), b'ours')
    assert (output / 'training.jsonl').read_bytes() == foreign


def test_formal_output_authority_holds_cached_child_directory(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    held_commits = output / 'epoch-commits-held'
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        authority.mkdir(Path('epoch-commits'))
        (output / 'epoch-commits').rename(held_commits)
        (output / 'epoch-commits').mkdir()
        authority.write_immutable(
            Path('epoch-commits/epoch_1.json'), b'{}\n')
        with pytest.raises(FormalTrainingError, match='output.*authority'):
            authority.revalidate()
        assert not tuple((output / 'epoch-commits').iterdir())
        assert (held_commits / 'epoch_1.json').read_bytes() == b'{}\n'


def test_formal_output_authority_rejects_forked_mutation_and_closes_fds(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    authority = formal_training._FormalOutputAuthority(tmp_path, init)
    descriptor = authority.output_fd
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_end)
        try:
            authority.write_immutable(Path('child.pth'), b'forbidden')
        except FormalTrainingError:
            os.write(write_end, b'rejected')
        else:
            os.write(write_end, b'accepted')
        finally:
            os.close(write_end)
            os._exit(0)
    os.close(write_end)
    observed = os.read(read_end, 32)
    os.close(read_end)
    os.waitpid(child, 0)
    assert observed == b'rejected'
    assert not (output / 'child.pth').exists()
    authority.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert not tuple(formal_training._ACTIVE_FORMAL_OUTPUT_AUTHORITIES)


def test_private_runner_staging_cleanup_is_fd_bound_and_victim_safe(tmp_path):
    staging = formal_training._PrivateRunnerStaging('full-seed0')
    assert stat.S_IMODE(staging.path.stat().st_mode) == 0o700
    (staging.path / 'runner.log').write_bytes(b'log')
    held = tmp_path / 'held-staging'
    staging.path.rename(held)
    victim = tmp_path / 'victim'
    victim.mkdir()
    (victim / 'valuable.bin').write_bytes(b'valuable')
    staging.path.symlink_to(victim, target_is_directory=True)
    with pytest.raises(FormalTrainingError, match='staging authority'):
        staging.cleanup()
    assert (held / 'runner.log').read_bytes() == b'log'
    assert (victim / 'valuable.bin').read_bytes() == b'valuable'


def test_private_runner_staging_normal_cleanup_and_fork_fd_closure():
    staging = formal_training._PrivateRunnerStaging('full-seed0')
    path = staging.path
    (path / 'runner.log').write_bytes(b'log')
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_end)
        try:
            os.fstat(staging._directory_fd)
        except OSError:
            os.write(write_end, b'closed')
        else:
            os.write(write_end, b'open')
        finally:
            os.close(write_end)
            os._exit(0)
    os.close(write_end)
    observed = os.read(read_end, 16)
    os.close(read_end)
    os.waitpid(child, 0)
    assert observed == b'closed'
    staging.cleanup()
    assert not path.exists()
    assert not tuple(formal_training._ACTIVE_PRIVATE_RUNNER_STAGINGS)


@pytest.mark.parametrize('failure', ['child_open', 'fstat', 'mode'])
def test_private_runner_staging_constructor_failure_closes_and_removes_exact_dir(
        monkeypatch, failure):
    token = f'constructor-{failure}'
    name = f'mambapose-formal-runner-full-seed0-{token}'
    path = Path('/tmp') / name
    assert not path.exists()
    monkeypatch.setattr(formal_training.secrets, 'token_hex', lambda _n: token)
    original_open = formal_training.os.open
    original_fstat = formal_training.os.fstat
    child_fd = None

    def injected_open(target, flags, *args, **kwargs):
        nonlocal child_fd
        if target == name and failure == 'child_open':
            raise OSError('injected child open failure')
        descriptor = original_open(target, flags, *args, **kwargs)
        if target == name:
            child_fd = descriptor
        return descriptor

    def injected_fstat(descriptor):
        observed = original_fstat(descriptor)
        if descriptor == child_fd and failure == 'fstat':
            raise OSError('injected child fstat failure')
        if descriptor == child_fd and failure == 'mode':
            values = list(observed)
            values[0] = (observed.st_mode & ~0o777) | 0o755
            return os.stat_result(values)
        return observed

    monkeypatch.setattr(formal_training.os, 'open', injected_open)
    monkeypatch.setattr(formal_training.os, 'fstat', injected_fstat)
    try:
        with pytest.raises((OSError, FormalTrainingError)):
            formal_training._PrivateRunnerStaging('full-seed0')
        assert not path.exists()
        assert not tuple(formal_training._ACTIVE_PRIVATE_RUNNER_STAGINGS)
        if child_fd is not None:
            with pytest.raises(OSError):
                original_fstat(child_fd)
    finally:
        if path.is_dir():
            path.rmdir()


def test_private_runner_staging_constructor_failure_preserves_replaced_name(
        monkeypatch):
    token = 'constructor-replaced-name'
    name = f'mambapose-formal-runner-full-seed0-{token}'
    path = Path('/tmp') / name
    held = Path('/tmp') / f'{name}.held'
    victim = Path('/tmp') / f'{name}.victim'
    for candidate in (path, held, victim):
        assert not candidate.exists() and not candidate.is_symlink()
    monkeypatch.setattr(formal_training.secrets, 'token_hex', lambda _n: token)
    original_open = formal_training.os.open

    def replace_before_child_open(target, flags, *args, **kwargs):
        if target == name:
            path.rename(held)
            victim.mkdir()
            (victim / 'valuable.bin').write_bytes(b'valuable')
            path.symlink_to(victim, target_is_directory=True)
            raise OSError('injected replaced-name child open failure')
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(formal_training.os, 'open', replace_before_child_open)
    try:
        with pytest.raises(OSError, match='replaced-name'):
            formal_training._PrivateRunnerStaging('full-seed0')
        assert path.is_symlink()
        assert held.is_dir()
        assert (victim / 'valuable.bin').read_bytes() == b'valuable'
        assert not tuple(formal_training._ACTIVE_PRIVATE_RUNNER_STAGINGS)
    finally:
        if path.is_symlink():
            path.unlink()
        if (victim / 'valuable.bin').exists():
            (victim / 'valuable.bin').unlink()
        if victim.is_dir():
            victim.rmdir()
        if held.is_dir():
            held.rmdir()


def test_public_recovery_rejects_preexisting_same_byte_output_replacement(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    original = tmp_path / 'run-original'
    output.rename(original)
    output.mkdir()
    (output / 'run-init.json').write_bytes(
        (original / 'run-init.json').read_bytes())
    with pytest.raises(FormalTrainingError, match='output.*(identity|mismatch)'):
        recover_training_lineage(tmp_path, init, dataset_size=11)
    (output / 'run-init.json').unlink()
    output.rmdir()
    original.rename(output)
    assert recover_training_lineage(
        tmp_path, init, dataset_size=11) is None


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
    method = ('write_mutable' if failure_target == 'training.jsonl'
              else 'write_immutable')
    _inject_authority_failure_once(
        monkeypatch, method, failure_target,
        'injected durable boundary failure')
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
    _inject_authority_failure_once(
        monkeypatch, 'unlink', 'epoch_1.pth', 'injected prune failure')
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
    _inject_authority_failure_once(
        monkeypatch, 'write_immutable', 'epoch_2.pth',
        'injected checkpoint publication failure',
        after=(boundary == 'parent_fsync'))
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


def test_epoch_recovery_removes_only_exact_next_fixed_pending(tmp_path):
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
    pending = output / '.epoch_2.pth.pending'
    pending.write_bytes(b'partial')
    recovered = recover_training_lineage(tmp_path, init, dataset_size=11)
    assert recovered is not None and recovered.completed_epoch == 1
    assert not pending.exists()


@pytest.mark.parametrize('kind', ['exact', 'mutable'])
def test_recovery_pending_cleanup_rejects_classified_name_replacement(
        tmp_path, monkeypatch, kind):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    name = ('.epoch_1.pth.pending' if kind == 'exact'
            else '.training.jsonl.pending')
    pending = output / name
    pending.write_bytes(b'classified')
    held_original = output / f'{name}.held'
    original_unlink = formal_training._FormalOutputAuthority.unlink
    swapped = False

    def replace_before_unlink(self, relative, *args, **kwargs):
        nonlocal swapped
        if Path(relative).as_posix() == name and not swapped:
            swapped = True
            pending.rename(held_original)
            pending.write_bytes(b'foreign-replacement')
        return original_unlink(self, relative, *args, **kwargs)

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, 'unlink',
        replace_before_unlink)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='authority'):
            if kind == 'exact':
                formal_training._recover_exact_pending_transaction(
                    authority, next_epoch=1, committed=0)
            else:
                formal_training._recover_linked_pending_publications(
                    authority, committed=0)
    assert pending.read_bytes() == b'foreign-replacement'
    assert held_original.read_bytes() == b'classified'


def test_uncommitted_next_checkpoint_cleanup_uses_logged_sha_and_identity(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    for epoch in (1, 2):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=epoch,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=orders[:epoch],
            dataset_size=11)
    (output / 'epoch-commits/epoch_2.json').unlink()
    checkpoint = output / 'epoch_2.pth'
    held_original = output / 'epoch_2.pth.uncommitted'
    original_unlink = formal_training._FormalOutputAuthority.unlink
    swapped = False

    def replace_before_rollback(self, relative, *args, **kwargs):
        nonlocal swapped
        if Path(relative) == Path('epoch_2.pth') and not swapped:
            swapped = True
            checkpoint.rename(held_original)
            checkpoint.write_bytes(b'foreign-replacement')
        return original_unlink(self, relative, *args, **kwargs)

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, 'unlink',
        replace_before_rollback)
    with pytest.raises(FormalTrainingError, match='authority'):
        recover_training_lineage(tmp_path, init, dataset_size=11)
    assert checkpoint.read_bytes() == b'foreign-replacement'
    assert held_original.exists()


def test_uncommitted_log_cleanup_uses_captured_sha_and_identity(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 2, dataset_size=11)
    for epoch in (1, 2):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=epoch,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=orders[:epoch],
            dataset_size=11)
    (output / 'epoch-commits/epoch_2.json').unlink()
    log = output / 'training.jsonl'
    held_original = output / 'training.jsonl.uncommitted'
    original_write = formal_training._FormalOutputAuthority.write_mutable
    swapped = False

    def replace_before_rollback(self, relative, *args, **kwargs):
        nonlocal swapped
        if Path(relative) == Path('training.jsonl') and not swapped:
            swapped = True
            log.rename(held_original)
            log.write_bytes(b'foreign-replacement')
        return original_write(self, relative, *args, **kwargs)

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, 'write_mutable',
        replace_before_rollback)
    with pytest.raises(FormalTrainingError, match='authority'):
        recover_training_lineage(tmp_path, init, dataset_size=11)
    assert log.read_bytes() == b'foreign-replacement'
    assert held_original.exists()


def test_uncommitted_best_cleanup_uses_captured_decision_identity(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 6, dataset_size=11)
    decision = None
    for epoch in range(1, 6):
        if epoch == 5:
            decision = _publish_test_best(tmp_path, init, 5, 0.5)
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init,
            completed_epoch=epoch,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={},
            order_hashes=orders[:epoch], dataset_size=11,
            best_decision_path=decision)
    _publish_test_best(tmp_path, init, 6, 0.6)
    extra = output / 'best-lineages/epoch_6.json'
    held_original = output / 'best-lineages/epoch_6.json.uncommitted'
    original_unlink = formal_training._FormalOutputAuthority.unlink
    swapped = False

    def replace_before_cleanup(self, relative, *args, **kwargs):
        nonlocal swapped
        if Path(relative) == Path('best-lineages/epoch_6.json') and not swapped:
            swapped = True
            extra.rename(held_original)
            extra.write_bytes(b'foreign-replacement')
        return original_unlink(self, relative, *args, **kwargs)

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, 'unlink',
        replace_before_cleanup)
    with pytest.raises(FormalTrainingError, match='authority'):
        recover_training_lineage(tmp_path, init, dataset_size=11)
    assert extra.read_bytes() == b'foreign-replacement'
    assert held_original.exists()


def test_epoch_recovery_rejects_unknown_or_multiple_fixed_pending(tmp_path):
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
    (output / '.epoch_7.pth.pending').write_bytes(b'unknown')
    with pytest.raises(FormalTrainingError, match='pending'):
        recover_training_lineage(tmp_path, init, dataset_size=11)


@pytest.mark.parametrize(
    ('relative', 'payload'), (
        (Path('epoch_1.pth'), b'checkpoint'),
        (Path('best_coco_AP_epoch_5.pth'), b'best'),
        (Path('epoch-commits/epoch_1.json'), b'commit\n'),
        (Path('best-lineages/epoch_5.json'), b'decision\n'),
        (Path('train-result.json'), b'result\n'),
    ))
def test_recovery_finishes_linked_immutable_pending_publication(
        tmp_path, relative, payload):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        if len(relative.parts) > 1:
            authority.mkdir(relative.parent)
        authority.write_immutable(relative, payload)
        parent = output / relative.parent
        pending = parent / f'.{relative.name}.pending'
        os.link(output / relative, pending)
        formal_training._recover_linked_pending_publications(
            authority,
            committed=(300 if relative == Path('train-result.json')
                       else None))
        assert not pending.exists()
        assert (output / relative).read_bytes() == payload


def test_recovery_rejects_foreign_final_for_immutable_pending_without_unlink(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    final = output / 'epoch_1.pth'
    pending = output / '.epoch_1.pth.pending'
    final.write_bytes(b'foreign-final')
    pending.write_bytes(b'our-pending')
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='pending.*publication'):
            formal_training._recover_linked_pending_publications(authority)
    assert final.read_bytes() == b'foreign-final'
    assert pending.read_bytes() == b'our-pending'


@pytest.mark.parametrize('second_name', (
    '.evil.pending', '.best-lineage.json.pending'))
def test_pending_inventory_rejects_before_mutating_any_member(
        tmp_path, second_name):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    first = output / '.training.jsonl.pending'
    second = output / second_name
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='pending.*inventory'):
            formal_training._recover_linked_pending_publications(authority)
    assert first.read_bytes() == b'first'
    assert second.read_bytes() == b'second'


def test_public_recovery_rejects_pre_final_linked_train_result_without_unlink(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    final = output / 'train-result.json'
    pending = output / '.train-result.json.pending'
    final.write_bytes(b'premature\n')
    os.link(final, pending)
    with pytest.raises(FormalTrainingError, match='precedes final'):
        recover_training_lineage(tmp_path, init, dataset_size=11)
    assert final.read_bytes() == b'premature\n'
    assert pending.read_bytes() == b'premature\n'


def test_public_completed_result_recovers_linked_pending_and_validates_bindings(
        tmp_path):
    raw_init = _run_init()
    init = replace(
        raw_init,
        initialization=replace(
            raw_init.initialization,
            asset=replace(raw_init.initialization.asset,
                          sha256=INITIALIZATION_SHA256)))
    output = tmp_path / init.output_root
    write_formal_run_init(init, output / 'run-init.json')
    order_hashes = trace_epoch_orders(init, 300, dataset_size=11)
    run_init_sha = _sha(output / 'run-init.json')
    best_name = 'best_coco_AP_epoch_5.pth'
    (output / best_name).write_bytes(b'best')
    best_sha = _sha(output / best_name)
    best_decisions = output / 'best-lineages'
    best_decisions.mkdir()
    decision_document = {
        'schema_version': 1,
        'identity': {
            'run_id': init.run_id, 'role': init.role, 'seed': init.seed},
        'completed_epoch': 5, 'metric': 0.5,
        'checkpoint': {'path': best_name, 'sha256': best_sha},
        'previous_decision': None,
    }
    decision_path = best_decisions / 'epoch_5.json'
    decision_path.write_bytes(
        formal_training._canonical_json_bytes(decision_document))
    decision_sha = _sha(decision_path)
    best_authority = {
        'decision': {
            'path': 'best-lineages/epoch_5.json', 'sha256': decision_sha},
        'checkpoint': {'path': best_name, 'sha256': best_sha},
    }
    (output / 'best-lineage.json').write_bytes(
        formal_training._canonical_json_bytes({
            'schema_version': 1,
            'decision': best_authority['decision'],
        }))

    checkpoint_shas = {epoch: hashlib.sha256(
        f'pruned-checkpoint-{epoch}'.encode()).hexdigest()
        for epoch in range(1, 299)}
    for epoch in (299, 300):
        document = formal_training._resume_document(
            expected=init, run_init_sha256=run_init_sha,
            completed_epoch=epoch, order_hashes=order_hashes[:epoch],
            model_state={'weight': torch.tensor([float(epoch)])},
            optimizer_state={}, scheduler_state={}, scaler_state={})
        torch.save(document, output / f'epoch_{epoch}.pth')
        checkpoint_shas[epoch] = _sha(output / f'epoch_{epoch}.pth')

    commits = output / 'epoch-commits'
    commits.mkdir()
    structured_log = b''
    previous_commit_sha = None
    for epoch in range(1, 301):
        record = {
            'schema_version': 1, 'run_id': init.run_id,
            'role': init.role, 'seed': init.seed, 'epoch': epoch,
            'order_sha256': order_hashes[epoch - 1],
            'resume_checkpoint': f'epoch_{epoch}.pth',
            'resume_sha256': checkpoint_shas[epoch],
        }
        structured_log += formal_training._canonical_json_bytes(record)
        commit = {
            'schema_version': 1,
            'identity': {
                'run_id': init.run_id, 'role': init.role, 'seed': init.seed,
                'run_init_sha256': run_init_sha,
            },
            'completed_epoch': epoch,
            'checkpoint': {
                'path': f'epoch_{epoch}.pth',
                'sha256': checkpoint_shas[epoch],
            },
            'structured_log': {
                'path': 'training.jsonl',
                'sha256': hashlib.sha256(structured_log).hexdigest(),
            },
            'log_record': record,
            'best': None if epoch < 5 else best_authority,
            'previous_commit': (
                None if epoch == 1 else {
                    'path': f'epoch_{epoch - 1}.json',
                    'sha256': previous_commit_sha,
                }),
        }
        commit_path = commits / f'epoch_{epoch}.json'
        commit_path.write_bytes(
            formal_training._canonical_json_bytes(commit))
        previous_commit_sha = _sha(commit_path)
    (output / 'training.jsonl').write_bytes(structured_log)
    bindings = {
        name: FileBinding(
            path=init.output_root / name,
            sha256=_sha(output / name))
        for name in (best_name, 'epoch_299.pth', 'epoch_300.pth',
                     'training.jsonl')
    }
    result = formal_training.FormalTrainResult(
        run_init_sha256=run_init_sha,
        initialization=init.initialization, run_id=init.run_id,
        role=init.role, seed=init.seed, output_root=init.output_root,
        best_checkpoint=bindings[best_name],
        resume_checkpoints=(bindings['epoch_299.pth'],
                            bindings['epoch_300.pth']),
        structured_log=bindings['training.jsonl'],
        order_hashes=order_hashes, final_epoch=300, status='complete')
    result_path = output / 'train-result.json'
    result_path.write_bytes(formal_training._canonical_json_bytes(
        formal_training._train_result_document(result)))
    pending = output / '.train-result.json.pending'
    os.link(result_path, pending)
    recovered = formal_training.recover_completed_training_result(
        tmp_path, init, dataset_size=11)
    assert recovered == result
    assert not pending.exists()


@pytest.mark.parametrize('target', [
    'train-result.json', 'epoch-commits/epoch_300.json',
    'best-lineages/epoch_5.json', 'best_coco_AP_epoch_5.pth',
    'epoch_299.pth', 'epoch_300.pth', 'training.jsonl',
    '__root_after_files__',
])
def test_completed_result_revalidates_every_captured_binding_before_return(
        tmp_path, monkeypatch, target):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    payloads = {
        'best_coco_AP_epoch_5.pth': b'best',
        'epoch_299.pth': b'299',
        'epoch_300.pth': b'300',
        'training.jsonl': b'log',
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    best_sha = _sha(output / 'best_coco_AP_epoch_5.pth')
    decision = {
        'schema_version': 1,
        'identity': {'run_id': init.run_id, 'role': init.role, 'seed': init.seed},
        'completed_epoch': 5, 'metric': 0.5,
        'checkpoint': {
            'path': 'best_coco_AP_epoch_5.pth', 'sha256': best_sha},
        'previous_decision': None,
    }
    decisions = output / 'best-lineages'
    decisions.mkdir()
    decision_path = decisions / 'epoch_5.json'
    decision_path.write_bytes(formal_training._canonical_json_bytes(decision))
    order_hashes = tuple(
        hashlib.sha256(f'order:{epoch}'.encode()).hexdigest()
        for epoch in range(1, 301))
    run_init_sha = _sha(output / 'run-init.json')
    result = formal_training.FormalTrainResult(
        run_init_sha256=run_init_sha, initialization=init.initialization,
        run_id=init.run_id, role=init.role, seed=init.seed,
        output_root=init.output_root,
        best_checkpoint=FileBinding(
            init.output_root / 'best_coco_AP_epoch_5.pth', best_sha),
        resume_checkpoints=(
            FileBinding(init.output_root / 'epoch_299.pth',
                        _sha(output / 'epoch_299.pth')),
            FileBinding(init.output_root / 'epoch_300.pth',
                        _sha(output / 'epoch_300.pth'))),
        structured_log=FileBinding(
            init.output_root / 'training.jsonl', _sha(output / 'training.jsonl')),
        order_hashes=order_hashes, final_epoch=300, status='complete')
    final_commit = {
        'schema_version': 1,
        'identity': {
            'run_id': init.run_id, 'role': init.role, 'seed': init.seed,
            'run_init_sha256': run_init_sha},
        'completed_epoch': 300,
        'checkpoint': {
            'path': 'epoch_300.pth',
            'sha256': result.resume_checkpoints[1].sha256},
        'structured_log': {
            'path': 'training.jsonl', 'sha256': result.structured_log.sha256},
        'log_record': {
            'schema_version': 1, 'run_id': init.run_id, 'role': init.role,
            'seed': init.seed, 'epoch': 300,
            'order_sha256': order_hashes[-1],
            'resume_checkpoint': 'epoch_300.pth',
            'resume_sha256': result.resume_checkpoints[1].sha256},
        'best': {
            'decision': {
                'path': 'best-lineages/epoch_5.json',
                'sha256': _sha(decision_path)},
            'checkpoint': {
                'path': 'best_coco_AP_epoch_5.pth', 'sha256': best_sha}},
        'previous_commit': {'path': 'epoch_299.json', 'sha256': 'a' * 64},
    }
    commits = output / 'epoch-commits'
    commits.mkdir()
    (commits / 'epoch_300.json').write_bytes(
        formal_training._canonical_json_bytes(final_commit))
    result_path = output / 'train-result.json'
    result_path.write_bytes(formal_training._canonical_json_bytes(
        formal_training._train_result_document(result)))
    monkeypatch.setattr(
        formal_training.FormalTrainResult, 'from_dict',
        classmethod(lambda _cls, *_args, **_kwargs: result))
    replaced = False
    held_original = (
        output.parent / 'run.captured' if target == '__root_after_files__'
        else output / f'{Path(target).name}.captured')
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        original_hold = authority.hold_streaming_regular

        def replace_target():
            nonlocal replaced
            if replaced:
                return
            replaced = True
            if target == '__root_after_files__':
                output.rename(held_original)
                output.mkdir()
                return
            live = output / target
            live.rename(held_original)
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_bytes(b'foreign-replacement')

        def hold(relative):
            captured = original_hold(relative)
            if Path(relative).as_posix() == target:
                replace_target()
            elif target == '__root_after_files__' \
                    and Path(relative) == Path('training.jsonl'):
                original_revalidate = captured.revalidate

                def revalidate_then_replace_root():
                    original_revalidate()
                    replace_target()

                monkeypatch.setattr(
                    captured, 'revalidate', revalidate_then_replace_root)
            return captured

        monkeypatch.setattr(authority, 'hold_streaming_regular', hold)
        with pytest.raises(FormalTrainingError, match='authority|binding|result'):
            formal_training._load_completed_training_result_held(
                tmp_path, init, authority)
    if target == '__root_after_files__':
        assert output.is_dir() and not tuple(output.iterdir())
        assert held_original.is_dir()
    else:
        assert (output / target).read_bytes() == b'foreign-replacement'
        assert held_original.exists()


@pytest.mark.parametrize('mutation', ['nonfinite_metric', 'invalid_predecessor'])
def test_training_completed_decision_rejects_incomplete_chain_semantics(
        mutation):
    init = _run_init(output_root='run')
    checkpoint = {
        'path': 'best_coco_AP_epoch_5.pth', 'sha256': 'b' * 64}
    decision = {
        'schema_version': 1,
        'identity': {
            'run_id': init.run_id, 'role': init.role, 'seed': init.seed},
        'completed_epoch': 5, 'metric': 0.5,
        'checkpoint': checkpoint, 'previous_decision': None,
    }
    if mutation == 'nonfinite_metric':
        decision['metric'] = float('nan')
    else:
        decision['previous_decision'] = {
            'path': 'best-lineages/../epoch_1.json', 'sha256': 'f' * 64}
    payload = json.dumps(decision, sort_keys=True).encode()
    with pytest.raises(FormalTrainingError, match='decision'):
        formal_training._validate_completed_best_decision_payload(
            payload, expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected=init, decision_path='best-lineages/epoch_5.json',
            expected_checkpoint=checkpoint)


def test_held_epoch_route_never_uses_pathname_or_legacy_atomic_io(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 1, dataset_size=11)

    def forbidden(*_args, **_kwargs):
        pytest.fail('pathname or legacy atomic I/O entered held route')

    for name in ('read_bytes', 'write_bytes', 'exists', 'iterdir', 'glob'):
        monkeypatch.setattr(Path, name, forbidden)
    for name in ('_atomic_write_bytes', '_atomic_torch_save',
                 '_write_atomic_file_at', '_durable_unlink'):
        monkeypatch.setattr(formal_training, name, forbidden)
    monkeypatch.setattr(formal_training.tempfile, 'mkdtemp', forbidden)
    original_save = torch.save
    original_load = torch.load

    def bytes_only_save(document, destination, *args, **kwargs):
        assert isinstance(destination, io.BytesIO)
        return original_save(document, destination, *args, **kwargs)

    def bytes_only_load(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, 'save', bytes_only_save)
    monkeypatch.setattr(torch, 'load', bytes_only_load)
    checkpoint = write_resume_checkpoint(
        repository_root=tmp_path, expected=init, completed_epoch=1,
        model_state={'weight': torch.ones(1)}, optimizer_state={},
        scheduler_state={}, scaler_state={}, order_hashes=orders,
        dataset_size=11)
    recovered = recover_training_lineage(tmp_path, init, dataset_size=11)
    assert checkpoint.name == 'epoch_1.pth'
    assert recovered is not None and recovered.completed_epoch == 1


def test_run_init_recovers_crash_before_and_after_immutable_link(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    identity = output.stat().st_dev, output.stat().st_ino
    payload = formal_training._canonical_json_bytes(
        formal_training._init_document(init, output_identity=identity))
    pending = output / '.run-init.json.pending'

    pending.write_bytes(payload)
    first = write_formal_run_init(init, output / 'run-init.json')
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert not pending.exists()

    os.link(output / 'run-init.json', pending)
    second = write_formal_run_init(init, output / 'run-init.json')
    assert second == first
    assert not pending.exists()


def test_run_init_pending_mismatch_and_foreign_final_fail_without_unlink(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    pending = output / '.run-init.json.pending'
    pending.write_bytes(b'wrong')
    with pytest.raises(FormalTrainingError, match='pending'):
        write_formal_run_init(init, output / 'run-init.json')
    assert pending.read_bytes() == b'wrong'

    pending.unlink()
    identity = output.stat().st_dev, output.stat().st_ino
    payload = formal_training._canonical_json_bytes(
        formal_training._init_document(init, output_identity=identity))
    pending.write_bytes(payload)
    (output / 'run-init.json').write_bytes(payload)
    with pytest.raises(FormalTrainingError, match='pending.*publication'):
        write_formal_run_init(init, output / 'run-init.json')
    assert pending.read_bytes() == payload
    assert (output / 'run-init.json').read_bytes() == payload


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
    _inject_authority_failure_once(
        monkeypatch, 'write_mutable', 'best-lineage.json',
        'injected best pointer failure')
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
    _inject_authority_failure_once(
        monkeypatch, 'unlink', first_best.name,
        'injected old best prune failure')
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
    _inject_authority_failure_once(
        monkeypatch, 'write_immutable',
        f'epoch-commits/epoch_{failed_epoch}.json',
        'injected sparse best commit failure')
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

    def replace_after_load(
            root, expected, *, completed_epoch, authority=None):
        nonlocal replaced
        result = original(
            root, expected, completed_epoch=completed_epoch,
            authority=authority)
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


def test_stale_checkpoint_prune_rejects_replacement_after_commit_classification(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    write_formal_run_init(init, output / 'run-init.json')
    orders = trace_epoch_orders(init, 3, dataset_size=11)
    for epoch in (1, 2):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=epoch,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=orders[:epoch],
            dataset_size=11)
    stale = output / 'epoch_1.pth'
    held_original = output / 'epoch_1.pth.committed'
    original_unlink = formal_training._FormalOutputAuthority.unlink
    swapped = False

    def replace_before_prune(self, relative, *args, **kwargs):
        nonlocal swapped
        if Path(relative) == Path('epoch_1.pth') and not swapped:
            swapped = True
            stale.rename(held_original)
            stale.write_bytes(b'foreign-replacement')
        return original_unlink(self, relative, *args, **kwargs)

    monkeypatch.setattr(
        formal_training._FormalOutputAuthority, 'unlink', replace_before_prune)
    with pytest.raises(FormalTrainingError, match='authority'):
        write_resume_checkpoint(
            repository_root=tmp_path, expected=init, completed_epoch=3,
            model_state={'weight': torch.ones(1)}, optimizer_state={},
            scheduler_state={}, scaler_state={}, order_hashes=orders,
            dataset_size=11)
    assert stale.read_bytes() == b'foreign-replacement'
    assert held_original.exists()


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
        assert isinstance(source, io.BytesIO)
        path.write_bytes(hostile)
        try:
            return original_load(source, *args, **kwargs)
        finally:
            path.write_bytes(approved)

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


@pytest.mark.parametrize(
    ('boundary', 'completed_epoch', 'prior_epoch'),
    [('before_val_epoch', 5, 0),
     ('after_val_iter', 10, 9),
     ('after_val_epoch', 300, 299)],
)
def test_validation_stop_acks_prior_commit_and_never_publishes_pending_epoch(
        tmp_path, monkeypatch, boundary, completed_epoch, prior_epoch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    (output / 'stop-request.json').write_bytes(b'pending-stop')
    hook = formal_training._build_training_hook(
        root=tmp_path, init=init, run_init_path=init_path, resume=None)
    hook.order_hashes = [f'{epoch:064x}' for epoch in range(1, completed_epoch + 1)]
    hook.pending_checkpoint = {'completed_epoch': completed_epoch}
    if prior_epoch:
        checkpoint = output / f'epoch_{prior_epoch}.pth'
        checkpoint.write_bytes(f'committed-{prior_epoch}'.encode())
        hook.last_checkpoint = checkpoint
    runner = SimpleNamespace(epoch=completed_epoch)
    synchronized = []
    acknowledgements = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: synchronized.append(True))
    monkeypatch.setattr(
        formal_training, '_poll_formal_training_stop',
        lambda _authority, safe_boundary: SimpleNamespace(
            request_id=f'stop-{boundary}', boundary=safe_boundary))

    def acknowledge(_authority, _request, resume):
        acknowledgements.append(resume)
        return SimpleNamespace(exit_code=75)

    monkeypatch.setattr(
        formal_training, '_write_formal_training_stop_ack', acknowledge)

    def forbidden_publish(*_args, **_kwargs):
        pytest.fail('interrupted validation published an epoch artifact')

    for name in ('publish_best_checkpoint', 'inherit_best_checkpoint',
                 'write_resume_checkpoint'):
        monkeypatch.setattr(formal_training, name, forbidden_publish)
    with pytest.raises(SystemExit) as stopped:
        if boundary == 'before_val_epoch':
            hook.before_val_epoch(runner)
        elif boundary == 'after_val_iter':
            hook.after_val_iter(runner, completed_epoch - 1)
        else:
            # This request arrives after the final after_val_iter callback.
            hook.after_val_epoch(runner, metrics=None)
    assert stopped.value.code == 75
    assert synchronized == [True]
    assert len(acknowledgements) == 1
    assert acknowledgements[0].completed_epoch == prior_epoch
    assert hook.pending_checkpoint == {'completed_epoch': completed_epoch}
    assert not (output / f'epoch_{completed_epoch}.pth').exists()
    assert not (output / f'epoch-commits/epoch_{completed_epoch}.json').exists()
    assert not tuple(output.glob(f'best_coco_AP_epoch_{completed_epoch}.pth'))


def test_formal_training_stop_uses_one_held_output_authority_across_ack(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    control = output / 'stop-request.json'
    control.write_bytes(formal_training._canonical_json_bytes({
        'schema_version': 1, 'request_id': 'held-stop',
        'stage': 'training', 'input_path': str(init_path),
        'input_sha256': _sha(init_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(output / 'stop-ack.json'),
        'partial_staging_path': str(output / 'partial-staging'),
    }))
    held_output = tmp_path / 'run-held'
    replacement = tmp_path / 'replacement'
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        request = formal_training._poll_formal_training_stop(
            authority, StageSafeBoundary('training', 'optimizer', 3, True))
        assert request is not None
        output.rename(held_output)
        output.mkdir()
        with pytest.raises(FormalTrainingError, match='authority'):
            formal_training._write_formal_training_stop_ack(
                authority, request,
                TrainingResumeAuthority(
                    init.run_id, 0, str(init_path),
                    _sha(held_output / 'run-init.json')))
        assert not tuple(output.iterdir())
    output.rename(replacement)
    held_output.rename(output)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        request = formal_training._poll_formal_training_stop(
            authority, StageSafeBoundary('training', 'optimizer', 3, True))
        assert request is not None
        acknowledgement = formal_training._write_formal_training_stop_ack(
            authority, request,
            TrainingResumeAuthority(
                init.run_id, 0, str(init_path), _sha(init_path)))
    assert acknowledgement.exit_code == 75
    assert (output / 'stop-ack.json').is_file()
    assert not tuple(replacement.iterdir())


def test_formal_training_hook_never_reopens_generic_stop_paths():
    source = inspect.getsource(formal_training._build_training_hook)
    assert 'poll_cooperative_stop(' not in source
    assert 'write_stop_acknowledgement(' not in source
    assert '_poll_formal_training_stop(' in source
    assert '_write_formal_training_stop_ack(' in source


def test_formal_training_stop_rejects_wrong_run_epoch_and_nonlatest_input(
        tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    control = output / 'stop-request.json'

    def publish(input_path):
        control.write_bytes(formal_training._canonical_json_bytes({
            'schema_version': 1, 'request_id': 'authority-stop',
            'stage': 'training', 'input_path': str(input_path),
            'input_sha256': _sha(input_path),
            'issued_at_ns': time.time_ns(),
            'acknowledgement_path': str(output / 'stop-ack.json'),
            'partial_staging_path': str(output / 'partial-staging'),
        }))

    publish(init_path)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        request = formal_training._poll_formal_training_stop(
            authority, StageSafeBoundary('training', 'optimizer', 1, True))
        assert request is not None
        with pytest.raises(FormalTrainingError, match='resume input'):
            formal_training._write_formal_training_stop_ack(
                authority, request,
                TrainingResumeAuthority(
                    'wrong-run', 0, str(init_path), _sha(init_path)))
        with pytest.raises(FormalTrainingError, match='latest committed'):
            formal_training._write_formal_training_stop_ack(
                authority, request,
                TrainingResumeAuthority(
                    init.run_id, 1, str(init_path), _sha(init_path)))

    training_log = output / 'training.jsonl'
    training_log.write_bytes(b'not-a-resume\n')
    publish(training_log)
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        with pytest.raises(FormalTrainingError, match='latest committed'):
            formal_training._poll_formal_training_stop(
                authority,
                StageSafeBoundary('training', 'optimizer', 1, True))


def test_formal_training_stop_rejects_control_swap_before_ack(tmp_path):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    control = output / 'stop-request.json'
    control.write_bytes(formal_training._canonical_json_bytes({
        'schema_version': 1, 'request_id': 'control-swap',
        'stage': 'training', 'input_path': str(init_path),
        'input_sha256': _sha(init_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(output / 'stop-ack.json'),
        'partial_staging_path': str(output / 'partial-staging'),
    }))
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        request = formal_training._poll_formal_training_stop(
            authority, StageSafeBoundary('training', 'optimizer', 1, True))
        assert request is not None
        payload = control.read_bytes()
        control.unlink()
        control.write_bytes(payload)
        with pytest.raises(FormalTrainingError, match='authority'):
            formal_training._write_formal_training_stop_ack(
                authority, request,
                TrainingResumeAuthority(
                    init.run_id, 0, str(init_path), _sha(init_path)))
    assert not (output / 'stop-ack.json').exists()


def test_formal_training_stop_staging_replacement_never_purges_victim(
        tmp_path, monkeypatch):
    init = _run_init(output_root='run')
    output = tmp_path / 'run'
    output.mkdir()
    init_path = output / 'run-init.json'
    write_formal_run_init(init, init_path)
    staging = output / 'partial-staging'
    staging.mkdir()
    (staging / 'partial.bin').write_bytes(b'partial')
    control = output / 'stop-request.json'
    control.write_bytes(formal_training._canonical_json_bytes({
        'schema_version': 1, 'request_id': 'staging-swap',
        'stage': 'training', 'input_path': str(init_path),
        'input_sha256': _sha(init_path), 'issued_at_ns': time.time_ns(),
        'acknowledgement_path': str(output / 'stop-ack.json'),
        'partial_staging_path': str(staging),
    }))
    outside = tmp_path / 'outside'
    victim = outside / 'victim'
    victim.mkdir(parents=True)
    (victim / 'valuable.bin').write_bytes(b'valuable')
    held_staging = output / 'held-staging'
    with formal_training._FormalOutputAuthority(tmp_path, init) as authority:
        request = formal_training._poll_formal_training_stop(
            authority, StageSafeBoundary('training', 'optimizer', 1, True))
        assert request is not None
        original = authority.purge_directory

        def replace_before_purge(relative):
            staging.rename(held_staging)
            staging.symlink_to(victim, target_is_directory=True)
            return original(relative)

        monkeypatch.setattr(authority, 'purge_directory', replace_before_purge)
        with pytest.raises(FormalTrainingError, match='purge'):
            formal_training._write_formal_training_stop_ack(
                authority, request,
                TrainingResumeAuthority(
                    init.run_id, 0, str(init_path), _sha(init_path)))
    assert (victim / 'valuable.bin').read_bytes() == b'valuable'
    assert (held_staging / 'partial.bin').read_bytes() == b'partial'
    assert not (output / 'stop-ack.json').exists()
