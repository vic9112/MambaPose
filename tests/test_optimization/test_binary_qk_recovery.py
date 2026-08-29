import json
import hashlib
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mmpose.registry import MODELS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _force_mmengine_runner_cpu(monkeypatch):
    monkeypatch.setattr(
        'mmengine.runner.runner.get_device', lambda: torch.device('cpu'))


def test_recovery_clone_uses_only_the_full_pose_checkpoint():
    from mmpose.models.distillers.binary_qk_distiller import _with_qk_mode

    base = {
        'backbone': {'pretrained': 'backbone-only.pth'},
        'head': {'tokenpose_cfg': {'qk_mode': 'float'}},
    }
    recovery = _with_qk_mode(base, 'binary_scaled')

    assert recovery['backbone']['pretrained'] is None
    assert recovery['head']['tokenpose_cfg']['qk_mode'] == 'binary_scaled'
    assert base['backbone']['pretrained'] == 'backbone-only.pth'


class _ToyBinaryQKPose(BaseModel):

    def __init__(self, head, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.qk_mode = head['tokenpose_cfg']['qk_mode']
        self.projection = nn.Linear(4, 4, bias=False)
        self.forward_calls = 0

    def forward(self, inputs, data_samples=None, mode='tensor'):
        self.forward_calls += 1
        output = self.projection(inputs)
        if self.qk_mode == 'binary_scaled':
            output = output * 0.5
        if mode == 'loss':
            return {'loss_kpt': (output - data_samples).square().mean()}
        if mode == 'predict':
            return output.detach()
        if mode == 'tensor':
            return output
        raise RuntimeError(mode)

    def loss_from_output(self, output, data_samples):
        return {'loss_kpt': (output - data_samples).square().mean()}


class _ResumeDataset(Dataset):

    metainfo = {}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            'inputs': torch.zeros(4),
            'data_samples': torch.zeros(4),
        }


@pytest.fixture
def toy_recovery_assets(tmp_path: Path):
    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    config = tmp_path / 'toy_pose.py'
    config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='float')))\n",
        encoding='utf-8')
    model = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'float'}})
    with torch.no_grad():
        model.projection.weight.copy_(torch.arange(16).reshape(4, 4) / 16)
    checkpoint = tmp_path / 'full_pose.pth'
    # Mirrors legacy MMEngine checkpoint metadata that PyTorch 2.6 rejects
    # unless loading uses the repository's restricted safe-globals path.
    torch.save({
        'meta': {'legacy_numpy': np.array([1], dtype=np.int64)},
        'state_dict': model.state_dict(),
    }, checkpoint)
    return (
        config, checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model.state_dict())


def _real_resume_runner(toy_recovery_assets, work_dir: Path):
    from mambapose_opt.binary_recovery import BoundBinaryRecoveryRunner
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, checkpoint_sha256, _ = toy_recovery_assets
    model = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256, distill_weight=0.25)
    runner_config = Config(dict(
        model=dict(type='BinaryQKSelfDistiller'),
        train_cfg=dict(by_epoch=True, max_epochs=60, val_interval=5),
        optim_wrapper=dict(optimizer=dict(type='Adam', lr=1e-4)),
        param_scheduler=[
            dict(
                type='LinearLR', begin=0, end=500, start_factor=0.001,
                by_epoch=False),
            dict(
                type='MultiStepLR', begin=0, end=60,
                milestones=[40, 52], gamma=0.1, by_epoch=True),
        ],
        randomness=dict(seed=0, deterministic=True),
    ))
    runner = BoundBinaryRecoveryRunner(
        model=model,
        work_dir=str(work_dir),
        train_dataloader=DataLoader(_ResumeDataset(), batch_size=1),
        train_cfg=runner_config.train_cfg,
        optim_wrapper=runner_config.optim_wrapper,
        param_scheduler=runner_config.param_scheduler,
        randomness=runner_config.randomness,
        default_scope='mmpose',
        cfg=runner_config,
    )
    # Build the same lazy MMEngine objects that checkpoint hooks persist.
    _ = runner.train_loop
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    runner.param_schedulers = runner.build_param_scheduler(
        runner.param_schedulers)
    return runner


def test_recovery_loads_same_full_checkpoint_and_freezes_float_teacher(
        toy_recovery_assets):
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, checkpoint_sha256, expected = toy_recovery_assets
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        distill_weight=0.25)
    distiller.init_weights()

    assert distiller.teacher.qk_mode == 'float'
    assert distiller.student.qk_mode == 'binary_scaled'
    assert not distiller.teacher.training
    assert all(not parameter.requires_grad
               for parameter in distiller.teacher.parameters())
    for key, value in expected.items():
        torch.testing.assert_close(distiller.teacher.state_dict()[key], value)
        torch.testing.assert_close(distiller.student.state_dict()[key], value)

    distiller.train()
    assert not distiller.teacher.training
    assert distiller.student.training


def test_recovery_combines_supervised_and_final_heatmap_mse(
        toy_recovery_assets):
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, checkpoint_sha256, _ = toy_recovery_assets
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        distill_weight=0.25)
    distiller.init_weights()
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    targets = torch.zeros_like(inputs)

    losses = distiller(inputs, targets, mode='loss')
    assert distiller.student.forward_calls == 1
    with torch.no_grad():
        teacher = distiller.teacher(inputs, targets, mode='tensor')
        student = distiller.student(inputs, targets, mode='tensor')

    assert set(losses) == {'loss_kpt', 'loss_binary_qk_distill'}
    torch.testing.assert_close(
        losses['loss_kpt'], (student - targets).square().mean())
    torch.testing.assert_close(
        losses['loss_binary_qk_distill'],
        torch.nn.functional.mse_loss(student, teacher) * 0.25)
    sum(losses.values()).backward()
    assert distiller.student.projection.weight.grad is not None
    assert distiller.teacher.projection.weight.grad is None


def test_recovery_student_state_export_is_strictly_base_compatible(
        toy_recovery_assets):
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, checkpoint_sha256, _ = toy_recovery_assets
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256)
    distiller.init_weights()
    exported = distiller.student_state_dict()
    float_model = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'float'}})

    float_model.load_state_dict(exported, strict=True)
    assert exported.keys() == float_model.state_dict().keys()


def test_recovery_checkpoint_loading_is_strict(toy_recovery_assets, tmp_path):
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, _, _, _ = toy_recovery_assets
    broken_checkpoint = tmp_path / 'broken.pth'
    torch.save(
        {'state_dict': {'foreign.weight': torch.ones(1)}}, broken_checkpoint)
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(broken_checkpoint),
        checkpoint_sha256=hashlib.sha256(
            broken_checkpoint.read_bytes()).hexdigest())

    with pytest.raises(RuntimeError, match='Missing key'):
        distiller.init_weights()


def test_recovery_checkpoint_hash_and_load_share_one_open(
        toy_recovery_assets, monkeypatch):
    from mambapose_opt import binary_recovery
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, checkpoint_sha256, _ = toy_recovery_assets
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256)
    real_open = binary_recovery.os.open
    opens = 0

    def monitored_open(path, *args, **kwargs):
        nonlocal opens
        if Path(path) == checkpoint:
            opens += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(binary_recovery.os, 'open', monitored_open)
    distiller.init_weights()

    assert opens == 1


def test_recovery_checkpoint_sha_is_verified_before_model_load(
        toy_recovery_assets):
    from mmpose.models.distillers import BinaryQKSelfDistiller

    config, checkpoint, _, _ = toy_recovery_assets
    distiller = BinaryQKSelfDistiller(
        base_model_config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256='0' * 64)

    with pytest.raises(ValueError, match='SHA-256'):
        distiller.init_weights()


def test_bounded_recovery_config_is_deterministic_scaled_and_sixty_epochs():
    from mmengine.config import Config

    config = Config.fromfile(
        REPOSITORY_ROOT
        / 'configs/optimization/recovery/binary_qk_s_v1.py')

    assert config.randomness == dict(seed=0, deterministic=True)
    assert config.model.type == 'BinaryQKSelfDistiller'
    assert config.model.base_model_config == (
        'configs/optimization/coco_s_v1_deterministic.py')
    assert config.model.checkpoint == (
        'work_dirs/reproduction/runs/coco-s-v1/'
        'best_coco_AP_epoch_300.pth')
    assert config.model.checkpoint_sha256 == (
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2')
    assert config.model.distill_weight == 0.25
    assert config.train_cfg.max_epochs == 60
    assert config.optim_wrapper.optimizer.lr == 1e-4
    assert config.param_scheduler[-1].end == 60
    assert max(config.param_scheduler[-1].milestones) < 60
    assert config.binary_qk_recovery.qk_mode == 'binary_scaled'
    assert config.binary_qk_recovery.schedule_extensible is True


def test_recovery_readiness_discloses_runtime_scaling_and_no_bias():
    from mambapose_opt.binary_recovery import (
        build_config_binding, load_binary_recovery_readiness)

    readiness = load_binary_recovery_readiness(REPOSITORY_ROOT)
    operation = readiness['operation']

    assert operation['qk_mode'] == 'binary_scaled'
    assert operation['q_scale'] == (
        'abs(q).mean(token).mean(channel) per batch/head')
    assert operation['k_scale'] == (
        'abs(k).mean(token).mean(channel) per batch/head')
    assert operation['runtime_operations'] == [
        'q_abs', 'q_token_mean', 'q_channel_mean',
        'k_abs', 'k_token_mean', 'k_channel_mean',
        'signed_dot_scale_multiply_q', 'signed_dot_scale_multiply_k',
        'original_head_dim_scale_multiply',
    ]
    assert operation['signed_dot_backend'] == 'torch-einsum-float-proxy'
    assert operation['bitwise_kernel_present'] is False
    assert operation['pure_bitwise_cost_claim'] is False
    assert readiness['student']['learnable_attention_bias'] is False
    assert readiness['student']['new_checkpoint_parameters'] == []
    extension = readiness['optional_extensions']['relative_attention_bias']
    assert extension['enabled'] is False
    assert '17 keypoint' in extension['reason']
    assert readiness['teacher']['frozen'] is True
    assert readiness['loss']['distillation_target'] == 'final-heatmap-mse'
    assert readiness['schedule']['epochs'] == 60
    assert readiness['schedule']['fraction_of_base_training'] == 0.2
    assert readiness['checkpoint']['sha256'] == (
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2')
    assert readiness['schema_version'] == 2
    assert readiness['config'] == build_config_binding(
        REPOSITORY_ROOT,
        Path('configs/optimization/recovery/binary_qk_s_v1.py'))
    assert readiness['deployment']['config'] == build_config_binding(
        REPOSITORY_ROOT,
        Path('configs/optimization/recovery/binary_qk_s_v1_deploy.py'))
    assert readiness['deployment']['qk_mode'] == 'binary_scaled'
    assert readiness['verification_commands'] == {
        'prepare': (
            'PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
            '.venv/bin/python tools/optimization/'
            'train_binary_qk_recovery.py --prepare-only'),
        'cpu_tests': (
            'PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
            ".venv/bin/python -m pytest -o addopts='' -q "
            'tests/test_optimization/test_binary_qk.py '
            'tests/test_optimization/test_binary_qk_recovery.py '
            'tests/test_optimization/test_binary_readiness.py'),
    }


def test_prepare_only_launcher_is_read_only_and_emits_bound_readiness(tmp_path):
    work_dir = tmp_path / 'must-not-be-created'
    result = subprocess.run(
        [sys.executable,
         str(REPOSITORY_ROOT
             / 'tools/optimization/train_binary_qk_recovery.py'),
         '--prepare-only', '--work-dir', str(work_dir)],
        cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)

    assert value['status'] == 'ready'
    assert value['config']['sha256']
    assert value['checkpoint']['sha256'] == (
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2')
    assert value['launch']['epochs'] == 60
    assert value['launch']['work_dir'] == str(work_dir.resolve())
    assert not work_dir.exists()


def test_launcher_requires_explicit_resume_path_and_never_disables_safe_load():
    source = (REPOSITORY_ROOT
              / 'tools/optimization/train_binary_qk_recovery.py').read_text(
                  encoding='utf-8')

    assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in source
    result = subprocess.run(
        [sys.executable,
         str(REPOSITORY_ROOT
             / 'tools/optimization/train_binary_qk_recovery.py'),
         '--resume'],
        cwd=REPOSITORY_ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert '--resume' in result.stderr


def _resume_payload():
    return {
        'meta': {
            'epoch': 1,
            'iter': 2,
            'cfg': 'model = dict()\n',
            'seed': 0,
            'experiment_name': 'binary-recovery',
            'time': '20260829_000000',
            'mmengine_version': '0.10.7',
            'dataset_meta': {},
        },
        'state_dict': {
            'student.layer.weight': torch.ones(1),
            'teacher.layer.weight': torch.ones(1)},
        'message_hub': {
            'log_scalars': {}, 'runtime_info': {}, 'resumed_keys': {}},
        'optimizer': {'state': {}, 'param_groups': [{
            'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8,
            'weight_decay': 0, 'amsgrad': False, 'maximize': False,
            'foreach': None, 'capturable': False, 'differentiable': False,
            'fused': None, 'decoupled_weight_decay': False,
            'initial_lr': 1e-4, 'params': [],
        }]},
        'param_schedulers': [
            {
                'start_factor': 0.001, 'end_factor': 1.0,
                'total_iters': 499, 'param_name': 'lr', 'begin': 0,
                'end': 500, 'by_epoch': False, 'base_values': [1e-4],
                'last_step': 1, '_global_step': 2, 'verbose': False,
                '_last_value': [1e-7],
            },
            {
                'milestones': {40: 1, 52: 1}, 'gamma': 0.1,
                'param_name': 'lr', 'begin': 0, 'end': 60,
                'by_epoch': True, 'base_values': [1e-4], 'last_step': 1,
                '_global_step': 2, 'verbose': False,
                '_last_value': [1e-4],
            },
        ],
    }


def test_real_mmengine_resume_accepts_linear_lr_state_and_strictly_restores(
        toy_recovery_assets, tmp_path):
    from mambapose_opt.binary_recovery import load_training_resume_checkpoint

    original = _real_resume_runner(
        toy_recovery_assets, tmp_path / 'original-runner')
    original.save_checkpoint(str(tmp_path), 'real-mmengine-resume.pth')
    checkpoint = tmp_path / 'real-mmengine-resume.pth'
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)

    assert raw['param_schedulers'][0]['total_iters'] == 499
    bound = load_training_resume_checkpoint(
        checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        expected_config=original.cfg.pretty_text,
        expected_state_dict=original.model.state_dict())

    resumed = _real_resume_runner(
        toy_recovery_assets, tmp_path / 'resumed-runner')
    resumed.bind_resume_checkpoint(bound)
    resumed.resume(str(checkpoint), map_location='cpu')

    assert resumed.epoch == raw['meta']['epoch']
    assert resumed.iter == raw['meta']['iter']
    assert resumed.param_schedulers[0].state_dict()['total_iters'] == 499
    for name, expected in raw['state_dict'].items():
        torch.testing.assert_close(resumed.model.state_dict()[name], expected)


@pytest.mark.parametrize('mutation', ['missing', 'foreign', 'shape', 'dtype'])
def test_resume_checkpoint_requires_exact_distiller_state_schema(
        toy_recovery_assets, tmp_path, mutation):
    from mambapose_opt.binary_recovery import load_training_resume_checkpoint

    runner = _real_resume_runner(toy_recovery_assets, tmp_path / 'runner')
    runner.save_checkpoint(str(tmp_path), 'resume-schema.pth')
    checkpoint = tmp_path / 'resume-schema.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = runner.model.state_dict()
    key = next(iter(payload['state_dict']))
    if mutation == 'missing':
        payload['state_dict'].pop(key)
    elif mutation == 'foreign':
        payload['state_dict']['student.foreign'] = torch.ones(1)
    elif mutation == 'shape':
        payload['state_dict'][key] = torch.ones(1)
    elif mutation == 'dtype':
        payload['state_dict'][key] = payload['state_dict'][key].double()
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match='state_dict'):
        load_training_resume_checkpoint(
            checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            expected_config=runner.cfg.pretty_text,
            expected_state_dict=expected)


def test_resume_checkpoint_is_restricted_sha_bound_and_schema_exact(tmp_path):
    from mambapose_opt.binary_recovery import load_training_resume_checkpoint

    checkpoint = tmp_path / 'epoch_1.pth'
    payload = _resume_payload()
    torch.save(payload, checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    bound = load_training_resume_checkpoint(
        checkpoint, expected, expected_config=payload['meta']['cfg'],
        expected_state_dict=payload['state_dict'])

    assert bound.sha256 == expected
    assert bound.path == checkpoint.resolve()
    assert set(bound.payload) == {
        'meta', 'state_dict', 'message_hub', 'optimizer', 'param_schedulers'}


@pytest.mark.parametrize('mutation', [
    lambda value: value.update(foreign=True),
    lambda value: value.pop('optimizer'),
    lambda value: value['message_hub'].update(foreign=True),
    lambda value: value['meta'].pop('iter'),
    lambda value: value['meta'].update(cfg='foreign = True\n'),
    lambda value: value.update(param_schedulers=[]),
])
def test_resume_checkpoint_schema_rejects_missing_or_foreign_fields(
        tmp_path, mutation):
    from mambapose_opt.binary_recovery import load_training_resume_checkpoint

    checkpoint = tmp_path / 'epoch_1.pth'
    payload = _resume_payload()
    mutation(payload)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match='resume checkpoint'):
        load_training_resume_checkpoint(
            checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            expected_config=_resume_payload()['meta']['cfg'],
            expected_state_dict=_resume_payload()['state_dict'])


def test_student_checkpoint_export_removes_wrapper_prefixes(tmp_path):
    from mambapose_opt.binary_recovery import export_student_checkpoint

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    source = tmp_path / 'wrapper.pth'
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    torch.save({
        'meta': {'epoch': 60},
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        },
        'optimizer': {'discard': True},
    }, source)

    smoke_inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    report = export_student_checkpoint(
        source, destination, deployment_config=deployment_config,
        smoke_inputs=smoke_inputs)
    exported = torch.load(destination, map_location='cpu', weights_only=True)

    assert report['tensors'] == len(reference.state_dict())
    assert exported['meta'] == {
        'source_checkpoint_sha256': report['source_sha256']}
    assert exported['state_dict'].keys() == reference.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(exported['state_dict'][key], value)
    assert report['checkpoint'] == {
        'path': str(destination.resolve()),
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    assert report['deployment_config'] == {
        'path': str(deployment_config.resolve()),
        'binding': {
            'path': 'deploy.py',
            'sha256': hashlib.sha256(
                deployment_config.read_bytes()).hexdigest(),
            'config_closure': [{
                'path': 'deploy.py',
                'sha256': hashlib.sha256(
                    deployment_config.read_bytes()).hexdigest(),
            }],
        },
    }
    assert len(report['transaction_token']) == 64
    assert len(report['state_schema_sha256']) == 64
    assert report['validation'] == {
        'qk_mode': 'binary_scaled',
        'strict_state_load': True,
        'cpu_smoke': True,
        'cpu_smoke_scope': 'full-model-test-double',
        'output_shape': [2, 4],
    }


def test_student_checkpoint_export_uses_restricted_tensor_loader(
        tmp_path, monkeypatch):
    from mambapose_opt import binary_recovery

    source = tmp_path / 'wrapper.pth'
    torch.save({
        'meta': {'epoch': 60},
        'state_dict': {
            'student.projection.weight': torch.eye(4),
            'teacher.projection.weight': torch.eye(4),
        },
        'optimizer': {'discard': True},
    }, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    real_open = binary_recovery.os.open
    opens = 0

    def monitored_open(path, *args, **kwargs):
        nonlocal opens
        if Path(path) == source:
            opens += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(binary_recovery.os, 'open', monitored_open)

    original_load = binary_recovery.torch.load

    def require_restricted(*args, **kwargs):
        assert kwargs.get('weights_only') is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(binary_recovery.torch, 'load', require_restricted)

    binary_recovery.export_student_checkpoint(
        source, destination, deployment_config=deployment_config,
        smoke_inputs=torch.zeros(1, 4))
    assert destination.is_file()
    assert opens == 1


def test_failed_deployment_validation_publishes_no_checkpoint(tmp_path):
    from mambapose_opt.binary_recovery import export_student_checkpoint

    source = tmp_path / 'wrapper.pth'
    destination = tmp_path / 'student.pth'
    torch.save({
        'state_dict': {
            'student.projection.weight': torch.eye(4),
            'teacher.projection.weight': torch.eye(4),
        }}, source)
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='float')))\n",
        encoding='utf-8')

    with pytest.raises(ValueError, match='binary_scaled'):
        export_student_checkpoint(
            source, destination, deployment_config=deployment_config,
            smoke_inputs=torch.zeros(1, 4))

    assert not destination.exists()


def test_config_snapshot_binds_base_closure_and_effective_config(tmp_path):
    from mambapose_opt.binary_recovery import (
        build_config_binding, load_bound_config)

    root = tmp_path / 'repository'
    (root / 'configs').mkdir(parents=True)
    base = root / 'configs/base.py'
    base.write_text(
        "train_dataloader = dict(batch_size=128)\n",
        encoding='utf-8')
    child = root / 'configs/recovery.py'
    child.write_text(
        "_base_ = ['./base.py']\nmodel = dict(type='Toy')\n",
        encoding='utf-8')
    binding = build_config_binding(root, Path('configs/recovery.py'))

    assert load_bound_config(
        root, binding, label='recovery config').train_dataloader.batch_size == 128
    assert binding['config_closure'] == [
        {
            'path': 'configs/base.py',
            'sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
        },
        {
            'path': 'configs/recovery.py',
            'sha256': hashlib.sha256(child.read_bytes()).hexdigest(),
        },
    ]

    base.write_text(
        "train_dataloader = dict(batch_size=129)\n",
        encoding='utf-8')
    with pytest.raises(ValueError, match='closure'):
        load_bound_config(root, binding, label='recovery config')


def test_readiness_rejects_effective_base_batch_size_drift(monkeypatch):
    from mambapose_opt import binary_recovery

    target = (
        REPOSITORY_ROOT
        / 'configs/body_2d_keypoint/tokenpose/'
        'mamba_tokenpose_T2_coco_256x192_300ep.py')
    real_read = binary_recovery._read_regular_bytes_once

    def drift_batch_size(path, label):
        bound = real_read(path, label)
        if Path(path) == target:
            data = bound.data.replace(b'batch_size=128', b'batch_size=129', 1)
            assert data != bound.data
            return binary_recovery.BoundFile(
                path=bound.path, data=data,
                sha256=hashlib.sha256(data).hexdigest(),
                device=bound.device, inode=bound.inode)
        return bound

    monkeypatch.setattr(
        binary_recovery, '_read_regular_bytes_once', drift_batch_size)
    with pytest.raises(ValueError, match='closure'):
        binary_recovery.load_binary_recovery_readiness(REPOSITORY_ROOT)


def test_export_parses_same_held_deployment_config_bytes(
        tmp_path, monkeypatch):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    source = tmp_path / 'wrapper.pth'
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    original = (
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n")
    deployment_config.write_text(original, encoding='utf-8')
    replacement = tmp_path / 'replacement.py'
    replacement.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='float')))\n",
        encoding='utf-8')
    original_sha = hashlib.sha256(original.encode()).hexdigest()
    real_read = binary_recovery._read_regular_bytes_once
    swapped = False

    def swap_after_held_read(path, label):
        nonlocal swapped
        result = real_read(path, label)
        if Path(path) == deployment_config and not swapped:
            swapped = True
            os.replace(replacement, deployment_config)
        return result

    monkeypatch.setattr(
        binary_recovery, '_read_regular_bytes_once', swap_after_held_read)
    report = binary_recovery.export_student_checkpoint(
        source, destination, deployment_config=deployment_config,
        smoke_inputs=torch.zeros(1, 4))

    assert swapped
    assert report['deployment_config']['binding']['sha256'] == original_sha
    assert destination.is_file()


def test_export_rejects_swapped_validated_temporary_inode(
        tmp_path, monkeypatch):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    source = tmp_path / 'wrapper.pth'
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    real_load = reference.load_state_dict
    swapped = False

    def swap_after_validation(state_dict, *args, **kwargs):
        nonlocal swapped
        temporary, = tmp_path.glob('.student.pth.*.tmp')
        attacker = tmp_path / 'attacker.pth'
        torch.save({'state_dict': {'attacker': torch.ones(1)}}, attacker)
        os.replace(attacker, temporary)
        swapped = True
        return real_load(state_dict, *args, **kwargs)

    monkeypatch.setattr(reference, 'load_state_dict', swap_after_validation)
    real_build = MODELS.build
    monkeypatch.setattr(
        MODELS, 'build', lambda config: reference
        if config['type'] == 'ToyBinaryQKPose' else real_build(config))

    with pytest.raises(ValueError, match='changed after validation'):
        binary_recovery.export_student_checkpoint(
            source, destination, deployment_config=deployment_config,
            smoke_inputs=torch.zeros(1, 4))

    assert swapped
    assert not destination.exists()


def test_atomic_launch_metadata_marks_absent_commit_as_incomplete(tmp_path):
    from mambapose_opt import binary_recovery

    fsync_calls = []
    real_fsync = binary_recovery.os.fsync

    def record_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(binary_recovery.os, 'fsync', record_fsync)
    try:
        binary_recovery.write_binary_recovery_launch_metadata(
            tmp_path, config_text='model = dict()\n',
            readiness={'schema_version': 2, 'status': 'ready'})
    finally:
        monkeypatch.undo()

    assert (tmp_path / 'resolved-binary-qk-recovery.py').read_text() == (
        'model = dict()\n')
    assert json.loads((tmp_path / 'readiness.json').read_text()) == {
        'schema_version': 2, 'status': 'ready'}
    assert binary_recovery.load_binary_recovery_completion(
        tmp_path, repository_root=tmp_path,
        deployment_binding={}) is None
    assert len(fsync_calls) >= 4

    completion = {
        'schema_version': 1,
        'status': 'complete',
        'report': {'checkpoint': {'sha256': 'a' * 64}},
    }
    binary_recovery.atomic_write_json(
        tmp_path / 'student-export.json', completion)
    with pytest.raises(ValueError, match='completion schema'):
        binary_recovery.load_binary_recovery_completion(
            tmp_path, repository_root=tmp_path,
            deployment_binding={})


def test_completion_transaction_rejects_destination_swap_after_post_stat(
        tmp_path, monkeypatch):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    source = tmp_path / 'wrapper.pth'
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    binding = binary_recovery.build_config_binding(
        tmp_path, Path('deploy.py'))
    real_fsync = binary_recovery.os.fsync
    swapped = False

    def swap_in_post_stat_directory_fsync(descriptor):
        nonlocal swapped
        if (not swapped and destination.exists()
                and stat.S_ISDIR(os.fstat(descriptor).st_mode)):
            attacker = tmp_path / 'attacker.pth'
            torch.save(
                {'state_dict': {'attacker': torch.ones(1)}}, attacker)
            os.replace(attacker, destination)
            swapped = True
        return real_fsync(descriptor)

    monkeypatch.setattr(
        binary_recovery.os, 'fsync', swap_in_post_stat_directory_fsync)
    with pytest.raises(ValueError, match='checkpoint|export'):
        binary_recovery.export_and_publish_binary_recovery_completion(
            source, destination,
            deployment_config=deployment_config,
            repository_root=tmp_path,
            deployment_binding=binding,
            smoke_inputs=torch.zeros(1, 4))

    assert swapped
    assert not (tmp_path / 'student-export.json').exists()


def test_completion_publisher_rejects_missing_checkpoint(tmp_path):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    source = tmp_path / 'wrapper.pth'
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    binding = binary_recovery.build_config_binding(
        tmp_path, Path('deploy.py'))
    report = binary_recovery.export_student_checkpoint(
        source, destination, deployment_config=deployment_config,
        deployment_repository_root=tmp_path,
        deployment_binding=binding, smoke_inputs=torch.zeros(1, 4))
    destination.unlink()

    with pytest.raises(ValueError, match='checkpoint'):
        binary_recovery.publish_binary_recovery_completion(
            tmp_path, report, repository_root=tmp_path,
            deployment_binding=binding,
            smoke_inputs=torch.zeros(1, 4))

    assert not (tmp_path / 'student-export.json').exists()


def test_completion_loader_revalidates_live_export_and_is_idempotent(tmp_path):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    source = tmp_path / 'wrapper.pth'
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    binding = binary_recovery.build_config_binding(
        tmp_path, Path('deploy.py'))
    first = binary_recovery.export_and_publish_binary_recovery_completion(
        source, destination, deployment_config=deployment_config,
        repository_root=tmp_path, deployment_binding=binding,
        smoke_inputs=torch.zeros(1, 4))
    loaded = binary_recovery.load_binary_recovery_completion(
        tmp_path, repository_root=tmp_path, deployment_binding=binding,
        smoke_inputs=torch.zeros(1, 4))
    source.unlink()
    second = binary_recovery.export_and_publish_binary_recovery_completion(
        source, destination, deployment_config=deployment_config,
        repository_root=tmp_path, deployment_binding=binding,
        smoke_inputs=torch.zeros(1, 4))

    assert isinstance(first, binary_recovery.BinaryRecoveryCompletion)
    assert loaded == first
    assert second == first
    assert first.report['transaction_token'] == first.transaction_token
    document = json.loads((tmp_path / 'student-export.json').read_text())
    assert document == first.to_dict()

    attacker = tmp_path / 'attacker.pth'
    torch.save({'state_dict': {'attacker': torch.ones(1)}}, attacker)
    os.replace(attacker, destination)
    with pytest.raises(ValueError, match='checkpoint'):
        binary_recovery.load_binary_recovery_completion(
            tmp_path, repository_root=tmp_path,
            deployment_binding=binding,
            smoke_inputs=torch.zeros(1, 4))


@pytest.mark.parametrize('commit_succeeds', [True, False])
def test_completion_reader_waits_for_inflight_publisher_before_absence_check(
        tmp_path, monkeypatch, commit_succeeds):
    from mambapose_opt import binary_recovery

    MODELS.register_module(
        name='ToyBinaryQKPose', module=_ToyBinaryQKPose, force=True)
    reference = _ToyBinaryQKPose(
        head={'tokenpose_cfg': {'qk_mode': 'binary_scaled'}})
    source = tmp_path / 'wrapper.pth'
    torch.save({
        'state_dict': {
            f'student.{key}': value
            for key, value in reference.state_dict().items()
        }}, source)
    destination = tmp_path / 'student.pth'
    deployment_config = tmp_path / 'deploy.py'
    deployment_config.write_text(
        "model = dict(type='ToyBinaryQKPose', "
        "head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')))\n",
        encoding='utf-8')
    binding = binary_recovery.build_config_binding(
        tmp_path, Path('deploy.py'))
    publisher_paused = threading.Event()
    release_publisher = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    publisher_values = []
    publisher_errors = []
    reader_values = []
    reader_errors = []
    real_atomic_write = binary_recovery.atomic_write_json

    def pause_before_completion_commit(path, value):
        if Path(path).name == 'student-export.json':
            publisher_paused.set()
            if not release_publisher.wait(5):
                raise RuntimeError('test publisher release timed out')
            if not commit_succeeds:
                raise OSError('injected completion commit failure')
        return real_atomic_write(path, value)

    monkeypatch.setattr(
        binary_recovery, 'atomic_write_json',
        pause_before_completion_commit)

    def publish():
        try:
            publisher_values.append(
                binary_recovery.export_and_publish_binary_recovery_completion(
                    source, destination,
                    deployment_config=deployment_config,
                    repository_root=tmp_path,
                    deployment_binding=binding,
                    smoke_inputs=torch.zeros(1, 4)))
        except Exception as error:
            publisher_errors.append(error)

    def read():
        reader_started.set()
        try:
            reader_values.append(
                binary_recovery.load_binary_recovery_completion(
                    tmp_path, repository_root=tmp_path,
                    deployment_binding=binding,
                    smoke_inputs=torch.zeros(1, 4)))
        except Exception as error:
            reader_errors.append(error)
        finally:
            reader_done.set()

    publisher_thread = threading.Thread(target=publish)
    reader_thread = threading.Thread(target=read)
    publisher_thread.start()
    assert publisher_paused.wait(5)
    reader_thread.start()
    assert reader_started.wait(5)
    try:
        assert not reader_done.wait(0.25)
    finally:
        release_publisher.set()
        publisher_thread.join(5)
        reader_thread.join(5)

    assert not publisher_thread.is_alive()
    assert not reader_thread.is_alive()
    assert not reader_errors
    if commit_succeeds:
        assert not publisher_errors
        assert len(publisher_values) == 1
        assert reader_values == publisher_values
    else:
        assert len(publisher_errors) == 1
        assert not publisher_values
        assert reader_values == [None]
        assert not (tmp_path / 'student-export.json').exists()


@pytest.mark.parametrize(('path', 'value'), [
    (('base_training_epochs',), 301),
    (('loss', 'distillation_weight'), 0.5),
    (('loss', 'foreign'), True),
    (('schedule', 'fraction_of_base_training'), 0.3),
    (('schedule', 'foreign'), True),
    (('optional_extensions', 'relative_attention_bias', 'enabled'), True),
    (('optional_extensions', 'foreign'), {}),
])
def test_recovery_readiness_nested_contract_is_exact_and_config_bound(
        monkeypatch, path, value):
    from mambapose_opt import binary_recovery

    document = json.loads((REPOSITORY_ROOT
                           / 'optimization/binary_qk_recovery.json').read_text(
                               encoding='utf-8'))
    cursor = document
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    monkeypatch.setattr(
        binary_recovery.json, 'loads', lambda _text: document)
    monkeypatch.setattr(
        binary_recovery, 'file_sha256',
        lambda candidate: (
            binary_recovery.CHECKPOINT_SHA256
            if candidate.name.endswith('.pth')
            else hashlib.sha256(candidate.read_bytes()).hexdigest()))

    with pytest.raises(ValueError, match='readiness'):
        binary_recovery.load_binary_recovery_readiness(REPOSITORY_ROOT)
