import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch
from torch import nn

from tools.optimization.calibrate_numeric import (
    _w8a8_activation_scales_from_records)


class SS2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, 8)
        self.out_proj = nn.Linear(4, 4)
        self.x_proj_weight = nn.Parameter(torch.ones(4, 2, 4))
        self.dt_projs_weight = nn.Parameter(torch.ones(4, 4, 2))
        self.dt_projs_bias = nn.Parameter(torch.ones(4, 4))
        self.A_logs = nn.Parameter(torch.zeros(16, 2))
        self.Ds = nn.Parameter(torch.ones(16))
        self.numeric_observer = None

    def set_numeric_observer(self, callback):
        self.numeric_observer = callback

    def forward(self, value):
        projected = self.in_proj(value).chunk(2, dim=-1)[0]
        return self.out_proj(projected)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(4, 12, bias=False)

    def forward(self, value):
        return self.to_qkv(value)


class PoseInteraction(nn.Identity):
    pass


class CalibrationFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList([SS2D()])
        self.attentions = nn.ModuleList([Attention() for _ in range(6)])
        self.pose_interaction = PoseInteraction()
        self.heatmap_projection = nn.Linear(4, 17)


def _activation_record(*, numeric_range=(-4.0, 2.0), underflow=7,
                       overflow=0, bounded=False):
    return {
        'granularity': 'tensor',
        'range': list(numeric_range),
        'underflow_count': underflow,
        'overflow_count': overflow,
        'percentile_bound_valid': bounded,
    }


def _root_determinism(seed=0):
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


def _schema_artifact(*, version, include_root):
    record = {
        'granularity': 'tensor', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 1.0,
        'range': [-1.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 1.0, '0.99': 1.0,
                        '0.999': 1.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }
    protocol = {
        'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
        'worker_count': 0, 'sample_count': 2,
        'sample_order_sha256': 'a' * 64,
    }
    if include_root:
        protocol['root_determinism'] = _root_determinism()
    return {
        'schema_version': version,
        'candidate_id': 'full-s-v1',
        'stage': 'calibrate',
        'source': {},
        'identity': _valid_identity(),
        'protocol': protocol,
        'hooks': {
            'records': {'attention.0.q': record},
            'required_records': ['attention.0.q'],
            'unsupported_internals': [],
        },
    }


def test_w8a8_max_abs_scale_accepts_histogram_underflow_without_overflow():
    scales = _w8a8_activation_scales_from_records(
        {'projection': 'projection.input'},
        {'projection.input': _activation_record()})

    assert scales == {
        'projection': {
            'source_record': 'projection.input',
            'granularity': 'tensor',
            'scale': 4.0 / 127.0,
        },
    }


@pytest.mark.parametrize(
    ('record', 'message'), (
        (_activation_record(numeric_range=(float('nan'), 2.0)),
         'finite measured range'),
        (_activation_record(numeric_range=(0.0, 0.0)),
         'positive max-abs scale'),
        (_activation_record(overflow=1), 'histogram overflow'),
    ))
def test_w8a8_max_abs_scale_rejects_unsafe_measured_ranges(record, message):
    with pytest.raises(ValueError, match=message):
        _w8a8_activation_scales_from_records(
            {'projection': 'projection.input'},
            {'projection.input': record})


def test_w8a8_max_abs_scale_rejects_wrong_source_record():
    with pytest.raises(ValueError, match='record is missing'):
        _w8a8_activation_scales_from_records(
            {'projection': 'projection.input'},
            {'different.input': _activation_record()})


@pytest.mark.parametrize(
    ('record', 'scale', 'message'), (
        (_activation_record(overflow=1), 4.0 / 127.0,
         'histogram overflow'),
        (_activation_record(numeric_range=(0.0, 0.0)), 1.0,
         'positive measured maximum'),
        (_activation_record(), 0.0, 'measured range'),
        (_activation_record(), -1.0, 'measured range'),
    ))
def test_w8a8_activation_scale_artifact_rejects_unsafe_scale_contract(
        record, scale, message):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_activation_scale_record)

    with pytest.raises(CalibrationContractError, match=message):
        validate_activation_scale_record(
            'projection', {
                'source_record': 'projection.input',
                'granularity': 'tensor',
                'scale': scale,
            }, record)


def test_calibration_inventory_fails_closed_on_missing_or_duplicate_roles():
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, discover_calibration_targets)

    model = CalibrationFixture()
    discovered = discover_calibration_targets(model)
    assert len(discovered.attention_qkv) == 6
    assert discovered.ss2d_boundaries == ('backbone.0',)
    assert set(discovered.transition_parameters) == {
        'backbone.0.A_logs', 'backbone.0.Ds',
        'backbone.0.dt_projs_bias', 'backbone.0.dt_projs_weight',
        'backbone.0.x_proj_weight',
    }
    assert discovered.functional_observers == ('backbone.0',)
    assert discovered.unsupported_internals == (
        'backbone.0.selective_scan_internal_state:opaque_cuda_kernel',)

    del model.attentions[-1]
    with pytest.raises(CalibrationContractError, match='six Transformer'):
        discover_calibration_targets(model)


def test_calibration_identity_is_source_checkpoint_policy_and_train_bound(
        tmp_path):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, calibration_identity)

    repo = tmp_path
    (repo / 'configs/reproduction').mkdir(parents=True)
    (repo / 'checkpoints').mkdir()
    (repo / 'data/coco/annotations').mkdir(parents=True)
    (repo / 'data/coco/train2017').mkdir(parents=True)
    config = repo / 'configs/reproduction/coco_s_v1.py'
    checkpoint = repo / 'checkpoints/full.pth'
    annotation = repo / 'data/coco/annotations/person_keypoints_train2017.json'
    policy = repo / 'configs/policy.py'
    config.write_text('full S-V1')
    checkpoint.write_bytes(b'checkpoint')
    annotation.write_text('{}')
    (repo / 'data/coco/train2017/000000000001.jpg').write_bytes(b'image')
    policy.parent.mkdir(exist_ok=True)
    policy.write_text('numeric policy')
    downloads = repo / 'work_dirs/reproduction/downloads'
    downloads.mkdir(parents=True)
    train_zip = downloads / 'train2017.zip'
    annotation_zip = downloads / 'annotations_trainval2017.zip'
    with zipfile.ZipFile(train_zip, 'w') as archive:
        archive.writestr('train2017/000000000001.jpg', b'image')
    with zipfile.ZipFile(annotation_zip, 'w') as archive:
        archive.writestr(
            'annotations/person_keypoints_train2017.json', b'{}')
    inventory = repo / 'data/inventory.json'
    inventory.write_text(json.dumps({
        'schema_version': 1,
        'assets': [
            {'id': 'coco-train2017',
             'path': 'work_dirs/reproduction/downloads/train2017.zip',
             'sha256': hashlib.sha256(train_zip.read_bytes()).hexdigest()},
            {'id': 'coco-annotations',
             'path': ('work_dirs/reproduction/downloads/'
                      'annotations_trainval2017.zip'),
             'sha256': hashlib.sha256(annotation_zip.read_bytes()).hexdigest()},
        ],
    }))

    identity = calibration_identity(
        repository_root=repo,
        candidate_id='full-s-v1',
        config=Path('configs/reproduction/coco_s_v1.py'),
        checkpoint=Path('checkpoints/full.pth'),
        expected_checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),
        policy=Path('configs/policy.py'),
        split='train2017',
        annotation=Path(
            'data/coco/annotations/person_keypoints_train2017.json'),
        image_prefix=Path('data/coco/train2017'))

    assert identity['split'] == 'train2017'
    assert identity['config_sha256'] == hashlib.sha256(b'full S-V1').hexdigest()
    assert identity['policy_sha256'] == hashlib.sha256(
        b'numeric policy').hexdigest()
    assert identity['dataset']['annotation_sha256'] == hashlib.sha256(
        b'{}').hexdigest()
    assert identity['dataset']['image_count'] == 1
    assert identity['dataset']['train_archive_sha256'] == hashlib.sha256(
        train_zip.read_bytes()).hexdigest()
    assert identity['dataset']['annotation_member_sha256'] == hashlib.sha256(
        b'{}').hexdigest()

    (repo / 'data/coco/train2017/000000000001.jpg').write_bytes(b'mutated')
    with pytest.raises(CalibrationContractError, match='content'):
        calibration_identity(
            repository_root=repo, candidate_id='full-s-v1',
            config=Path('configs/reproduction/coco_s_v1.py'),
            checkpoint=Path('checkpoints/full.pth'),
            expected_checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),
            policy=Path('configs/policy.py'), split='train2017',
            annotation=Path(
                'data/coco/annotations/person_keypoints_train2017.json'),
            image_prefix=Path('data/coco/train2017'))

    (repo / 'data/coco/train2017/000000000001.jpg').write_bytes(b'image')
    config_bytes = config.read_bytes()
    config.unlink()
    (config.parent / 'actual.py').write_bytes(config_bytes)
    config.symlink_to(config.parent / 'actual.py')
    with pytest.raises(CalibrationContractError, match='symlink'):
        calibration_identity(
            repository_root=repo, candidate_id='full-s-v1',
            config=Path('configs/reproduction/coco_s_v1.py'),
            checkpoint=Path('checkpoints/full.pth'),
            expected_checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),
            policy=Path('configs/policy.py'), split='train2017',
            annotation=Path(
                'data/coco/annotations/person_keypoints_train2017.json'),
            image_prefix=Path('data/coco/train2017'))
    config.unlink()
    config.write_bytes(config_bytes)

    with pytest.raises(CalibrationContractError, match='train2017'):
        calibration_identity(
            repository_root=repo, candidate_id='full-s-v1',
            config=Path('configs/reproduction/coco_s_v1.py'),
            checkpoint=Path('checkpoints/full.pth'),
            expected_checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),
            policy=Path('configs/policy.py'), split='val2017',
            annotation=Path(
                'data/coco/annotations/person_keypoints_train2017.json'),
            image_prefix=Path('data/coco/train2017'))


def test_calibration_artifact_requires_deterministic_order_and_exact_hooks():
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_artifact)

    record = {
        'granularity': 'tensor', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 1.0,
        'range': [-1.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 1.0, '0.99': 1.0,
                        '0.999': 1.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }
    artifact = {
        'schema_version': 2,
        'candidate_id': 'full-s-v1',
        'stage': 'calibrate',
        'source': {},
        'identity': _valid_identity(),
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 2,
            'sample_order_sha256': 'a' * 64,
            'root_determinism': _root_determinism(),
        },
        'hooks': {'records': {'attention.0.q': record},
                  'required_records': ['attention.0.q'],
                  'unsupported_internals': []},
    }
    assert validate_calibration_artifact(artifact)['protocol']['sample_count'] == 2

    artifact['protocol']['shuffle'] = True
    with pytest.raises(CalibrationContractError, match='shuffle'):
        validate_calibration_artifact(artifact)
    artifact['protocol']['shuffle'] = False
    artifact['hooks']['required_records'].append('attention.0.q')
    with pytest.raises(CalibrationContractError, match='duplicated'):
        validate_calibration_artifact(artifact)

    artifact['hooks']['required_records'] = ['attention.0.q']
    artifact['protocol']['root_determinism']['torch_seed'] = 1
    with pytest.raises(CalibrationContractError, match='root determinism'):
        validate_calibration_artifact(artifact)


def test_calibration_schema_rejects_strip_downgrade_and_unknown_version():
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_artifact)

    valid = _schema_artifact(version=2, include_root=True)
    assert validate_calibration_artifact(valid) is valid

    stripped = copy.deepcopy(valid)
    stripped['protocol'].pop('root_determinism')
    with pytest.raises(CalibrationContractError, match='schema v2'):
        validate_calibration_artifact(stripped)

    downgraded = copy.deepcopy(valid)
    downgraded['schema_version'] = 1
    with pytest.raises(CalibrationContractError, match='schema v1'):
        validate_calibration_artifact(downgraded)

    pwl = copy.deepcopy(valid)
    pwl['schema_version'] = 3
    pwl['candidate_id'] = 'pwl-silu-s-v1'
    from mambapose_opt.pwl_artifacts import fit_pwl_observations
    pwl['pwl_fit'] = fit_pwl_observations(
        candidate_id='pwl-silu-s-v1',
        policy={
            'enabled_function': 'silu', 'source': 'module',
            'roles': ('block.act',), 'domain': (-2.0, 2.0),
            'segments': 4, 'grid_points': 129, 'saturation': 'clamp',
            'qat_form': 'differentiable',
            'selection_policy': 'observed-range-max-then-mean-v1',
        },
        observations={'block.act': [torch.tensor([-1.0, 0.0, 1.0])]})
    assert validate_calibration_artifact(pwl) is pwl

    for unknown_version in (4, True, [], None):
        unknown = copy.deepcopy(valid)
        unknown['schema_version'] = unknown_version
        with pytest.raises(CalibrationContractError, match='version'):
            validate_calibration_artifact(unknown)


def test_calibration_artifact_keeps_legacy_protocol_compatible():
    from mambapose_opt.numeric_calibration import validate_calibration_artifact

    record = {
        'granularity': 'tensor', 'sample_count': 1, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 1.0,
        'range': [-1.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 1.0, '0.99': 1.0,
                        '0.999': 1.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }
    artifact = {
        'schema_version': 1, 'candidate_id': 'legacy', 'stage': 'calibrate',
        'source': {}, 'identity': _valid_identity(),
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 1,
            'sample_order_sha256': 'a' * 64,
        },
        'hooks': {
            'records': {'layer.input': record},
            'required_records': ['layer.input'],
            'unsupported_internals': [],
        },
    }

    assert validate_calibration_artifact(artifact) is artifact


def test_calibration_v2_provenance_rejects_candidate_seed_downgrade(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_provenance)
    from mambapose_opt.schema import CandidateSpec

    candidate = CandidateSpec.from_dict({
        'id': 'w8a8', 'route': 'ssm-quant-pwl', 'kind': 'fake-quant',
        'config': 'configs/w8a8.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 7,
        'features': {'numeric_kind': 'w8a8'},
    })
    artifact = _schema_artifact(version=2, include_root=True)
    artifact['candidate_id'] = candidate.id
    artifact['protocol']['root_determinism'] = _root_determinism(seed=0)
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {
            'git_commit': 'b' * 40, 'policy_path': 'configs/w8a8.py',
            'policy_sha256': 'a' * 64,
            'authority_path': 'optimization/authority.json'})
    monkeypatch.setattr(
        'mambapose_opt.checkpoints.authorize_tracked_config',
        lambda *_args: SimpleNamespace(load_config=lambda: {
            'numeric_optimization': {
                'calibration': {'artifact_schema_version': 2}}}))

    with pytest.raises(CalibrationContractError, match='candidate'):
        validate_calibration_provenance(
            artifact, expected_candidate=candidate,
            repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_calibration_provenance_rejects_combined_strip_and_v1_downgrade(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_provenance)
    from mambapose_opt.schema import CandidateSpec

    candidate = CandidateSpec.from_dict({
        'id': 'w8a8', 'route': 'ssm-quant-pwl', 'kind': 'fake-quant',
        'config': 'configs/w8a8.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'w8a8'},
    })
    downgraded = _schema_artifact(version=1, include_root=False)
    downgraded['candidate_id'] = candidate.id
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {
            'git_commit': 'b' * 40, 'policy_path': 'configs/w8a8.py',
            'policy_sha256': 'a' * 64,
            'authority_path': 'optimization/authority.json'})
    monkeypatch.setattr(
        'mambapose_opt.checkpoints.authorize_tracked_config',
        lambda *_args: SimpleNamespace(load_config=lambda: {
            'numeric_optimization': {
                'calibration': {'artifact_schema_version': 2}}}))

    with pytest.raises(CalibrationContractError, match='source policy'):
        validate_calibration_provenance(
            downgraded, expected_candidate=candidate,
            repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_calibration_provenance_rejects_tracked_alternate_policy_downgrade(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_provenance)
    from mambapose_opt.schema import CandidateSpec

    candidate = CandidateSpec.from_dict({
        'id': 'numeric-observer-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'fake-quant',
        'config': 'configs/optimization/numeric/observer_only.py',
        'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'observer'},
    })
    alternate_policy = 'configs/reproduction/coco_s_v1.py'
    rebound = _schema_artifact(version=1, include_root=False)
    rebound['candidate_id'] = candidate.id
    rebound['identity']['policy'] = alternate_policy
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {
            'git_commit': 'b' * 40, 'policy_path': alternate_policy,
            'policy_sha256': 'a' * 64,
            'authority_path': 'optimization/authority.json'})
    monkeypatch.setattr(
        'mmengine.config.Config.fromfile',
        lambda _path: {'numeric_optimization': {'calibration': {}}})

    with pytest.raises(CalibrationContractError,
                       match='target candidate config'):
        validate_calibration_provenance(
            rebound, expected_candidate=candidate,
            repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_numeric_calibration_policies_declare_schema_v2():
    from mmengine.config import Config

    for path in (
            'configs/optimization/numeric/observer_only.py',
            'configs/optimization/numeric/w8a8.py'):
        config = Config.fromfile(path)
        assert config.numeric_optimization.calibration[
            'artifact_schema_version'] == 2


def test_calibrate_seeds_candidate_before_model_and_worker_zero_loader(
        tmp_path, monkeypatch):
    import tools.optimization.calibrate_numeric as tool
    from mambapose_opt.determinism import seed_deterministic_root

    source = SimpleNamespace(
        id='full-s-v1', seed=7,
        config=Path('configs/reproduction/coco_s_v1.py'))
    target = SimpleNamespace(
        id='pwl-silu-s-v1', seed=7, config=Path('policy.py'),
        features={'numeric_kind': 'pwl'})
    authorized = SimpleNamespace(
        candidate=source, config_path=tmp_path / 'config.py',
        checkpoint_path=tmp_path / 'checkpoint.pth')
    config = SimpleNamespace(
        train_dataloader={},
        numeric_optimization={'quant_policy': {'activation_observers': {}}})
    calls = []
    identity_candidates = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def test_step(self, _batch):
            return None

    class Session:
        def __init__(self, _model, _targets):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def records(self):
            return {
                'layer.input': {
                    'granularity': 'tensor', 'sample_count': 1,
                    'zero_count': 0, 'underflow_count': 0,
                    'overflow_count': 0, 'max_abs': 1.0,
                    'range': [-1.0, 1.0],
                    'percentiles': {
                        '0.5': 1.0, '0.9': 1.0, '0.99': 1.0,
                        '0.999': 1.0},
                    'algorithm': 'fixed-log2-histogram-v1',
                    'histogram_bins': 256,
                    'histogram_domain': [2 ** -32, 2 ** 32],
                    'percentile_bound_valid': True,
                    'relative_error_bound': 2 ** 0.25 - 1,
                    'outlier_ratio_above_p99_bin': 0.0,
                    'token_ids': None, 'observed_shape': [],
                },
            }

    def deterministic_root(seed):
        calls.append(('seed', seed))
        return seed_deterministic_root(seed)

    def init_model(*_args, **_kwargs):
        calls.append(('model', None))
        return Model()

    def build_dataloader(_config, *, seed, diff_rank_seed):
        calls.append(('loader', seed, diff_rank_seed))
        return [{'data_samples': [SimpleNamespace(img_id=11)]}]

    def identity(candidate, _policy):
        identity_candidates.append(candidate.id)
        return _valid_identity()

    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', tmp_path)
    (tmp_path / 'policy.py').write_text(
        'numeric_optimization = dict()\n', encoding='utf-8')
    monkeypatch.setattr(tool, 'authorize_manifest_candidate',
                        lambda *_args: authorized)
    monkeypatch.setattr(
        tool, 'authorize_tracked_config',
        lambda *_args: SimpleNamespace(load_config=lambda: config))
    monkeypatch.setattr(tool, '_identity', identity)
    monkeypatch.setattr(tool, 'seed_deterministic_root', deterministic_root)
    monkeypatch.setattr(tool, 'discover_calibration_targets', lambda _model:
                        SimpleNamespace(unsupported_internals=()))
    monkeypatch.setattr(tool, '_required_records',
                        lambda _targets: ('layer.input',))
    monkeypatch.setattr(tool, '_HookSession', Session)
    monkeypatch.setattr(tool, 'build_numeric_source_binding',
                        lambda **_kwargs: {})
    monkeypatch.setattr('mmengine.config.Config.fromfile',
                        lambda _path: config)
    monkeypatch.setattr(
        tool, 'build_manifest_authorized_model',
        lambda *_args, **_kwargs: init_model())
    monkeypatch.setattr('mmengine.runner.Runner.build_dataloader',
                        build_dataloader)
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        artifact = tool.calibrate(
            source, tmp_path / 'policy.py', samples=1, device='cuda:0',
            target_candidate=target,
            manifest_path=tmp_path / 'manifest.json')
    finally:
        torch.use_deterministic_algorithms(original_algorithms)
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic

    assert calls == [('seed', 7), ('model', None), ('loader', 7, False)]
    assert identity_candidates == ['full-s-v1', 'full-s-v1']
    assert artifact['candidate_id'] == 'pwl-silu-s-v1'
    assert artifact['schema_version'] == 2
    assert artifact['protocol']['root_determinism']['seed'] == 7


@pytest.mark.parametrize('operation', ('audit', 'calibrate'))
def test_calibration_producer_rejects_alternate_target_policy_before_load(
        tmp_path, monkeypatch, operation):
    import tools.optimization.calibrate_numeric as tool

    source = SimpleNamespace(
        id='full-s-v1', config=Path('configs/reproduction/coco_s_v1.py'))
    target = SimpleNamespace(
        id='numeric-observer-s-v1',
        config=Path('configs/optimization/numeric/observer_only.py'))
    alternate = tmp_path / 'configs/reproduction/coco_s_v1.py'
    expected = tmp_path / target.config
    alternate.parent.mkdir(parents=True)
    expected.parent.mkdir(parents=True)
    alternate.write_text('model = dict()\n', encoding='utf-8')
    expected.write_text('numeric_optimization = dict()\n', encoding='utf-8')
    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(
        tool, 'authorize_manifest_candidate',
        lambda *_args: (_ for _ in ()).throw(
            ValueError('source checkpoint load reached')))

    with pytest.raises(ValueError, match='target candidate config'):
        if operation == 'audit':
            tool.audit(
                source, alternate, target_candidate=target,
                manifest_path=tmp_path / 'manifest.json')
        else:
            tool.calibrate(
                source, alternate, samples=1, device='cuda:0',
                target_candidate=target,
                manifest_path=tmp_path / 'manifest.json')


def test_calibration_cli_audit_rejects_alternate_target_policy(
        tmp_path, monkeypatch, capsys):
    import tools.optimization.calibrate_numeric as tool

    source = SimpleNamespace(
        id='full-s-v1', route='baseline',
        config=Path('configs/reproduction/coco_s_v1.py'))
    target = SimpleNamespace(
        id='numeric-observer-s-v1', route='ssm-quant-pwl',
        config=Path('configs/optimization/numeric/observer_only.py'))
    alternate = tmp_path / 'configs/reproduction/coco_s_v1.py'
    expected = tmp_path / target.config
    alternate.parent.mkdir(parents=True)
    expected.parent.mkdir(parents=True)
    alternate.write_text('model = dict()\n', encoding='utf-8')
    expected.write_text('numeric_optimization = dict()\n', encoding='utf-8')
    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(
        tool, '_candidate',
        lambda _manifest, identifier: (
            source if identifier == 'full-s-v1' else target))
    monkeypatch.setattr(
        tool, 'authorize_manifest_candidate',
        lambda *_args: (_ for _ in ()).throw(
            ValueError('source checkpoint load reached')))
    monkeypatch.setattr(sys, 'argv', [
        'calibrate_numeric.py', '--candidate', target.id,
        '--manifest', str(tmp_path / 'manifest.json'),
        '--policy', str(alternate), '--audit-only'])

    with pytest.raises(SystemExit) as error:
        tool.main()

    assert error.value.code == 2
    assert 'target candidate config' in capsys.readouterr().err


def test_calibration_provenance_rejects_identity_commit_mismatch(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_calibration import (
        CalibrationContractError, validate_calibration_provenance)
    from mambapose_opt.schema import CandidateSpec

    candidate = CandidateSpec.from_dict({
        'id': 'w8a8', 'route': 'ssm-quant-pwl', 'kind': 'fake-quant',
        'config': 'configs/w8a8.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'w8a8'},
    })
    record = {
        'granularity': 'tensor', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 1.0,
        'range': [-1.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 1.0, '0.99': 1.0,
                        '0.999': 1.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }
    artifact = {
        'schema_version': 1, 'candidate_id': 'w8a8', 'stage': 'calibrate',
        'source': {}, 'identity': _valid_identity(),
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 1,
            'sample_order_sha256': 'a' * 64},
        'hooks': {'records': {'layer.input': record},
                  'required_records': ['layer.input'],
                  'unsupported_internals': []},
    }
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {
            'git_commit': 'c' * 40, 'policy_path': 'configs/w8a8.py',
            'policy_sha256': 'a' * 64})
    monkeypatch.setattr(
        'mmengine.config.Config.fromfile',
        lambda _path: {'numeric_optimization': {'calibration': {}}})

    with pytest.raises(CalibrationContractError, match='commit'):
        validate_calibration_provenance(
            artifact, expected_candidate=candidate,
            repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_ss2d_numeric_callback_is_opt_in_and_default_state_output_exact():
    from mmpose.models.backbones.Vmamba.vmamba import SS2D as ProductionSS2D

    class Cross:
        @staticmethod
        def apply(value):
            flat = value.flatten(2)
            return torch.stack((flat, flat, flat, flat), dim=1)

    class Scan:
        @staticmethod
        def apply(u, delta, A, B, C, D, delta_bias, delta_softplus,
                  _a, _b, _c):
            return u

    class Merge:
        @staticmethod
        def apply(value):
            return value.sum(dim=1)

    torch.manual_seed(9)
    module = ProductionSS2D(
        d_model=4, d_state=2, ssm_ratio=1.0, dt_rank=1,
        d_conv=1, channel_first=True, forward_type='v2').eval()
    value = torch.randn(1, 4, 2, 2)
    keys = tuple(module.state_dict())
    baseline = module.forward_corev2(
        value, SelectiveScan=Scan, CrossScan=Cross, CrossMerge=Merge)
    records = []
    module.set_numeric_observer(lambda role, tensor: records.append(
        (role, tuple(tensor.shape))))
    observed = module.forward_corev2(
        value, SelectiveScan=Scan, CrossScan=Cross, CrossMerge=Merge)

    assert tuple(module.state_dict()) == keys
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    assert {role for role, _ in records} == {
        'x_proj', 'dt_proj', 'scan_input_u', 'scan_input_dt',
        'transition_A', 'transition_B', 'transition_C', 'transition_D',
        'transition_delta_bias', 'scan_output', 'transition_exp_input',
        'transition_softplus_input',
    }


def test_calibration_cli_exposes_only_manifest_bound_inputs():
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, 'tools/optimization/calibrate_numeric.py', '--help'],
        cwd=root, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout
    assert '--candidate' in result.stdout
    assert '--policy' in result.stdout
    assert '--npy' not in result.stdout
    assert '--metadata' not in result.stdout
def _valid_identity():
    sha = 'a' * 64
    return {
        'candidate_id': 'full-s-v1',
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': sha, 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': sha, 'policy': 'policy.py',
        'policy_sha256': sha, 'split': 'train2017',
        'git_commit': 'b' * 40,
        'dataset': {
            'annotation':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_sha256': sha,
            'image_prefix': 'data/coco/train2017',
            'inventory': 'data/inventory.json', 'inventory_sha256': sha,
            'train_archive': 'downloads/train2017.zip',
            'train_archive_sha256': sha, 'image_count': 118287,
            'image_content_algorithm': 'sha256-zip-member-bytes-v1',
            'image_content_aggregate_sha256': sha,
            'image_order_algorithm':
                'sha256-zip-central-directory-order-v1',
            'image_order_sha256': sha,
            'annotation_archive': 'downloads/annotations.zip',
            'annotation_archive_sha256': sha,
            'annotation_member':
                'annotations/person_keypoints_train2017.json',
            'annotation_member_sha256': sha,
        },
    }
