import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch
from torch import nn


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

    artifact = {
        'schema_version': 1,
        'candidate_id': 'full-s-v1',
        'stage': 'calibrate',
        'identity': {'candidate_id': 'full-s-v1', 'split': 'train2017'},
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 2,
            'sample_order_sha256': 'a' * 64,
        },
        'hooks': {'records': {'attention.0.q': {'sample_count': 2}},
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
        'transition_delta_bias', 'scan_output',
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
