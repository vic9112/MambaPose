import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer('counter', torch.zeros(1, dtype=torch.int64))


def _write_marker(path, value):
    Path(path).write_text(value, encoding='utf-8')


def _git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True,
                   capture_output=True)


def _authorized_repo(tmp_path, payload):
    _git('init', '-q', cwd=tmp_path)
    (tmp_path / 'data').mkdir()
    checkpoint = tmp_path / 'work_dirs/reproduction/model.pth'
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)
    config = tmp_path / 'configs/model.py'
    config.parent.mkdir()
    config.write_text('model = dict(type="Fixture")\n', encoding='utf-8')
    authority = tmp_path / 'optimization/coco_val2017_authority.json'
    authority.parent.mkdir()
    authority.write_text('{}\n', encoding='utf-8')
    manifest = tmp_path / 'optimization/candidates.json'
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'pwl-silu-s-v1',
            'route': 'ssm-quant-pwl',
            'kind': 'pwl',
            'config': 'configs/model.py',
            'checkpoint': 'work_dirs/reproduction/model.pth',
            'checkpoint_sha256': hashlib.sha256(
                checkpoint.read_bytes()).hexdigest(),
            'seed': 0,
            'features': {
                'numeric_kind': 'pwl', 'conditional': True,
                'auto_run': False, 'pwl_function': 'silu'},
        }],
    }), encoding='utf-8')
    (tmp_path / '.gitignore').write_text(
        'work_dirs/\n', encoding='utf-8')
    _git('add', '.gitignore', 'configs/model.py',
         'optimization/candidates.json',
         'optimization/coco_val2017_authority.json', cwd=tmp_path)
    _git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
         'commit', '-qm', 'fixture', cwd=tmp_path)
    return manifest, checkpoint


def test_manifest_authorized_builder_neutralizes_initializers_and_loads_state(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import build_manifest_authorized_model

    manifest, checkpoint = _authorized_repo(tmp_path, {
        'state_dict': {
            'weight': torch.tensor([1.5, -2.0], dtype=torch.float32),
            'counter': torch.tensor([7], dtype=torch.int64),
        },
    })
    captured = {}

    def init_model(config, checkpoint_value, *, device):
        captured['config'] = config
        captured['checkpoint'] = checkpoint_value
        captured['device'] = device
        return _ToyModel()

    monkeypatch.setattr('mmpose.apis.init_model', init_model)
    config = Config(dict(model=dict(
        type='Fixture', pretrained='implicit.pth',
        init_cfg=dict(type='Pretrained', checkpoint='implicit.pth'),
        nested=(dict(pretrained='tuple.pth'),
                [dict(init_cfg=dict(type='Pretrained'))]))))

    model = build_manifest_authorized_model(
        tmp_path, manifest, 'pwl-silu-s-v1', config=config, device='cpu')

    assert captured['checkpoint'] is None
    assert captured['device'] == 'cpu'
    assert captured['config'].model.pretrained is None
    assert captured['config'].model.init_cfg is None
    assert captured['config'].model.nested[0]['pretrained'] is None
    assert captured['config'].model.nested[1][0]['init_cfg'] is None
    assert torch.equal(model.weight, torch.tensor([1.5, -2.0]))
    assert torch.equal(model.counter, torch.tensor([7]))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != '0' * 64


@pytest.mark.parametrize(
    ('state', 'message'),
    [
        ({'weight': torch.ones(2)}, 'missing'),
        ({'weight': torch.ones(2), 'counter': torch.zeros(1, dtype=torch.int64),
          'extra': torch.ones(1)}, 'unexpected'),
        ({'weight': torch.ones(3), 'counter': torch.zeros(1, dtype=torch.int64)},
         'shape'),
        ({'weight': torch.ones(2, dtype=torch.float64),
          'counter': torch.zeros(1, dtype=torch.int64)}, 'dtype'),
        ({'weight': torch.tensor([1.0, float('nan')]),
          'counter': torch.zeros(1, dtype=torch.int64)}, 'finite'),
    ],
)
def test_strict_tensor_injection_rejects_incompatible_state(state, message):
    from mambapose_opt.checkpoints import load_tensor_state_strict

    with pytest.raises(ValueError, match=message):
        load_tensor_state_strict(_ToyModel(), state)


def test_manifest_authorized_builder_rejects_hostile_pickle_without_execution(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import build_manifest_authorized_model

    marker = tmp_path / 'executed'

    class Hostile:
        def __reduce__(self):
            return _write_marker, (marker, 'executed')

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {'state_dict': {'weight': Hostile()}})
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda *_args, **_kwargs: _ToyModel())
    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    with pytest.raises(ValueError, match='weights-only|restricted|checkpoint'):
        build_manifest_authorized_model(
            tmp_path, manifest, 'pwl-silu-s-v1',
            config=Config(dict(model=dict(type='Fixture'))), device='cpu')

    assert not marker.exists()


def test_test_cli_injects_safe_model_and_disables_runner_checkpoint(
        tmp_path, monkeypatch):
    import tools.test as tool
    import mambapose_opt.checkpoints as checkpoints

    config_path = tmp_path / 'config.py'
    config_path.write_text('model = dict(type="Fixture")\n', encoding='utf-8')
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'fixture')
    manifest = tmp_path / 'candidates.json'
    manifest.write_text('{}\n', encoding='utf-8')
    model = _ToyModel()
    captured = {}

    monkeypatch.setattr(
        tool.Config, 'fromfile',
        lambda _path: Config(dict(model=dict(type='Fixture'))))
    monkeypatch.setattr(
        checkpoints, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(checkpoint_path=checkpoint.resolve()))
    monkeypatch.setattr(
        checkpoints, 'build_manifest_authorized_model',
        lambda repository_root, manifest_path, candidate, *, config, device:
        model)

    class FakeRunner:
        def register_hook(self, *_args, **_kwargs):
            raise AssertionError('no output hook expected')

        def test(self):
            captured['tested'] = True

    def from_cfg(config):
        captured['load_from'] = config.load_from
        captured['model'] = config.model
        return FakeRunner()

    monkeypatch.setattr(tool.Runner, 'from_cfg', from_cfg)
    monkeypatch.setattr(sys, 'argv', [
        'tools/test.py', str(config_path), str(checkpoint),
        '--safe-manifest', str(manifest),
        '--safe-candidate', 'pwl-silu-s-v1'])

    tool.main()

    assert captured == {
        'load_from': None, 'model': model, 'tested': True}
