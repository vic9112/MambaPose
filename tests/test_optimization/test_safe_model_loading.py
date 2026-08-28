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


def _authorized_repo(
        tmp_path, payload,
        config_text='model = dict(type="Fixture")\n'):
    _git('init', '-q', cwd=tmp_path)
    (tmp_path / 'data').mkdir()
    checkpoint = tmp_path / 'work_dirs/reproduction/model.pth'
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)
    config = tmp_path / 'configs/model.py'
    config.parent.mkdir()
    config.write_text(config_text, encoding='utf-8')
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
    from mambapose_opt.checkpoints import (
        authorize_tracked_config, build_manifest_authorized_model)

    manifest, checkpoint = _authorized_repo(
        tmp_path, {
            'state_dict': {
                'weight': torch.tensor([1.5, -2.0], dtype=torch.float32),
                'counter': torch.tensor([7], dtype=torch.int64),
            },
        },
        'model = dict(\n'
        '    type="Fixture", pretrained="implicit.pth",\n'
        '    init_cfg=dict(type="Pretrained", checkpoint="implicit.pth"),\n'
        '    nested=(dict(pretrained="tuple.pth"),\n'
        '            [dict(init_cfg=dict(type="Pretrained"))]))\n')
    captured = {}

    def init_model(config, checkpoint_value, *, device):
        captured['config'] = config
        captured['checkpoint'] = checkpoint_value
        captured['device'] = device
        return _ToyModel()

    monkeypatch.setattr('mmpose.apis.init_model', init_model)
    authority = authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    model = build_manifest_authorized_model(
        tmp_path, manifest, 'pwl-silu-s-v1',
        config_authority=authority, device='cpu')

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
    from mambapose_opt.checkpoints import (
        authorize_tracked_config, build_manifest_authorized_model)

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

    authority = authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    with pytest.raises(ValueError, match='weights-only|restricted|checkpoint'):
        build_manifest_authorized_model(
            tmp_path, manifest, 'pwl-silu-s-v1',
            config_authority=authority, device='cpu')

    assert not marker.exists()


def test_manifest_authorized_builder_rejects_naked_alternate_config(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import build_manifest_authorized_model

    manifest, _checkpoint = _authorized_repo(tmp_path, {
        'state_dict': {
            'weight': torch.tensor([1.0, 2.0], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
        },
    })
    monkeypatch.setattr(
        'mmpose.apis.init_model', lambda *_args, **_kwargs: _ToyModel())
    alternate = Config(dict(model=dict(
        type='Fixture', authority_tag='alternate')))

    with pytest.raises(TypeError, match='config_authority|unexpected'):
        build_manifest_authorized_model(
            tmp_path, manifest, 'pwl-silu-s-v1',
            config=alternate, device='cpu')


def test_config_authority_returns_reconstructed_config_not_mutable_state(tmp_path):
    from mambapose_opt.checkpoints import authorize_tracked_config

    manifest, _checkpoint = _authorized_repo(tmp_path, {
        'state_dict': {
            'weight': torch.tensor([1.0, 2.0], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
        },
    })
    authority = authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    config = authority.load_config()
    config.model.type = 'Alternate'

    assert authority.load_config().model.type == 'Fixture'
    authority.verify()


def test_public_config_authority_constructor_rejects_forgery(tmp_path):
    import mambapose_opt.checkpoints as checkpoints

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {
            'state_dict': {
                'weight': torch.tensor([1.0, 2.0], dtype=torch.float32),
                'counter': torch.tensor([3], dtype=torch.int64),
            },
        }, 'model = dict(type="Fixture", authority_tag="tracked")\n')
    authorized = checkpoints.authorize_manifest_candidate(
        tmp_path, manifest, 'pwl-silu-s-v1')
    alternate = Config(dict(model=dict(
        type='Fixture', authority_tag='alternate')))
    with pytest.raises(TypeError, match='created by an authorizer'):
        forged = checkpoints.ConfigAuthority(
            object(),
            repository_root=tmp_path, manifest_path=manifest,
            candidate=authorized.candidate, path=authorized.config_path,
            sha256='0' * 64, config=alternate, verify=lambda: None,
            reference={'kind': 'forged'})


def test_builder_rejects_compound_mutation_of_authority_locator(
        tmp_path, monkeypatch):
    import mambapose_opt.checkpoints as checkpoints

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {
            'state_dict': {
                'weight': torch.tensor([1.0, 2.0], dtype=torch.float32),
                'counter': torch.tensor([3], dtype=torch.int64),
            },
        }, 'model = dict(type="Fixture", authority_tag="tracked")\n')
    authority = checkpoints.authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    alternate = Config(dict(model=dict(
        type='Fixture', authority_tag='alternate')))
    mutations = {
        '_config': alternate,
        '_config_sha256': checkpoints.ConfigAuthority._config_fingerprint(
            alternate),
        '_verify_callback': lambda: None,
        '_reference': {'kind': 'forged'},
        '_seal': object(),
    }
    for name, value in mutations.items():
        try:
            object.__setattr__(authority, name, value)
        except (AttributeError, TypeError):
            pass
    captured = {}
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda config, _checkpoint, *, device: (
            captured.update(tag=config.model.authority_tag) or _ToyModel()))

    with pytest.raises(ValueError, match='locator differs'):
        checkpoints.build_manifest_authorized_model(
            tmp_path, manifest, 'pwl-silu-s-v1',
            config_authority=authority, device='cpu')

    assert captured == {}


def test_builder_rejects_caller_selected_noncanonical_stage_locators(tmp_path):
    import mambapose_opt.checkpoints as checkpoints

    manifest, _checkpoint = _authorized_repo(tmp_path, {
        'state_dict': {
            'weight': torch.tensor([1.0, 2.0], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
        },
    })
    authority = checkpoints.authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    candidate = authority.candidate
    candidate_root = (
        tmp_path / 'work_dirs/optimization' / candidate.route /
        candidate.id / str(candidate.seed))

    with pytest.raises(ValueError, match='does not identify a model stage'):
        checkpoints.build_manifest_authorized_model(
            tmp_path, manifest, candidate, config_authority=authority,
            downstream_output=(candidate_root / 'alternate/profile.json'))

    with pytest.raises(ValueError, match='not canonical for stage'):
        checkpoints.build_manifest_authorized_model(
            tmp_path, manifest, candidate, config_authority=authority,
            materialized_config_path=(tmp_path / 'alternate/resolved-flip.py'),
            materialized_authority_path=(
                tmp_path / 'alternate/resolved-flip.config-authority.json'))


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
    authority_path = tmp_path / 'config-authority.json'
    authority_path.write_text('{}\n', encoding='utf-8')

    class Authority:
        path = config_path.resolve()

        def load_config(self):
            return Config(dict(model=dict(type='Fixture')))

        def verify(self):
            captured['verified'] = captured.get('verified', 0) + 1

    authority = Authority()

    monkeypatch.setattr(
        tool.Config, 'fromfile',
        lambda _path: Config(dict(model=dict(type='Fixture'))))
    monkeypatch.setattr(
        checkpoints, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(checkpoint_path=checkpoint.resolve()))
    monkeypatch.setattr(
        checkpoints, 'load_materialized_config_authority',
        lambda *_args: authority)
    monkeypatch.setattr(
        checkpoints, 'build_manifest_authorized_model',
        lambda repository_root, manifest_path, candidate, *,
        config_authority, materialized_authority_path,
        materialized_config_path, device:
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
        '--safe-candidate', 'pwl-silu-s-v1',
        '--safe-config-authority', str(authority_path)])

    tool.main()

    assert captured == {
        'load_from': None, 'model': model, 'tested': True, 'verified': 1}


def test_test_cli_postverifies_materialized_config_after_runner(
        tmp_path, monkeypatch):
    import tools.test as tool
    import mambapose_opt.checkpoints as checkpoints

    config_path = tmp_path / 'resolved.py'
    config_path.write_text('model = dict(type="Fixture")\n')
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'fixture')
    manifest = tmp_path / 'candidates.json'
    manifest.write_text('{}\n')
    authority_path = tmp_path / 'authority.json'
    authority_path.write_text('{}\n')

    class Authority:
        path = config_path.resolve()

        def load_config(self):
            return Config(dict(model=dict(type='Fixture')))

        def verify(self):
            if config_path.read_text() != 'model = dict(type="Fixture")\n':
                raise ValueError('materialized evaluation config changed')

    authority = Authority()
    monkeypatch.setattr(
        checkpoints, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(checkpoint_path=checkpoint.resolve()))
    monkeypatch.setattr(
        checkpoints, 'load_materialized_config_authority',
        lambda *_args: authority)
    monkeypatch.setattr(
        checkpoints, 'build_manifest_authorized_model',
        lambda *_args, **_kwargs: _ToyModel())

    class FakeRunner:
        def register_hook(self, *_args, **_kwargs):
            raise AssertionError('no output hook expected')

        def test(self):
            config_path.write_text(
                'model = dict(type="Alternate")\n', encoding='utf-8')

    monkeypatch.setattr(tool.Runner, 'from_cfg', lambda _cfg: FakeRunner())
    monkeypatch.setattr(sys, 'argv', [
        'tools/test.py', str(config_path), str(checkpoint),
        '--safe-manifest', str(manifest),
        '--safe-candidate', 'pwl-silu-s-v1',
        '--safe-config-authority', str(authority_path)])

    with pytest.raises(ValueError, match='materialized.*changed'):
        tool.main()


def test_test_cli_rejects_cfg_options_in_safe_authority_mode(
        tmp_path, monkeypatch):
    import tools.test as tool

    config_path = tmp_path / 'resolved.py'
    checkpoint = tmp_path / 'model.pth'
    manifest = tmp_path / 'candidates.json'
    authority = tmp_path / 'authority.json'
    for path in (config_path, checkpoint, manifest, authority):
        path.write_text('{}\n', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'tools/test.py', str(config_path), str(checkpoint),
        '--safe-manifest', str(manifest),
        '--safe-candidate', 'pwl-silu-s-v1',
        '--safe-config-authority', str(authority),
        '--cfg-options', 'model.authority_tag="alternate"'])

    with pytest.raises(ValueError, match='cfg-options'):
        tool.main()
