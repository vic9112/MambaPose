import hashlib
import json
import os
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


def _install_tracked_parse_window_swap(monkeypatch, source_path):
    original = Config.fromfile
    authorized = source_path.read_bytes()
    alternate = authorized.replace(
        b'authority_tag="tracked"', b'authority_tag="alternate"')
    assert alternate != authorized
    observed = []

    def swapped(path, *args, **kwargs):
        path = Path(path)
        if path.resolve() != source_path.resolve():
            return original(path, *args, **kwargs)
        path.write_bytes(alternate)
        try:
            value = original(path, *args, **kwargs)
            observed.append(value.model.authority_tag)
            return value
        finally:
            path.write_bytes(authorized)

    monkeypatch.setattr(Config, 'fromfile', swapped)
    return authorized, observed


def test_tracked_authority_never_parses_mutable_source_path(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import authorize_tracked_config

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {'state_dict': {'unused': torch.ones(1)}},
        'model = dict(type="Fixture", authority_tag="tracked")\n')
    authority = authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    source = tmp_path / 'configs/model.py'
    authorized, observed = _install_tracked_parse_window_swap(
        monkeypatch, source)

    loaded = authority.load_config()

    assert loaded.model.authority_tag == 'tracked'
    assert source.read_bytes() == authorized
    assert observed == []


def test_builder_never_parses_mutable_tracked_source_path(
        tmp_path, monkeypatch):
    import mambapose_opt.checkpoints as checkpoints

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {
            'state_dict': {
                'weight': torch.tensor([3.0, 4.0], dtype=torch.float32),
                'counter': torch.tensor([5], dtype=torch.int64),
            },
        }, 'model = dict(type="Fixture", authority_tag="tracked")\n')
    authority = checkpoints.authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1')
    source = tmp_path / 'configs/model.py'
    authorized, observed = _install_tracked_parse_window_swap(
        monkeypatch, source)
    captured = {}
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda config, checkpoint, *, device: (
            captured.update(
                tag=config.model.authority_tag,
                checkpoint=checkpoint,
                device=device) or _ToyModel()))

    model = checkpoints.build_manifest_authorized_model(
        tmp_path, manifest, 'pwl-silu-s-v1',
        config_authority=authority, device='cpu')

    assert captured == {
        'tag': 'tracked', 'checkpoint': None, 'device': 'cpu'}
    assert torch.equal(model.weight, torch.tensor([3.0, 4.0]))
    assert source.read_bytes() == authorized
    assert observed == []


def test_tracked_authority_preserves_inherited_config_in_private_tree(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import authorize_tracked_config

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {'state_dict': {'unused': torch.ones(1)}})
    base = tmp_path / 'configs/base.py'
    base.write_text(
        'model = dict(type="Fixture", inherited=True)\n', encoding='utf-8')
    source = tmp_path / 'configs/model.py'
    source.write_text(
        '_base_ = "base.py"\nmodel = dict(authority_tag="tracked")\n',
        encoding='utf-8')
    _git('add', 'configs/base.py', 'configs/model.py', cwd=tmp_path)
    _git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
         'commit', '-qm', 'add inherited config', cwd=tmp_path)
    original = Config.fromfile
    parsed = []

    def observe_private_tree(path, *args, **kwargs):
        path = Path(path)
        assert path.resolve() != source.resolve()
        private_root = next(
            parent for parent in path.parents
            if parent.name.startswith('mambapose-config-'))
        parsed.append(private_root)
        assert not path.is_symlink()
        assert os.stat(path).st_mode & 0o777 == 0o400
        assert os.stat(private_root).st_mode & 0o777 == 0o500
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Config, 'fromfile', observe_private_tree)

    config = authorize_tracked_config(
        tmp_path, manifest, 'pwl-silu-s-v1').load_config()

    assert config.model.type == 'Fixture'
    assert config.model.inherited is True
    assert config.model.authority_tag == 'tracked'
    assert parsed
    assert all(not path.exists() for path in parsed)


def test_tracked_authority_rejects_lazy_temp_path_backed_config(
        tmp_path, monkeypatch):
    from mambapose_opt.checkpoints import authorize_tracked_config

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {'state_dict': {'unused': torch.ones(1)}})

    class LazyConfig:
        def __init__(self, path):
            self.path = Path(path)

        def dump(self):
            return self.path.read_text(encoding='utf-8')

    monkeypatch.setattr(Config, 'fromfile', lambda path: LazyConfig(path))

    with pytest.raises(ValueError, match='not serializable'):
        authorize_tracked_config(
            tmp_path, manifest, 'pwl-silu-s-v1')


@pytest.mark.parametrize('base', ['/tmp/outside.py', '../../outside.py'])
def test_tracked_authority_rejects_base_escaping_private_closure(
        tmp_path, base):
    from mambapose_opt.checkpoints import authorize_tracked_config

    manifest, _checkpoint = _authorized_repo(
        tmp_path, {'state_dict': {'unused': torch.ones(1)}})
    source = tmp_path / 'configs/model.py'
    source.write_text(
        f'_base_ = {base!r}\nmodel = dict(type="Fixture")\n',
        encoding='utf-8')
    _git('add', 'configs/model.py', cwd=tmp_path)
    _git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
         'commit', '-qm', 'add escaping base', cwd=tmp_path)

    with pytest.raises(ValueError, match='base.*absolute|base.*escapes'):
        authorize_tracked_config(
            tmp_path, manifest, 'pwl-silu-s-v1')


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
    model.data_preprocessor = torch.nn.Identity()
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

    def runner_init(self, model, *, cfg, load_from, **_kwargs):
        captured['load_from'] = load_from
        captured['pretty_text'] = cfg.pretty_text
        captured['config_model'] = cfg.model.to_dict()
        self.model = self.build_model(model)

    def runner_test(self):
        captured['tested'] = True
        captured['runner_model'] = self.model

    monkeypatch.setattr(tool.Runner, '__init__', runner_init)
    monkeypatch.setattr(tool.Runner, 'test', runner_test)
    monkeypatch.setattr(sys, 'argv', [
        'tools/test.py', str(config_path), str(checkpoint),
        '--safe-manifest', str(manifest),
        '--safe-candidate', 'pwl-silu-s-v1',
        '--safe-config-authority', str(authority_path)])

    tool.main()

    assert captured['load_from'] is None
    assert captured['config_model'] == {'type': 'Fixture'}
    assert "model = dict(type='Fixture')" in captured['pretty_text']
    assert captured['runner_model'] is model
    assert captured['tested'] is True
    assert captured['verified'] == 1


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
