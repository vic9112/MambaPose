import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch


def _policy(function_name='silu'):
    source = 'module' if function_name in {'silu', 'gelu'} \
        else 'ss2d-transition'
    return {
        'enabled_function': function_name,
        'source': source,
        'roles': ('block.act',),
        'domain': (-2.0, 2.0),
        'segments': 4,
        'grid_points': 129,
        'saturation': 'clamp',
        'qat_form': 'differentiable',
        'selection_policy': 'observed-range-max-then-mean-v1',
    }


def _fit(function_name='silu', values=None):
    from mambapose_opt.pwl_artifacts import fit_pwl_observations

    policy = _policy(function_name)
    return fit_pwl_observations(
        candidate_id=f'pwl-{function_name}-s-v1',
        policy=policy,
        observations={
            'block.act': [
                torch.tensor(
                    values if values is not None else [-3.0, -1.0, 0.0, 2.5],
                    dtype=torch.float64)
            ]
        })


def _write_reference(root: Path, artifact: dict, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'pwl_fit': artifact}), encoding='utf-8')
    return path, {
        'path': relative,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_pwl_fit_records_exact_role_ranges_errors_and_saturation():
    fit = _fit(values=[-3.0, -1.0, 0.0, 2.5])

    assert fit['candidate_id'] == 'pwl-silu-s-v1'
    assert fit['input_roles'] == [{
        'operation_role': 'block.act',
        'exact_input_role': 'block.act.input',
    }]
    assert fit['observed_range'] == [-3.0, 2.5]
    assert fit['in_domain_error']['samples'] == 129
    assert fit['observed_range_error']['samples'] == 129
    assert fit['in_domain_error']['max'] >= fit['in_domain_error']['mean'] >= 0
    assert fit['observed_range_error']['max'] >= (
        fit['observed_range_error']['mean']) >= 0
    assert fit['observed_samples_error']['samples'] == 4
    assert fit['clamp'] == {
        'below': 1, 'above': 1, 'total': 4, 'ratio': 0.5}
    role = fit['role_observations'][0]
    assert role['operation_role'] == 'block.act'
    assert role['exact_input_role'] == 'block.act.input'
    assert role['observed_range'] == [-3.0, 2.5]
    assert role['clamp']['ratio'] == 0.5


def test_pwl_artifact_ranking_is_measured_not_function_name_order():
    from mambapose_opt.pwl_artifacts import rank_pwl_fit_artifacts

    silu = _fit('silu')
    gelu = _fit('gelu')

    assert [item['candidate_id'] for item in rank_pwl_fit_artifacts(
            [silu, gelu])] == ['pwl-gelu-s-v1', 'pwl-silu-s-v1']


def test_pwl_fit_rejects_forged_role_range_and_recomputed_error():
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, validate_pwl_fit_report)

    fit = _fit()
    invalid_range = json.loads(json.dumps(fit))
    invalid_range['role_observations'][0]['observed_range'][0] = float('nan')
    with pytest.raises(PWLArtifactError, match='role observation'):
        validate_pwl_fit_report(invalid_range)

    forged_error = json.loads(json.dumps(fit))
    forged_error['observed_range_error']['max'] += 1.0
    with pytest.raises(PWLArtifactError, match='recomputed'):
        validate_pwl_fit_report(forged_error)


def test_exp_fit_retains_exact_export_time_constant_folding_comparator():
    fit = _fit('exp', values=[-1.0, 0.0, 1.0])

    assert fit['exact_comparator'] == {
        'kind': 'exact-export-time-constant-folding',
        'applicable_source': 'static-parameter',
        'runtime_nonlinear_operations': 0,
        'max_error': 0.0,
        'mean_error': 0.0,
        'preferred_over_pwl_when_exportable': True,
    }


def test_pwl_fit_reference_is_relative_hash_bound_and_rejects_symlinks(
        tmp_path):
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, load_pwl_fit_reference)

    fit = _fit()
    path, reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')

    loaded = load_pwl_fit_reference(
        reference, repository_root=tmp_path,
        expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())
    assert loaded == fit

    absolute = dict(reference, path=str(path))
    with pytest.raises(PWLArtifactError, match='repository-relative'):
        load_pwl_fit_reference(
            absolute, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())
    traversal = dict(reference, path='work_dirs/optimization/../fit.json')
    with pytest.raises(PWLArtifactError, match='unsafe'):
        load_pwl_fit_reference(
            traversal, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())

    symlink = tmp_path / 'work_dirs/optimization/pwl-link.json'
    symlink.symlink_to(path)
    linked = {
        'path': 'work_dirs/optimization/pwl-link.json',
        'sha256': reference['sha256'],
    }
    with pytest.raises(PWLArtifactError, match='symlink'):
        load_pwl_fit_reference(
            linked, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())


def test_pwl_fit_reference_rejects_alternate_policy_downgrade(tmp_path):
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, load_pwl_fit_reference)

    fit = _fit('silu')
    _, reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')

    with pytest.raises(PWLArtifactError, match='policy'):
        load_pwl_fit_reference(
            reference, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1',
            expected_policy=_policy('gelu'))


def test_serialized_pwl_installation_binds_fit_and_operation_manifest():
    from dataclasses import asdict

    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, validate_pwl_installation_manifest)

    fit = _fit()
    fit_reference = {
        'path': ('work_dirs/optimization/pwl-silu-s-v1/'
                 'calibrate/calibrate.json'),
        'sha256': 'a' * 64,
    }
    manifest = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    validated = validate_pwl_installation_manifest(
        manifest, expected_candidate_id='pwl-silu-s-v1',
        expected_fit_reference=fit_reference)

    assert validated['report'] == asdict(validated['report_object'])
    assert validated['operation_manifest'] == {
        'function': 'silu',
        'source': 'module',
        'operation_roles': ['block.act'],
        'exact_input_roles': ['block.act.input'],
        'segments_per_role': 4,
        'saturation': 'clamp',
        'hardware_latency_claimed': False,
    }
    forged = json.loads(json.dumps(manifest))
    forged['report']['observed_range_max_error'] += 1.0
    with pytest.raises(ValueError, match='measured fit'):
        validate_pwl_installation_manifest(
            forged, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=fit)


def test_calibration_hook_session_observes_declared_module_exact_inputs():
    from mambapose_opt.numeric_calibration import CalibrationTargets
    from tools.optimization.calibrate_numeric import _HookSession

    model = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})
    empty = CalibrationTargets(
        ss2d_boundaries=(), vmamba_in_proj=(), vmamba_out_proj=(),
        attention_qkv=(), pif_boundaries=(), heatmap_projection=(),
        transition_parameters=(), functional_observers=(),
        unsupported_internals=())

    with _HookSession(model, empty, pwl_policy=_policy()) as session:
        model['block']['act'](torch.tensor([-3.0, 0.0, 1.0]))
        fit = session.pwl_report(candidate_id='pwl-silu-s-v1')

    assert fit['input_roles'][0]['exact_input_role'] == 'block.act.input'
    assert fit['observed_range'] == [-3.0, 1.0]


def test_calibration_hook_session_observes_ss2d_pre_nonlinear_exact_input():
    from mambapose_opt.numeric_calibration import CalibrationTargets
    from tools.optimization.calibrate_numeric import _HookSession

    class FunctionalHost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.callback = None

        def set_numeric_observer(self, callback):
            self.callback = callback

    model = torch.nn.ModuleDict({'block': FunctionalHost()})
    targets = CalibrationTargets(
        ss2d_boundaries=(), vmamba_in_proj=(), vmamba_out_proj=(),
        attention_qkv=(), pif_boundaries=(), heatmap_projection=(),
        transition_parameters=(), functional_observers=('block',),
        unsupported_internals=())
    policy = _policy('softplus')
    policy['roles'] = ('block',)

    with _HookSession(model, targets, pwl_policy=policy) as session:
        model['block'].callback(
            'transition_softplus_input', torch.tensor([-7.0, 2.0]))
        fit = session.pwl_report(candidate_id='pwl-softplus-s-v1')

    assert fit['input_roles'][0]['exact_input_role'] == (
        'block.transition_softplus_input')
    assert fit['observed_range'] == [-7.0, 2.0]


def test_numeric_runtime_installs_only_hash_bound_fit_and_manifest(
        tmp_path, monkeypatch):
    from mambapose_opt import numeric_conversion
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, NumericRuntimeHook)
    from mambapose_opt.pwl_artifacts import build_pwl_installation_manifest
    from mmpose.models.utils.hardware_friendly import (
        PiecewiseLinearApproximation)

    fit = _fit()
    fit_path, fit_reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')
    installation = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    install_path = (
        tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
        'convert/pwl-installation.json')
    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(json.dumps(installation), encoding='utf-8')
    install_reference = {
        'path': install_path.relative_to(tmp_path).as_posix(),
        'sha256': hashlib.sha256(install_path.read_bytes()).hexdigest(),
    }
    policy = _policy()
    policy.update({
        'candidate_id': 'pwl-silu-s-v1',
        'fit_artifact': fit_reference,
        'installation_manifest': install_reference,
    })
    monkeypatch.setattr(numeric_conversion, 'REPOSITORY_ROOT', tmp_path)
    model = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})

    report = NumericRuntimeHook.apply_to_model(
        model, {'candidate_kind': 'pwl', 'pwl': policy})

    assert report.fit_artifact_sha256 == fit_reference['sha256']
    assert isinstance(model['block']['act'], PiecewiseLinearApproximation)
    fit_path.write_text(json.dumps({'pwl_fit': _fit('gelu')}), encoding='utf-8')
    second = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})
    with pytest.raises(NumericBindingError, match='hash changed'):
        NumericRuntimeHook.apply_to_model(
            second, {'candidate_kind': 'pwl', 'pwl': policy})


@pytest.mark.parametrize('name', ['silu', 'gelu', 'softplus', 'exp'])
def test_pwl_configs_declare_artifact_ranking_and_five_stage_campaign(name):
    from mmengine.config import Config

    from mambapose_opt.numeric_conversion import numeric_stage_plan

    config = Config.fromfile(f'configs/optimization/numeric/pwl_{name}.py')
    numeric = config.numeric_optimization
    assert tuple(numeric.stage_order) == (
        'calibrate', 'convert', 'profile', 'evaluate', 'latency')
    assert numeric.pwl.selection_policy == (
        'observed-range-max-then-mean-v1')
    assert numeric.calibration.artifact_schema_version == 3
    assert numeric_stage_plan('pwl', conditional=True) == tuple(
        numeric.stage_order)


def test_pwl_convert_requires_fit_artifact_before_model_load(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.optimization import convert_numeric

    candidate = SimpleNamespace(
        id='pwl-silu-s-v1', route='ssm-quant-pwl',
        features={'numeric_kind': 'pwl'}, config=Path('configs/pwl.py'),
        checkpoint=Path('checkpoint.pth'), checkpoint_sha256='a' * 64)
    entered = []
    monkeypatch.setattr(
        convert_numeric, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(candidate=candidate))
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda *_args, **_kwargs: entered.append(True))

    with pytest.raises(ValueError, match='PWL.*calibration|fit artifact'):
        convert_numeric.convert(
            candidate, stage='convert',
            output=tmp_path / 'work_dirs/optimization/pwl/convert/convert.json',
            manifest_path=tmp_path / 'optimization/candidates.json')
    assert entered == []


def test_pwl_campaign_convert_consumes_canonical_calibrate_output(
        tmp_path, monkeypatch):
    from mambapose_opt.schema import CandidateSpec
    from tools.optimization import run_campaign
    from tools.optimization.run_campaign import SubprocessStageRunner

    candidate = CandidateSpec.from_dict({
        'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': 'configs/optimization/numeric/pwl_silu.py',
        'checkpoint': 'checkpoint.pth', 'checkpoint_sha256': 'a' * 64,
        'seed': 0, 'features': {
            'numeric_kind': 'pwl', 'conditional': True, 'auto_run': False},
    })
    output = (
        tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
        'convert/convert.json')
    monkeypatch.setattr(run_campaign, 'REPO_ROOT', tmp_path)
    runner = SubprocessStageRunner(
        tmp_path / 'work_dirs/optimization',
        tmp_path / 'optimization/candidates.json')

    command = runner._command(candidate, 'convert', output)

    index = command.index('--calibration-artifact')
    assert command[index + 1].endswith(
        'pwl-silu-s-v1/calibrate/calibrate.json')


def test_pwl_runtime_resolver_fails_closed_before_convert(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime)
    from mambapose_opt.schema import CandidateSpec

    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    candidate = CandidateSpec.from_dict({
        'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': 'configs/pwl.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(),
        'seed': 0, 'features': {
            'numeric_kind': 'pwl', 'conditional': True, 'auto_run': False},
    })
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(
            candidate=candidate, checkpoint_path=checkpoint))

    with pytest.raises(NumericRuntimeError, match='completed convert'):
        resolve_numeric_runtime(
            candidate, repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json',
            downstream_output=(
                tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
                'profile/profile.json'))


def test_pwl_producer_controller_and_all_downstream_share_install_provenance(
        tmp_path, monkeypatch):
    from torch import nn

    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.numeric_runtime import (
        resolve_numeric_runtime, validate_numeric_convert_artifact)
    from mambapose_opt.schema import load_candidate_manifest
    from tools.optimization import convert_numeric

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'work_dirs/reproduction').mkdir(parents=True)
    (tmp_path / 'work_dirs/optimization').mkdir(parents=True)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/.keep').write_text('', encoding='utf-8')
    (tmp_path / '.gitignore').write_text('work_dirs/\n', encoding='utf-8')
    checkpoint = tmp_path / 'work_dirs/reproduction/checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / 'configs/pwl.py'
    config.write_text(
        "numeric_optimization = dict(\n"
        "    candidate_kind='pwl',\n"
        "    calibration=dict(artifact_schema_version=3),\n"
        "    pwl=dict(\n"
        "        candidate_id='pwl-silu-s-v1', enabled_function='silu',\n"
        "        source='module', roles=('layer',), domain=(-2.0, 2.0),\n"
        "        segments=4, grid_points=129, saturation='clamp',\n"
        "        qat_form='differentiable',\n"
        "        selection_policy='observed-range-max-then-mean-v1'))\n",
        encoding='utf-8')
    manifest = tmp_path / 'optimization/candidates.json'
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl',
            'kind': 'pwl', 'config': 'configs/pwl.py',
            'checkpoint': 'work_dirs/reproduction/checkpoint.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'pwl', 'conditional': True,
                         'auto_run': False},
        }],
    }), encoding='utf-8')
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text(
        '{}\n', encoding='utf-8')
    (tmp_path / 'optimization/coco_val2017_authority.json').write_text(
        '{}\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '-qm', 'pwl fixture'], cwd=tmp_path, check=True)
    candidate = load_candidate_manifest(manifest)[0]
    root = tmp_path / 'work_dirs/optimization/pwl-silu-s-v1'
    calibration_path = root / 'calibrate/calibrate.json'
    calibration_path.parent.mkdir(parents=True)
    from mambapose_opt.pwl_artifacts import fit_pwl_observations
    fixture_policy = _policy()
    fixture_policy['roles'] = ('layer',)
    fixture_fit = fit_pwl_observations(
        candidate_id=candidate.id, policy=fixture_policy,
        observations={'layer': [torch.tensor([-1.0, 0.0, 1.0])]})
    calibration_path.write_text(json.dumps({
        'schema_version': 3, 'candidate_id': candidate.id,
        'stage': 'calibrate', 'source': {}, 'identity': {},
        'protocol': {}, 'hooks': {}, 'pwl_fit': fixture_fit,
    }), encoding='utf-8')
    output = root / 'convert/convert.json'
    output.parent.mkdir(parents=True)
    model = nn.Module()
    model.layer = nn.SiLU()
    monkeypatch.setattr(convert_numeric, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(
        'mmpose.apis.init_model', lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        convert_numeric, 'validate_calibration_provenance',
        lambda value, **_kwargs: value)
    monkeypatch.setattr(
        'mambapose_opt.numeric_calibration.validate_calibration_provenance',
        lambda value, **_kwargs: value)

    produced = convert_numeric.convert(
        candidate, stage='convert', output=output, manifest_path=manifest,
        calibration_artifact=calibration_path)
    output.write_text(json.dumps(produced), encoding='utf-8')
    round_tripped = json.loads(output.read_text(encoding='utf-8'))

    assert validate_numeric_convert_artifact(
        round_tripped, candidate=candidate, repository_root=tmp_path,
        manifest_path=manifest, artifact_path=output) == round_tripped
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *_args: None, repository_root=tmp_path,
        manifest_path=manifest, stages=('convert',))
    assert controller._artifact_schema('convert', output) == (
        'optimization-stage-envelope-v1')
    runtimes = [resolve_numeric_runtime(
        candidate, repository_root=tmp_path, manifest_path=manifest,
        downstream_output=root / stage / f'{stage}.json')
        for stage in ('profile', 'evaluate', 'latency')]
    assert len({item['config_sha256'] for item in runtimes}) == 1
    assert len({item['pwl_installation']['sha256'] for item in runtimes}) == 1
    assert all(item['pwl_installation'] == produced['result']['installation']
               for item in runtimes)
