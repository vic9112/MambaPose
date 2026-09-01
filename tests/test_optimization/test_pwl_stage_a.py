import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from timm.layers import DropPath


_STRUCTURAL_SMOKE_PROTOCOL = {
    'kind': 'deterministic-structural-gradient-v1',
    'model_mode': 'train',
    'stochastic_depth': 'disabled-timm-drop-path-only',
    'covered_training_passes': [
        'target-gradient-and-adam-step', 'optimizer-resume-step'],
    'gradient_batches': 1,
    'target_gradient_requirement': (
        'finite-nonzero-input-and-output-every-target'),
}


def _policy():
    return {
        'enabled_function': 'silu', 'source': 'module',
        'roles': ('activation',), 'domain': (-2.0, 2.0), 'segments': 4,
        'grid_points': 129,
        'saturation': 'continuous-asymptotic-tail-v1',
        'qat_form': 'differentiable',
        'selection_policy': 'observed-range-max-then-mean-v1',
    }


def _fit_and_installation():
    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, fit_pwl_observations)

    fit = fit_pwl_observations(
        candidate_id='pwl-silu-s-v1', policy=_policy(),
        observations={
            'activation': [torch.tensor([-1.0, 0.0, 1.0])]},
    )
    reference = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/pwl-silu-s-v1/'
                 '0/calibrate/calibrate.json'),
        'sha256': 'a' * 64,
    }
    installation = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=reference)
    return fit, reference, installation


class TinyPose(nn.Module):
    def __init__(self, *, identity=False):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)
        self.activation = nn.SiLU()
        self.head = nn.Linear(4, 2, bias=False)
        self.identity = identity

    def forward(self, inputs, data_samples=None, mode='tensor'):
        output = self.head(self.activation(self.projection(inputs)))
        if mode == 'loss':
            return {'loss_kpt': ((output - data_samples) ** 2).mean()}
        if mode == 'tensor':
            return output
        raise ValueError(mode)


class TinyDropPathPose(TinyPose):
    """A PWL target in a whole-sample stochastic-depth residual branch."""

    def __init__(self, *, identity=False):
        super().__init__(identity=identity)
        self.drop_path = DropPath(0.075)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        residual = self.projection(inputs)
        output = self.head(
            residual + self.drop_path(self.activation(residual)))
        if mode == 'loss':
            return {'loss_kpt': ((output - data_samples) ** 2).mean()}
        if mode == 'tensor':
            return output
        raise ValueError(mode)


def test_pwl_stage_a_replay_factories_reuse_exact_authorized_state(
        monkeypatch):
    """Regression: replay factories must not create uninitialized tensors."""
    from mambapose_opt import pwl_smoke

    factory_builder = getattr(pwl_smoke, '_stage_a_model_factories', None)
    assert factory_builder is not None, (
        'PWL Stage-A requires an exact-state replay factory')
    source = {
        'weight': torch.tensor([1.25, -2.5], dtype=torch.float32),
        'buffer': torch.tensor([3, 5], dtype=torch.int64),
    }
    source_before = {
        name: value.detach().clone() for name, value in source.items()}
    authority = object()
    restored = object()
    calls = []

    def capture_build(config_authority, state, device, *, install):
        calls.append((config_authority, state, device, install))
        return restored

    monkeypatch.setattr(pwl_smoke, '_build_model', capture_build)
    monkeypatch.setattr(
        torch, 'empty_like',
        lambda value: torch.full_like(
            value, float('nan') if value.is_floating_point() else -1))
    fitted_factory, identity_factory = factory_builder(authority, source)

    assert fitted_factory() is restored
    assert identity_factory() is restored
    assert [call[0] for call in calls] == [authority, authority]
    assert [call[1] is source for call in calls] == [True, True]
    assert [call[2] for call in calls] == [
        torch.device('cpu'), torch.device('cpu')]
    assert [call[3] for call in calls] == [True, False]
    assert all(torch.equal(calls[0][1][name], value)
               for name, value in source_before.items())
    assert all(torch.equal(calls[1][1][name], value)
               for name, value in source_before.items())
    assert all(torch.equal(source[name], value)
               for name, value in source_before.items())


def test_pwl_stage_a_core_proves_target_grad_adam_identity_and_resume(tmp_path):
    from mambapose_opt.numeric_conversion import install_pwl_fit
    from mambapose_opt.pwl_artifacts import validate_pwl_installation_manifest
    from mambapose_opt.pwl_smoke import execute_pwl_stage_a_model

    fit, reference, installation = _fit_and_installation()
    report = validate_pwl_installation_manifest(
        installation, expected_candidate_id='pwl-silu-s-v1',
        expected_fit_reference=reference, expected_fit=fit)['report_object']

    def fitted_factory():
        model = TinyPose()
        install_pwl_fit(model, fit=fit, expected_report=report)
        return model

    torch.manual_seed(12)
    model = fitted_factory()
    inputs = torch.tensor([[1.0, -0.5, 0.25]])
    targets = torch.tensor([[0.25, -0.75]])
    export = tmp_path / 'round-trip.pth'
    result = execute_pwl_stage_a_model(
        model=model, inputs=inputs, data_samples=targets,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        model_factory=fitted_factory,
        optimizer_factory=lambda restored: torch.optim.Adam(
            restored.parameters(), lr=1e-3),
        identity_factory=TinyPose, export_path=export,
        function_name='silu', roles=('activation',),
        coefficients=fit['coefficients'],
        operation_manifest=installation['operation_manifest'])

    assert result['checks'] == {
        'forward': True, 'loss': True, 'backward': True,
        'optimizer_step': True, 'finite_loss': True,
        'finite_gradients': True, 'pwl_target_count': 1,
        'pwl_target_gradients': True, 'identity_mode': True,
        'coefficients_exact': True, 'roles_exact': True,
        'restricted_tensor_only_load': True,
        'state_round_trip_exact': True, 'output_round_trip_exact': True,
        'optimizer_resume_step': True,
    }
    assert result['pwl_targets'][0]['operation_role'] == 'activation'
    assert result['pwl_targets'][0]['input_gradient_norm'] > 0
    assert result['pwl_targets'][0]['output_gradient_norm'] > 0
    assert result['optimizer']['kind'] == 'Adam'
    assert result['optimizer']['resumed_min_step'] >= 2
    assert result['identity']['enabled'] is True
    payload = torch.load(export, map_location='cpu', weights_only=True)

    def tensor_tree(value):
        return isinstance(value, torch.Tensor) or (
            isinstance(value, dict) and value
            and all(isinstance(key, str) and tensor_tree(item)
                    for key, item in value.items()))

    assert tensor_tree(payload)
    assert result['export']['format'] == (
        'torch-weights-only-model-and-adam-tensors-v1')


def test_pwl_stage_a_core_disables_only_drop_path_for_structural_smoke(
        tmp_path):
    """Regression: batch-one DropPath must not hide an installed PWL target."""
    from mambapose_opt.numeric_conversion import install_pwl_fit
    from mambapose_opt.pwl_artifacts import validate_pwl_installation_manifest
    from mambapose_opt.pwl_smoke import execute_pwl_stage_a_model

    fit, reference, installation = _fit_and_installation()
    report = validate_pwl_installation_manifest(
        installation, expected_candidate_id='pwl-silu-s-v1',
        expected_fit_reference=reference, expected_fit=fit)['report_object']

    def fitted_factory():
        model = TinyDropPathPose()
        install_pwl_fit(model, fit=fit, expected_report=report)
        return model

    model = fitted_factory()
    torch.manual_seed(0)  # timm DropPath(0.075) drops this batch-one branch.
    result = execute_pwl_stage_a_model(
        model=model, inputs=torch.tensor([[1.0, -0.5, 0.25]]),
        data_samples=torch.tensor([[0.25, -0.75]]),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        model_factory=fitted_factory,
        optimizer_factory=lambda restored: torch.optim.Adam(
            restored.parameters(), lr=1e-3),
        identity_factory=TinyDropPathPose,
        export_path=tmp_path / 'drop-path-round-trip.pth',
        function_name='silu', roles=('activation',),
        coefficients=fit['coefficients'],
        operation_manifest=installation['operation_manifest'])

    assert result['protocol'] == _STRUCTURAL_SMOKE_PROTOCOL
    assert result['pwl_targets'][0]['input_gradient_norm'] > 0
    assert result['pwl_targets'][0]['output_gradient_norm'] > 0
    assert model.drop_path.drop_prob == pytest.approx(0.075)


def test_pwl_stage_a_core_rejects_forged_tail_as_clamp(tmp_path):
    """Break caught: Stage-A must attest the exported tail operation."""
    from mambapose_opt.numeric_conversion import install_pwl_fit
    from mambapose_opt.pwl_artifacts import validate_pwl_installation_manifest
    from mambapose_opt.pwl_smoke import execute_pwl_stage_a_model

    fit, reference, installation = _fit_and_installation()
    report = validate_pwl_installation_manifest(
        installation, expected_candidate_id='pwl-silu-s-v1',
        expected_fit_reference=reference, expected_fit=fit)['report_object']
    model = TinyPose()
    install_pwl_fit(model, fit=fit, expected_report=report)
    forged = dict(installation['operation_manifest'])
    forged['domain_handling'] = {'kind': 'clamp'}

    with pytest.raises(RuntimeError, match='operation manifest'):
        execute_pwl_stage_a_model(
            model=model, inputs=torch.tensor([[1.0, -0.5, 0.25]]),
            data_samples=torch.tensor([[0.25, -0.75]]),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            model_factory=TinyPose,
            optimizer_factory=lambda restored: torch.optim.Adam(
                restored.parameters(), lr=1e-3),
            identity_factory=TinyPose, export_path=tmp_path / 'forged.pth',
            function_name='silu', roles=('activation',),
            coefficients=fit['coefficients'], operation_manifest=forged)


def _artifact(root: Path):
    fit, fit_reference, installation = _fit_and_installation()
    path = root / (
        'work_dirs/optimization/ssm-quant-pwl/pwl-silu-s-v1/0/'
        'smoke-stage-a/smoke.json')
    export = path.parent / 'round-trip.pth'
    export.parent.mkdir(parents=True)
    torch.save({
        'model_state': {
            f'activation.{name}': torch.as_tensor(values, dtype=torch.float64)
            for name, values in fit['coefficients'].items()},
        'adam_state': {
            'projection.weight': {
                'step': torch.tensor(1.0),
                'exp_avg': torch.zeros(1),
                'exp_avg_sq': torch.zeros(1)}}}, export)
    sha = hashlib.sha256(export.read_bytes()).hexdigest()
    selection = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/pwl-selection/'
                 'selection.json'), 'sha256': 'b' * 64}
    install_reference = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/pwl-silu-s-v1/0/'
                 'convert/pwl-installation.json'), 'sha256': 'c' * 64}
    value = {
        'schema_version': 2,
        'artifact_kind': 'pwl-stage-a-full-model-smoke',
        'candidate_id': 'pwl-silu-s-v1',
        'source': {'git_commit': 'd' * 40},
        'config': {'path': 'configs/optimization/numeric/pwl_silu.py',
                   'sha256': 'e' * 64},
        'checkpoint': {'path': 'work_dirs/reproduction/full.pth',
                       'sha256': 'f' * 64},
        'fit_artifact': fit_reference,
        'selection_artifact': selection,
        'installation': install_reference,
        'policy': _policy(),
        'data': {
            'dataset': 'coco', 'split': 'train2017', 'batch_size': 1,
            'packed_production_pipeline': True,
            'input_shape': [1, 3, 256, 192], 'sample_ids': [42]},
        'gpu': {
            'logical': 'cuda:0', 'physical_index': 0,
            'lease': {'validated-fixture': True}},
        'execution': {
            'protocol': _STRUCTURAL_SMOKE_PROTOCOL,
            'checks': {
                'forward': True, 'loss': True, 'backward': True,
                'optimizer_step': True, 'finite_loss': True,
                'finite_gradients': True, 'pwl_target_count': 1,
                'pwl_target_gradients': True, 'identity_mode': True,
                'coefficients_exact': True, 'roles_exact': True,
                'restricted_tensor_only_load': True,
                'state_round_trip_exact': True,
                'output_round_trip_exact': True,
                'optimizer_resume_step': True},
            'losses': {'loss_kpt': 0.25},
            'pwl_targets': [{
                'operation_role': 'activation',
                'module_path': 'activation', 'input_gradient_norm': 1.0,
                'output_gradient_norm': 1.0, 'gradient_finite': True}],
            'optimizer': {
                'kind': 'Adam', 'updated_parameter_count': 2,
                'state_tensor_count': 6, 'first_min_step': 1,
                'resumed_min_step': 2, 'state_finite': True},
            'identity': {'enabled': True, 'output_shape': [1, 17, 64, 48],
                         'finite': True},
            'output': {'shape': [1, 17, 64, 48],
                       'dtype': 'torch.float32', 'sha256': '1' * 64},
            'operation': installation['operation_manifest'],
            'export': {
                'path': export.relative_to(root).as_posix(), 'sha256': sha,
                'bytes': export.stat().st_size,
                'format': 'torch-weights-only-model-and-adam-tensors-v1'},
        },
        'claim_limits': {
            'hardware_latency_claimed': False,
            'fpga_speedup_claimed': False,
            'fastmamba_composite_mechanisms_inherited': False},
    }
    path.write_text(json.dumps(value), encoding='utf-8')
    return path, value, fit, installation, selection, install_reference


def test_public_pwl_stage_a_validator_reconstructs_all_authority(
        tmp_path, monkeypatch):
    from mambapose_opt import pwl_smoke

    artifact, value, fit, installation, selection, install_reference = (
        _artifact(tmp_path))
    monkeypatch.setattr(
        pwl_smoke, '_production_dependencies',
        lambda **unused: {
            'candidate_id': 'pwl-silu-s-v1',
            'git_commit': 'd' * 40,
            'config': value['config'], 'checkpoint': value['checkpoint'],
            'policy': json.loads(json.dumps(_policy())),
            'fit_reference': value['fit_artifact'],
            'fit': fit, 'selection_reference': selection,
            'installation_reference': install_reference,
            'installation': installation})
    monkeypatch.setattr(
        pwl_smoke, 'validate_gpu_lease',
        lambda value: {'stage_id': 'pwl-silu-s-v1:smoke-stage-a',
                       'device_index': 0})

    assert pwl_smoke.validate_pwl_stage_a_artifact(
        artifact, repository_root=tmp_path,
        manifest_path=Path('optimization/candidates.json')) == json.loads(
            artifact.read_text(encoding='utf-8'))

    forged = json.loads(json.dumps(value))
    forged['policy']['segments'] = 2
    artifact.write_text(json.dumps(forged), encoding='utf-8')
    with pytest.raises(ValueError, match='policy'):
        pwl_smoke.validate_pwl_stage_a_artifact(
            artifact, repository_root=tmp_path,
            manifest_path=Path('optimization/candidates.json'))

    forged = json.loads(json.dumps(value))
    forged['execution']['protocol']['stochastic_depth'] = 'training-enabled'
    artifact.write_text(json.dumps(forged), encoding='utf-8')
    with pytest.raises(ValueError, match='protocol'):
        pwl_smoke.validate_pwl_stage_a_artifact(
            artifact, repository_root=tmp_path,
            manifest_path=Path('optimization/candidates.json'))

    missing = json.loads(json.dumps(value))
    missing['execution'].pop('protocol')
    artifact.write_text(json.dumps(missing), encoding='utf-8')
    with pytest.raises(ValueError, match='fields|protocol'):
        pwl_smoke.validate_pwl_stage_a_artifact(
            artifact, repository_root=tmp_path,
            manifest_path=Path('optimization/candidates.json'))


def test_production_dependencies_reads_canonical_calibration_envelope(
        tmp_path, monkeypatch):
    from mmengine.config import Config

    from mambapose_opt import checkpoints, numeric_calibration, pwl_smoke

    fit, fit_reference, installation = _fit_and_installation()
    candidate = SimpleNamespace(
        id='pwl-silu-s-v1', route='ssm-quant-pwl', seed=0,
        features={'numeric_kind': 'pwl'},
        config=Path('configs/optimization/numeric/pwl_silu.py'),
        checkpoint=Path('work_dirs/reproduction/full.pth'),
        checkpoint_sha256='f' * 64)
    authorized = SimpleNamespace(
        candidate=candidate, config_path=tmp_path / candidate.config,
        source={'git_commit': 'd' * 40})
    calibration = tmp_path / fit_reference['path']
    calibration.parent.mkdir(parents=True)
    calibration.write_text(json.dumps({'pwl_fit': fit}), encoding='utf-8')
    authorized.config_path.parent.mkdir(parents=True)
    authorized.config_path.write_text('# fixture\n', encoding='utf-8')
    selection_reference = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/pwl-selection/'
                 'selection.json'), 'sha256': 'b' * 64}
    installation_reference = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/pwl-silu-s-v1/0/'
                 'convert/pwl-installation.json'), 'sha256': 'c' * 64}
    value = {
        'candidate_id': candidate.id, 'fit_artifact': fit_reference,
        'selection_artifact': selection_reference,
        'installation': installation_reference}
    monkeypatch.setattr(
        checkpoints, 'authorize_manifest_candidate',
        lambda *unused, **ignored: authorized)
    tracked_config = SimpleNamespace(
        numeric_optimization=SimpleNamespace(pwl=_policy()))
    monkeypatch.setattr(
        checkpoints, 'authorize_tracked_config',
        lambda *unused, **ignored: SimpleNamespace(
            load_config=lambda: tracked_config))
    monkeypatch.setattr(
        Config, 'fromfile', staticmethod(lambda unused: (_ for _ in ()).throw(
            AssertionError('tracked PWL config bypassed ConfigAuthority'))))
    monkeypatch.setattr(
        pwl_smoke, 'load_pwl_fit_reference',
        lambda *unused, **ignored: fit)
    monkeypatch.setattr(
        pwl_smoke, 'load_pwl_selection_reference',
        lambda *unused, **ignored: {
            'selected_candidate_id': candidate.id,
            'candidates': [{
                'candidate_id': candidate.id,
                'calibration_artifact': fit_reference}]})
    monkeypatch.setattr(
        pwl_smoke, 'load_pwl_installation_reference',
        lambda *unused, **ignored: installation)
    monkeypatch.setattr(
        numeric_calibration, 'validate_calibration_provenance',
        lambda *unused, **ignored: None)

    dependency = pwl_smoke._production_dependencies(
        value=value, repository_root=tmp_path,
        manifest_path=Path('optimization/candidates.json'))

    assert dependency['fit'] == fit
    assert dependency['selection_reference'] == selection_reference
    assert dependency['installation'] == installation


def test_nested_pretrain_initializers_are_neutralized_without_mutating_input():
    from mmengine.config import Config
    from mambapose_opt.checkpoints import neutralize_model_initializers

    source = {
        'type': 'TopdownPoseEstimator',
        'backbone': {
            'type': 'MM_VSSM', 'pretrained': 'upstream.pth',
            'child': {'init_cfg': {'type': 'Pretrained'}}},
        'head': {'init_cfg': {'type': 'Normal'}, 'channels': 17}}

    result = neutralize_model_initializers(Config(dict(model=source))).model

    assert result['backbone']['pretrained'] is None
    assert result['backbone']['child']['init_cfg'] is None
    assert result['head']['init_cfg'] is None
    assert result['head']['channels'] == 17
    assert source['backbone']['pretrained'] == 'upstream.pth'


def test_pwl_stage_a_cli_is_directly_executable():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, 'tools/optimization/smoke_pwl.py', '--help'],
        capture_output=True, text=True, check=False,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    assert result.returncode == 0, result.stderr
    assert '--candidate' in result.stdout
    assert '--output-root' in result.stdout
    assert '--device-index' in result.stdout


def test_controller_precreated_smoke_directory_preserves_only_attempt_log(
        tmp_path):
    from mambapose_opt.pwl_smoke import _prepare_smoke_output_directory

    output = tmp_path / 'smoke-stage-a'
    output.mkdir()
    log = output / 'attempt-1.log'
    log.write_text('controller owned\n', encoding='utf-8')

    _prepare_smoke_output_directory(output)
    assert log.read_text(encoding='utf-8') == 'controller owned\n'
    (output / 'unexpected.json').write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='unexpected'):
        _prepare_smoke_output_directory(output)


def test_controller_precreated_smoke_directory_rejects_attempt_log_symlink(
        tmp_path):
    from mambapose_opt.pwl_smoke import _prepare_smoke_output_directory

    output = tmp_path / 'smoke-stage-a'
    output.mkdir()
    outside = tmp_path / 'outside.log'
    outside.write_text('sentinel\n', encoding='utf-8')
    (output / 'attempt-1.log').symlink_to(outside)

    with pytest.raises(FileExistsError, match='unexpected|symlink'):
        _prepare_smoke_output_directory(output)


def test_pwl_smoke_consumes_controller_lease_without_relocking(
        tmp_path, monkeypatch):
    from dataclasses import asdict
    from datetime import datetime, timezone
    import fcntl
    import os

    from mambapose_opt.gpu_guard import GpuLease
    from mambapose_opt import pwl_smoke

    lock = tmp_path / 'gpu.lock'
    timestamp = datetime.now(timezone.utc)
    lease = GpuLease(
        stage_id='pwl-silu-s-v1:smoke-stage-a', pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        timestamp=timestamp.isoformat(), device_index=2,
        allowed_pids=(os.getpid(),), lease_id='7' * 64)
    lock.write_text(json.dumps({
        **asdict(lease), 'allowed_pids': list(lease.allowed_pids)}),
        encoding='utf-8')
    monkeypatch.setattr(pwl_smoke, '_canonical_gpu_lock', lambda _root: lock)
    with lock.open('r+', encoding='utf-8') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = pwl_smoke._active_controller_lease(
            'pwl-silu-s-v1', 2, repository_root=tmp_path,
            now=lambda: timestamp)

    assert observed['lease_id'] == lease.lease_id


def test_pwl_smoke_refreshes_artifact_lease_after_long_execution(
        tmp_path, monkeypatch):
    """Regression: Stage-A artifacts must use the current lock heartbeat."""
    from dataclasses import asdict, replace
    from datetime import datetime, timedelta, timezone
    import fcntl
    import os

    from mambapose_opt.gpu_guard import GpuLease
    from mambapose_opt import pwl_smoke

    refresh = getattr(
        pwl_smoke, '_refresh_controller_lease_for_artifact', None)
    assert refresh is not None, (
        'PWL Stage-A requires a final controller lease refresh')
    lock = tmp_path / 'gpu.lock'
    started_at = datetime(2026, 8, 30, tzinfo=timezone.utc)
    initial = GpuLease(
        stage_id='pwl-silu-s-v1:smoke-stage-a', pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        timestamp=started_at.isoformat(), device_index=2,
        allowed_pids=(os.getpid(),), lease_id='7' * 64)
    heartbeat = replace(
        initial, timestamp=(started_at + timedelta(seconds=301)).isoformat())
    lock.write_text(json.dumps({
        **asdict(heartbeat),
        'allowed_pids': list(heartbeat.allowed_pids),
    }), encoding='utf-8')
    monkeypatch.setattr(pwl_smoke, '_canonical_gpu_lock', lambda _root: lock)

    with lock.open('r+', encoding='utf-8') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = refresh(
            {**asdict(initial),
             'allowed_pids': list(initial.allowed_pids)},
            'pwl-silu-s-v1', 2, repository_root=tmp_path,
            now=lambda: started_at + timedelta(seconds=302))

    assert observed == {
        **asdict(heartbeat),
        'allowed_pids': list(heartbeat.allowed_pids),
    }


def test_pwl_smoke_rejects_replacement_lease_during_final_refresh(
        tmp_path, monkeypatch):
    """A fresh timestamp cannot substitute a different controller lease."""
    from dataclasses import asdict, replace
    from datetime import datetime, timedelta, timezone
    import fcntl
    import os

    from mambapose_opt.gpu_guard import GpuLease
    from mambapose_opt import pwl_smoke

    refresh = getattr(
        pwl_smoke, '_refresh_controller_lease_for_artifact', None)
    assert refresh is not None, (
        'PWL Stage-A requires a final controller lease refresh')
    lock = tmp_path / 'gpu.lock'
    started_at = datetime(2026, 8, 30, tzinfo=timezone.utc)
    initial = GpuLease(
        stage_id='pwl-silu-s-v1:smoke-stage-a', pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        timestamp=started_at.isoformat(), device_index=2,
        allowed_pids=(os.getpid(),), lease_id='7' * 64)
    replacement = replace(
        initial, timestamp=(started_at + timedelta(seconds=301)).isoformat(),
        lease_id='8' * 64)
    lock.write_text(json.dumps({
        **asdict(replacement),
        'allowed_pids': list(replacement.allowed_pids),
    }), encoding='utf-8')
    monkeypatch.setattr(pwl_smoke, '_canonical_gpu_lock', lambda _root: lock)

    with lock.open('r+', encoding='utf-8') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='identity'):
            refresh(
                {**asdict(initial),
                 'allowed_pids': list(initial.allowed_pids)},
                'pwl-silu-s-v1', 2, repository_root=tmp_path,
                now=lambda: started_at + timedelta(seconds=302))
