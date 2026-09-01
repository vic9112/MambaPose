"""Real one-batch full-MambaPose Stage-A proof for an admitted PWL."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

import torch
from timm.layers import DropPath

from .gpu_guard import controller_process_tree
from .latency import (
    LEASE_MAX_AGE_SECONDS, LEASE_MAX_FUTURE_SKEW_SECONDS,
    LatencyError, validate_gpu_lease, validate_refreshed_gpu_lease)
from .pwl_artifacts import (
    load_pwl_fit_reference, load_pwl_installation_reference)
from .combined_candidate import load_pwl_admission_reference
from .pwl_paths import canonical_path, canonical_relative_path


# Preserve the existing patch seam while admitting either selection schema.
load_pwl_selection_reference = load_pwl_admission_reference


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_LEASE_MAX_AGE = timedelta(seconds=LEASE_MAX_AGE_SECONDS)
_LEASE_MAX_FUTURE_SKEW = timedelta(seconds=LEASE_MAX_FUTURE_SKEW_SECONDS)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode('ascii'))
    digest.update(str(tuple(contiguous.shape)).encode('ascii'))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def _state_exact(first: Mapping[str, torch.Tensor],
                 second: Mapping[str, torch.Tensor]) -> bool:
    return (set(first) == set(second)
            and all(torch.equal(first[name], second[name]) for name in first))


def _disable_stochastic_depth_for_structural_smoke(
        model: torch.nn.Module) -> None:
    """Keep train semantics while ensuring each residual branch is exercised."""
    for module in model.modules():
        if isinstance(module, DropPath):
            module.eval()


def _tensor_tree(value: object) -> bool:
    return isinstance(value, torch.Tensor) or (
        isinstance(value, Mapping) and bool(value)
        and all(isinstance(key, str) and _tensor_tree(item)
                for key, item in value.items()))


def _step(value: object) -> int | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        number = float(value.detach().cpu())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        return None
    return int(number) if math.isfinite(number) and number >= 1 else None


def _pwl_modules(
        model: torch.nn.Module, *, function_name: str,
        roles: Sequence[str]) -> tuple[tuple[str, str, torch.nn.Module], ...]:
    from mmpose.models.utils.hardware_friendly.pwl import (
        PiecewiseLinearApproximation)

    modules = dict(model.named_modules())
    result = []
    for role in roles:
        direct = modules.get(role)
        child_path = f'{role}._numeric_pwl_{function_name}'
        child = modules.get(child_path)
        if isinstance(direct, PiecewiseLinearApproximation):
            result.append((role, role, direct))
        elif isinstance(child, PiecewiseLinearApproximation):
            result.append((role, child_path, child))
        else:
            raise RuntimeError(f'installed PWL target is missing: {role}')
    return tuple(result)


def _coefficients_exact(
        targets: Sequence[tuple[str, str, torch.nn.Module]],
        coefficients: Mapping[str, Any]) -> bool:
    if not isinstance(coefficients, Mapping) or set(coefficients) != {
            'breakpoints', 'slopes', 'intercepts'}:
        return False
    expected = {
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in coefficients.items()}
    return all(
        all(torch.equal(getattr(module, name).detach().cpu(), value)
            for name, value in expected.items())
        for _, _, module in targets)


def _loss(model, inputs, data_samples) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    losses = model(inputs, data_samples=data_samples, mode='loss')
    selected = {
        name: value for name, value in losses.items()
        if isinstance(name, str) and name.startswith('loss')
    } if isinstance(losses, Mapping) else {}
    if not selected or any(not isinstance(item, torch.Tensor)
                           for item in selected.values()):
        raise RuntimeError('Stage-A full-model loss has no tensor losses')
    if any(not torch.isfinite(item).all() for item in selected.values()):
        raise RuntimeError('Stage-A full-model loss is non-finite')
    total = sum(item.mean() for item in selected.values())
    if not torch.isfinite(total):
        raise RuntimeError('Stage-A total loss is non-finite')
    return selected, total


def _adam_tensors(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer) -> tuple[
            dict[str, dict[str, torch.Tensor]], int, int, int]:
    if not isinstance(optimizer, torch.optim.Adam):
        raise RuntimeError('PWL Stage-A requires Adam')
    rows = {}
    steps = []
    tensor_count = 0
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not state:
            continue
        normalized = {}
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise RuntimeError(f'Adam state is missing/non-finite: {name}.{key}')
            normalized[key] = value.detach().cpu().clone()
            tensor_count += 1
        current = _step(normalized['step'])
        if current is None:
            raise RuntimeError(f'Adam step is invalid: {name}')
        steps.append(current)
        rows[name] = normalized
    if not rows:
        raise RuntimeError('PWL Stage-A Adam state is empty')
    return rows, min(steps), len(rows), tensor_count


def _load_adam_tensors(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        rows: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    parameters = dict(model.named_parameters())
    if set(rows) - set(parameters):
        raise RuntimeError('Stage-A Adam resume parameter identity changed')
    for name, state in rows.items():
        parameter = parameters[name]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise RuntimeError('Stage-A Adam resume fields are invalid')
        optimizer.state[parameter] = {
            key: value.detach().to(
                'cpu' if key == 'step' else parameter.device).clone()
            for key, value in state.items()}


def execute_pwl_stage_a_model(
        *, model: torch.nn.Module, inputs: torch.Tensor, data_samples: Any,
        optimizer: torch.optim.Optimizer,
        model_factory: Callable[[], torch.nn.Module],
        optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
        identity_factory: Callable[[], torch.nn.Module], export_path: Path,
        function_name: str, roles: Sequence[str],
        coefficients: Mapping[str, Any],
        operation_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Execute full forward/loss/backward/Adam/export/load/resume evidence."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError('PWL Stage-A model must be a torch module')
    if not isinstance(inputs, torch.Tensor) or inputs.numel() == 0:
        raise ValueError('PWL Stage-A inputs must be a non-empty tensor')
    export = Path(export_path)
    if export.exists() or export.is_symlink():
        raise FileExistsError(f'refusing to overwrite Stage-A export: {export}')
    export.parent.mkdir(parents=True, exist_ok=True)
    targets = _pwl_modules(model, function_name=function_name, roles=roles)
    if not _coefficients_exact(targets, coefficients):
        raise RuntimeError('installed PWL coefficients differ from fit')
    expected_operation = {
        'function': function_name,
        'source': operation_manifest.get('source'),
        'operation_roles': list(roles),
        'exact_input_roles': operation_manifest.get('exact_input_roles'),
        'segments_per_role': len(coefficients['slopes']),
        'domain_handling': (
            {'kind': 'continuous-asymptotic-tail-v1',
             'left': 'constant-endpoint',
             'right': 'identity-plus-endpoint-offset'}
            if function_name in {'silu', 'gelu', 'softplus'}
            else {'kind': 'clamp'}),
        'hardware_latency_claimed': False}
    if dict(operation_manifest) != expected_operation:
        raise RuntimeError('PWL operation manifest does not match targets')

    captures = {role: {'input': [], 'output': []}
                for role, _, _ in targets}
    handles = []
    for role, _, module in targets:
        def pre_hook(_module, arguments, *, target_role=role):
            value = arguments[0] if arguments else None
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                raise RuntimeError(
                    f'PWL target input has no gradient path: {target_role}')
            value.register_hook(
                lambda gradient, selected=target_role:
                captures[selected]['input'].append(gradient.detach()))

        def forward_hook(_module, _arguments, output, *, target_role=role):
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                raise RuntimeError(
                    f'PWL target output has no gradient path: {target_role}')
            output.register_hook(
                lambda gradient, selected=target_role:
                captures[selected]['output'].append(gradient.detach()))

        handles.extend([
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(forward_hook)])
    model.train()
    _disable_stochastic_depth_for_structural_smoke(model)
    optimizer.zero_grad(set_to_none=True)
    before = {name: value.detach().cpu().clone()
              for name, value in model.named_parameters()}
    try:
        selected, total = _loss(model, inputs, data_samples)
        total.backward()
    finally:
        for handle in handles:
            handle.remove()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    if not gradients or any(not torch.isfinite(item).all()
                            for item in gradients):
        raise RuntimeError('PWL Stage-A gradients are missing/non-finite')
    target_evidence = []
    for role, module_path, _ in targets:
        row = captures[role]
        if len(row['input']) != 1 or len(row['output']) != 1:
            raise RuntimeError(f'PWL target gradient count is invalid: {role}')
        input_norm = float(row['input'][0].float().norm().cpu())
        output_norm = float(row['output'][0].float().norm().cpu())
        if (not math.isfinite(input_norm) or input_norm <= 0
                or not math.isfinite(output_norm) or output_norm <= 0):
            raise RuntimeError(f'PWL target gradient has no signal: {role}')
        target_evidence.append({
            'operation_role': role, 'module_path': module_path,
            'input_gradient_norm': input_norm,
            'output_gradient_norm': output_norm,
            'gradient_finite': True})
    optimizer.step()
    after_parameters = dict(model.named_parameters())
    updated = sum(
        not torch.equal(value, after_parameters[name].detach().cpu())
        for name, value in before.items())
    if updated <= 0:
        raise RuntimeError('PWL Stage-A Adam step changed no parameters')
    state = _state_cpu(model)
    adam, first_step, state_parameter_count, state_tensor_count = (
        _adam_tensors(model, optimizer))
    payload = {'model_state': state, 'adam_state': adam}
    if not _tensor_tree(payload):
        raise RuntimeError('PWL Stage-A payload is not tensor-only')
    torch.save(payload, export)
    try:
        loaded = torch.load(export, map_location='cpu', weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f'PWL Stage-A restricted tensor load failed: {error}') from error
    if not _tensor_tree(loaded) or set(loaded) != {'model_state', 'adam_state'}:
        raise RuntimeError('PWL Stage-A restricted payload is invalid')

    restored = model_factory()
    restored.load_state_dict(dict(loaded['model_state']), strict=True)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = inputs.device
    restored.to(device).eval()
    state_exact = _state_exact(state, _state_cpu(restored))
    model.eval()
    with torch.inference_mode():
        reference = model(inputs, data_samples=data_samples, mode='tensor')
        replay = restored(inputs, data_samples=data_samples, mode='tensor')
    if (not isinstance(reference, torch.Tensor)
            or not torch.isfinite(reference).all()
            or not isinstance(replay, torch.Tensor)
            or not torch.equal(reference.detach(), replay.detach())
            or not state_exact):
        raise RuntimeError('PWL Stage-A export/load identity failed')

    resumed_optimizer = optimizer_factory(restored)
    if not isinstance(resumed_optimizer, torch.optim.Adam):
        raise RuntimeError('PWL Stage-A resume optimizer must be Adam')
    _load_adam_tensors(restored, resumed_optimizer, loaded['adam_state'])
    restored.train()
    _disable_stochastic_depth_for_structural_smoke(restored)
    resumed_optimizer.zero_grad(set_to_none=True)
    _, resumed_loss = _loss(restored, inputs, data_samples)
    resumed_loss.backward()
    resumed_optimizer.step()
    _, resumed_step, _, _ = _adam_tensors(restored, resumed_optimizer)
    if resumed_step < first_step + 1:
        raise RuntimeError('PWL Stage-A Adam resume did not advance')

    identity = identity_factory().to(device)
    filtered = dict(state)
    for _, module_path, _ in targets:
        for name in ('breakpoints', 'slopes', 'intercepts'):
            filtered.pop(f'{module_path}.{name}', None)
    identity.load_state_dict(filtered, strict=True)
    from mmpose.models.utils.hardware_friendly.pwl import (
        PiecewiseLinearApproximation)
    if any(isinstance(item, PiecewiseLinearApproximation)
           for item in identity.modules()):
        raise RuntimeError('PWL identity mode still contains approximation')
    identity.eval()
    with torch.inference_mode():
        identity_output = identity(
            inputs, data_samples=data_samples, mode='tensor')
    if (not isinstance(identity_output, torch.Tensor)
            or not torch.isfinite(identity_output).all()
            or identity_output.shape != reference.shape):
        raise RuntimeError('PWL identity mode is invalid')

    return {
        'protocol': copy.deepcopy(_STRUCTURAL_SMOKE_PROTOCOL),
        'checks': {
            'forward': True, 'loss': True, 'backward': True,
            'optimizer_step': True, 'finite_loss': True,
            'finite_gradients': True, 'pwl_target_count': len(targets),
            'pwl_target_gradients': True, 'identity_mode': True,
            'coefficients_exact': True, 'roles_exact': True,
            'restricted_tensor_only_load': True,
            'state_round_trip_exact': state_exact,
            'output_round_trip_exact': True,
            'optimizer_resume_step': True,
        },
        'losses': {name: float(value.detach().mean().cpu())
                   for name, value in sorted(selected.items())},
        'pwl_targets': target_evidence,
        'optimizer': {
            'kind': 'Adam', 'updated_parameter_count': updated,
            'state_tensor_count': state_tensor_count,
            'first_min_step': first_step,
            'resumed_min_step': resumed_step, 'state_finite': True,
        },
        'identity': {
            'enabled': True, 'output_shape': list(identity_output.shape),
            'finite': True},
        'output': {
            'shape': list(reference.shape), 'dtype': str(reference.dtype),
            'sha256': _tensor_sha256(reference)},
        'operation': dict(operation_manifest),
        'export': {
            'path': str(export), 'sha256': _sha256(export),
            'bytes': export.stat().st_size,
            'format': 'torch-weights-only-model-and-adam-tensors-v1'},
    }


def _strict_file(root: Path, value: object, *, label: str) -> Path:
    relative = canonical_relative_path(value, label=label)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not use symlinks')
    if not cursor.is_file():
        raise ValueError(f'{label} is missing')
    return cursor


def _binding(value: object, *, label: str) -> dict[str, str]:
    if (not isinstance(value, Mapping)
            or set(value) != {'path', 'sha256'}
            or not isinstance(value.get('path'), str)
            or not isinstance(value.get('sha256'), str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise ValueError(f'{label} binding is invalid')
    try:
        relative = canonical_relative_path(value['path'], label=label)
    except ValueError as error:
        raise ValueError(f'{label} binding is invalid: {error}') from error
    return {'path': relative.as_posix(), 'sha256': value['sha256']}


def _policy(config) -> dict[str, Any]:
    value = config.numeric_optimization.pwl
    result = {name: value[name] for name in (
        'enabled_function', 'source', 'roles', 'domain', 'segments',
        'grid_points', 'saturation', 'qat_form', 'selection_policy')}
    result['roles'] = list(result['roles'])
    result['domain'] = list(result['domain'])
    return result


def _production_dependencies(
        *, value: Mapping[str, Any], repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    from .checkpoints import (
        authorize_manifest_candidate, authorize_tracked_config)

    root = Path(repository_root).resolve(strict=True)
    candidate_id = value.get('candidate_id')
    authorized = authorize_manifest_candidate(
        root, manifest_path, candidate_id)
    candidate = authorized.candidate
    if candidate.features.get('numeric_kind') not in {
            'pwl', 'pwl-combined'}:
        raise ValueError('PWL Stage-A candidate kind is invalid')
    config = authorize_tracked_config(
        root, manifest_path, candidate).load_config()
    policy = _policy(config)
    fit_reference = _binding(value.get('fit_artifact'), label='PWL fit')
    expected_fit_path = (
        Path('work_dirs/optimization') / candidate.route / candidate.id /
        str(candidate.seed) / 'calibrate/calibrate.json')
    if Path(fit_reference['path']) != expected_fit_path:
        raise ValueError('PWL Stage-A fit path is not canonical')
    fit = load_pwl_fit_reference(
        fit_reference, repository_root=root,
        expected_candidate_id=candidate.id, expected_policy=policy)
    selection_reference = _binding(
        value.get('selection_artifact'), label='PWL selection')
    selection = load_pwl_selection_reference(
        selection_reference, repository_root=root,
        manifest_path=manifest_path)
    if selection.get('selected_candidate_id') != candidate.id:
        raise ValueError('PWL Stage-A candidate is not selected')
    if selection.get('admission_kind') != 'combined-parent-authority-v1':
        selected_rows = [
            row for row in selection.get('candidates', ())
            if isinstance(row, Mapping)
            and row.get('candidate_id') == candidate.id]
        if (len(selected_rows) != 1
                or selected_rows[0].get(
                    'calibration_artifact') != fit_reference):
            raise ValueError(
                'PWL Stage-A fit differs from selected calibration')
    installation_reference = _binding(
        value.get('installation'), label='PWL installation')
    expected_installation = (
        Path('work_dirs/optimization') / candidate.route / candidate.id /
        str(candidate.seed) / 'convert/pwl-installation.json')
    if Path(installation_reference['path']) != expected_installation:
        raise ValueError('PWL Stage-A installation path is not canonical')
    installation = load_pwl_installation_reference(
        installation_reference, repository_root=root,
        expected_candidate_id=candidate.id,
        expected_fit_reference=fit_reference, expected_fit=fit)
    calibration_path = root / expected_fit_path
    try:
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        from .numeric_calibration import validate_calibration_provenance
        validate_calibration_provenance(
            calibration, expected_candidate=candidate,
            repository_root=root, manifest_path=manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f'PWL Stage-A calibration provenance is invalid: {error}') \
            from error
    if calibration.get('pwl_fit') != fit:
        raise ValueError('PWL Stage-A fit differs from calibration envelope')
    return {
        'candidate_id': candidate.id,
        'git_commit': authorized.source['git_commit'],
        'config': {'path': candidate.config.as_posix(),
                   'sha256': _sha256(authorized.config_path)},
        'checkpoint': {'path': candidate.checkpoint.as_posix(),
                       'sha256': candidate.checkpoint_sha256},
        'policy': policy, 'fit_reference': fit_reference, 'fit': fit,
        'selection_reference': selection_reference,
        'installation_reference': installation_reference,
        'installation': installation,
        'candidate': candidate,
    }


def validate_pwl_stage_a_artifact(
        artifact_path: Path, *, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    """Public validator reconstructing tracked fit, policy and selection."""
    root = Path(repository_root).resolve(strict=True)
    supplied = canonical_path(
        str(artifact_path), label='PWL Stage-A artifact',
        allow_absolute=True)
    if supplied.is_absolute():
        try:
            relative = supplied.absolute().relative_to(root)
        except ValueError as error:
            raise ValueError('PWL Stage-A artifact escapes repository') from error
    else:
        relative = supplied
    artifact = _strict_file(root, relative.as_posix(), label='PWL Stage-A')
    if (relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name != 'smoke.json'
            or relative.parent.name != 'smoke-stage-a'):
        raise ValueError('PWL Stage-A artifact path is not canonical')
    try:
        value = json.loads(artifact.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('PWL Stage-A artifact is invalid JSON') from error
    fields = {
        'schema_version', 'artifact_kind', 'candidate_id', 'source', 'config',
        'checkpoint', 'fit_artifact', 'selection_artifact', 'installation',
        'policy', 'data', 'gpu', 'execution', 'claim_limits'}
    if (not isinstance(value, Mapping) or set(value) != fields
            or value.get('schema_version') != 2
            or value.get('artifact_kind') != 'pwl-stage-a-full-model-smoke'):
        raise ValueError('PWL Stage-A artifact identity is invalid')
    dependency = _production_dependencies(
        value=value, repository_root=root, manifest_path=manifest_path)
    if (value['candidate_id'] != dependency['candidate_id']
            or value['source'] != {'git_commit': dependency['git_commit']}
            or value['config'] != dependency['config']
            or value['checkpoint'] != dependency['checkpoint']
            or value['fit_artifact'] != dependency['fit_reference']
            or value['selection_artifact'] != dependency['selection_reference']
            or value['installation'] != dependency['installation_reference']):
        raise ValueError('PWL Stage-A source authority is invalid')
    if value['policy'] != dependency['policy']:
        raise ValueError('PWL Stage-A policy binding is invalid')
    candidate = dependency.get('candidate')
    if candidate is not None:
        expected_parent = (Path('work_dirs/optimization') / candidate.route /
                           candidate.id / str(candidate.seed) / 'smoke-stage-a')
        if relative.parent != expected_parent:
            raise ValueError('PWL Stage-A candidate path is not canonical')
    data = value['data']
    if (not isinstance(data, Mapping) or set(data) != {
            'dataset', 'split', 'batch_size', 'packed_production_pipeline',
            'input_shape', 'sample_ids'}
            or data['dataset'] != 'coco' or data['split'] != 'train2017'
            or data['batch_size'] != 1
            or data['packed_production_pipeline'] is not True
            or data['input_shape'] != [1, 3, 256, 192]
            or not isinstance(data['sample_ids'], list)
            or len(data['sample_ids']) != 1
            or isinstance(data['sample_ids'][0], bool)
            or not isinstance(data['sample_ids'][0], int)
            or data['sample_ids'][0] < 0):
        raise ValueError('PWL Stage-A data contract is invalid')
    gpu = value['gpu']
    if (not isinstance(gpu, Mapping)
            or set(gpu) != {'logical', 'physical_index', 'lease'}
            or gpu['logical'] != 'cuda:0'
            or isinstance(gpu['physical_index'], bool)
            or not isinstance(gpu['physical_index'], int)
            or gpu['physical_index'] < 0):
        raise ValueError('PWL Stage-A GPU identity is invalid')
    try:
        lease = validate_gpu_lease(gpu['lease'])
    except LatencyError as error:
        raise ValueError(f'PWL Stage-A GPU lease is invalid: {error}') from error
    if (lease['stage_id'] != f'{value["candidate_id"]}:smoke-stage-a'
            or lease['device_index'] != gpu['physical_index']):
        raise ValueError('PWL Stage-A GPU lease identity is invalid')
    execution = value['execution']
    if not isinstance(execution, Mapping) or set(execution) != {
            'protocol', 'checks', 'losses', 'pwl_targets', 'optimizer',
            'identity',
            'output', 'operation', 'export'}:
        raise ValueError('PWL Stage-A execution fields are invalid')
    if execution['protocol'] != _STRUCTURAL_SMOKE_PROTOCOL:
        raise ValueError('PWL Stage-A structural smoke protocol is invalid')
    checks = execution['checks']
    boolean = {
        'forward', 'loss', 'backward', 'optimizer_step', 'finite_loss',
        'finite_gradients', 'pwl_target_gradients', 'identity_mode',
        'coefficients_exact', 'roles_exact', 'restricted_tensor_only_load',
        'state_round_trip_exact', 'output_round_trip_exact',
        'optimizer_resume_step'}
    roles = list(dependency['policy']['roles'])
    if (not isinstance(checks, Mapping)
            or set(checks) != boolean | {'pwl_target_count'}
            or any(checks.get(name) is not True for name in boolean)
            or checks.get('pwl_target_count') != len(roles)):
        raise ValueError('PWL Stage-A checks are incomplete')
    losses = execution['losses']
    if (not isinstance(losses, Mapping) or not losses
            or any(not isinstance(name, str) or not name.startswith('loss')
                   or isinstance(item, bool)
                   or not isinstance(item, (int, float))
                   or not math.isfinite(float(item))
                   for name, item in losses.items())):
        raise ValueError('PWL Stage-A losses are invalid')
    targets = execution['pwl_targets']
    expected_paths = [
        role if dependency['policy']['source'] == 'module'
        else f'{role}._numeric_pwl_{dependency["policy"]["enabled_function"]}'
        for role in roles]
    if (not isinstance(targets, list) or len(targets) != len(roles)
            or [row.get('operation_role') for row in targets] != roles
            or [row.get('module_path') for row in targets] != expected_paths
            or any(set(row) != {
                'operation_role', 'module_path', 'input_gradient_norm',
                'output_gradient_norm', 'gradient_finite'}
                or row['gradient_finite'] is not True
                or any(isinstance(row[name], bool)
                       or not isinstance(row[name], (int, float))
                       or not math.isfinite(float(row[name]))
                       or float(row[name]) <= 0
                       for name in ('input_gradient_norm',
                                    'output_gradient_norm'))
                for row in targets)):
        raise ValueError('PWL Stage-A target gradients are invalid')
    optimizer = execution['optimizer']
    if (not isinstance(optimizer, Mapping) or set(optimizer) != {
            'kind', 'updated_parameter_count', 'state_tensor_count',
            'first_min_step', 'resumed_min_step', 'state_finite'}
            or optimizer['kind'] != 'Adam'
            or optimizer['state_finite'] is not True
            or any(isinstance(optimizer[name], bool)
                   or not isinstance(optimizer[name], int)
                   or optimizer[name] < 1 for name in (
                       'updated_parameter_count', 'state_tensor_count',
                       'first_min_step'))
            or optimizer['resumed_min_step'] < optimizer['first_min_step'] + 1):
        raise ValueError('PWL Stage-A Adam evidence is invalid')
    output = execution['output']
    identity = execution['identity']
    if (not isinstance(output, Mapping)
            or set(output) != {'shape', 'dtype', 'sha256'}
            or output['shape'] != [1, 17, 64, 48]
            or output['dtype'] not in {'torch.float16', 'torch.float32'}
            or not _SHA256.fullmatch(str(output['sha256']))
            or identity != {'enabled': True,
                            'output_shape': [1, 17, 64, 48],
                            'finite': True}):
        raise ValueError('PWL Stage-A output/identity evidence is invalid')
    installation = dependency['installation']
    operation = installation.get('operation_manifest')
    if execution['operation'] != operation:
        raise ValueError('PWL Stage-A operation manifest is invalid')
    export = execution['export']
    if (not isinstance(export, Mapping) or set(export) != {
            'path', 'sha256', 'bytes', 'format'}
            or export.get('format') !=
            'torch-weights-only-model-and-adam-tensors-v1'):
        raise ValueError('PWL Stage-A export binding is invalid')
    exported = _strict_file(root, export.get('path'), label='PWL Stage-A export')
    if (exported.parent != artifact.parent
            or not _SHA256.fullmatch(str(export.get('sha256')))
            or _sha256(exported) != export['sha256']
            or isinstance(export.get('bytes'), bool)
            or not isinstance(export.get('bytes'), int)
            or export['bytes'] != exported.stat().st_size):
        raise ValueError('PWL Stage-A export hash/location is invalid')
    try:
        payload = torch.load(exported, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('PWL Stage-A restricted export load failed') from error
    if (not _tensor_tree(payload) or not isinstance(payload, Mapping)
            or set(payload) != {'model_state', 'adam_state'}):
        raise ValueError('PWL Stage-A export is not tensor-only')
    state = payload['model_state']
    coefficients = dependency['fit']['coefficients']
    for module_path in expected_paths:
        for name, expected_value in coefficients.items():
            actual = state.get(f'{module_path}.{name}')
            if (not isinstance(actual, torch.Tensor)
                    or not torch.equal(
                        actual.detach().cpu(),
                        torch.as_tensor(expected_value, dtype=torch.float64))):
                raise ValueError(
                    'PWL Stage-A exported coefficients/roles are invalid')
    limits = value['claim_limits']
    if limits != {
            'hardware_latency_claimed': False,
            'fpga_speedup_claimed': False,
            'fastmamba_composite_mechanisms_inherited': False}:
        raise ValueError('PWL Stage-A claim limits are invalid')
    return dict(value)


def pwl_stage_a_binding(
        candidate, *, repository_root: Path, manifest_path: Path) -> dict[str, str]:
    root = Path(repository_root).resolve(strict=True)
    path = (root / 'work_dirs/optimization' / candidate.route /
            candidate.id / str(candidate.seed) / 'smoke-stage-a/smoke.json')
    value = validate_pwl_stage_a_artifact(
        path, repository_root=root, manifest_path=manifest_path)
    if value['candidate_id'] != candidate.id:
        raise ValueError('PWL Stage-A candidate identity changed')
    return {'path': path.relative_to(root).as_posix(), 'sha256': _sha256(path)}


def _canonical_gpu_lock(repository_root: Path) -> Path:
    try:
        common = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'], cwd=repository_root,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot derive canonical GPU lock') from error
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = repository_root / common_path
    common_path = common_path.resolve(strict=True)
    if common_path.name != '.git':
        raise ValueError('Git common directory is not a checkout .git')
    return common_path.parent / 'work_dirs/optimization/gpu.lock'


def _active_controller_lease(
        candidate_id: str, device_index: int, *, repository_root: Path,
        now: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """Read the controller's already-held lease without acquiring it again."""
    lock_path = _canonical_gpu_lock(repository_root)
    try:
        stream = lock_path.open('r+', encoding='utf-8')
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            value = json.load(stream)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            raise ValueError('canonical GPU lease is not actively held')
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read active GPU lease: {error}') from error
    finally:
        if 'stream' in locals():
            stream.close()
    value = validate_gpu_lease(value)
    if value['stage_id'] != f'{candidate_id}:smoke-stage-a':
        raise ValueError('active GPU lease does not match PWL smoke stage')
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if value['boot_id'] != boot_id:
        raise ValueError('active GPU lease belongs to a different boot')
    if value['device_index'] != device_index:
        raise ValueError('active GPU lease device does not match PWL smoke')
    timestamp = datetime.fromisoformat(value['timestamp'])
    current = now() if now is not None else datetime.now(timezone.utc)
    if current - timestamp > _LEASE_MAX_AGE:
        raise ValueError('active GPU lease timestamp is stale')
    if timestamp - current > _LEASE_MAX_FUTURE_SKEW:
        raise ValueError('active GPU lease timestamp is in the future')
    if os.getpid() not in controller_process_tree({value['pid']}):
        raise ValueError('PWL smoke process is not a controller descendant')
    return value


def _refresh_controller_lease_for_artifact(
        initial: Mapping[str, Any], candidate_id: str, device_index: int, *,
        repository_root: Path,
        now: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """Refresh a long-running smoke lease without accepting a new owner."""
    refreshed = _active_controller_lease(
        candidate_id, device_index, repository_root=repository_root, now=now)
    return validate_refreshed_gpu_lease(initial, refreshed)


def _prepare_smoke_output_directory(output: Path) -> None:
    """Admit the controller-created directory while preserving its log."""
    output = Path(output)
    if output.is_symlink():
        raise FileExistsError('PWL smoke output must not be a symlink')
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not output.is_dir():
        raise FileExistsError('PWL smoke output is not a directory')
    unexpected = tuple(
        path.name for path in output.iterdir()
        if not (
            not path.is_symlink()
            and path.is_file()
            and re.fullmatch(r'attempt-[1-9][0-9]*\.log', path.name)))
    if unexpected:
        raise FileExistsError(
            f'PWL smoke output contains unexpected entries: {unexpected}')


def _smoke_dataloader(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    dataset = result.get('dataset')
    if (not isinstance(dataset, Mapping)
            or dataset.get('type') != 'CocoDataset'
            or dataset.get('data_mode') != 'topdown'
            or dataset.get('data_root') not in {'data/coco', 'data/coco/'}
            or dataset.get('ann_file') !=
            'annotations/person_keypoints_train2017.json'
            or not isinstance(dataset.get('pipeline'), (list, tuple))
            or not dataset['pipeline']
            or dataset['pipeline'][-1].get('type') != 'PackPoseInputs'):
        raise ValueError(
            'PWL Stage-A requires packed production COCO train pipeline')
    result.update({
        'batch_size': 1, 'num_workers': 0, 'persistent_workers': False,
        'drop_last': False,
        'sampler': dict(type='DefaultSampler', shuffle=False)})
    result.pop('batch_sampler', None)
    result.pop('worker_init_fn', None)
    return result


def _build_model(config_authority, state, device, *, install: bool):
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS

    from .checkpoints import (
        ConfigAuthority, load_tensor_state_strict,
        neutralize_model_initializers)
    from .numeric_conversion import NumericRuntimeHook

    if not isinstance(config_authority, ConfigAuthority):
        raise TypeError('PWL Stage-A requires sealed config authority')
    config = config_authority.load_config()
    init_default_scope(config.get('default_scope', 'mmpose'))
    safe_config = neutralize_model_initializers(config)
    model = MODELS.build(safe_config.model)
    load_tensor_state_strict(model, state)
    if config.numeric_optimization.get(
            'candidate_kind') == 'pwl-combined':
        from .combined_candidate import prune_disabled_pif
        prune_disabled_pif(model)
    if install:
        NumericRuntimeHook.apply_to_model(
            model, config.numeric_optimization)
    return model.to(device)


def _stage_a_model_factories(
        config_authority,
        state: Mapping[str, torch.Tensor],
) -> tuple[Callable[[], torch.nn.Module], Callable[[], torch.nn.Module]]:
    """Build replay closures from the already-validated checkpoint state."""
    device = torch.device('cpu')

    def fitted_factory() -> torch.nn.Module:
        return _build_model(
            config_authority, state, device, install=True)

    def identity_factory() -> torch.nn.Module:
        return _build_model(
            config_authority, state, device, install=False)

    return fitted_factory, identity_factory


def build_stage_a_optimizer(model, optim_wrapper) -> torch.optim.Adam:
    value = optim_wrapper.get('optimizer') \
        if isinstance(optim_wrapper, Mapping) else None
    if not isinstance(value, Mapping) or value.get('type') != 'Adam':
        raise ValueError('PWL Stage-A resolved paper config must use Adam')
    allowed = {'type', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
    if set(value) - allowed or 'lr' not in value:
        raise ValueError('PWL Stage-A Adam config is invalid')
    return torch.optim.Adam(
        model.parameters(), **{name: item for name, item in value.items()
                               if name != 'type'})


def run_pwl_stage_a_smoke(
        *, repository_root: Path, manifest_path: Path, candidate_id: str,
        output_relative: Path, device_index: int) -> Path:
    """Run the real selected PWL on one packed COCO batch under GPU lease."""
    if device_index < 0:
        raise ValueError('PWL smoke device index must be non-negative')
    root = Path(repository_root).resolve(strict=True)
    relative = canonical_relative_path(
        str(output_relative), label='PWL smoke output')
    if (relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name != 'smoke-stage-a'):
        raise ValueError('PWL smoke output path is invalid')
    output = root / relative
    from mmengine.runner import Runner

    from .checkpoints import (
        authorize_manifest_candidate, authorize_pwl_runtime_config,
        tensor_state)
    from .numeric_runtime import validate_numeric_convert_artifact

    authorized = authorize_manifest_candidate(root, manifest_path, candidate_id)
    candidate = authorized.candidate
    if candidate.features.get('numeric_kind') not in {
            'pwl', 'pwl-combined'}:
        raise ValueError('PWL smoke requires a PWL candidate')
    expected_output = (Path('work_dirs/optimization') / candidate.route /
                       candidate.id / str(candidate.seed) / 'smoke-stage-a')
    if relative != expected_output:
        raise ValueError('PWL smoke output is not canonical for candidate')
    conversion_path = (root / 'work_dirs/optimization' / candidate.route /
                       candidate.id / str(candidate.seed) /
                       'convert/convert.json')
    conversion = json.loads(conversion_path.read_text(encoding='utf-8'))
    validate_numeric_convert_artifact(
        conversion, candidate=candidate, repository_root=root,
        manifest_path=manifest_path, artifact_path=conversion_path)
    config_authority = authorize_pwl_runtime_config(
        root, manifest_path, candidate, conversion_path=conversion_path)
    config = config_authority.load_config()
    policy = _policy(config)
    fit_reference = conversion['result']['runtime_bindings']['calibration']
    fit = load_pwl_fit_reference(
        fit_reference, repository_root=root,
        expected_candidate_id=candidate.id, expected_policy=policy)
    installation_reference = conversion['result']['installation']
    installation = load_pwl_installation_reference(
        installation_reference, repository_root=root,
        expected_candidate_id=candidate.id,
        expected_fit_reference=fit_reference, expected_fit=fit)
    selection_reference = conversion['result']['selection']
    selection = load_pwl_selection_reference(
        selection_reference, repository_root=root, manifest_path=manifest_path)
    if selection['selected_candidate_id'] != candidate.id:
        raise ValueError('PWL smoke candidate is not selected')
    loader_config = _smoke_dataloader(config.train_dataloader)
    state = tensor_state(
        authorized.checkpoint_path,
        expected_sha256=authorized.candidate.checkpoint_sha256)
    _prepare_smoke_output_directory(output)
    try:
        lease_value = _active_controller_lease(
            candidate.id, device_index, repository_root=root)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(device_index):
            raise RuntimeError(
                'CUDA_VISIBLE_DEVICES must expose exactly the leased GPU')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable for PWL Stage-A smoke')
        torch.manual_seed(candidate.seed)
        torch.cuda.manual_seed_all(candidate.seed)
        torch.use_deterministic_algorithms(True)
        device = torch.device('cuda:0')
        model = _build_model(
            config_authority, state, device, install=True)
        loader = Runner.build_dataloader(
            loader_config, seed=candidate.seed, diff_rank_seed=False)
        batch = next(iter(loader))
        processed = model.data_preprocessor(batch, training=True)
        inputs, samples = processed['inputs'], processed['data_samples']
        if (not isinstance(inputs, torch.Tensor)
                or list(inputs.shape) != [1, 3, 256, 192]
                or not isinstance(samples, list) or len(samples) != 1):
            raise RuntimeError('PWL Stage-A packed batch is invalid')

        fitted_factory, identity_factory = _stage_a_model_factories(
            config_authority, state)

        execution = execute_pwl_stage_a_model(
            model=model, inputs=inputs, data_samples=samples,
            optimizer=build_stage_a_optimizer(model, config.optim_wrapper),
            model_factory=fitted_factory,
            optimizer_factory=lambda restored: build_stage_a_optimizer(
                restored, config.optim_wrapper),
            identity_factory=identity_factory,
            export_path=output / 'round-trip.pth',
            function_name=policy['enabled_function'],
            roles=policy['roles'], coefficients=fit['coefficients'],
            operation_manifest=installation['operation_manifest'])
        execution['export']['path'] = (
            output / 'round-trip.pth').relative_to(root).as_posix()
        sample_id = samples[0].metainfo.get('img_id')
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise RuntimeError('PWL Stage-A sample image id is invalid')
        lease_value = _refresh_controller_lease_for_artifact(
            lease_value, candidate.id, device_index, repository_root=root)
        artifact = {
            'schema_version': 2,
            'artifact_kind': 'pwl-stage-a-full-model-smoke',
            'candidate_id': candidate.id,
            'source': {'git_commit': authorized.source['git_commit']},
            'config': {'path': candidate.config.as_posix(),
                       'sha256': _sha256(authorized.config_path)},
            'checkpoint': {'path': candidate.checkpoint.as_posix(),
                           'sha256': candidate.checkpoint_sha256},
            'fit_artifact': fit_reference,
            'selection_artifact': selection_reference,
            'installation': installation_reference,
            'policy': policy,
            'data': {
                'dataset': 'coco', 'split': 'train2017', 'batch_size': 1,
                'packed_production_pipeline': True,
                'input_shape': list(inputs.shape),
                'sample_ids': [sample_id]},
            'gpu': {'logical': 'cuda:0',
                    'physical_index': device_index, 'lease': lease_value},
            'execution': execution,
            'claim_limits': {
                'hardware_latency_claimed': False,
                'fpga_speedup_claimed': False,
                'fastmamba_composite_mechanisms_inherited': False},
        }
        artifact_path = output / 'smoke.json'
        temporary = output / f'.smoke.{os.getpid()}.tmp'
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact_path)
        validate_pwl_stage_a_artifact(
            artifact_path, repository_root=root, manifest_path=manifest_path)
        return artifact_path
    except BaseException:
        for path in (
                output / 'smoke.json', output / 'round-trip.pth'):
            path.unlink(missing_ok=True)
        for path in output.glob('.smoke.*.tmp'):
            path.unlink(missing_ok=True)
        raise


__all__ = [
    'build_stage_a_optimizer', 'execute_pwl_stage_a_model',
    'pwl_stage_a_binding', 'run_pwl_stage_a_smoke',
    'validate_pwl_stage_a_artifact']
