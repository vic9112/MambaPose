"""One-batch Stage-A training and round-trip smoke for Binary Q/K."""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

import torch

from .binary_operation import (
    CANONICAL_BINARY_ATTENTION_MODULES, build_binary_operation_manifest,
    validate_binary_operation_manifest)
from .artifacts import lexical_repository_root
from .latency import LatencyError, validate_gpu_lease


_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode('ascii'))
    digest.update(str(tuple(contiguous.shape)).encode('ascii'))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _state_exact(
        first: Mapping[str, torch.Tensor],
        second: Mapping[str, torch.Tensor]) -> bool:
    return (
        set(first) == set(second)
        and all(torch.equal(first[name], second[name]) for name in first)
    )


def _binary_qk_targets(
        model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    modules = dict(model.named_modules())
    rows = []
    for name in CANONICAL_BINARY_ATTENTION_MODULES:
        module = modules.get(name)
        if (
                module is None or getattr(module, 'qk_mode', None) != 'binary'
                or not isinstance(getattr(module, 'to_qkv', None), torch.nn.Linear)
                or not isinstance(module.to_qkv.weight, torch.nn.Parameter)):
            raise RuntimeError(
                f'Binary Q/K gradient target is missing or invalid: {name}')
        rows.append((f'{name}.to_qkv.weight', module.to_qkv.weight))
    return tuple(rows)


def _optimizer_step_value(value: object) -> float | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def execute_binary_stage_a_model(
        *, model: torch.nn.Module, inputs: torch.Tensor, data_samples: Any,
        optimizer: torch.optim.Optimizer,
        model_factory: Callable[[], torch.nn.Module], export_path: Path,
        operation_builder: Callable[[torch.nn.Module], Mapping[str, Any]] = (
            build_binary_operation_manifest),
        require_binary_targets: bool = True,
        ) -> dict[str, Any]:
    """Execute the same train/export/load core used by the real-model CLI."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Stage-A smoke model must be a torch module')
    if not isinstance(inputs, torch.Tensor) or inputs.numel() == 0:
        raise ValueError('Stage-A smoke inputs must be a non-empty tensor')
    export = Path(export_path)
    if export.exists() or export.is_symlink():
        raise FileExistsError(f'refusing to overwrite Stage-A export: {export}')
    export.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(inputs, data_samples=data_samples, mode='loss')
    if not isinstance(losses, Mapping) or not losses:
        raise RuntimeError('Stage-A full-model loss must be a non-empty mapping')
    selected = {
        name: value for name, value in losses.items()
        if isinstance(name, str) and name.startswith('loss')
    }
    if not selected or any(not isinstance(value, torch.Tensor)
                           for value in selected.values()):
        raise RuntimeError('Stage-A loss mapping contains no tensor losses')
    if any(not torch.isfinite(value).all() for value in selected.values()):
        raise RuntimeError('Stage-A full-model loss is non-finite')
    total = sum(value.mean() for value in selected.values())
    if not torch.isfinite(total):
        raise RuntimeError('Stage-A total loss is non-finite')
    before = _state_cpu(model)
    total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError('Stage-A backward produced no gradients')
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError('Stage-A gradients are non-finite')
    targets = _binary_qk_targets(model) if require_binary_targets else ()
    target_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in targets
    }
    target_gradients = {}
    for name, parameter in targets:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f'Binary Q/K gradient is missing/non-finite: {name}')
        if gradient.shape[0] % 3:
            raise RuntimeError(f'Binary Q/K gradient shape is invalid: {name}')
        width = gradient.shape[0] // 3
        q_norm = float(gradient[:width].detach().float().norm().cpu())
        k_norm = float(gradient[width:2 * width].detach().float().norm().cpu())
        if not math.isfinite(q_norm) or not math.isfinite(k_norm) \
                or q_norm <= 0.0 or k_norm <= 0.0:
            raise RuntimeError(
                f'Binary Q/K gradient has no finite Q/K signal: {name}')
        target_gradients[name] = (q_norm, k_norm, width)
    optimizer.step()
    after = _state_cpu(model)
    if _state_exact(before, after):
        raise RuntimeError('Stage-A optimizer step changed no model state')
    target_evidence = []
    for name, parameter in targets:
        previous = target_before[name]
        current = parameter.detach().cpu()
        width = target_gradients[name][2]
        q_changed = not torch.equal(previous[:width], current[:width])
        k_changed = not torch.equal(
            previous[width:2 * width], current[width:2 * width])
        state = optimizer.state.get(parameter)
        step = (
            _optimizer_step_value(state.get('step'))
            if isinstance(state, Mapping) else None)
        if (
                not q_changed or not k_changed or step is None or step < 1.0
                or not all(
                    isinstance(state.get(key), torch.Tensor)
                    and torch.isfinite(state[key]).all()
                    for key in ('exp_avg', 'exp_avg_sq'))):
            raise RuntimeError(
                f'Binary Q/K parameter/Adam state did not update: {name}')
        q_norm, k_norm, _ = target_gradients[name]
        target_evidence.append({
            'parameter': name,
            'gradient_finite': True,
            'q_gradient_norm': q_norm,
            'k_gradient_norm': k_norm,
            'q_parameter_changed': q_changed,
            'k_parameter_changed': k_changed,
            'optimizer': 'Adam',
            'optimizer_step': int(step),
            'optimizer_state_finite': True,
        })

    operation = dict(operation_builder(model))
    validate_binary_operation_manifest(operation)
    model.eval()
    with torch.inference_mode():
        reference = model(inputs, data_samples=data_samples, mode='tensor')
    if not isinstance(reference, torch.Tensor) or not torch.isfinite(reference).all():
        raise RuntimeError('Stage-A tensor forward is missing or non-finite')

    torch.save({'state_dict': after}, export)
    try:
        payload = torch.load(export, map_location='cpu', weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f'Stage-A restricted round-trip load failed: {error}') from error
    if (
            not isinstance(payload, Mapping)
            or set(payload) != {'state_dict'}
            or not isinstance(payload['state_dict'], Mapping)
            or not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in payload['state_dict'].items())):
        raise RuntimeError('Stage-A export is not a tensor-only state_dict')
    restored = model_factory()
    if not isinstance(restored, torch.nn.Module):
        raise TypeError('Stage-A model factory must return a torch module')
    restored.load_state_dict(dict(payload['state_dict']), strict=True)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = inputs.device
    restored.to(device).eval()
    restored_state = _state_cpu(restored)
    state_exact = _state_exact(after, restored_state)
    with torch.inference_mode():
        replay = restored(inputs, data_samples=data_samples, mode='tensor')
    output_exact = (
        isinstance(replay, torch.Tensor)
        and torch.equal(reference.detach(), replay.detach()))
    if not state_exact or not output_exact:
        raise RuntimeError('Stage-A export/load round-trip identity failed')

    return {
        'checks': {
            'forward': True,
            'loss': True,
            'backward': True,
            'optimizer_step': True,
            'finite_loss': True,
            'finite_gradients': True,
            'gradient_tensor_count': len(gradients),
            'binary_target_count': len(target_evidence),
            'state_round_trip_exact': state_exact,
            'output_round_trip_exact': output_exact,
        },
        'losses': {
            name: float(value.detach().mean().cpu())
            for name, value in sorted(selected.items())
        },
        'binary_targets': target_evidence,
        'output': {
            'shape': list(reference.shape),
            'dtype': str(reference.dtype),
            'sha256': _tensor_sha256(reference),
        },
        'operation': operation,
        'export': {
            'path': str(export),
            'sha256': _sha256(export),
            'bytes': export.stat().st_size,
            'format': 'torch-weights-only-state-dict-v1',
        },
    }


def _relative_file(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} path is invalid')
    raw = str(value)
    relative = Path(raw)
    if relative.is_absolute() or any(
            part in {'', '.', '..'} for part in raw.split('/')):
        raise ValueError(f'{label} path is invalid')
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not contain symlinks')
    try:
        effective = cursor.resolve(strict=True)
        effective.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes the repository') from error
    if not effective.is_file():
        raise ValueError(f'{label} must be a regular file')
    return effective


def _binding(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {'path', 'sha256'}:
        raise ValueError(f'{label} binding is invalid')
    if (
            not isinstance(value['path'], str) or not value['path']
            or Path(value['path']).is_absolute()
            or any(
                part in {'', '.', '..'} for part in value['path'].split('/'))
            or not isinstance(value['sha256'], str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise ValueError(f'{label} binding is invalid')
    return value


def _stage_a_config_binding(
        value: object, *, label: str = 'Stage-A config') -> Mapping[str, Any]:
    fields = {
        'path', 'sha256', 'config_closure', 'resolved_config_sha256'}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f'{label} binding is invalid')
    _binding(
        {'path': value.get('path'), 'sha256': value.get('sha256')},
        label=label)
    closure = value.get('config_closure')
    if (
            not isinstance(closure, list) or not closure
            or any(
                not isinstance(row, Mapping)
                or set(row) != {'path', 'sha256'}
                or not isinstance(row['path'], str) or not row['path']
                or Path(row['path']).is_absolute()
                or any(part in {'', '.', '..'}
                       for part in row['path'].split('/'))
                or not isinstance(row['sha256'], str)
                or not _SHA256.fullmatch(row['sha256'])
                for row in closure)
            or not isinstance(value.get('resolved_config_sha256'), str)
            or not _SHA256.fullmatch(value['resolved_config_sha256'])):
        raise ValueError(f'{label} binding is invalid')
    return value


def _stage_a_config_authority(
        repository_root: Path, *, candidate: Any,
        git_commit: str) -> dict[str, Any]:
    """Rebuild the exact inherited and resolved Binary Stage-A config."""
    from mmengine.config import Config
    from .numeric_source import validate_numeric_config_closure

    root = lexical_repository_root(repository_root)
    try:
        closure = list(validate_numeric_config_closure(
            root, candidate.config, git_commit=git_commit))
    except ValueError as error:
        raise ValueError(
            f'Stage-A config closure differs from recorded commit: {error}') \
            from error
    direct = next(
        (row for row in closure if row['path'] == candidate.config.as_posix()),
        None)
    if direct is None:
        raise ValueError('Stage-A config closure omits the direct config')
    config_path = _relative_file(
        root, candidate.config.as_posix(), label='Stage-A config')
    config = Config.fromfile(config_path)
    serialized = config.dump()
    if not isinstance(serialized, str):
        raise ValueError('Stage-A resolved config is not serializable')
    return {
        'path': candidate.config.as_posix(),
        'sha256': direct['sha256'],
        'config_closure': closure,
        'resolved_config_sha256': hashlib.sha256(
            serialized.encode('utf-8')).hexdigest(),
    }


def _validate_smoke_authority(
        value: Mapping[str, Any], *, repository_root: Path) -> None:
    """Rebuild source/checkpoint/PWL authority without loading tensor state."""
    from .binary_readiness import validate_binary_qk_admission
    from .checkpoints import authorize_candidate_checkpoint_reference
    from .evaluation import build_source_binding
    from .schema import load_candidate_manifest

    root = lexical_repository_root(repository_root)
    source = value.get('source')
    source_fields = {
        'git_commit', 'manifest_path', 'manifest_sha256', 'config_path',
        'config_sha256', 'authority_path', 'authority_sha256'}
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise ValueError('Stage-A smoke source identity is invalid')
    manifest = _relative_file(
        root, source.get('manifest_path'), label='Stage-A source manifest')
    matches = tuple(
        candidate for candidate in load_candidate_manifest(manifest)
        if candidate.id == value.get('candidate_id'))
    if len(matches) != 1 or matches[0].kind != 'binary-qk':
        raise ValueError('Stage-A smoke candidate authority is invalid')
    candidate = matches[0]
    expected_source = build_source_binding(
        repository_root=root, candidate=candidate,
        manifest_path=manifest, git_commit=str(source.get('git_commit')))
    if dict(source) != expected_source:
        raise ValueError('Stage-A smoke source binding mismatch')
    config = _stage_a_config_binding(value.get('config'))
    config_path = _relative_file(
        root, config['path'], label='Stage-A config')
    expected_config = _stage_a_config_authority(
        root, candidate=candidate, git_commit=str(source.get('git_commit')))
    if (
            config['path'] != candidate.config.as_posix()
            or config['sha256'] != _sha256(config_path)
            or config['sha256'] != source['config_sha256']
            or dict(config) != expected_config):
        raise ValueError('Stage-A smoke config authority mismatch')
    checkpoint = _binding(
        value.get('checkpoint'), label='Stage-A checkpoint')
    authorize_candidate_checkpoint_reference(
        root, manifest, candidate, checkpoint)
    dependency = _binding(
        value.get('pwl_stage_b'), label='Stage-A PWL dependency')
    if dependency != {
            'path': candidate.features.get('pwl_stage_b_artifact'),
            'sha256': candidate.features.get('pwl_stage_b_sha256')}:
        raise ValueError('Stage-A smoke PWL dependency authority mismatch')
    validate_binary_qk_admission(
        candidate, repository_root=root, manifest_path=manifest)


def validate_binary_stage_a_artifact(
        artifact_path: Path | str, *, repository_root: Path) -> Mapping[str, Any]:
    """Validate the public full-model Stage-A smoke artifact and its export."""
    root = lexical_repository_root(repository_root)
    artifact = _relative_file(
        root, str(artifact_path), label='Stage-A smoke artifact')
    relative_artifact = artifact.relative_to(root)
    if (
            relative_artifact.parts[:2] != ('work_dirs', 'optimization')
            or relative_artifact.name != 'smoke.json'
            or relative_artifact.parent.name != 'smoke-stage-a'):
        raise ValueError('Stage-A smoke artifact path is not canonical')
    try:
        value = json.loads(artifact.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Stage-A smoke artifact is invalid JSON: {error}') from error
    top_fields = {
        'schema_version', 'artifact_kind', 'candidate_id', 'source', 'config',
        'checkpoint', 'pwl_stage_b', 'data', 'gpu', 'execution',
    }
    if (
            not isinstance(value, Mapping) or set(value) != top_fields
            or value['schema_version'] != 2
            or value['artifact_kind'] != (
                'binary-qk-stage-a-full-model-smoke')
            or not isinstance(value['candidate_id'], str)
            or not value['candidate_id']):
        raise ValueError('Stage-A smoke artifact identity is invalid')
    source = value['source']
    if (
            not isinstance(source, Mapping)
            or set(source) != {
                'git_commit', 'manifest_path', 'manifest_sha256',
                'config_path', 'config_sha256', 'authority_path',
                'authority_sha256'}
            or not isinstance(source['git_commit'], str)
            or not re.fullmatch(r'[0-9a-f]{40}', source['git_commit'])
            or any(
                not isinstance(source.get(name), str)
                or not _SHA256.fullmatch(source[name])
                for name in (
                    'manifest_sha256', 'config_sha256',
                    'authority_sha256'))):
        raise ValueError('Stage-A smoke source identity is invalid')
    _stage_a_config_binding(value['config'])
    _binding(value['checkpoint'], label='Stage-A checkpoint')
    _binding(value['pwl_stage_b'], label='Stage-A PWL dependency')
    data = value['data']
    if (
            not isinstance(data, Mapping)
            or set(data) != {
                'dataset', 'split', 'batch_size',
                'packed_production_pipeline', 'input_shape', 'sample_ids'}
            or data['dataset'] != 'coco' or data['split'] != 'train2017'
            or data['batch_size'] != 1
            or data['packed_production_pipeline'] is not True
            or data['input_shape'] != [1, 3, 256, 192]
            or not isinstance(data['sample_ids'], list)
            or len(data['sample_ids']) != 1
            or isinstance(data['sample_ids'][0], bool)
            or not isinstance(data['sample_ids'][0], int)
            or data['sample_ids'][0] < 0):
        raise ValueError('Stage-A smoke data contract is invalid')
    gpu = value['gpu']
    if (
            not isinstance(gpu, Mapping)
            or set(gpu) != {'logical', 'physical_index', 'lease'}
            or gpu['logical'] != 'cuda:0'
            or isinstance(gpu['physical_index'], bool)
            or not isinstance(gpu['physical_index'], int)
            or gpu['physical_index'] < 0):
        raise ValueError('Stage-A smoke GPU identity is invalid')
    try:
        lease = validate_gpu_lease(gpu['lease'])
    except LatencyError as error:
        raise ValueError(f'Stage-A smoke GPU lease is invalid: {error}') from error
    if (
            lease['stage_id'] != f'binary-smoke:{value["candidate_id"]}'
            or lease['device_index'] != gpu['physical_index']):
        raise ValueError('Stage-A smoke GPU lease identity mismatch')
    execution = value['execution']
    if not isinstance(execution, Mapping) or set(execution) != {
            'checks', 'losses', 'binary_targets', 'output', 'operation',
            'export'}:
        raise ValueError('Stage-A smoke execution fields are invalid')
    checks = execution['checks']
    boolean_checks = {
        'forward', 'loss', 'backward', 'optimizer_step', 'finite_loss',
        'finite_gradients', 'state_round_trip_exact',
        'output_round_trip_exact',
    }
    if (
            not isinstance(checks, Mapping)
            or set(checks) != boolean_checks | {
                'gradient_tensor_count', 'binary_target_count'}
            or any(checks.get(name) is not True for name in boolean_checks)
            or isinstance(checks['gradient_tensor_count'], bool)
            or not isinstance(checks['gradient_tensor_count'], int)
            or checks['gradient_tensor_count'] <= 0
            or checks['binary_target_count'] != 6):
        raise ValueError('Stage-A smoke checks are incomplete')
    losses = execution['losses']
    if (
            not isinstance(losses, Mapping) or not losses
            or any(
                not isinstance(name, str) or not name.startswith('loss')
                or isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for name, item in losses.items())):
        raise ValueError('Stage-A smoke losses are invalid')
    targets = execution['binary_targets']
    target_fields = {
        'parameter', 'gradient_finite', 'q_gradient_norm', 'k_gradient_norm',
        'q_parameter_changed', 'k_parameter_changed', 'optimizer',
        'optimizer_step', 'optimizer_state_finite'}
    expected_parameters = [
        f'{name}.to_qkv.weight'
        for name in CANONICAL_BINARY_ATTENTION_MODULES]
    if (
            not isinstance(targets, list) or len(targets) != 6
            or [row.get('parameter') if isinstance(row, Mapping) else None
                for row in targets] != expected_parameters
            or any(
                set(row) != target_fields
                or row['gradient_finite'] is not True
                or row['q_parameter_changed'] is not True
                or row['k_parameter_changed'] is not True
                or row['optimizer'] != 'Adam'
                or isinstance(row['optimizer_step'], bool)
                or not isinstance(row['optimizer_step'], int)
                or row['optimizer_step'] < 1
                or row['optimizer_state_finite'] is not True
                or any(
                    isinstance(row[name], bool)
                    or not isinstance(row[name], (int, float))
                    or not math.isfinite(float(row[name]))
                    or float(row[name]) <= 0.0
                    for name in ('q_gradient_norm', 'k_gradient_norm'))
                for row in targets)):
        raise ValueError('Stage-A Binary Q/K target evidence is invalid')
    output = execution['output']
    if (
            not isinstance(output, Mapping)
            or set(output) != {'shape', 'dtype', 'sha256'}
            or output['shape'] != [1, 17, 64, 48]
            or output['dtype'] not in {'torch.float16', 'torch.float32'}
            or not isinstance(output['sha256'], str)
            or not _SHA256.fullmatch(output['sha256'])):
        raise ValueError('Stage-A smoke output identity is invalid')
    validate_binary_operation_manifest(execution['operation'])
    export = execution['export']
    if not isinstance(export, Mapping) or set(export) != {
            'path', 'sha256', 'bytes', 'format'}:
        raise ValueError('Stage-A smoke export binding is invalid')
    if export.get('format') != 'torch-weights-only-state-dict-v1':
        raise ValueError('Stage-A smoke export format is invalid')
    exported = _relative_file(root, export.get('path'), label='Stage-A export')
    if (
            exported.parent != artifact.parent
            or not isinstance(export.get('sha256'), str)
            or not _SHA256.fullmatch(export['sha256'])
            or _sha256(exported) != export['sha256']
            or isinstance(export.get('bytes'), bool)
            or not isinstance(export.get('bytes'), int)
            or export['bytes'] != exported.stat().st_size):
        raise ValueError('Stage-A smoke export hash or location mismatch')
    _validate_smoke_authority(value, repository_root=root)
    return value


def _canonical_gpu_lock(repository_root: Path) -> Path:
    try:
        common = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'],
            cwd=repository_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot derive canonical GPU lock') from error
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = repository_root / common_path
    common_path = common_path.resolve(strict=True)
    if common_path.name != '.git':
        raise ValueError('Git common directory is not a checkout .git')
    return common_path.parent / 'work_dirs/optimization/gpu.lock'


def _smoke_dataloader(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    dataset = result.get('dataset')
    if (
            not isinstance(dataset, Mapping)
            or dataset.get('type') != 'CocoDataset'
            or dataset.get('data_mode') != 'topdown'
            or dataset.get('data_root') not in {'data/coco', 'data/coco/'}
            or dataset.get('ann_file') != (
                'annotations/person_keypoints_train2017.json')
            or not isinstance(dataset.get('pipeline'), (list, tuple))
            or not dataset['pipeline']
            or dataset['pipeline'][-1].get('type') != 'PackPoseInputs'):
        raise ValueError(
            'binary Stage-A requires the packed production COCO train pipeline')
    result.update({
        'batch_size': 1,
        'num_workers': 0,
        'persistent_workers': False,
        'drop_last': False,
        'sampler': dict(type='DefaultSampler', shuffle=False),
    })
    result.pop('batch_sampler', None)
    result.pop('worker_init_fn', None)
    return result


def _neutralized_model_config(value: Any) -> Any:
    result = copy.deepcopy(value)

    def neutralize(item: Any) -> None:
        if isinstance(item, Mapping):
            for name in tuple(item):
                if name in {'init_cfg', 'pretrained'}:
                    item[name] = None
                else:
                    neutralize(item[name])
        elif isinstance(item, (list, tuple)):
            for child in item:
                neutralize(child)

    neutralize(result)
    return result


def _build_binary_model(config, state: Mapping[str, torch.Tensor], device):
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS
    from .numeric_conversion import NumericRuntimeHook

    init_default_scope(config.get('default_scope', 'mmpose'))
    model = MODELS.build(_neutralized_model_config(config.model))
    model.load_state_dict(dict(state), strict=True)
    numeric = config.get('numeric_optimization')
    if numeric is not None:
        NumericRuntimeHook.apply_to_model(model, numeric)
    return model.to(device)


def build_stage_a_optimizer(
        model: torch.nn.Module,
        optim_wrapper: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build the one-step optimizer from the resolved paper config."""
    if not isinstance(optim_wrapper, Mapping):
        raise ValueError('Stage-A optim_wrapper must be a mapping')
    value = optim_wrapper.get('optimizer')
    if not isinstance(value, Mapping) or value.get('type') != 'Adam':
        raise ValueError('Stage-A resolved paper config must use Adam')
    allowed = {'type', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
    if set(value) - allowed or 'lr' not in value:
        raise ValueError('Stage-A Adam config has unsupported or missing fields')
    kwargs = {name: item for name, item in value.items() if name != 'type'}
    return torch.optim.Adam(model.parameters(), **kwargs)


def _prepare_smoke_output(output: Path) -> bool:
    """Create a standalone output or admit only controller attempt logs."""
    if output.is_symlink():
        raise FileExistsError(f'refusing symlinked binary smoke: {output}')
    if not output.exists():
        output.mkdir(parents=True)
        return True
    if not output.is_dir():
        raise FileExistsError(f'binary smoke output is not a directory: {output}')
    unexpected = tuple(
        child.name for child in output.iterdir()
        if not re.fullmatch(r'attempt-[1-9][0-9]*\.log', child.name)
        or not child.is_file() or child.is_symlink())
    if unexpected:
        raise FileExistsError(
            f'binary smoke has unexpected existing files: {unexpected}')
    return False


def run_binary_stage_a_smoke(
        *, repository_root: Path, manifest_path: Path, candidate_id: str,
        output_relative: Path | str, device_index: int) -> Path:
    """Run one real packed COCO batch under the canonical exclusive GPU lease."""
    if device_index < 0:
        raise ValueError('binary smoke device index must be non-negative')
    root = lexical_repository_root(repository_root)
    raw = str(output_relative)
    relative = Path(raw)
    if (
            relative.is_absolute()
            or any(part in {'', '.', '..'} for part in raw.split('/'))
            or relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name != 'smoke-stage-a'):
        raise ValueError(
            'binary smoke output must be a smoke-stage-a directory under '
            'work_dirs/optimization')
    output = root / relative
    _prepare_smoke_output(output)

    from mmengine.config import Config
    from mmengine.runner import Runner
    from .binary_readiness import validate_binary_qk_admission
    from .checkpoints import authorize_manifest_candidate, tensor_state
    from .gpu_guard import exclusive_cuda_stage

    authorized = authorize_manifest_candidate(root, manifest_path, candidate_id)
    candidate = authorized.candidate
    if candidate.kind != 'binary-qk':
        raise ValueError('binary smoke requires a binary-qk candidate')
    validate_binary_qk_admission(
        candidate, repository_root=root, manifest_path=manifest_path)
    config_authority = _stage_a_config_authority(
        root, candidate=candidate,
        git_commit=str(authorized.source.get('git_commit')))
    config = Config.fromfile(authorized.config_path)
    loader_config = _smoke_dataloader(config.train_dataloader)
    state = tensor_state(
        authorized.checkpoint_path,
        expected_sha256=authorized.candidate.checkpoint_sha256)
    lock = _canonical_gpu_lock(root)
    stage_id = f'binary-smoke:{candidate.id}'
    try:
        with exclusive_cuda_stage(
                lock, device_index, (os.getpid(),), stage_id=stage_id) as lease:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != str(device_index):
                raise RuntimeError(
                    'CUDA_VISIBLE_DEVICES must expose exactly the leased GPU')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable for binary Stage-A smoke')
            device = torch.device('cuda:0')
            torch.manual_seed(candidate.seed)
            torch.cuda.manual_seed_all(candidate.seed)
            torch.use_deterministic_algorithms(True)
            model = _build_binary_model(config, state, device)
            loader = Runner.build_dataloader(
                loader_config, seed=candidate.seed, diff_rank_seed=False)
            batch = next(iter(loader))
            processed = model.data_preprocessor(batch, training=True)
            inputs = processed['inputs']
            samples = processed['data_samples']
            if (
                    not isinstance(inputs, torch.Tensor)
                    or list(inputs.shape) != [1, 3, 256, 192]
                    or not isinstance(samples, list) or len(samples) != 1):
                raise RuntimeError(
                    'binary Stage-A did not receive one packed 256x192 batch')
            optimizer = build_stage_a_optimizer(model, config.optim_wrapper)

            def factory():
                empty = {name: torch.empty_like(value) for name, value in state.items()}
                return _build_binary_model(config, empty, torch.device('cpu'))

            execution = execute_binary_stage_a_model(
                model=model, inputs=inputs, data_samples=samples,
                optimizer=optimizer, model_factory=factory,
                export_path=output / 'round-trip.pth')
            execution['export']['path'] = (
                output / 'round-trip.pth').relative_to(root).as_posix()
            sample_id = samples[0].metainfo.get('img_id')
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise RuntimeError('binary Stage-A sample has no integer image id')
            lease_value = asdict(lease)
            lease_value['allowed_pids'] = list(lease_value['allowed_pids'])
            artifact = {
                'schema_version': 2,
                'artifact_kind': 'binary-qk-stage-a-full-model-smoke',
                'candidate_id': candidate.id,
                'source': dict(authorized.source),
                'config': config_authority,
                'checkpoint': {
                    'path': candidate.checkpoint.as_posix(),
                    'sha256': candidate.checkpoint_sha256,
                },
                'pwl_stage_b': {
                    'path': str(candidate.features['pwl_stage_b_artifact']),
                    'sha256': str(candidate.features['pwl_stage_b_sha256']),
                },
                'data': {
                    'dataset': 'coco', 'split': 'train2017', 'batch_size': 1,
                    'packed_production_pipeline': True,
                    'input_shape': list(inputs.shape),
                    'sample_ids': [sample_id],
                },
                'gpu': {
                    'logical': 'cuda:0', 'physical_index': device_index,
                    'lease': lease_value,
                },
                'execution': execution,
            }
            artifact_path = output / 'smoke.json'
            temporary = output / f'.smoke.{os.getpid()}.tmp'
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, artifact_path)
        validate_binary_stage_a_artifact(
            (output / 'smoke.json').relative_to(root), repository_root=root)
        return output / 'smoke.json'
    except BaseException:
        for name in ('smoke.json', 'round-trip.pth'):
            path = output / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        for temporary in output.glob('.smoke.*.tmp'):
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
        raise
