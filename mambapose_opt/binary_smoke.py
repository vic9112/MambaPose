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
import shutil
import subprocess
from typing import Any, Callable, Mapping

import torch

from .binary_operation import (
    build_binary_operation_manifest, validate_binary_operation_manifest)
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


def execute_binary_stage_a_model(
        *, model: torch.nn.Module, inputs: torch.Tensor, data_samples: Any,
        optimizer: torch.optim.Optimizer,
        model_factory: Callable[[], torch.nn.Module], export_path: Path,
        operation_builder: Callable[[torch.nn.Module], Mapping[str, Any]] = (
            build_binary_operation_manifest),
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
    optimizer.step()
    after = _state_cpu(model)
    if _state_exact(before, after):
        raise RuntimeError('Stage-A optimizer step changed no model state')

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
            'state_round_trip_exact': state_exact,
            'output_round_trip_exact': output_exact,
        },
        'losses': {
            name: float(value.detach().mean().cpu())
            for name, value in sorted(selected.items())
        },
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
    relative = Path(value)
    if relative.is_absolute() or any(
            part in {'.', '..'} for part in relative.parts):
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
            or any(part in {'.', '..'} for part in Path(value['path']).parts)
            or not isinstance(value['sha256'], str)
            or not _SHA256.fullmatch(value['sha256'])):
        raise ValueError(f'{label} binding is invalid')
    return value


def validate_binary_stage_a_artifact(
        artifact_path: Path, *, repository_root: Path) -> Mapping[str, Any]:
    """Validate the public full-model Stage-A smoke artifact and its export."""
    root = lexical_repository_root(repository_root)
    artifact = _relative_file(
        root,
        Path(artifact_path).relative_to(root).as_posix()
        if Path(artifact_path).is_absolute() else str(artifact_path),
        label='Stage-A smoke artifact')
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
            or value['schema_version'] != 1
            or value['artifact_kind'] != (
                'binary-qk-stage-a-full-model-smoke')
            or not isinstance(value['candidate_id'], str)
            or not value['candidate_id']):
        raise ValueError('Stage-A smoke artifact identity is invalid')
    source = value['source']
    if (
            not isinstance(source, Mapping)
            or set(source) != {'git_commit'}
            or not isinstance(source['git_commit'], str)
            or not re.fullmatch(r'[0-9a-f]{40}', source['git_commit'])):
        raise ValueError('Stage-A smoke source identity is invalid')
    _binding(value['config'], label='Stage-A config')
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
            'checks', 'losses', 'output', 'operation', 'export'}:
        raise ValueError('Stage-A smoke execution fields are invalid')
    checks = execution['checks']
    boolean_checks = {
        'forward', 'loss', 'backward', 'optimizer_step', 'finite_loss',
        'finite_gradients', 'state_round_trip_exact',
        'output_round_trip_exact',
    }
    if (
            not isinstance(checks, Mapping)
            or set(checks) != boolean_checks | {'gradient_tensor_count'}
            or any(checks.get(name) is not True for name in boolean_checks)
            or isinstance(checks['gradient_tensor_count'], bool)
            or not isinstance(checks['gradient_tensor_count'], int)
            or checks['gradient_tensor_count'] <= 0):
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


def _build_binary_model(config, state: Mapping[str, torch.Tensor], device):
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS
    from .numeric_conversion import NumericRuntimeHook

    init_default_scope(config.get('default_scope', 'mmpose'))
    model = MODELS.build(config.model)
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


def run_binary_stage_a_smoke(
        *, repository_root: Path, manifest_path: Path, candidate_id: str,
        output_relative: Path, device_index: int) -> Path:
    """Run one real packed COCO batch under the canonical exclusive GPU lease."""
    if device_index < 0:
        raise ValueError('binary smoke device index must be non-negative')
    root = lexical_repository_root(repository_root)
    relative = Path(output_relative)
    if (
            relative.is_absolute()
            or any(part in {'.', '..'} for part in relative.parts)
            or relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name != 'smoke-stage-a'):
        raise ValueError(
            'binary smoke output must be a smoke-stage-a directory under '
            'work_dirs/optimization')
    output = root / relative
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'refusing to overwrite binary smoke: {relative}')

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
    config = Config.fromfile(authorized.config_path)
    loader_config = _smoke_dataloader(config.train_dataloader)
    state = tensor_state(authorized.checkpoint_path)
    lock = _canonical_gpu_lock(root)
    stage_id = f'binary-smoke:{candidate.id}'
    output.mkdir(parents=True)
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
                'schema_version': 1,
                'artifact_kind': 'binary-qk-stage-a-full-model-smoke',
                'candidate_id': candidate.id,
                'source': {'git_commit': authorized.source['git_commit']},
                'config': {
                    'path': candidate.config.as_posix(),
                    'sha256': _sha256(authorized.config_path),
                },
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
            output / 'smoke.json', repository_root=root)
        return output / 'smoke.json'
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
