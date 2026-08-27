#!/usr/bin/env python3
"""Create a reproducible inventory for a frozen candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True

import torch
from mmengine.config import Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_ARTIFACTS = Path('work_dirs/optimization')
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.inventory import collect_module_inventory, count_parameters
from mambapose_opt.numeric_conversion import NumericRuntimeHook
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest
from mambapose_opt.numeric_runtime import resolve_numeric_runtime
from mambapose_opt.numeric_source import build_numeric_source_binding
from mambapose_opt.source import clean_git_commit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, tuple):
        return [_shape_tree(item) for item in value]
    if isinstance(value, list):
        return [_shape_tree(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _shape_tree(item) for key, item in value.items()}
    return type(value).__name__


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('shape must be comma-separated integers') from error
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError('shape must be four positive dimensions')
    return shape


def _output_path(value: str) -> Path:
    return optimization_output_path(value, repository_root=REPOSITORY_ROOT)


def _candidate(manifest: Path, identifier: str) -> CandidateSpec:
    for candidate in load_candidate_manifest(manifest):
        if candidate.id == identifier:
            return candidate
    raise ValueError(f'candidate not found: {identifier}')


def _profile_device(value: str) -> dict[str, Any]:
    if value == 'cuda:0':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable for formal VMamba profiling')
        try:
            physical = int(os.environ['MAMBAPOSE_PHYSICAL_DEVICE_INDEX'])
        except (KeyError, ValueError) as error:
            raise ValueError(
                'formal CUDA profile requires controller physical device index') \
                from error
        if physical < 0:
            raise ValueError('physical CUDA device index must be nonnegative')
        return {'logical': 'cuda:0', 'physical_index': physical, 'kind': 'cuda'}
    if value == 'cpu':
        return {'logical': 'cpu', 'physical_index': None, 'kind': 'cpu'}
    raise ValueError('profile device must be logical cuda:0 or cpu')


def profile(
        candidate: CandidateSpec, input_shape: tuple[int, ...], *,
        manifest_path: Path | None = None,
        output: Path | None = None,
        device: str = 'cuda:0') -> dict[str, Any]:
    """Execute and inventory one candidate on its explicitly admitted device."""
    device_record = _profile_device(device)
    if device == 'cpu' and candidate.features.get(
            'cpu_profile_supported') is not True:
        raise ValueError('candidate is not proven CPU-capable for profiling')
    manifest_path = manifest_path or REPOSITORY_ROOT / 'optimization/candidates.json'
    commit = clean_git_commit(REPOSITORY_ROOT)
    runtime = resolve_numeric_runtime(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=manifest_path,
        downstream_output=(output or REPOSITORY_ROOT / 'work_dirs/optimization/'
                           'profile/profile.json'))
    checkpoint = runtime['checkpoint_path']
    actual_checksum = _sha256(checkpoint)
    if actual_checksum != runtime['checkpoint_sha256']:
        raise ValueError(
            f'checkpoint sha256 mismatch for {candidate.id}: '
            f'expected {candidate.checkpoint_sha256}, got {actual_checksum}')

    from mmpose.apis import init_model

    config_path = runtime['config_path']
    config = Config.fromfile(config_path)
    model = init_model(str(config_path),
                       str(checkpoint), device=device)
    numeric = config.get('numeric_optimization')
    if numeric is not None:
        NumericRuntimeHook.apply_to_model(model, numeric)
    input_tensor = torch.zeros(input_shape, device=device)
    if device_record['kind'] == 'cuda':
        torch.cuda.synchronize()
    with torch.inference_mode():
        outputs = model(input_tensor, data_samples=None, mode='tensor')
    if device_record['kind'] == 'cuda':
        torch.cuda.synchronize()
    parameters = count_parameters(model)
    if (_sha256(config_path) != runtime['config_sha256']
            or _sha256(checkpoint) != runtime['checkpoint_sha256']):
        raise ValueError('profile runtime inputs changed during execution')
    result = {
        'schema_version': 2,
        'git_commit': commit,
        'candidate': candidate.id,
        'config': config_path.relative_to(REPOSITORY_ROOT).as_posix(),
        'checkpoint': runtime['checkpoint_name'],
        'checkpoint_sha256': actual_checksum,
        'device': device_record,
        'parent': {
            'config': candidate.config.as_posix(),
            'checkpoint': candidate.checkpoint.as_posix(),
            'checkpoint_sha256': candidate.checkpoint_sha256,
        },
        'runtime': {
            'config': {
                'path': config_path.relative_to(REPOSITORY_ROOT).as_posix(),
                'sha256': runtime['config_sha256'],
            },
            'checkpoint': {
                'path': runtime['checkpoint_name'],
                'sha256': runtime['checkpoint_sha256'],
            },
        },
        'input_shapes': _shape_tree(input_tensor),
        'output_shapes': _shape_tree(outputs),
        'parameters': {
            'total': parameters.total,
            'trainable': parameters.trainable,
            'bytes_by_dtype': dict(parameters.bytes_by_dtype),
            'by_prefix': dict(parameters.by_prefix),
        },
        'modules': [
            {
                'name': record.name,
                'kind': record.kind,
                'parameters': record.parameters,
                'hazard': record.hazard,
            }
            for record in collect_module_inventory(model)
        ],
    }
    if candidate.route == 'ssm-quant-pwl':
        result['source'] = build_numeric_source_binding(
            repository_root=REPOSITORY_ROOT, candidate=candidate,
            manifest_path=manifest_path,
            policy_path=REPOSITORY_ROOT / candidate.config,
            git_commit=commit)
    if candidate.kind == 'binary-qk':
        from mambapose_opt.binary_operation import (
            binary_smoke_binding_for_profile,
            build_binary_operation_manifest)
        operation = build_binary_operation_manifest(model)
        result['binary_qk_operation'] = operation
        result['binary_qk_smoke'] = binary_smoke_binding_for_profile(
            output.relative_to(REPOSITORY_ROOT),
            repository_root=REPOSITORY_ROOT, candidate_id=candidate.id,
            operation=operation)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--manifest', type=Path,
                        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', type=_output_path, required=True)
    parser.add_argument('--input-shape', type=_parse_shape, default=(1, 3, 256, 192))
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    args = parser.parse_args()

    candidate = _candidate(args.manifest, args.candidate_id)
    _atomic_json(args.output, profile(
        candidate, args.input_shape, manifest_path=args.manifest,
        output=args.output, device=args.device))


if __name__ == '__main__':
    main()
