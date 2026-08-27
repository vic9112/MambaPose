"""Measured software-operation identity for the S-V1 Binary Q/K proxy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from mmpose.models.heads.heatmap_heads.tokenbase import Attention
from mmpose.models.utils.hardware_friendly.binary_qk import (
    binary_qk_operation_report, ste_sign)

from .artifacts import lexical_repository_root


_FIELDS = {
    'schema_version', 'artifact_kind', 'implementation', 'qk_mode', 'layers',
    'module_names', 'heads', 'query_tokens', 'key_tokens', 'head_dim',
    'zero_sign', 'ste_gradient', 'scale', 'softmax', 'value',
    'attention_accumulation', 'output_projection',
    'theoretical_qk_multiplications_replaced', 'bitwise_kernel_present',
    'measured_integer_latency', 'hardware_claim',
}
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_PROFILE_BINDING_FIELDS = {'path', 'sha256', 'operation_sha256'}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repository_file(
        value: Path, *, repository_root: Path, label: str) -> Path:
    root = lexical_repository_root(repository_root)
    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = root / lexical
    try:
        relative = lexical.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f'{label} escapes the repository') from error
    if relative.parts[:2] != ('work_dirs', 'optimization'):
        raise ValueError(f'{label} must stay under work_dirs/optimization')
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not contain symlinks')
    try:
        effective = lexical.resolve(strict=True)
        effective.relative_to(root / 'work_dirs/optimization')
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes artifact authority') from error
    if not effective.is_file():
        raise ValueError(f'{label} must be a regular file')
    return effective


def _tokenpose_geometry(model: torch.nn.Module, names: list[str]) -> tuple[int, int]:
    matches = []
    for prefix, module in model.named_modules():
        if not hasattr(module, 'num_patches') or not hasattr(module, 'num_keypoints'):
            continue
        child_prefix = f'{prefix}.' if prefix else ''
        if all(name.startswith(child_prefix) for name in names):
            matches.append(module)
    if len(matches) != 1:
        raise ValueError(
            'binary operation manifest requires one enclosing TokenPose geometry')
    patches = getattr(matches[0], 'num_patches')
    keypoints = getattr(matches[0], 'num_keypoints')
    if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in (patches, keypoints)):
        raise ValueError('TokenPose patch/keypoint geometry is invalid')
    return patches, keypoints


def build_binary_operation_manifest(model: torch.nn.Module) -> dict[str, Any]:
    """Derive the exact Binary Q/K operation contract from a live full model."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError('binary operation manifest requires a torch model')
    rows = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, Attention)
    ]
    if len(rows) != 6:
        raise ValueError('Binary Q/K requires exactly six Attention layers')
    names = [name for name, _ in rows]
    if any(module.qk_mode != 'binary' for _, module in rows):
        raise ValueError('all six Attention layers must use binary qk_mode')
    heads = {module.heads for _, module in rows}
    dimensions = {module.to_qkv.in_features for _, module in rows}
    if len(heads) != 1 or len(dimensions) != 1:
        raise ValueError('Binary Q/K layers disagree on heads or dimensions')
    head_count = heads.pop()
    dimension = dimensions.pop()
    if dimension % head_count:
        raise ValueError('Binary Q/K dimension must divide evenly across heads')
    head_dim = dimension // head_count
    expected_scale = head_dim ** -0.5
    if any(
            not math.isclose(
                float(module.scale), expected_scale,
                rel_tol=0.0, abs_tol=1e-15)
            for _, module in rows):
        raise ValueError('Binary Q/K must preserve the original head scale')
    if any(
            module.to_qkv.out_features != dimension * 3
            or not isinstance(module.to_out[0], torch.nn.Linear)
            or module.to_out[0].in_features != dimension
            or module.to_out[0].out_features != dimension
            for _, module in rows):
        raise ValueError('Binary Q/K QKV/value/output projection topology drifted')
    patches, keypoints = _tokenpose_geometry(model, names)
    tokens = patches + keypoints
    operation = binary_qk_operation_report(
        query_tokens=tokens, key_tokens=tokens, head_dim=head_dim,
        heads=head_count, layers=len(rows))

    probe = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    signed = ste_sign(probe)
    signed.sum().backward()
    if (
            signed.detach().tolist() != [-1.0, 1.0, 1.0]
            or probe.grad is None
            or probe.grad.detach().tolist() != [1.0, 1.0, 1.0]):
        raise ValueError('Binary Q/K sign or identity STE semantics drifted')

    return {
        'schema_version': 1,
        'artifact_kind': 'binary-qk-operation-manifest',
        'implementation': 'ste-sign-einsum-software-proxy',
        'qk_mode': 'binary',
        'layers': operation.layers,
        'module_names': names,
        'heads': operation.heads,
        'query_tokens': operation.query_tokens,
        'key_tokens': operation.key_tokens,
        'head_dim': operation.head_dim,
        'zero_sign': 1,
        'ste_gradient': 'identity',
        'scale': {
            'kind': 'floating-original',
            'formula': '1/sqrt(head_dim)',
            'value': expected_scale,
        },
        'softmax': 'floating',
        'value': 'floating',
        'attention_accumulation': 'floating',
        'output_projection': 'floating',
        'theoretical_qk_multiplications_replaced': (
            operation.theoretical_changed_multiplies),
        'bitwise_kernel_present': False,
        'measured_integer_latency': False,
        'hardware_claim': 'none-software-proxy',
    }


def validate_binary_operation_manifest(value: object) -> Mapping[str, Any]:
    """Reject any manifest that overstates or drifts from the S-V1 proxy."""
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError('binary operation manifest has invalid fields')
    expected = {
        'schema_version': 1,
        'artifact_kind': 'binary-qk-operation-manifest',
        'implementation': 'ste-sign-einsum-software-proxy',
        'qk_mode': 'binary',
        'layers': 6,
        'heads': 8,
        'query_tokens': 65,
        'key_tokens': 65,
        'head_dim': 32,
        'zero_sign': 1,
        'ste_gradient': 'identity',
        'scale': {
            'kind': 'floating-original',
            'formula': '1/sqrt(head_dim)',
            'value': 32 ** -0.5,
        },
        'softmax': 'floating',
        'value': 'floating',
        'attention_accumulation': 'floating',
        'output_projection': 'floating',
        'theoretical_qk_multiplications_replaced': 6_489_600,
        'bitwise_kernel_present': False,
        'measured_integer_latency': False,
        'hardware_claim': 'none-software-proxy',
    }
    for name, item in expected.items():
        if value.get(name) != item:
            raise ValueError(f'binary operation manifest {name} is invalid')
    names = value['module_names']
    if (
            not isinstance(names, list) or len(names) != 6
            or len(set(names)) != 6
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError('binary operation manifest module names are invalid')
    return value


def binary_profile_binding_for_stage(
        stage_output: Path, *, repository_root: Path) -> dict[str, str]:
    """Bind an evaluate/latency output to its canonical sibling profile."""
    root = lexical_repository_root(repository_root)
    output = Path(stage_output)
    if not output.is_absolute():
        output = root / output
    try:
        relative = output.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError('binary stage output escapes the repository') from error
    if (
            relative.parts[:2] != ('work_dirs', 'optimization')
            or relative.name not in {'evaluate.json', 'latency.json'}
            or relative.parent.name not in {'evaluate', 'latency'}):
        raise ValueError('binary stage output has no canonical profile sibling')
    profile = _safe_repository_file(
        relative.parent.parent / 'profile/profile.json',
        repository_root=root, label='binary profile')
    try:
        value = json.loads(profile.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'binary profile is invalid JSON: {error}') from error
    if not isinstance(value, Mapping):
        raise ValueError('binary profile root must be an object')
    operation = validate_binary_operation_manifest(
        value.get('binary_qk_operation'))
    return {
        'path': profile.relative_to(root).as_posix(),
        'sha256': _sha256_file(profile),
        'operation_sha256': canonical_json_sha256(operation),
    }


def _validated_profile_binding(
        value: object, *, profile: Path,
        operation: Mapping[str, Any]) -> Mapping[str, str]:
    validated = validate_binary_profile_binding(value)
    if validated.get('path') != profile.as_posix():
        raise ValueError('binary profile binding path is invalid')
    return validated


def validate_binary_profile_binding(value: object) -> Mapping[str, str]:
    """Validate the portable syntax of a profile hash binding."""
    if not isinstance(value, Mapping) or set(value) != _PROFILE_BINDING_FIELDS:
        raise ValueError('binary profile binding is invalid')
    path = value.get('path')
    if (
            not isinstance(path, str) or not path
            or Path(path).is_absolute()
            or any(part in {'.', '..'} for part in Path(path).parts)
            or Path(path).parts[:2] != ('work_dirs', 'optimization')
            or Path(path).name != 'profile.json'
            or Path(path).parent.name != 'profile'):
        raise ValueError('binary profile binding path is invalid')
    for name in ('sha256', 'operation_sha256'):
        if (
                not isinstance(value.get(name), str)
                or not _SHA256.fullmatch(value[name])):
            raise ValueError(f'binary profile binding {name} is invalid')
    return value


def validate_binary_stage_binding(
        *, profile_path: Path, stage_path: Path, repository_root: Path,
        candidate_id: str, stage: str) -> Mapping[str, Any]:
    """Validate one evaluate/latency binding as soon as it is produced."""
    if stage not in {'evaluate', 'latency'}:
        raise ValueError('binary bound stage must be evaluate or latency')
    root = lexical_repository_root(repository_root)
    profile = _safe_repository_file(
        profile_path, repository_root=root, label='binary profile')
    path = _safe_repository_file(
        stage_path, repository_root=root, label=f'binary {stage}')
    try:
        profile_value = json.loads(profile.read_text(encoding='utf-8'))
        envelope = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'binary {stage} binding JSON is invalid: {error}') from error
    if (
            not isinstance(profile_value, Mapping)
            or profile_value.get('candidate') != candidate_id):
        raise ValueError('binary profile candidate identity mismatch')
    operation = validate_binary_operation_manifest(
        profile_value.get('binary_qk_operation'))
    if (
            not isinstance(envelope, Mapping)
            or envelope.get('candidate_id') != candidate_id
            or envelope.get('stage') != stage
            or not isinstance(envelope.get('result'), Mapping)):
        raise ValueError(f'binary {stage} identity is invalid')
    relative_profile = profile.relative_to(root)
    binding = _validated_profile_binding(
        envelope['result'].get('binary_qk_profile'),
        profile=relative_profile, operation=operation)
    expected = {
        'path': relative_profile.as_posix(),
        'sha256': _sha256_file(profile),
        'operation_sha256': canonical_json_sha256(operation),
    }
    if dict(binding) != expected:
        raise ValueError(f'binary {stage} profile binding mismatch')
    return operation


def validate_binary_artifact_bundle(
        *, profile_path: Path, evaluation_path: Path, latency_path: Path,
        repository_root: Path, candidate_id: str) -> Mapping[str, Any]:
    """Prove profile, AP and latency all describe the same Binary Q/K graph."""
    root = lexical_repository_root(repository_root)
    operation = validate_binary_stage_binding(
        profile_path=profile_path, stage_path=evaluation_path,
        repository_root=root, candidate_id=candidate_id, stage='evaluate')
    latency_operation = validate_binary_stage_binding(
        profile_path=profile_path, stage_path=latency_path,
        repository_root=root, candidate_id=candidate_id, stage='latency')
    if dict(operation) != dict(latency_operation):
        raise ValueError('binary evaluate and latency operations disagree')
    return operation
