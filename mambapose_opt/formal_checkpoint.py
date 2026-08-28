"""Authenticated, weights-only checkpoint loading for formal Stage C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Mapping

import torch


class FormalCheckpointError(ValueError):
    """A checkpoint is not the exact authenticated tensor artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in '0123456789abcdef' for character in value):
        raise FormalCheckpointError(f'{label} SHA-256 is invalid')
    return value


def _relative(value: str) -> Path:
    if not isinstance(value, str) or not value or '\\' in value:
        raise FormalCheckpointError('checkpoint path is not canonical')
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value \
            or any(part in {'', '.', '..'} for part in pure.parts):
        raise FormalCheckpointError('checkpoint path is not canonical')
    return Path(value)


@dataclass(frozen=True)
class FileAuthority:
    authority_root: str
    path: str
    sha256: str

    def validate(self) -> tuple[Path, Path]:
        root = Path(self.authority_root)
        if not root.is_absolute() or '..' in root.parts \
                or '/./' in self.authority_root \
                or root.is_symlink() or not root.is_dir():
            raise FormalCheckpointError(
                'checkpoint authority root must be an existing absolute directory')
        cursor = Path(root.anchor)
        for part in root.parts[1:]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise FormalCheckpointError(
                    'checkpoint authority root contains a symlink')
        relative = _relative(self.path)
        _sha(self.sha256, 'checkpoint')
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            try:
                mode = cursor.lstat().st_mode
            except FileNotFoundError as error:
                raise FormalCheckpointError('checkpoint path is missing') from error
            if stat.S_ISLNK(mode):
                raise FormalCheckpointError('checkpoint path contains a symlink')
        if not cursor.is_file():
            raise FormalCheckpointError('checkpoint authority is not a file')
        return root, cursor


@dataclass(frozen=True)
class TensorShape:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in self.shape):
            raise FormalCheckpointError('tensor shape is invalid')
        if not isinstance(self.dtype, str) or not self.dtype.startswith('torch.'):
            raise FormalCheckpointError('tensor dtype is invalid')


def _check_tensor_tree(value: object, path: str = 'checkpoint') -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise FormalCheckpointError(f'{path} contains a non-finite tensor')
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise FormalCheckpointError(
                f'{path} contains a non-string mapping key')
        for key, item in value.items():
            _check_tensor_tree(item, f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_tensor_tree(item, f'{path}[{index}]')
        return
    raise FormalCheckpointError(f'{path} violates the tensor-only contract')


def load_formal_tensor_checkpoint(
        path: Path,
        authority: FileAuthority,
        expected_keys: Mapping[str, TensorShape],
        ) -> Mapping[str, torch.Tensor]:
    """Load an exact tensor mapping without any unsafe fallback.

    A checkpoint may be the tensor mapping itself or a single ``model`` /
    ``state_dict`` wrapper.  No metadata leaf is admitted.
    """
    if os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD') == '1':
        raise FormalCheckpointError(
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is forbidden')
    if not isinstance(authority, FileAuthority):
        raise FormalCheckpointError('checkpoint authority type is invalid')
    _root, canonical = authority.validate()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    if supplied.absolute() != canonical.absolute():
        raise FormalCheckpointError('checkpoint path differs from authority')
    if _sha256(canonical) != authority.sha256:
        raise FormalCheckpointError('checkpoint SHA-256 differs from authority')
    if not isinstance(expected_keys, Mapping) or not expected_keys \
            or any(not isinstance(key, str)
                   or not isinstance(value, TensorShape)
                   for key, value in expected_keys.items()):
        raise FormalCheckpointError('expected tensor contract is invalid')
    try:
        document = torch.load(
            canonical, map_location='cpu', weights_only=True)
    except Exception as error:
        raise FormalCheckpointError(
            'checkpoint could not be loaded in weights-only mode') from error
    _check_tensor_tree(document)
    if not isinstance(document, Mapping):
        raise FormalCheckpointError('checkpoint must be a tensor mapping')
    if set(document) in ({'model'}, {'state_dict'}):
        document = document[next(iter(document))]
    if not isinstance(document, Mapping):
        raise FormalCheckpointError('checkpoint tensor wrapper is invalid')
    if set(document) != set(expected_keys):
        raise FormalCheckpointError('checkpoint tensor keys differ from contract')
    result: dict[str, torch.Tensor] = {}
    for key in sorted(expected_keys):
        tensor = document[key]
        contract = expected_keys[key]
        if not isinstance(tensor, torch.Tensor):
            raise FormalCheckpointError(f'{key} is not a tensor')
        if tuple(tensor.shape) != contract.shape:
            raise FormalCheckpointError(f'{key} shape differs from contract')
        if str(tensor.dtype) != contract.dtype:
            raise FormalCheckpointError(f'{key} dtype differs from contract')
        if (tensor.is_floating_point() or tensor.is_complex()) \
                and not torch.isfinite(tensor).all().item():
            raise FormalCheckpointError(f'{key} is non-finite')
        result[key] = tensor.detach().cpu()
    return MappingProxyType(result)


@dataclass(frozen=True)
class BackboneLoadReport:
    compatible_tensors: int
    missing_tensors: int
    unexpected_tensors: int
    compatible_keys: tuple[str, ...]


def load_formal_backbone_initialization(
        path: Path, authority: FileAuthority, backbone: torch.nn.Module,
        *, minimum_compatible_tensors: int = 100) -> BackboneLoadReport:
    """Inject a safe upstream VMamba ``model`` mapping into a backbone.

    The upstream ImageNet file is a training bundle rather than a tensor-only
    pose artifact.  It is nevertheless admitted only through ``weights_only``;
    every tensor below ``model`` is recursively finite before the backbone's
    standard key-translation method sees it.  No generic loader is called.
    """
    if os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD') == '1':
        raise FormalCheckpointError(
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is forbidden')
    _root, canonical = authority.validate()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    if supplied.absolute() != canonical.absolute() \
            or _sha256(canonical) != authority.sha256:
        raise FormalCheckpointError(
            'backbone initialization differs from authority')
    if isinstance(minimum_compatible_tensors, bool) \
            or not isinstance(minimum_compatible_tensors, int) \
            or minimum_compatible_tensors < 1:
        raise FormalCheckpointError('minimum compatible tensor count is invalid')
    try:
        document = torch.load(canonical, map_location='cpu', weights_only=True)
    except Exception as error:
        raise FormalCheckpointError(
            'backbone initialization failed weights-only loading') from error
    if not isinstance(document, Mapping) or 'model' not in document \
            or not isinstance(document['model'], Mapping):
        raise FormalCheckpointError(
            'backbone initialization has no model tensor mapping')
    model = document['model']
    _check_tensor_tree(model, 'checkpoint.model')
    before = tuple(backbone.state_dict())
    try:
        incompatible = backbone.load_state_dict(dict(model), strict=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise FormalCheckpointError(
            'backbone initialization tensors are incompatible') from error
    missing = tuple(incompatible.missing_keys)
    compatible = tuple(sorted(set(before) - set(missing)))
    if len(compatible) < minimum_compatible_tensors:
        raise FormalCheckpointError(
            'backbone initialization compatible tensor count is too small')
    state = backbone.state_dict()
    for key in compatible:
        tensor = state[key]
        if (tensor.is_floating_point() or tensor.is_complex()) \
                and not torch.isfinite(tensor).all().item():
            raise FormalCheckpointError(
                'backbone initialization produced a non-finite tensor')
    return BackboneLoadReport(
        compatible_tensors=len(compatible),
        missing_tensors=len(missing),
        unexpected_tensors=len(incompatible.unexpected_keys),
        compatible_keys=compatible)
