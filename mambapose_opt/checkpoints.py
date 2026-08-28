"""Manifest-authorized, data-only candidate checkpoint loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .artifacts import lexical_repository_root
from .evaluation import (
    build_source_binding, resolve_project_asset_root,
    resolve_shared_asset_exposure)
from .schema import CandidateSpec, load_candidate_manifest
from .source import clean_git_commit


@dataclass(frozen=True)
class AuthorizedCandidate:
    candidate: CandidateSpec
    config_path: Path
    checkpoint_path: Path
    source: Mapping[str, str]


_APPROVED_CHECKPOINT_ROOTS = (Path('work_dirs/reproduction'),)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _lexical_file(root: Path, relative: Path, *, label: str) -> Path:
    """Reject a symlink in any caller-visible path component before resolve."""
    if relative.is_absolute() or any(part in {'.', '..'} for part in relative.parts):
        raise ValueError(f'{label} must be a safe relative path')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'{label} path must not use symlinks')
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} must be an existing contained file') from error
    if not resolved.is_file():
        raise ValueError(f'{label} must be a regular file')
    return resolved


def _approved_checkpoint_root(relative: Path) -> Path:
    for approved in _APPROVED_CHECKPOINT_ROOTS:
        if relative.parts[:len(approved.parts)] == approved.parts:
            return approved
    raise ValueError(
        'candidate checkpoint is not under an approved shared asset root')


def authorize_manifest_candidate(
        repository_root: Path, manifest_path: Path, candidate_id: str,
        ) -> AuthorizedCandidate:
    """Select and bind exactly one tracked candidate before checkpoint load."""
    root = lexical_repository_root(repository_root)
    manifest_relative = Path(manifest_path)
    if manifest_relative.is_absolute():
        try:
            manifest_relative = manifest_relative.relative_to(root)
        except ValueError as error:
            raise ValueError('candidate manifest must be repository-contained') from error
    manifest = _lexical_file(root, manifest_relative, label='candidate manifest')
    matches = [
        item for item in load_candidate_manifest(manifest)
        if item.id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f'manifest must contain exactly one candidate {candidate_id!r}')
    candidate = matches[0]
    commit = clean_git_commit(root)
    source = build_source_binding(
        repository_root=root, candidate=candidate,
        manifest_path=manifest, git_commit=commit)
    config = _lexical_file(root, candidate.config, label='candidate config')
    asset_root = resolve_project_asset_root(root)
    checkpoint_root = _approved_checkpoint_root(candidate.checkpoint)
    resolve_shared_asset_exposure(
        root, asset_root, checkpoint_root, label='candidate checkpoint')
    checkpoint = _lexical_file(
        asset_root, candidate.checkpoint, label='candidate checkpoint')
    actual = _sha256(checkpoint)
    if actual != candidate.checkpoint_sha256:
        raise ValueError(
            f'candidate checkpoint sha256 mismatch: expected '
            f'{candidate.checkpoint_sha256}, got {actual}')
    return AuthorizedCandidate(candidate, config, checkpoint, source)


def authorize_candidate_checkpoint_reference(
        repository_root: Path, manifest_path: Path, candidate: CandidateSpec,
        reference: object) -> Path:
    """Bind one serialized logical checkpoint reference to its approved file."""
    if not isinstance(reference, Mapping) or set(reference) != {
            'path', 'sha256'}:
        raise ValueError('candidate checkpoint reference is invalid')
    expected = {
        'path': candidate.checkpoint.as_posix(),
        'sha256': candidate.checkpoint_sha256,
    }
    if dict(reference) != expected:
        raise ValueError(
            'candidate checkpoint reference disagrees with manifest')
    try:
        authorized = authorize_manifest_candidate(
            repository_root, manifest_path, candidate.id)
    except (
            OSError, RuntimeError, subprocess.SubprocessError,
            ValueError) as error:
        raise ValueError(
            f'candidate checkpoint is not authorized: {error}') from error
    if authorized.candidate != candidate:
        raise ValueError(
            'candidate checkpoint reference differs from authorized manifest')
    return authorized.checkpoint_path


def authorized_tracked_file(
        repository_root: Path, relative: Path | str, *, commit: str,
        label: str) -> Path:
    """Resolve a nonsymlink source file and require its exact Git blob."""
    root = lexical_repository_root(repository_root)
    path = _lexical_file(root, Path(relative), label=label)
    relative_text = path.relative_to(root).as_posix()
    try:
        blob = subprocess.run(
            ['git', 'show', f'{commit}:{relative_text}'], cwd=root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f'{label} is not tracked at the authorized commit') from error
    if path.read_bytes() != blob:
        raise ValueError(f'{label} differs from the authorized Git blob')
    return path


def tensor_state(checkpoint: Path) -> dict[str, Tensor]:
    """Load the known MMPose mapping with PyTorch's restricted unpickler."""
    from mmengine.logging.history_buffer import HistoryBuffer

    safe: list[object] = [
        np.core.multiarray._reconstruct,
        np.core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        HistoryBuffer,
        getattr,
    ]
    safe.extend(
        value for value in vars(np.dtypes).values()
        if isinstance(value, type))
    try:
        with torch.serialization.safe_globals(safe):
            payload = torch.load(
                checkpoint, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(
            f'authorized checkpoint is not weights-only compatible: {error}') from error
    if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
        payload = payload['state_dict']
    if (
            not isinstance(payload, dict)
            or not payload
            or not all(isinstance(key, str) and isinstance(value, Tensor)
                       for key, value in payload.items())):
        raise ValueError('checkpoint must contain a non-empty tensor state_dict')
    return dict(payload)


def _neutralize_initializers(value: Any) -> Any:
    """Return a copy with every implicit model initializer disabled."""
    if isinstance(value, Mapping):
        result = copy.deepcopy(value)
        for key in list(result):
            if key in {'pretrained', 'init_cfg'}:
                result[key] = None
            else:
                result[key] = _neutralize_initializers(result[key])
        return result
    if isinstance(value, list):
        return [_neutralize_initializers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_neutralize_initializers(item) for item in value)
    return copy.deepcopy(value)


def neutralize_model_initializers(config: Any) -> Any:
    """Copy a Config and prevent model construction from loading any asset."""
    result = copy.deepcopy(config)
    if not hasattr(result, 'model'):
        raise ValueError('config must define a model')
    result.model = _neutralize_initializers(result.model)
    return result


def load_tensor_state_strict(model: Any, state: Mapping[str, Tensor]) -> None:
    """Inject an exact, finite tensor state with no missing/extra coercions."""
    if (
            not isinstance(state, Mapping)
            or not state
            or not all(isinstance(key, str) and isinstance(value, Tensor)
                       for key, value in state.items())):
        raise ValueError('state must be a non-empty string-to-tensor mapping')
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing:
        raise ValueError(f'checkpoint has missing state keys: {missing}')
    if unexpected:
        raise ValueError(f'checkpoint has unexpected state keys: {unexpected}')
    for key, target in expected.items():
        value = state[key]
        if value.shape != target.shape:
            raise ValueError(
                f'checkpoint tensor shape mismatch for {key}: '
                f'{tuple(value.shape)} != {tuple(target.shape)}')
        if value.dtype != target.dtype:
            raise ValueError(
                f'checkpoint tensor dtype mismatch for {key}: '
                f'{value.dtype} != {target.dtype}')
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()):
            raise ValueError(f'checkpoint tensor is nonfinite for {key}')

    from mmengine.runner.checkpoint import _load_checkpoint_to_model

    _load_checkpoint_to_model(
        model, {'state_dict': dict(state)}, strict=True)
    loaded = model.state_dict()
    for key, value in state.items():
        actual = loaded[key]
        if (
                actual.shape != value.shape
                or actual.dtype != value.dtype
                or not torch.equal(actual.detach().cpu(), value.detach().cpu())):
            raise ValueError(
                f'checkpoint post-load verification failed for {key}')


def _build_authorized_model(
        authorized: AuthorizedCandidate, *, config: Any | None = None,
        device: str = 'cpu') -> Any:
    """Construct without implicit I/O, then inject authorized tensor state."""
    from mmengine.config import Config
    from mmpose.apis import init_model

    source_config = (
        Config.fromfile(str(authorized.config_path))
        if config is None else config)
    safe_config = neutralize_model_initializers(source_config)
    model = init_model(safe_config, None, device=device)
    load_tensor_state_strict(model, tensor_state(authorized.checkpoint_path))
    return model


def build_manifest_authorized_model(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec, *, config: Any | None = None,
        device: str = 'cpu') -> Any:
    """Authorize manifest identity and safely construct its exact model."""
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    authorized = authorize_manifest_candidate(
        repository_root, manifest_path, candidate_id)
    if isinstance(candidate, CandidateSpec) and authorized.candidate != candidate:
        raise ValueError('candidate differs from the authorized manifest entry')
    return _build_authorized_model(
        authorized, config=config, device=device)
