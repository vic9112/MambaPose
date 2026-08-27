"""Manifest-authorized, data-only candidate checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Mapping

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
