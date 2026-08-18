"""Checkpoint integrity, provenance, resume, and evaluation selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch


class PermanentCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointValidation:
    path: Path
    valid: bool
    error: str | None
    epoch: int | None
    iteration: int | None
    sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(
        path: Path | str,
        *,
        require_training_state: bool = False) -> CheckpointValidation:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError('checkpoint root is not a mapping')
        state_dict = checkpoint.get('state_dict')
        if (not isinstance(state_dict, dict) or not state_dict
                or not any(isinstance(value, torch.Tensor)
                           for value in state_dict.values())):
            raise ValueError('checkpoint has no model state_dict tensors')
        meta = checkpoint.get('meta')
        if not isinstance(meta, dict) or not isinstance(meta.get('epoch'), int):
            raise ValueError('checkpoint has no integer meta.epoch')
        if require_training_state:
            if not any(key in checkpoint for key in ('optimizer', 'optim_wrapper')):
                raise ValueError('checkpoint has no optimizer training state')
            if not any(key in checkpoint for key in (
                    'param_schedulers', 'param_scheduler')):
                raise ValueError('checkpoint has no param scheduler training state')
        return CheckpointValidation(
            path=path,
            valid=True,
            error=None,
            epoch=meta['epoch'],
            iteration=meta.get('iter'),
            sha256=_sha256(path))
    except Exception as error:
        return CheckpointValidation(
            path=path,
            valid=False,
            error=str(error),
            epoch=None,
            iteration=None,
            sha256=None)


def _validate_provenance(
        checkpoint_dir: Path, expected: Mapping[str, Any]) -> None:
    path = checkpoint_dir / 'provenance.json'
    try:
        actual = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise PermanentCheckpointError(
            f'checkpoint provenance is unavailable: {error}') from error
    mismatches = {
        key: {'expected': value, 'actual': actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise PermanentCheckpointError(
            f'checkpoint provenance mismatch: {mismatches}')


def _epoch_from_name(path: Path) -> int:
    match = re.search(r'epoch_(\d+)', path.name)
    return int(match.group(1)) if match else -1


def select_resume(
        checkpoint_dir: Path | str,
        expected_provenance: Mapping[str, Any]) -> CheckpointValidation | None:
    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(
        checkpoint_dir.glob('epoch_*.pth'),
        key=_epoch_from_name,
        reverse=True)
    if not candidates:
        return None
    _validate_provenance(checkpoint_dir, expected_provenance)
    for candidate in candidates:
        validation = validate_checkpoint(
            candidate, require_training_state=True)
        if validation.valid:
            return validation
    return None


def select_evaluation_checkpoint(checkpoint_dir: Path | str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    best = sorted(
        checkpoint_dir.glob('best_*.pth'),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True)
    candidates = best or sorted(
        checkpoint_dir.glob('epoch_*.pth'),
        key=_epoch_from_name,
        reverse=True)
    for candidate in candidates:
        if validate_checkpoint(candidate).valid:
            return candidate
    raise PermanentCheckpointError(
        f'no valid evaluation checkpoint in {checkpoint_dir}')

