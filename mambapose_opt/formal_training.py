"""Authenticated formal Stage C training, replay, and stop primitives.

The production entry points in this module are intentionally separated from
the standard-library launcher.  Torch is imported only after the launcher has
established the environment authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import stat
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch

from .formal_checkpoint import FileAuthority
from .formal_checkpoint import load_formal_backbone_initialization
from .formal_checkpoint import (
    FormalCheckpointError,
    load_authenticated_tensor_document,
)
from .formal_determinism import trace_epoch_orders
from .formal_environment import (
    EnvironmentAuthority,
    validate_environment_authority,
)
from .formal_schema import (
    FileBinding,
    FormalRunInit,
    FormalStageCManifest,
    FormalTrainResult,
    config_closure_sha256,
    canonical_main_root,
    formal_resolved_config_sha256,
    load_formal_manifest,
    load_formal_run_init,
)


class FormalRepeatabilityError(ValueError):
    """A model preflight cannot establish deterministic replay."""


class FormalTrainingError(ValueError):
    """A formal training or resume authority is invalid."""


FORMAL_TOLERANCES = (
    ('atol', 1e-6),
    ('rtol', 1e-5),
)
_EVIDENCE_STAGES = (
    'backward', 'forward', 'gradient', 'loss', 'optimizer_update')


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in '0123456789abcdef' for character in value)


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in '0123456789abcdef' for character in value)


def _valid_seed(seed: object) -> bool:
    return isinstance(seed, int) and not isinstance(seed, bool) \
        and 0 <= seed <= 2**32 - 1


def _tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(items):
        if not isinstance(name, str) or not name \
                or not isinstance(tensor, torch.Tensor):
            raise FormalRepeatabilityError('tensor digest input is invalid')
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) \
                and not torch.isfinite(value).all().item():
            raise FormalRepeatabilityError('tensor evidence must be finite')
        encoded = name.encode('utf-8')
        dtype = str(value.dtype).encode('ascii')
        digest.update(struct.pack('>I', len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack('>I', len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack('>I', value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack('>Q', dimension))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    digest.update(struct.pack('>Q', count))
    return digest.hexdigest()


def _metrics(tensor: torch.Tensor) -> tuple[tuple[str, float | int], ...]:
    value = tensor.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if not torch.isfinite(value).all().item():
        raise FormalRepeatabilityError('tensor evidence must be finite')
    if value.numel() == 0:
        return (('count', 0), ('l2', 0.0), ('max_abs', 0.0), ('mean', 0.0))
    return (
        ('count', value.numel()),
        ('l2', float(torch.linalg.vector_norm(value).item())),
        ('max_abs', float(value.abs().max().item())),
        ('mean', float(value.mean().item())),
    )


@dataclass(frozen=True)
class FormalRepeatabilityResult:
    schema_version: int
    role: Literal['baseline', 'no_pif']
    seed: int
    config_path: str
    config_closure_sha256: str
    resolved_config_sha256: str
    source_commit: str
    environment_inventory_sha256: str
    initialization_sha256: str
    complete_initial_state_sha256: str
    common_state_sha256: str
    common_state_keys: tuple[str, ...]
    evidence_sha256: Mapping[str, str]
    evidence_metrics: Mapping[str, tuple[tuple[str, float | int], ...]]
    gradients_finite: bool
    custom_scan_exercised: bool
    tolerances: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PairedInitializationResult:
    schema_version: int
    seed: int
    common_state_sha256: str
    baseline_complete_sha256: str
    no_pif_complete_sha256: str


def build_formal_repeatability_result(
        *, role: str, seed: int, config_path: str,
        config_closure_sha256: str, resolved_config_sha256: str,
        source_commit: str, environment_inventory_sha256: str,
        initialization_sha256: str,
        initial_state: Mapping[str, torch.Tensor],
        common_state_keys: Sequence[str],
        evidence: Mapping[str, torch.Tensor], gradients_finite: bool,
        custom_scan_exercised: bool,
        tolerances: Sequence[tuple[str, float]],
        ) -> FormalRepeatabilityResult:
    if not isinstance(initial_state, Mapping) or not initial_state:
        raise FormalRepeatabilityError('initial state is missing')
    keys = tuple(sorted(common_state_keys))
    if not keys or len(keys) != len(set(keys)) \
            or any(key not in initial_state for key in keys):
        raise FormalRepeatabilityError('common state keys are invalid')
    if set(evidence) != set(_EVIDENCE_STAGES):
        raise FormalRepeatabilityError('repeatability evidence is incomplete')
    if gradients_finite is not True:
        raise FormalRepeatabilityError('gradients must be finite')
    result = FormalRepeatabilityResult(
        schema_version=1,
        role=role,
        seed=seed,
        config_path=config_path,
        config_closure_sha256=config_closure_sha256,
        resolved_config_sha256=resolved_config_sha256,
        source_commit=source_commit,
        environment_inventory_sha256=environment_inventory_sha256,
        initialization_sha256=initialization_sha256,
        complete_initial_state_sha256=_tensor_digest(initial_state.items()),
        common_state_sha256=_tensor_digest(
            (key, initial_state[key]) for key in keys),
        common_state_keys=keys,
        evidence_sha256=MappingProxyType({
            key: _tensor_digest(((key, evidence[key]),))
            for key in sorted(evidence)}),
        evidence_metrics=MappingProxyType({
            key: _metrics(evidence[key]) for key in sorted(evidence)}),
        gradients_finite=True,
        custom_scan_exercised=custom_scan_exercised,
        tolerances=tuple(tolerances),
    )
    validate_formal_repeatability_result(result)
    return result


def validate_formal_repeatability_result(
        result: FormalRepeatabilityResult) -> None:
    if not isinstance(result, FormalRepeatabilityResult):
        raise FormalRepeatabilityError('repeatability result type is invalid')
    if result.schema_version != 1 or isinstance(result.schema_version, bool):
        raise FormalRepeatabilityError('repeatability schema_version must be 1')
    if result.role not in {'baseline', 'no_pif'}:
        raise FormalRepeatabilityError('repeatability role is invalid')
    if not _valid_seed(result.seed):
        raise FormalRepeatabilityError('repeatability seed is invalid')
    if not isinstance(result.config_path, str) \
            or not result.config_path.startswith('configs/') \
            or any(part in {'', '.', '..'}
                   for part in Path(result.config_path).parts):
        raise FormalRepeatabilityError('repeatability config path is invalid')
    for value, label in (
            (result.config_closure_sha256, 'config closure'),
            (result.resolved_config_sha256, 'resolved config'),
            (result.environment_inventory_sha256, 'environment'),
            (result.initialization_sha256, 'initialization'),
            (result.complete_initial_state_sha256, 'complete initial state'),
            (result.common_state_sha256, 'common state')):
        if not _is_sha(value):
            raise FormalRepeatabilityError(f'{label} SHA-256 is invalid')
    if not _is_commit(result.source_commit):
        raise FormalRepeatabilityError('source commit is invalid')
    if result.common_state_keys != tuple(sorted(set(result.common_state_keys))) \
            or not result.common_state_keys:
        raise FormalRepeatabilityError('common state keys are invalid')
    if set(result.evidence_sha256) != set(_EVIDENCE_STAGES) \
            or set(result.evidence_metrics) != set(_EVIDENCE_STAGES):
        raise FormalRepeatabilityError('repeatability evidence is incomplete')
    if any(not _is_sha(value) for value in result.evidence_sha256.values()):
        raise FormalRepeatabilityError('evidence SHA-256 is invalid')
    for stage, metrics in result.evidence_metrics.items():
        if tuple(name for name, _value in metrics) != (
                'count', 'l2', 'max_abs', 'mean'):
            raise FormalRepeatabilityError(
                f'{stage} evidence metrics are invalid')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for _name, value in metrics):
            raise FormalRepeatabilityError(
                f'{stage} evidence metrics are non-finite')
    if result.gradients_finite is not True:
        raise FormalRepeatabilityError('gradients must be finite')
    if result.custom_scan_exercised is not True:
        raise FormalRepeatabilityError('custom selective scan evidence is required')
    if result.tolerances != FORMAL_TOLERANCES:
        raise FormalRepeatabilityError('repeatability tolerances differ from policy')


def repeatability_result_to_dict(
        result: FormalRepeatabilityResult) -> dict[str, Any]:
    validate_formal_repeatability_result(result)
    return {
        'schema_version': 1,
        'identity': {
            'role': result.role,
            'seed': result.seed,
            'config_path': result.config_path,
            'config_closure_sha256': result.config_closure_sha256,
            'resolved_config_sha256': result.resolved_config_sha256,
            'source_commit': result.source_commit,
            'environment_inventory_sha256': (
                result.environment_inventory_sha256),
            'initialization_sha256': result.initialization_sha256,
        },
        'initial_state': {
            'complete_sha256': result.complete_initial_state_sha256,
            'common_sha256': result.common_state_sha256,
            'common_keys': list(result.common_state_keys),
        },
        'evidence': {
            stage: {
                'sha256': result.evidence_sha256[stage],
                'metrics': dict(result.evidence_metrics[stage]),
            } for stage in sorted(result.evidence_sha256)
        },
        'gradients_finite': result.gradients_finite,
        'custom_scan_exercised': result.custom_scan_exercised,
        'tolerances': dict(result.tolerances),
    }


def formal_repeatability_result_from_dict(
        value: Mapping[str, Any]) -> FormalRepeatabilityResult:
    if not isinstance(value, Mapping) or set(value) != {
            'schema_version', 'identity', 'initial_state', 'evidence',
            'gradients_finite', 'custom_scan_exercised', 'tolerances'}:
        raise FormalRepeatabilityError('repeatability document fields are invalid')
    identity = value['identity']
    state = value['initial_state']
    evidence = value['evidence']
    tolerances = value['tolerances']
    if not isinstance(identity, Mapping) or set(identity) != {
            'role', 'seed', 'config_path', 'config_closure_sha256',
            'resolved_config_sha256', 'source_commit',
            'environment_inventory_sha256', 'initialization_sha256'}:
        raise FormalRepeatabilityError('repeatability identity fields are invalid')
    if not isinstance(state, Mapping) or set(state) != {
            'complete_sha256', 'common_sha256', 'common_keys'}:
        raise FormalRepeatabilityError('repeatability state fields are invalid')
    if not isinstance(evidence, Mapping) or set(evidence) != set(
            _EVIDENCE_STAGES):
        raise FormalRepeatabilityError('repeatability evidence fields are invalid')
    for stage, record in evidence.items():
        if not isinstance(record, Mapping) or set(record) != {
                'sha256', 'metrics'} or not isinstance(
                    record['metrics'], Mapping) or set(
                        record['metrics']) != {'count', 'l2', 'max_abs', 'mean'}:
            raise FormalRepeatabilityError(
                f'{stage} repeatability evidence record is invalid')
    if not isinstance(tolerances, Mapping) or set(tolerances) != {
            'atol', 'rtol'}:
        raise FormalRepeatabilityError('repeatability tolerance fields are invalid')
    try:
        result = FormalRepeatabilityResult(
            schema_version=value['schema_version'], role=identity['role'],
            seed=identity['seed'], config_path=identity['config_path'],
            config_closure_sha256=identity['config_closure_sha256'],
            resolved_config_sha256=identity['resolved_config_sha256'],
            source_commit=identity['source_commit'],
            environment_inventory_sha256=(
                identity['environment_inventory_sha256']),
            initialization_sha256=identity['initialization_sha256'],
            complete_initial_state_sha256=state['complete_sha256'],
            common_state_sha256=state['common_sha256'],
            common_state_keys=tuple(state['common_keys']),
            evidence_sha256=MappingProxyType({
                stage: record['sha256'] for stage, record in evidence.items()}),
            evidence_metrics=MappingProxyType({
                stage: tuple(sorted(record['metrics'].items()))
                for stage, record in evidence.items()}),
            gradients_finite=value['gradients_finite'],
            custom_scan_exercised=value['custom_scan_exercised'],
            tolerances=tuple(sorted(tolerances.items())),
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise FormalRepeatabilityError(
            'repeatability document types are invalid') from error
    validate_formal_repeatability_result(result)
    return result


def write_immutable_artifact(
        destination: Path, payload: bytes, repository_root: Path) -> None:
    root = Path(repository_root).absolute()
    target = Path(destination).absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise FormalTrainingError('artifact path escapes the frozen root') from error
    if not relative.parts or '..' in relative.parts:
        raise FormalTrainingError('artifact path is not canonical')
    cursor = root
    for part in relative.parent.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise FormalTrainingError('artifact path contains a symlink')
    if target.exists():
        if target.is_symlink() or not target.is_file() \
                or target.read_bytes() != payload:
            raise FormalTrainingError('immutable artifact already differs')
        return
    _atomic_write_bytes(target, payload)


def _same_authority(
        first: FormalRepeatabilityResult,
        second: FormalRepeatabilityResult,
        *, paired: bool) -> None:
    if first.seed != second.seed:
        raise FormalRepeatabilityError('repeatability seed differs')
    for field in (
            'source_commit', 'environment_inventory_sha256',
            'initialization_sha256', 'tolerances'):
        if getattr(first, field) != getattr(second, field):
            raise FormalRepeatabilityError(
                f'repeatability authority differs at {field}')
    if not paired:
        for field in (
                'role', 'config_path', 'config_closure_sha256',
                'resolved_config_sha256'):
            if getattr(first, field) != getattr(second, field):
                raise FormalRepeatabilityError(
                    f'repeatability replay differs at {field}')


def compare_repeatability_replay(
        first: FormalRepeatabilityResult,
        second: FormalRepeatabilityResult) -> None:
    validate_formal_repeatability_result(first)
    validate_formal_repeatability_result(second)
    _same_authority(first, second, paired=False)
    if first.complete_initial_state_sha256 \
            != second.complete_initial_state_sha256:
        raise FormalRepeatabilityError('complete initial state did not replay')
    atol = dict(FORMAL_TOLERANCES)['atol']
    rtol = dict(FORMAL_TOLERANCES)['rtol']
    for stage in _EVIDENCE_STAGES:
        first_metrics = dict(first.evidence_metrics[stage])
        second_metrics = dict(second.evidence_metrics[stage])
        if first_metrics['count'] != second_metrics['count']:
            raise FormalRepeatabilityError(
                f'{stage} evidence count did not replay')
        for metric in ('l2', 'max_abs', 'mean'):
            left = first_metrics[metric]
            right = second_metrics[metric]
            if abs(left - right) > atol + rtol * abs(left):
                raise FormalRepeatabilityError(
                    f'{stage} evidence exceeded tolerance')


def compare_paired_initial_state(
        baseline: FormalRepeatabilityResult,
        no_pif: FormalRepeatabilityResult) -> PairedInitializationResult:
    validate_formal_repeatability_result(baseline)
    validate_formal_repeatability_result(no_pif)
    if baseline.role != 'baseline' or no_pif.role != 'no_pif':
        raise FormalRepeatabilityError('paired roles are not canonical')
    _same_authority(baseline, no_pif, paired=True)
    if baseline.common_state_keys != no_pif.common_state_keys \
            or baseline.common_state_sha256 != no_pif.common_state_sha256:
        raise FormalRepeatabilityError('paired common initial state differs')
    return PairedInitializationResult(
        schema_version=1,
        seed=baseline.seed,
        common_state_sha256=baseline.common_state_sha256,
        baseline_complete_sha256=baseline.complete_initial_state_sha256,
        no_pif_complete_sha256=no_pif.complete_initial_state_sha256,
    )


def _tensor_leaves(value: object, prefix: str = 'value') \
        -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        leaves: list[tuple[str, torch.Tensor]] = []
        for key in sorted(value):
            leaves.extend(_tensor_leaves(value[key], f'{prefix}.{key}'))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(item, f'{prefix}[{index}]'))
        return leaves
    raise FormalRepeatabilityError('model evidence contains a non-tensor leaf')


def _summary_vector(value: object) -> torch.Tensor:
    records: list[torch.Tensor] = []
    for _name, tensor in _tensor_leaves(value):
        flat = tensor.detach().to(dtype=torch.float64).reshape(-1)
        if not torch.isfinite(flat).all().item():
            raise FormalRepeatabilityError('model evidence must be finite')
        if flat.numel():
            records.append(torch.tensor([
                float(flat.numel()), float(flat.sum().item()),
                float(flat.abs().max().item()),
                float(torch.linalg.vector_norm(flat).item()),
            ], dtype=torch.float64))
        else:
            records.append(torch.zeros(4, dtype=torch.float64))
    if not records:
        raise FormalRepeatabilityError('model evidence contains no tensors')
    return torch.cat(records)


def _graph_exercised_custom_scan(loss: torch.Tensor) -> bool:
    pending = [loss.grad_fn]
    observed: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in observed:
            continue
        observed.add(id(node))
        if 'SelectiveScan' in type(node).__name__:
            return True
        pending.extend(
            child for child, _index in getattr(node, 'next_functions', ())
            if child is not None)
    return False


def _neutralize_pretrained(config: Any) -> None:
    model = config.model
    if 'pretrained' in model:
        model.pretrained = None
    if 'init_cfg' in model:
        model.init_cfg = None
    if 'backbone' not in model:
        raise FormalRepeatabilityError('formal model has no backbone config')
    model.backbone.pretrained = None
    if 'init_cfg' in model.backbone:
        model.backbone.init_cfg = None


def _read_authenticated_config_file(root: Path, relative: Path) -> bytes:
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise FormalTrainingError('config closure path is not canonical')
    path = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FormalTrainingError('config closure contains a symlink')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FormalTrainingError('config closure file cannot be opened') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FormalTrainingError('config closure member is not a file')
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise FormalTrainingError('config closure changed during capture')
    return b''.join(blocks)


def _capture_config_closure(
        root: Path, leaf: Path) -> Mapping[str, bytes]:
    records: dict[str, bytes] = {}
    visiting: set[str] = set()

    def visit(relative: Path) -> None:
        try:
            normalized = Path(os.path.abspath(root / relative)).relative_to(root)
        except ValueError as error:
            raise FormalTrainingError('config base escapes frozen root') from error
        if '..' in normalized.parts:
            raise FormalTrainingError('config base path is not canonical')
        key = normalized.as_posix()
        if key in visiting:
            raise FormalTrainingError('config inheritance cycle')
        if key in records:
            return
        visiting.add(key)
        data = _read_authenticated_config_file(root, normalized)
        try:
            tree = ast.parse(data.decode('utf-8'), filename=key)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise FormalTrainingError('config closure member is malformed') from error
        bases: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id == '_base_':
                try:
                    bases = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as error:
                    raise FormalTrainingError(
                        'config base declaration is not literal') from error
                break
        if isinstance(bases, str):
            bases = [bases]
        if not isinstance(bases, (list, tuple)) or any(
                not isinstance(base, str) for base in bases):
            raise FormalTrainingError('config base declaration is invalid')
        for base in bases:
            if '\\' in base or Path(base).is_absolute():
                raise FormalTrainingError('config base path is not relative')
            visit(normalized.parent / base)
        records[key] = data
        visiting.remove(key)

    visit(leaf)
    return MappingProxyType(dict(sorted(records.items())))


def _captured_closure_sha256(records: Mapping[str, bytes]) -> str:
    payload = json.dumps([
        {'path': path, 'sha256': hashlib.sha256(data).hexdigest()}
        for path, data in sorted(records.items())],
        sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _ConfigLoadAuthority:
    config: Path
    config_closure_sha256: str
    git_commit: str


def _validate_trace_source(root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
            cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalTrainingError('trace source Git authority failed') from error
    if not _is_commit(commit):
        raise FormalTrainingError('trace source commit is invalid')
    if dirty:
        raise FormalTrainingError('trace source must be clean')
    return commit


def _revalidate_live_config_authority(
        authority: _ConfigLoadAuthority, root: Path, *, source_kind: str
        ) -> None:
    if source_kind == 'frozen':
        commit = _validate_frozen_source(root)
    elif source_kind == 'trace':
        commit = _validate_trace_source(root)
    else:
        raise FormalTrainingError('formal config source kind is invalid')
    if commit != authority.git_commit:
        raise FormalTrainingError('formal source commit changed')
    if config_closure_sha256(root, authority.config) \
            != authority.config_closure_sha256:
        raise FormalTrainingError('formal config closure changed')


def _deterministic_config_fingerprint(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise FormalTrainingError(
            'resolved formal config is not deterministically serializable') \
            from error
    return hashlib.sha256(payload).hexdigest()


def _validate_private_config_snapshot(
        snapshot: Path, records: Mapping[str, bytes], leaf: Path) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir() \
            or stat.S_IMODE(snapshot.stat().st_mode) != 0o500:
        raise FormalTrainingError('private config snapshot root is unsafe')
    paths: set[str] = set()
    for candidate in snapshot.rglob('*'):
        if candidate.is_symlink():
            raise FormalTrainingError('private config snapshot contains a symlink')
        mode = stat.S_IMODE(candidate.stat().st_mode)
        if candidate.is_dir():
            if mode != 0o500:
                raise FormalTrainingError(
                    'private config snapshot directory mode changed')
        elif candidate.is_file():
            if mode != 0o400:
                raise FormalTrainingError(
                    'private config snapshot file mode changed')
            paths.add(candidate.relative_to(snapshot).as_posix())
        else:
            raise FormalTrainingError(
                'private config snapshot contains a special file')
    if paths != set(records) \
            or _capture_config_closure(snapshot, leaf) != records:
        raise FormalTrainingError('private config snapshot differs from capture')


def _remove_private_config_snapshot(snapshot: Path) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise FormalTrainingError('private config snapshot cleanup is unsafe')
    for current, directories, files in os.walk(
            snapshot, topdown=False, followlinks=False):
        directory = Path(current)
        for name in files:
            path = directory / name
            if path.is_symlink():
                path.unlink()
            else:
                path.chmod(0o600)
        for name in directories:
            path = directory / name
            if path.is_symlink():
                path.unlink()
            else:
                path.chmod(0o700)
        directory.chmod(0o700)
    shutil.rmtree(snapshot)


def _load_authenticated_config(
        authority: _ConfigLoadAuthority, repository_root: Path, *,
        source_kind: str):
    root = Path(repository_root).absolute()
    _revalidate_live_config_authority(
        authority, root, source_kind=source_kind)
    records = _capture_config_closure(root, authority.config)
    if _captured_closure_sha256(records) \
            != authority.config_closure_sha256:
        raise FormalTrainingError('captured formal config closure mismatch')
    _revalidate_live_config_authority(
        authority, root, source_kind=source_kind)
    temporary_root = Path('/tmp')
    if temporary_root.is_symlink() or not temporary_root.is_dir():
        raise FormalTrainingError('private system temporary root is unsafe')
    snapshot = Path(tempfile.mkdtemp(
        prefix='mambapose-formal-config-', dir=temporary_root))
    os.chmod(snapshot, 0o700)
    parsed = None
    detached = None
    fingerprint: str | None = None
    parse_error: Exception | None = None
    try:
        for relative_text, data in records.items():
            destination = snapshot / relative_text
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600)
            try:
                with os.fdopen(descriptor, 'wb', closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
            destination.chmod(0o400)
        directories = sorted(
            (path for path in snapshot.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True)
        for directory in directories:
            directory.chmod(0o500)
        snapshot.chmod(0o500)
        _validate_private_config_snapshot(snapshot, records, authority.config)
        from mmengine.config import Config
        parsed = Config.fromfile(snapshot / authority.config)
        _validate_private_config_snapshot(
            snapshot, records, authority.config)
        document = parsed.to_dict()
        fingerprint = _deterministic_config_fingerprint(document)
        detached = Config(document)
        if detached.filename is not None \
                or _deterministic_config_fingerprint(detached.to_dict()) \
                != fingerprint:
            raise FormalTrainingError(
                'detached formal config fingerprint mismatch')
    except Exception as error:
        parse_error = error
    finally:
        authority_error: Exception | None = None
        try:
            _revalidate_live_config_authority(
                authority, root, source_kind=source_kind)
        except Exception as error:
            authority_error = error
        try:
            _remove_private_config_snapshot(snapshot)
        except Exception as error:
            if authority_error is None:
                authority_error = error
        try:
            _revalidate_live_config_authority(
                authority, root, source_kind=source_kind)
        except Exception as error:
            if authority_error is None:
                authority_error = error
        if authority_error is not None:
            if isinstance(authority_error, FormalTrainingError):
                raise authority_error
            raise FormalTrainingError(
                'formal config authority failed after parse') from authority_error
    if parse_error is not None:
        if isinstance(parse_error, FormalTrainingError):
            raise parse_error
        raise FormalTrainingError(
            'authenticated formal config could not be parsed') from parse_error
    if parsed is None or detached is None or fingerprint is None:
        raise FormalTrainingError('authenticated formal config is missing')
    if _deterministic_config_fingerprint(detached.to_dict()) != fingerprint:
        raise FormalTrainingError(
            'formal config changed after private snapshot cleanup')
    return detached


def load_authenticated_formal_config(
        init: FormalRunInit, repository_root: Path):
    """Parse a private immutable copy of a run-init config closure."""
    if not isinstance(init, FormalRunInit):
        raise FormalTrainingError('formal config init type is invalid')
    authority = _ConfigLoadAuthority(
        config=init.config,
        config_closure_sha256=init.config_closure_sha256,
        git_commit=init.git_commit)
    return _load_authenticated_config(
        authority, repository_root, source_kind='frozen')


def load_authenticated_trace_config(
        manifest: FormalStageCManifest, run_id: str, repository_root: Path):
    """Parse the selected public-manifest config from captured exact bytes."""
    if not isinstance(manifest, FormalStageCManifest):
        raise FormalTrainingError('trace manifest type is invalid')
    root = Path(repository_root).absolute()
    if manifest.repository_root.absolute() != root:
        raise FormalTrainingError('trace manifest repository root mismatch')
    matches = tuple(spec for spec in manifest.runs if spec.run_id == run_id)
    if len(matches) != 1:
        raise FormalTrainingError('trace run id is invalid')
    spec = matches[0]
    commit = _validate_trace_source(root)
    authority = _ConfigLoadAuthority(
        config=spec.config,
        config_closure_sha256=spec.config_sha256,
        git_commit=commit)
    return _load_authenticated_config(
        authority, root, source_kind='trace')


def run_formal_model_preflight(
        pair_seed: int, role: str) -> FormalRepeatabilityResult:
    """Exercise one real MambaPose batch after process-first admission.

    This function intentionally has no injectable production callback.  Unit
    tests exercise the pure artifact builder; the production path always
    rebuilds the canonical config/model/dataloader from its run-init authority.
    """
    if role not in {'baseline', 'no_pif'} or not _valid_seed(pair_seed) \
            or pair_seed not in range(5):
        raise FormalRepeatabilityError('formal preflight identity is invalid')
    root = Path.cwd().absolute()
    stem = 'full' if role == 'baseline' else 'no-pif'
    run_id = f'{stem}-seed{pair_seed}'
    init_path = (root / 'work_dirs/optimization/formal-stage-c' /
                 run_id / 'run-init.json')
    init = load_formal_run_init(init_path, repository_root=root)
    if init.role != role or init.seed != pair_seed:
        raise FormalRepeatabilityError('preflight run-init identity mismatch')
    environment = EnvironmentAuthority.capture(root)
    if environment.inventory_sha256 != init.environment_inventory_sha256:
        raise FormalRepeatabilityError('preflight environment authority drift')

    # Root determinism must precede config, registry, model, and loader setup.
    from .formal_determinism import configure_root_determinism
    configure_root_determinism(pair_seed)
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmpose.registry import MODELS
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    config = load_authenticated_formal_config(init, root)
    _neutralize_pretrained(config)
    init_default_scope(config.get('default_scope', 'mmpose'))
    model = MODELS.build(config.model)
    model.init_weights()
    canonical = canonical_main_root(root)
    asset = init.initialization.asset
    initialization_path = (
        canonical / asset.authority_root / asset.asset_relative_path)
    initialization_authority = FileAuthority(
        authority_root=str(canonical / asset.authority_root),
        path=asset.asset_relative_path.as_posix(), sha256=asset.sha256)
    report = load_formal_backbone_initialization(
        initialization_path, initialization_authority, model.backbone,
        minimum_compatible_tensors=100)
    if report.compatible_tensors < 100:
        raise FormalRepeatabilityError(
            'preflight backbone initialization is incomplete')
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()}
    common_keys = tuple(sorted(initial_state))

    if not torch.cuda.is_available():
        raise FormalRepeatabilityError('formal model preflight requires CUDA')
    model = model.to(torch.device('cuda:0'))
    dataloader = Runner.build_dataloader(
        config.train_dataloader, seed=pair_seed, diff_rank_seed=False)
    batch = next(iter(dataloader))
    processed = model.data_preprocessor(batch, training=True)
    prediction = model(**processed, mode='tensor')
    losses = model(**processed, mode='loss')
    loss, _log_vars = model.parse_losses(losses)
    custom_scan = _graph_exercised_custom_scan(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {
        key: parameter.grad.detach().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None}
    if not gradients or any(
            not torch.isfinite(value).all().item()
            for value in gradients.values()):
        raise FormalRepeatabilityError('preflight gradients are not finite')
    before = {
        key: parameter.detach().clone()
        for key, parameter in model.named_parameters()}
    optimizer.step()
    updates = {
        key: parameter.detach() - before[key]
        for key, parameter in model.named_parameters()}
    evidence = {
        'forward': _summary_vector(prediction),
        'loss': loss.detach().reshape(1),
        'backward': _summary_vector(losses),
        'gradient': _summary_vector(gradients),
        'optimizer_update': _summary_vector(updates),
    }
    return build_formal_repeatability_result(
        role=role, seed=pair_seed, config_path=init.config.as_posix(),
        config_closure_sha256=init.config_closure_sha256,
        resolved_config_sha256=init.resolved_config_sha256,
        source_commit=init.git_commit,
        environment_inventory_sha256=init.environment_inventory_sha256,
        initialization_sha256=init.initialization.sha256,
        initial_state=initial_state, common_state_keys=common_keys,
        evidence=evidence, gradients_finite=True,
        custom_scan_exercised=custom_scan, tolerances=FORMAL_TOLERANCES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _init_document(init: FormalRunInit) -> dict[str, Any]:
    asset = init.initialization.asset
    return {
        'schema_version': 1,
        'manifest_sha256': init.manifest_sha256,
        'source': {'git_commit': init.git_commit, 'clean_tree': True},
        'config': {
            'path': init.config.as_posix(),
            'closure_sha256': init.config_closure_sha256,
            'resolved_sha256': init.resolved_config_sha256,
        },
        'environment_inventory_sha256': init.environment_inventory_sha256,
        'data_authority': dict(sorted(init.data_authority.items())),
        'run': {
            'run_id': init.run_id,
            'role': init.role,
            'seed': init.seed,
            'epochs': init.epochs,
            'effective_batch_size': init.effective_batch_size,
            'worker_count': init.worker_count,
            'persistent_workers': init.persistent_workers,
            'output_root': init.output_root.as_posix(),
        },
        'initialization': {
            'id': init.initialization.id,
            'kind': init.initialization.kind,
            'asset': {
                'authority_root': asset.authority_root,
                'target_root': asset.target_root,
                'asset_relative_path': asset.asset_relative_path.as_posix(),
                'sha256': asset.sha256,
            },
        },
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        + '\n').encode('utf-8')


def _destination_root(destination: Path, relative_parent: Path) -> Path:
    absolute = destination.absolute()
    expected_suffix = relative_parent / destination.name
    if tuple(absolute.parts[-len(expected_suffix.parts):]) \
            != expected_suffix.parts:
        raise FormalTrainingError('formal output path is not canonical')
    root = absolute.parents[len(relative_parent.parts)]
    cursor = root
    for part in relative_parent.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise FormalTrainingError('formal output path contains a symlink')
    return root


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_formal_run_init(
        init: FormalRunInit, destination: Path) -> FileAuthority:
    if not isinstance(init, FormalRunInit):
        raise FormalTrainingError('run init type is invalid')
    target = Path(destination)
    root = _destination_root(target, init.output_root)
    if target.name != 'run-init.json':
        raise FormalTrainingError('formal run init filename is not canonical')
    payload = _canonical_json_bytes(_init_document(init))
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise FormalTrainingError('immutable run init target is unsafe')
        if target.read_bytes() != payload:
            raise FormalTrainingError('immutable run init already differs')
    else:
        _atomic_write_bytes(target, payload)
    return FileAuthority(
        authority_root=str(root),
        path=(init.output_root / target.name).as_posix(),
        sha256=_sha256_file(target))


def _validate_frozen_source(frozen_root: Path) -> str:
    raw = str(frozen_root)
    root = Path(frozen_root)
    if not root.is_absolute() or '..' in root.parts or '/./' in raw \
            or root.is_symlink() or not root.is_dir():
        raise FormalTrainingError(
            'frozen source root must be an exact absolute directory')
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        attached = subprocess.run(
            ['git', 'symbolic-ref', '-q', 'HEAD'], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
            cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalTrainingError('frozen source Git authority failed') from error
    if not _is_commit(commit):
        raise FormalTrainingError('frozen source commit is invalid')
    if attached.returncode == 0:
        raise FormalTrainingError('frozen source must be detached')
    if attached.returncode != 1:
        raise FormalTrainingError('frozen source attachment is unknown')
    if dirty:
        raise FormalTrainingError('frozen source must be clean')
    return commit


def _build_run_init_unchecked(
        manifest: FormalStageCManifest, run_id: str,
        environment: EnvironmentAuthority, frozen_root: Path,
        *, commit: str, manifest_sha256: str) -> FormalRunInit:
    matching = tuple(spec for spec in manifest.runs if spec.run_id == run_id)
    if len(matching) != 1:
        raise FormalTrainingError('formal run id is absent or duplicated')
    spec = matching[0]
    observed_closure = config_closure_sha256(frozen_root, spec.config)
    if observed_closure != spec.config_sha256:
        raise FormalTrainingError('formal run config closure drift')
    resolved = formal_resolved_config_sha256(
        frozen_root, role=spec.role, seed=spec.seed)
    protocol = manifest.protocol
    data = {
        f'{role}_sha256': authority.sha256
        for role, authority in manifest.data_authority.items()}
    return FormalRunInit(
        manifest_sha256=manifest_sha256,
        git_commit=commit,
        config=spec.config,
        config_closure_sha256=observed_closure,
        resolved_config_sha256=resolved,
        environment_inventory_sha256=environment.inventory_sha256,
        data_authority=MappingProxyType(dict(sorted(data.items()))),
        run_id=spec.run_id,
        role=spec.role,
        seed=spec.seed,
        epochs=protocol.epochs,
        effective_batch_size=protocol.effective_batch_size,
        worker_count=protocol.worker_count,
        persistent_workers=protocol.persistent_workers,
        output_root=spec.output_root,
        initialization=manifest.initialization,
    )


def build_formal_run_init(
        manifest: FormalStageCManifest, run_id: str,
        environment: EnvironmentAuthority, frozen_root: Path) -> FormalRunInit:
    root = Path(frozen_root)
    if not isinstance(manifest, FormalStageCManifest) \
            or not isinstance(environment, EnvironmentAuthority) \
            or manifest.repository_root != root:
        raise FormalTrainingError('manifest is not bound to the frozen root')
    commit = _validate_frozen_source(root)
    validate_environment_authority(environment, root)
    manifest_path = root / 'optimization/formal_stage_c.json'
    observed = load_formal_manifest(manifest_path, repository_root=root)
    if observed != manifest:
        raise FormalTrainingError('manifest differs from frozen observation')
    return _build_run_init_unchecked(
        manifest, run_id, environment, root, commit=commit,
        manifest_sha256=_sha256_file(manifest_path))


def build_all_formal_run_inits(
        manifest: FormalStageCManifest, environment: EnvironmentAuthority,
        frozen_root: Path) -> tuple[FormalRunInit, ...]:
    """Validate the root once, then build all ten documents before any write."""
    root = Path(frozen_root)
    if not isinstance(manifest, FormalStageCManifest) \
            or not isinstance(environment, EnvironmentAuthority) \
            or manifest.repository_root != root:
        raise FormalTrainingError('manifest is not bound to the frozen root')
    commit = _validate_frozen_source(root)
    validate_environment_authority(environment, root)
    manifest_path = root / 'optimization/formal_stage_c.json'
    observed = load_formal_manifest(manifest_path, repository_root=root)
    if observed != manifest or len(manifest.runs) != 10:
        raise FormalTrainingError('formal manifest matrix is not exact')
    manifest_sha = _sha256_file(manifest_path)
    results = tuple(_build_run_init_unchecked(
        manifest, spec.run_id, environment, root, commit=commit,
        manifest_sha256=manifest_sha) for spec in manifest.runs)
    if len({(item.role, item.seed) for item in results}) != 10:
        raise FormalTrainingError('formal run-init matrix is not exact')
    return results


def _jsonable_sequence(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_sequence(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_sequence(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise FormalTrainingError('RNG state contains an unsupported value')


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda = []
    if torch.cuda.is_initialized():
        cuda = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        'python': _jsonable_sequence(random.getstate()),
        'numpy': {
            'algorithm': numpy_state[0],
            'keys': torch.from_numpy(numpy_state[1].astype(np.int64)),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch': torch.get_rng_state().cpu(),
        'cuda': cuda,
    }


def _validate_finite_tree(value: object, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) \
                and not torch.isfinite(value).all().item():
            raise FormalTrainingError(f'{label} state is not finite')
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) or isinstance(key, bool)
               or (isinstance(key, int) and key < 0) for key in value):
            raise FormalTrainingError(f'{label} state has an invalid key')
        for key, item in value.items():
            _validate_finite_tree(item, f'{label}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f'{label}[{index}]')
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise FormalTrainingError(f'{label} state is not finite')
        return
    raise FormalTrainingError(f'{label} state has an unsupported value')


@dataclass(frozen=True)
class ResumeState:
    run_init_sha256: str
    checkpoint_sha256: str
    commit_path: str
    commit_sha256: str
    structured_log_sha256: str
    completed_epoch: int
    order_hashes: tuple[str, ...]
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    scaler_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]


@dataclass(frozen=True)
class _EpochCommitChain:
    documents: tuple[Mapping[str, Any], ...]
    paths: tuple[Path, ...]
    sha256s: tuple[str, ...]
    structured_log: bytes


def _resume_document(
        *, expected: FormalRunInit, run_init_sha256: str,
        completed_epoch: int, order_hashes: Sequence[str],
        model_state: Mapping[str, Any], optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any], scaler_state: Mapping[str, Any],
        ) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'run_init_sha256': run_init_sha256,
        'identity': {
            'manifest_sha256': expected.manifest_sha256,
            'git_commit': expected.git_commit,
            'config_path': expected.config.as_posix(),
            'config_closure_sha256': expected.config_closure_sha256,
            'resolved_config_sha256': expected.resolved_config_sha256,
            'environment_inventory_sha256': (
                expected.environment_inventory_sha256),
            'initialization_id': expected.initialization.id,
            'initialization_sha256': expected.initialization.sha256,
            'run_id': expected.run_id,
            'role': expected.role,
            'seed': expected.seed,
        },
        'completed_epoch': completed_epoch,
        'order_hashes': list(order_hashes),
        'model': dict(model_state),
        'optimizer': dict(optimizer_state),
        'scheduler': dict(scheduler_state),
        'scaler': dict(scaler_state),
        'rng': _capture_rng_state(),
    }


def _atomic_torch_save(document: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        torch.save(dict(document), temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(destination.parent)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_unlink(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _training_log_record(
        expected: FormalRunInit, completed_epoch: int,
        order_sha256: str, checkpoint: Path) -> dict[str, Any]:
    return {
        'schema_version': 1, 'run_id': expected.run_id,
        'role': expected.role, 'seed': expected.seed,
        'epoch': completed_epoch, 'order_sha256': order_sha256,
        'resume_checkpoint': checkpoint.name,
        'resume_sha256': _sha256_file(checkpoint),
    }


def _epoch_commit_path(output: Path, epoch: int) -> Path:
    return output / 'epoch-commits' / f'epoch_{epoch}.json'


def _epoch_commit_document(
        *, expected: FormalRunInit, completed_epoch: int,
        checkpoint: Path, structured_log: Path,
        log_record: Mapping[str, Any], previous_commit: Path | None,
        best_decision: Path | None,
        ) -> dict[str, Any]:
    previous = None
    if previous_commit is not None:
        previous = {
            'path': previous_commit.name,
            'sha256': _sha256_file(previous_commit),
        }
    return {
        'schema_version': 1,
        'identity': {
            'run_id': expected.run_id, 'role': expected.role,
            'seed': expected.seed,
            'run_init_sha256': _sha256_file(
                checkpoint.parent / 'run-init.json'),
        },
        'completed_epoch': completed_epoch,
        'checkpoint': {
            'path': checkpoint.name, 'sha256': _sha256_file(checkpoint)},
        'structured_log': {
            'path': structured_log.name,
            'sha256': _sha256_file(structured_log)},
        'log_record': dict(log_record),
        'best': (None if best_decision is None else
                 _best_epoch_authority(
                     checkpoint.parent, expected, completed_epoch,
                     best_decision)),
        'previous_commit': previous,
    }


def _load_epoch_commit(
        output: Path, expected: FormalRunInit, epoch: int
        ) -> tuple[Mapping[str, Any], Path, str]:
    path = _epoch_commit_path(output, epoch)
    if path.is_symlink() or not path.is_file():
        raise FormalTrainingError('resume epoch commit is unavailable')
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalTrainingError('resume epoch commit is malformed') from error
    fields = {
        'schema_version', 'identity', 'completed_epoch', 'checkpoint',
        'structured_log', 'log_record', 'best', 'previous_commit'}
    if not isinstance(document, Mapping) or set(document) != fields \
            or document['schema_version'] != 1 \
            or document['completed_epoch'] != epoch:
        raise FormalTrainingError('resume epoch commit fields are invalid')
    identity = document['identity']
    expected_identity = {
        'run_id': expected.run_id, 'role': expected.role,
        'seed': expected.seed,
        'run_init_sha256': _sha256_file(output / 'run-init.json')}
    if not isinstance(identity, Mapping) or set(identity) != set(
            expected_identity):
        raise FormalTrainingError('resume epoch commit identity fields invalid')
    for field, value in expected_identity.items():
        if identity[field] != value:
            raise FormalTrainingError(
                f'resume epoch commit {field.replace("_", " ")} mismatch')
    checkpoint = document['checkpoint']
    log = document['structured_log']
    if checkpoint != {
            'path': f'epoch_{epoch}.pth',
            'sha256': checkpoint.get('sha256') if isinstance(
                checkpoint, Mapping) else None} \
            or not _is_sha(checkpoint['sha256']):
        raise FormalTrainingError('resume epoch commit checkpoint is invalid')
    if log != {
            'path': 'training.jsonl',
            'sha256': log.get('sha256') if isinstance(log, Mapping) else None} \
            or not _is_sha(log['sha256']):
        raise FormalTrainingError('resume epoch commit log is invalid')
    previous = document['previous_commit']
    if epoch == 1:
        if previous is not None:
            raise FormalTrainingError('first epoch commit has a predecessor')
    else:
        predecessor = _epoch_commit_path(output, epoch - 1)
        if previous != {
                'path': predecessor.name,
                'sha256': _sha256_file(predecessor)}:
            raise FormalTrainingError('resume epoch commit chain mismatch')
    record = document['log_record']
    if not isinstance(record, Mapping) or record != {
            'schema_version': 1, 'run_id': expected.run_id,
            'role': expected.role, 'seed': expected.seed, 'epoch': epoch,
            'order_sha256': record.get('order_sha256'),
            'resume_checkpoint': f'epoch_{epoch}.pth',
            'resume_sha256': checkpoint['sha256']} \
            or not _is_sha(record['order_sha256']):
        raise FormalTrainingError('resume epoch commit log record is invalid')
    return document, path, hashlib.sha256(payload).hexdigest()


def _load_epoch_commit_chain(
        output: Path, expected: FormalRunInit) -> _EpochCommitChain:
    commits = output / 'epoch-commits'
    if not commits.exists():
        return _EpochCommitChain((), (), (), b'')
    if commits.is_symlink() or not commits.is_dir():
        raise FormalTrainingError('resume epoch commit root is unsafe')
    epochs: list[int] = []
    for child in commits.iterdir():
        match = re.fullmatch(r'epoch_([1-9][0-9]*)\.json', child.name)
        if match is None or child.is_symlink() or not child.is_file():
            raise FormalTrainingError('resume epoch commit inventory is invalid')
        epochs.append(int(match.group(1)))
    epochs.sort()
    if epochs and epochs != list(range(1, epochs[-1] + 1)):
        raise FormalTrainingError('resume epoch commit sequence is incomplete')
    documents: list[Mapping[str, Any]] = []
    paths: list[Path] = []
    sha256s: list[str] = []
    structured_log = b''
    previous_best: Mapping[str, Any] | None = None
    for epoch in epochs:
        document, path, sha256 = _load_epoch_commit(output, expected, epoch)
        structured_log += _canonical_json_bytes(document['log_record'])
        expected_log_sha = hashlib.sha256(structured_log).hexdigest()
        if document['structured_log']['sha256'] != expected_log_sha:
            raise FormalTrainingError(
                'resume epoch commit structured log chain mismatch')
        best = document['best']
        if best is not None:
            _validate_epoch_best_authority(
                output, expected, epoch, best)
            if previous_best is None or best != previous_best:
                decision_epoch = _best_authority_decision_epoch(best)
                if decision_epoch != epoch:
                    raise FormalTrainingError(
                        'new best decision does not match its epoch')
                best_document, _, _ = _load_best_decision(
                    output, expected, decision_epoch)
                expected_previous = (
                    None if previous_best is None
                    else previous_best['decision'])
                if best_document['previous_decision'] != expected_previous:
                    raise FormalTrainingError(
                        'epoch best decision chain mismatch')
            previous_best = best
        elif previous_best is not None:
            raise FormalTrainingError('epoch best authority is not continuous')
        documents.append(document)
        paths.append(path)
        sha256s.append(sha256)
    return _EpochCommitChain(
        tuple(documents), tuple(paths), tuple(sha256s), structured_log)


def _resume_checkpoint_inventory(output: Path) -> Mapping[int, Path]:
    checkpoints: dict[int, Path] = {}
    for child in output.iterdir():
        if not child.name.startswith('epoch_') or child.suffix != '.pth':
            continue
        match = re.fullmatch(r'epoch_([1-9][0-9]*)\.pth', child.name)
        if match is None or child.is_symlink() or not child.is_file():
            raise FormalTrainingError('resume checkpoint inventory is invalid')
        epoch = int(match.group(1))
        if epoch in checkpoints:
            raise FormalTrainingError('duplicate resume checkpoint epoch')
        checkpoints[epoch] = child
    return MappingProxyType(checkpoints)


def _validate_uncommitted_log_record(
        line: bytes, *, expected: FormalRunInit, epoch: int,
        checkpoint: Path) -> None:
    try:
        record = json.loads(line.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalTrainingError(
            'uncommitted structured log record is malformed') from error
    checkpoint_sha = _sha256_file(checkpoint)
    if not isinstance(record, Mapping) or record != {
            'schema_version': 1, 'run_id': expected.run_id,
            'role': expected.role, 'seed': expected.seed, 'epoch': epoch,
            'order_sha256': record.get('order_sha256'),
            'resume_checkpoint': checkpoint.name,
            'resume_sha256': checkpoint_sha} \
            or not _is_sha(record['order_sha256']):
        raise FormalTrainingError(
            'uncommitted structured log record authority mismatch')


def recover_training_lineage(
        repository_root: Path, expected: FormalRunInit, *,
        dataset_size: int = 118287) -> ResumeState | None:
    """Recover only the exact next interrupted epoch transaction.

    Epoch commit records are the durable authority.  A lone canonical next
    checkpoint (and its complete atomic log record, when present) may be
    rolled back.  Any other uncommitted or malformed inventory fails closed.
    """
    root = Path(repository_root).absolute()
    output = root / expected.output_root
    if output.is_symlink() or not output.is_dir():
        raise FormalTrainingError('formal output root is unavailable')
    chain = _load_epoch_commit_chain(output, expected)
    committed = len(chain.documents)
    checkpoints = dict(_resume_checkpoint_inventory(output))
    for epoch in range(1, committed + 1):
        checkpoint = checkpoints.get(epoch)
        if checkpoint is not None and _sha256_file(checkpoint) \
                != chain.documents[epoch - 1]['checkpoint']['sha256']:
            raise FormalTrainingError('committed resume checkpoint drift')
    if committed and committed not in checkpoints:
        raise FormalTrainingError('latest committed checkpoint is unavailable')
    uncommitted = sorted(set(checkpoints) - set(range(1, committed + 1)))
    next_epoch = committed + 1
    if uncommitted not in ([], [next_epoch]):
        raise FormalTrainingError(
            'resume inventory contains non-next uncommitted artifacts')

    log = output / 'training.jsonl'
    if log.exists() and (log.is_symlink() or not log.is_file()):
        raise FormalTrainingError('resume training log is unsafe')
    observed_log = log.read_bytes() if log.exists() else b''
    if observed_log != chain.structured_log:
        if uncommitted != [next_epoch] \
                or not observed_log.startswith(chain.structured_log):
            raise FormalTrainingError(
                'resume structured log is not recoverable')
        suffix = observed_log[len(chain.structured_log):]
        if not suffix.endswith(b'\n') or suffix.count(b'\n') != 1:
            raise FormalTrainingError(
                'resume structured log has multiple incomplete records')
        _validate_uncommitted_log_record(
            suffix, expected=expected, epoch=next_epoch,
            checkpoint=checkpoints[next_epoch])
        if chain.structured_log:
            _atomic_write_bytes(log, chain.structured_log)
        else:
            _durable_unlink(log)
    _recover_best_lineage_state(root, expected, chain)
    if uncommitted:
        # The exact canonical next transaction is never loaded or resumed.
        # Removing this already-authenticated regular child cannot affect any
        # other path and permits a deterministic replay of the same epoch.
        _durable_unlink(checkpoints[next_epoch])

    if not committed:
        return None
    for epoch, checkpoint in sorted(checkpoints.items()):
        if epoch <= committed - 2 and checkpoint.exists():
            _durable_unlink(checkpoint)
    latest = output / f'epoch_{committed}.pth'
    return validate_resume_checkpoint(
        latest, expected, repository_root=root, dataset_size=dataset_size)


def write_resume_checkpoint(
        *, repository_root: Path, expected: FormalRunInit,
        completed_epoch: int, model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any], scaler_state: Mapping[str, Any],
        order_hashes: Sequence[str], dataset_size: int = 118287,
        validate_after_write: bool = True,
        best_decision_path: Path | None = None) -> Path:
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) \
            or completed_epoch < 1 or completed_epoch > expected.epochs:
        raise FormalTrainingError('completed epoch is invalid')
    root = Path(repository_root).absolute()
    output = root / expected.output_root
    if output.is_symlink() or not output.is_dir():
        raise FormalTrainingError('formal output root is unavailable')
    run_init = output / 'run-init.json'
    if run_init.is_symlink() or not run_init.is_file():
        raise FormalTrainingError('run init is unavailable')
    if len(order_hashes) != completed_epoch \
            or any(not _is_sha(value) for value in order_hashes):
        raise FormalTrainingError('order hash prefix is invalid')
    if tuple(order_hashes) != trace_epoch_orders(
            expected, completed_epoch, dataset_size=dataset_size):
        raise FormalTrainingError('order hash prefix mismatch')
    if validate_after_write:
        prior_epoch = len(_load_epoch_commit_chain(output, expected).documents)
        if completed_epoch != prior_epoch + 1:
            raise FormalTrainingError(
                'resume checkpoint does not extend the committed lineage')
    for label, state in (
            ('model', model_state), ('optimizer', optimizer_state),
            ('scheduler', scheduler_state), ('scaler', scaler_state)):
        if not isinstance(state, Mapping):
            raise FormalTrainingError(f'{label} state is invalid')
        _validate_finite_tree(state, label)
    target = output / f'epoch_{completed_epoch}.pth'
    if target.exists():
        raise FormalTrainingError('resume checkpoint is immutable')
    document = _resume_document(
        expected=expected, run_init_sha256=_sha256_file(run_init),
        completed_epoch=completed_epoch, order_hashes=order_hashes,
        model_state=model_state, optimizer_state=optimizer_state,
        scheduler_state=scheduler_state, scaler_state=scaler_state)
    _atomic_torch_save(document, target)
    if validate_after_write:
        previous_commit = (
            None if completed_epoch == 1
            else _epoch_commit_path(output, completed_epoch - 1))
        if previous_commit is not None and not previous_commit.is_file():
            raise FormalTrainingError('previous epoch commit is unavailable')
        log = output / 'training.jsonl'
        prior = b'' if completed_epoch == 1 else log.read_bytes()
        if len(prior.splitlines()) != completed_epoch - 1:
            raise FormalTrainingError('prior structured log is incomplete')
        record = _training_log_record(
            expected, completed_epoch, order_hashes[-1], target)
        _atomic_write_bytes(log, prior + _canonical_json_bytes(record))
        commits = output / 'epoch-commits'
        if not commits.exists():
            commits.mkdir()
            _fsync_directory(output)
        commit = _epoch_commit_path(output, completed_epoch)
        if commit.exists():
            raise FormalTrainingError('resume epoch commit is immutable')
        _atomic_write_bytes(commit, _canonical_json_bytes(
            _epoch_commit_document(
                expected=expected, completed_epoch=completed_epoch,
                checkpoint=target, structured_log=log,
                log_record=record, previous_commit=previous_commit,
                best_decision=best_decision_path)))
        validate_resume_checkpoint(
            target, expected, repository_root=root,
            dataset_size=dataset_size)
        _recover_best_lineage_state(
            root, expected, _load_epoch_commit_chain(output, expected))
        committed_epochs = tuple(range(1, completed_epoch + 1))
        for stale_epoch in committed_epochs[:-2]:
            stale = output / f'epoch_{stale_epoch}.pth'
            if stale.exists():
                _durable_unlink(stale)
    return target


def _load_resume_document(
        path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    authority = FileAuthority(
        authority_root=str(path.parent), path=path.name,
        sha256=expected_sha256)
    try:
        with load_authenticated_tensor_document(path, authority) as value:
            if not isinstance(value, Mapping):
                raise FormalTrainingError(
                    'resume checkpoint must be a mapping')
            document = dict(value)
    except FormalCheckpointError as error:
        raise FormalTrainingError(
            'resume checkpoint is not a weights-only document') from error
    return document


def validate_resume_checkpoint(
        path: Path, expected: FormalRunInit, *,
        repository_root: Path | None = None,
        dataset_size: int = 118287) -> ResumeState:
    source = Path(path).absolute()
    root = (Path(repository_root).absolute() if repository_root is not None
            else source.parents[len(expected.output_root.parts) + 1])
    output = root / expected.output_root
    if source.parent != output or source.is_symlink() or not source.is_file():
        raise FormalTrainingError('resume checkpoint path is not canonical')
    run_init_path = output / 'run-init.json'
    if run_init_path.is_symlink() or not run_init_path.is_file():
        raise FormalTrainingError('resume run init is unavailable')
    match = re.fullmatch(r'epoch_([1-9][0-9]*)\.pth', source.name)
    if match is None:
        raise FormalTrainingError('resume checkpoint path is not canonical')
    requested_epoch = int(match.group(1))
    chain = _load_epoch_commit_chain(output, expected)
    if requested_epoch > len(chain.documents):
        raise FormalTrainingError('resume epoch commit is unavailable')
    commit_document = chain.documents[requested_epoch - 1]
    commit_path = chain.paths[requested_epoch - 1]
    commit_sha256 = chain.sha256s[requested_epoch - 1]
    checkpoint_sha256 = commit_document['checkpoint']['sha256']
    log_path = output / 'training.jsonl'
    if log_path.is_symlink() or not log_path.is_file() \
            or log_path.read_bytes() != chain.structured_log:
        raise FormalTrainingError('resume structured log authority mismatch')
    document = _load_resume_document(
        source, expected_sha256=checkpoint_sha256)
    fields = {
        'schema_version', 'run_init_sha256', 'identity', 'completed_epoch',
        'order_hashes', 'model', 'optimizer', 'scheduler', 'scaler', 'rng'}
    if set(document) != fields or document['schema_version'] != 1:
        raise FormalTrainingError('resume checkpoint fields are invalid')
    identity = document['identity']
    if not isinstance(identity, Mapping):
        raise FormalTrainingError('resume identity is invalid')
    expected_identity = _resume_document(
        expected=expected, run_init_sha256='0' * 64,
        completed_epoch=1, order_hashes=('0' * 64,), model_state={},
        optimizer_state={}, scheduler_state={}, scaler_state={})['identity']
    # The helper above captures RNG but the identity itself is pure.
    for field, value in expected_identity.items():
        if identity.get(field) != value:
            label = field.replace('_', ' ')
            raise FormalTrainingError(f'resume {label} mismatch')
    if set(identity) != set(expected_identity):
        raise FormalTrainingError('resume identity fields are invalid')
    supplied_init_sha = document['run_init_sha256']
    if supplied_init_sha != _sha256_file(run_init_path):
        raise FormalTrainingError('resume run init SHA-256 mismatch')
    epoch = document['completed_epoch']
    if isinstance(epoch, bool) or not isinstance(epoch, int) \
            or epoch < 1 or epoch > expected.epochs:
        raise FormalTrainingError('resume completed epoch is invalid')
    if source.name != f'epoch_{epoch}.pth':
        raise FormalTrainingError('resume checkpoint epoch path mismatch')
    order_hashes = document['order_hashes']
    if not isinstance(order_hashes, list) or len(order_hashes) != epoch \
            or any(not _is_sha(value) for value in order_hashes):
        raise FormalTrainingError('resume order hash prefix is invalid')
    observed_order = trace_epoch_orders(
        expected, epoch, dataset_size=dataset_size)
    if tuple(order_hashes) != observed_order:
        raise FormalTrainingError('resume order hash prefix mismatch')
    for label in ('model', 'optimizer', 'scheduler', 'scaler'):
        if not isinstance(document[label], Mapping):
            raise FormalTrainingError(f'resume {label} state is invalid')
        _validate_finite_tree(document[label], label)
    rng = document['rng']
    if not isinstance(rng, Mapping) or set(rng) != {
            'python', 'numpy', 'torch', 'cuda'}:
        raise FormalTrainingError('resume RNG state is incomplete')
    _validate_finite_tree(rng, 'RNG')
    return ResumeState(
        run_init_sha256=supplied_init_sha,
        checkpoint_sha256=checkpoint_sha256,
        commit_path=commit_path.relative_to(output).as_posix(),
        commit_sha256=commit_sha256,
        structured_log_sha256=hashlib.sha256(
            chain.structured_log).hexdigest(),
        completed_epoch=epoch,
        order_hashes=tuple(order_hashes),
        model_state=MappingProxyType(dict(document['model'])),
        optimizer_state=MappingProxyType(dict(document['optimizer'])),
        scheduler_state=MappingProxyType(dict(document['scheduler'])),
        scaler_state=MappingProxyType(dict(document['scaler'])),
        rng_state=MappingProxyType(dict(rng)),
    )


def _revalidate_resume_state_authority(
        root: Path, init: FormalRunInit, source: Path,
        state: ResumeState) -> None:
    output = root / init.output_root
    expected_source = output / f'epoch_{state.completed_epoch}.pth'
    if Path(source).absolute() != expected_source:
        raise FormalTrainingError('resume checkpoint is not the committed latest')
    expected_commit = (
        Path('epoch-commits') / f'epoch_{state.completed_epoch}.json')
    if state.commit_path != expected_commit.as_posix():
        raise FormalTrainingError('resume commit path authority mismatch')
    for label, path, expected_sha in (
            ('resume checkpoint', expected_source, state.checkpoint_sha256),
            ('resume epoch commit', output / expected_commit,
             state.commit_sha256),
            ('resume structured log', output / 'training.jsonl',
             state.structured_log_sha256)):
        payload, _ = _read_absolute_regular_file_nofollow(
            path, label=label)
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise FormalTrainingError(f'{label} authority drift')


_STATELESS_STAGES = frozenset(
    {'profile', 'evaluate', 'latency', 'export', 'stage_a'})


@dataclass(frozen=True)
class StageSafeBoundary:
    stage: str
    kind: str
    unit: int
    synchronized: bool


@dataclass(frozen=True)
class CooperativeStopRequest:
    request_id: str
    stage: str
    input_path: str
    input_sha256: str
    issued_at_ns: int
    acknowledgement_path: str
    partial_staging_path: str
    stage_output_root: str
    control_path: str
    control_sha256: str
    root_device: int
    root_inode: int
    control_device: int
    control_inode: int
    input_device: int
    input_inode: int


@dataclass(frozen=True)
class TrainingResumeAuthority:
    run_id: str
    completed_epoch: int
    path: str
    sha256: str


@dataclass(frozen=True)
class StatelessStageInputAuthority:
    stage: str
    path: str
    sha256: str


StopResumeAuthority = TrainingResumeAuthority | StatelessStageInputAuthority


@dataclass(frozen=True)
class StopAcknowledgement:
    request_id: str
    stage: str
    resume: StopResumeAuthority
    exit_code: Literal[75]


@dataclass(frozen=True)
class SyntheticTrainingSmokeResult:
    order_hashes: tuple[str, str]
    resume_checkpoints: tuple[Path, Path]
    final_model_sha256: str


def run_synthetic_training_smoke(
        expected: FormalRunInit, repository_root: Path
        ) -> SyntheticTrainingSmokeResult:
    """CPU-only two-epoch proof of the formal checkpoint/resume core.

    This API is deliberately separate from ``train_formal_candidate``; no
    smoke flag can weaken a production 300-epoch run.
    """
    root = Path(repository_root).absolute()
    if expected.epochs != 300:
        raise FormalTrainingError('synthetic smoke still requires a formal init')
    output = root / expected.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_formal_run_init(expected, output / 'run-init.json')
    from .formal_determinism import (
        build_seeded_sampler,
        configure_root_determinism,
        hash_sample_order,
    )
    configure_root_determinism(expected.seed)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    order_hashes: list[str] = []
    checkpoints: list[Path] = []
    dataset = tuple(range(11))
    for epoch in range(2):
        order = tuple(build_seeded_sampler(dataset, expected.seed, epoch))
        for index in order:
            value = torch.tensor([[float(index), float(index + 1)]])
            target = torch.tensor([[float(index % 3)]])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(value), target)
            loss.backward()
            if any(parameter.grad is None
                   or not torch.isfinite(parameter.grad).all().item()
                   for parameter in model.parameters()):
                raise FormalTrainingError('synthetic gradient is not finite')
            optimizer.step()
        scheduler.step()
        order_hashes.append(hash_sample_order(order))
        checkpoints.append(write_resume_checkpoint(
            repository_root=root, expected=expected,
            completed_epoch=epoch + 1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state={'0': scheduler.state_dict()}, scaler_state={},
            order_hashes=tuple(order_hashes), dataset_size=len(dataset)))
    final_digest = _tensor_digest(model.state_dict().items())
    return SyntheticTrainingSmokeResult(
        order_hashes=(order_hashes[0], order_hashes[1]),
        resume_checkpoints=(checkpoints[0], checkpoints[1]),
        final_model_sha256=final_digest)


_STOP_CONTROL_NAME = 'stop-request.json'
_STOP_ACKNOWLEDGEMENT_NAME = 'stop-ack.json'
_STOP_STAGING_NAME = 'partial-staging'
_STOP_MAX_AGE_NS = 300 * 1_000_000_000
_STOP_MAX_FUTURE_NS = 5 * 1_000_000_000


def _canonical_absolute_path(value: Path | str, *, label: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.startswith('/') \
            or raw == '/' or raw.endswith('/') or '//' in raw:
        raise FormalTrainingError(f'{label} is not a canonical absolute path')
    components = raw.split('/')[1:]
    if not components or any(part in {'', '.', '..'} for part in components):
        raise FormalTrainingError(f'{label} is not a canonical absolute path')
    path = Path(raw)
    if str(path) != raw:
        raise FormalTrainingError(f'{label} is not a canonical absolute path')
    return path


def _open_absolute_directory_nofollow(
        path: Path, *, label: str = 'stage output root'
        ) -> tuple[int, os.stat_result]:
    canonical = _canonical_absolute_path(path, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open('/', flags)
    try:
        for component in canonical.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise FormalTrainingError('stage output root is not a directory')
        return descriptor, observed
    except OSError as error:
        os.close(descriptor)
        raise FormalTrainingError(f'{label} is unavailable or unsafe') from error
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_file_at(
        directory_fd: int, name: str, *, label: str
        ) -> tuple[bytes, os.stat_result]:
    if not isinstance(name, str) or not name or '/' in name \
            or name in {'.', '..'}:
        raise FormalTrainingError(f'{label} name is invalid')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise FormalTrainingError(f'{label} is unavailable or unsafe') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FormalTrainingError(f'{label} is not a regular file')
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
                after.st_dev, after.st_ino, after.st_size):
            raise FormalTrainingError(f'{label} changed while reading')
        payload = b''.join(chunks)
        if len(payload) != before.st_size:
            raise FormalTrainingError(f'{label} size changed while reading')
        return payload, before
    finally:
        os.close(descriptor)


def _read_absolute_regular_file_nofollow(
        path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    canonical = _canonical_absolute_path(path, label=label)
    parent_fd, _parent_stat = _open_absolute_directory_nofollow(
        canonical.parent, label=f'{label} parent')
    try:
        return _read_regular_file_at(
            parent_fd, canonical.name, label=label)
    finally:
        os.close(parent_fd)


def _stat_child_nofollow(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _purge_directory_fd(directory_fd: int) -> None:
    with os.scandir(directory_fd) as entries:
        names = tuple(entry.name for entry in entries)
    for name in names:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            child_fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (
                        observed.st_dev, observed.st_ino):
                    raise FormalTrainingError(
                        'partial staging changed during cleanup')
                _purge_directory_fd(child_fd)
            finally:
                os.close(child_fd)
            current = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (
                    observed.st_dev, observed.st_ino):
                raise FormalTrainingError(
                    'partial staging changed during cleanup')
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _write_atomic_file_at(
        directory_fd: int, destination_name: str, payload: bytes,
        *, temporary_name: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(
            temporary_name, destination_name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
        raise


def poll_cooperative_stop(
        control_path: Path, *, safe_boundary: StageSafeBoundary,
        stage_output_root: Path | None = None
        ) -> CooperativeStopRequest | None:
    if not isinstance(safe_boundary, StageSafeBoundary) \
            or isinstance(safe_boundary.unit, bool) \
            or not isinstance(safe_boundary.unit, int) \
            or safe_boundary.unit < 0:
        raise FormalTrainingError('stage safe boundary is invalid')
    expected_kind = ('optimizer' if safe_boundary.stage == 'training'
                     else 'cuda' if safe_boundary.stage in _STATELESS_STAGES
                     else None)
    if safe_boundary.kind != expected_kind:
        raise FormalTrainingError('stage safe boundary kind is invalid')
    if stage_output_root is None:
        raise FormalTrainingError('stage output root authority is required')
    root = _canonical_absolute_path(
        stage_output_root, label='stage output root')
    source = _canonical_absolute_path(control_path, label='stop request path')
    if source != root / _STOP_CONTROL_NAME:
        raise FormalTrainingError('stop request path is not canonical')
    root_fd, root_stat = _open_absolute_directory_nofollow(root)
    try:
        try:
            payload, control_stat = _read_regular_file_at(
                root_fd, _STOP_CONTROL_NAME, label='stop request')
        except FormalTrainingError as error:
            if _stat_child_nofollow(root_fd, _STOP_CONTROL_NAME) is None:
                return None
            raise error
        try:
            document = json.loads(payload.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalTrainingError('stop request is malformed') from error
    finally:
        os.close(root_fd)
    fields = {
        'schema_version', 'request_id', 'stage', 'input_path',
        'input_sha256', 'issued_at_ns', 'acknowledgement_path',
        'partial_staging_path'}
    if not isinstance(document, Mapping) or set(document) != fields \
            or document['schema_version'] != 1:
        raise FormalTrainingError('stop request fields are invalid')
    if document['stage'] != safe_boundary.stage:
        raise FormalTrainingError('stop request stage differs from worker')
    if not isinstance(document['request_id'], str) \
            or not document['request_id']:
        raise FormalTrainingError('stop request identity is invalid')
    if isinstance(document['issued_at_ns'], bool) \
            or not isinstance(document['issued_at_ns'], int) \
            or document['issued_at_ns'] < 0:
        raise FormalTrainingError('stop request timestamp is invalid')
    now = time.time_ns()
    if document['issued_at_ns'] < now - _STOP_MAX_AGE_NS \
            or document['issued_at_ns'] > now + _STOP_MAX_FUTURE_NS:
        raise FormalTrainingError('stop request is stale')
    if not _is_sha(document['input_sha256']):
        raise FormalTrainingError('stop request input SHA-256 is invalid')
    for field in ('input_path', 'acknowledgement_path', 'partial_staging_path'):
        if not isinstance(document[field], str):
            raise FormalTrainingError(f'stop request {field} is invalid')
        _canonical_absolute_path(
            document[field], label=f'stop request {field}')
    if document['acknowledgement_path'] != str(
            root / _STOP_ACKNOWLEDGEMENT_NAME) \
            or document['partial_staging_path'] != str(
                root / _STOP_STAGING_NAME):
        raise FormalTrainingError('stop request output paths are not canonical')
    if len(document['request_id']) > 128 or any(
            not (character.isalnum() or character in {'-', '_'})
            for character in document['request_id']):
        raise FormalTrainingError('stop request identity is invalid')
    input_bytes, input_stat = _read_absolute_regular_file_nofollow(
        Path(document['input_path']), label='stop request input')
    if hashlib.sha256(input_bytes).hexdigest() != document['input_sha256']:
        raise FormalTrainingError('stop request input authority mismatch')
    acknowledgement = Path(document['acknowledgement_path'])
    if acknowledgement.exists():
        raise FormalTrainingError('stop request is stale or already acknowledged')
    if safe_boundary.synchronized is not True:
        return None
    return CooperativeStopRequest(
        request_id=document['request_id'], stage=document['stage'],
        input_path=document['input_path'],
        input_sha256=document['input_sha256'],
        issued_at_ns=document['issued_at_ns'],
        acknowledgement_path=document['acknowledgement_path'],
        partial_staging_path=document['partial_staging_path'],
        stage_output_root=str(root), control_path=str(source),
        control_sha256=hashlib.sha256(payload).hexdigest(),
        root_device=root_stat.st_dev, root_inode=root_stat.st_ino,
        control_device=control_stat.st_dev,
        control_inode=control_stat.st_ino,
        input_device=input_stat.st_dev, input_inode=input_stat.st_ino)


def write_stop_acknowledgement(
        request: CooperativeStopRequest,
        resume: StopResumeAuthority) -> StopAcknowledgement:
    if not isinstance(request, CooperativeStopRequest):
        raise FormalTrainingError('stop request type is invalid')
    root = _canonical_absolute_path(
        request.stage_output_root, label='stage output root')
    if request.control_path != str(root / _STOP_CONTROL_NAME) \
            or request.acknowledgement_path != str(
                root / _STOP_ACKNOWLEDGEMENT_NAME) \
            or request.partial_staging_path != str(root / _STOP_STAGING_NAME):
        raise FormalTrainingError('stop request path authority is invalid')
    root_fd, root_stat = _open_absolute_directory_nofollow(root)
    if (root_stat.st_dev, root_stat.st_ino) != (
            request.root_device, request.root_inode):
        os.close(root_fd)
        raise FormalTrainingError('stage output root authority changed')
    try:
        control_bytes, control_stat = _read_regular_file_at(
            root_fd, _STOP_CONTROL_NAME, label='stop request control')
        if (control_stat.st_dev, control_stat.st_ino) != (
                request.control_device, request.control_inode) \
                or hashlib.sha256(control_bytes).hexdigest() \
                != request.control_sha256:
            raise FormalTrainingError('stop request control authority changed')
        input_bytes, input_stat = _read_absolute_regular_file_nofollow(
            Path(request.input_path), label='stop request input')
        if (input_stat.st_dev, input_stat.st_ino) != (
                request.input_device, request.input_inode) \
                or hashlib.sha256(input_bytes).hexdigest() \
                != request.input_sha256:
            raise FormalTrainingError('stop request input authority changed')
        if isinstance(resume, TrainingResumeAuthority):
            if request.stage != 'training':
                raise FormalTrainingError('training resume stage mismatch')
            if isinstance(resume.completed_epoch, bool) \
                    or not isinstance(resume.completed_epoch, int) \
                    or resume.completed_epoch < 0:
                raise FormalTrainingError('training resume epoch is invalid')
            if resume.path != request.input_path \
                    or resume.sha256 != request.input_sha256:
                raise FormalTrainingError('training resume input mismatch')
        elif isinstance(resume, StatelessStageInputAuthority):
            if request.stage != resume.stage \
                    or resume.stage not in _STATELESS_STAGES:
                raise FormalTrainingError('stateless resume stage mismatch')
            if resume.path != request.input_path \
                    or resume.sha256 != request.input_sha256:
                raise FormalTrainingError('stateless resume input mismatch')
        else:
            raise FormalTrainingError('stop resume authority type is invalid')
        if not _is_sha(resume.sha256) \
                or resume.sha256 != hashlib.sha256(input_bytes).hexdigest():
            raise FormalTrainingError('stop resume input authority changed')
        acknowledgement = StopAcknowledgement(
            request_id=request.request_id, stage=request.stage,
            resume=resume, exit_code=75)
        if _stat_child_nofollow(
                root_fd, _STOP_ACKNOWLEDGEMENT_NAME) is not None:
            raise FormalTrainingError('stop acknowledgement already exists')
        staging_stat = _stat_child_nofollow(root_fd, _STOP_STAGING_NAME)
        if staging_stat is not None:
            if not stat.S_ISDIR(staging_stat.st_mode):
                raise FormalTrainingError('partial staging authority is unsafe')
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            staging_fd = os.open(_STOP_STAGING_NAME, flags, dir_fd=root_fd)
            try:
                opened = os.fstat(staging_fd)
                if (opened.st_dev, opened.st_ino) != (
                        staging_stat.st_dev, staging_stat.st_ino):
                    raise FormalTrainingError(
                        'partial staging authority changed')
                abandoned_name = (
                    f'.{_STOP_STAGING_NAME}.abandoned-{request.request_id}')
                if _stat_child_nofollow(root_fd, abandoned_name) is not None:
                    raise FormalTrainingError(
                        'partial staging abandonment collides')
                os.rename(
                    _STOP_STAGING_NAME, abandoned_name,
                    src_dir_fd=root_fd, dst_dir_fd=root_fd)
                os.fsync(root_fd)
                renamed = os.stat(
                    abandoned_name, dir_fd=root_fd,
                    follow_symlinks=False)
                if (renamed.st_dev, renamed.st_ino) != (
                        opened.st_dev, opened.st_ino):
                    raise FormalTrainingError(
                        'partial staging authority changed')
                _purge_directory_fd(staging_fd)
            finally:
                os.close(staging_fd)
            renamed = os.stat(
                abandoned_name, dir_fd=root_fd, follow_symlinks=False)
            if (renamed.st_dev, renamed.st_ino) != (
                    staging_stat.st_dev, staging_stat.st_ino):
                raise FormalTrainingError('partial staging authority changed')
            os.rmdir(abandoned_name, dir_fd=root_fd)
            os.fsync(root_fd)
        resume_document = {
            'kind': ('training' if isinstance(resume, TrainingResumeAuthority)
                     else 'stateless'),
            **resume.__dict__,
        }
        payload = _canonical_json_bytes({
            'schema_version': 1,
            'request_id': request.request_id,
            'stage': request.stage,
            'resume': resume_document,
            'exit_code': 75,
        })
        _write_atomic_file_at(
            root_fd, _STOP_ACKNOWLEDGEMENT_NAME, payload,
            temporary_name=(
                f'.{_STOP_ACKNOWLEDGEMENT_NAME}.tmp-{request.request_id}'))
        return acknowledgement
    finally:
        os.close(root_fd)


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    def tuples(item: object) -> object:
        if isinstance(item, list):
            return tuple(tuples(child) for child in item)
        return item

    numpy_state = value['numpy']
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            'algorithm', 'keys', 'position', 'has_gauss',
            'cached_gaussian'}:
        raise FormalTrainingError('resume NumPy RNG state is invalid')
    random.setstate(tuples(value['python']))
    keys = numpy_state['keys']
    if not isinstance(keys, torch.Tensor):
        raise FormalTrainingError('resume NumPy RNG keys are invalid')
    np.random.set_state((
        numpy_state['algorithm'], keys.cpu().numpy().astype(np.uint32),
        numpy_state['position'], numpy_state['has_gauss'],
        numpy_state['cached_gaussian']))
    if not isinstance(value['torch'], torch.Tensor):
        raise FormalTrainingError('resume Torch RNG state is invalid')
    torch.set_rng_state(value['torch'].cpu())
    cuda = value['cuda']
    if not isinstance(cuda, list) or any(
            not isinstance(item, torch.Tensor) for item in cuda):
        raise FormalTrainingError('resume CUDA RNG state is invalid')
    if cuda:
        if not torch.cuda.is_initialized():
            raise FormalTrainingError(
                'resume CUDA RNG cannot precede device initialization')
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda])


def _atomic_save_tensor_mapping(
        tensors: Mapping[str, torch.Tensor], destination: Path) -> None:
    if not tensors or any(not isinstance(key, str)
                          or not isinstance(value, torch.Tensor)
                          for key, value in tensors.items()):
        raise FormalTrainingError('pose checkpoint tensor mapping is invalid')
    _validate_finite_tree(tensors, 'pose checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save({
            key: value.detach().cpu() for key, value in tensors.items()},
            temporary)
        loaded = torch.load(temporary, map_location='cpu', weights_only=True)
        if set(loaded) != set(tensors) \
                or any(not isinstance(value, torch.Tensor)
                       for value in loaded.values()):
            raise FormalTrainingError('pose checkpoint roundtrip failed')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(destination.parent)


def _file_binding(root: Path, path: Path) -> FileBinding:
    return FileBinding(path=path.relative_to(root), sha256=_sha256_file(path))


def _train_result_document(result: FormalTrainResult) -> dict[str, Any]:
    asset = result.initialization.asset
    return {
        'schema_version': 1,
        'run_init_sha256': result.run_init_sha256,
        'initialization': {
            'id': result.initialization.id,
            'kind': result.initialization.kind,
            'asset': {
                'authority_root': asset.authority_root,
                'target_root': asset.target_root,
                'asset_relative_path': asset.asset_relative_path.as_posix(),
                'sha256': asset.sha256,
            },
        },
        'run': {
            'run_id': result.run_id,
            'role': result.role,
            'seed': result.seed,
            'output_root': result.output_root.as_posix(),
        },
        'best_checkpoint': {
            'path': result.best_checkpoint.path.as_posix(),
            'sha256': result.best_checkpoint.sha256,
        },
        'resume_checkpoints': [
            {'path': item.path.as_posix(), 'sha256': item.sha256}
            for item in result.resume_checkpoints],
        'structured_log': {
            'path': result.structured_log.path.as_posix(),
            'sha256': result.structured_log.sha256,
        },
        'order_hashes': [
            {'epoch': epoch, 'sha256': sha256}
            for epoch, sha256 in enumerate(result.order_hashes, start=1)],
        'final_epoch': 300,
        'status': 'complete',
    }


def _best_decision_path(output: Path, epoch: int) -> Path:
    return output / 'best-lineages' / f'epoch_{epoch}.json'


def _best_pointer_bytes(output: Path, decision: Path) -> bytes:
    payload, _ = _read_absolute_regular_file_nofollow(
        decision, label='best decision')
    return _canonical_json_bytes({
        'schema_version': 1,
        'decision': {
            'path': decision.relative_to(output).as_posix(),
            'sha256': hashlib.sha256(payload).hexdigest(),
        },
    })


def _load_best_decision(
        output: Path, init: FormalRunInit, epoch: int
        ) -> tuple[Mapping[str, Any], Path, str]:
    path = _best_decision_path(output, epoch)
    try:
        payload, authority = _read_absolute_regular_file_nofollow(
            path, label='best decision')
        document = json.loads(payload.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalTrainingError('best decision is malformed') from error
    fields = {
        'schema_version', 'identity', 'completed_epoch', 'metric',
        'checkpoint', 'previous_decision'}
    if not isinstance(document, Mapping) or set(document) != fields \
            or document['schema_version'] != 1 \
            or document['identity'] != {
                'run_id': init.run_id, 'role': init.role, 'seed': init.seed} \
            or document['completed_epoch'] != epoch:
        raise FormalTrainingError('best decision identity is invalid')
    metric = document['metric']
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) \
            or not math.isfinite(metric):
        raise FormalTrainingError('best decision metric is invalid')
    checkpoint = document['checkpoint']
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            'path', 'sha256'} or not isinstance(checkpoint['path'], str) \
            or re.fullmatch(
                r'best_coco_AP_epoch_[1-9][0-9]*\.pth',
                checkpoint['path']) is None or not _is_sha(
                    checkpoint['sha256']):
        raise FormalTrainingError('best decision checkpoint is invalid')
    previous = document['previous_decision']
    if previous is not None:
        if not isinstance(previous, Mapping) or set(previous) != {
                'path', 'sha256'} or not isinstance(previous['path'], str) \
                or not _is_sha(previous['sha256']):
            raise FormalTrainingError('best decision predecessor is invalid')
        match = re.fullmatch(
            r'best-lineages/epoch_([1-9][0-9]*)\.json', previous['path'])
        if match is None or int(match.group(1)) >= epoch:
            raise FormalTrainingError('best decision predecessor is invalid')
        predecessor = output / previous['path']
        predecessor_payload, _ = _read_absolute_regular_file_nofollow(
            predecessor, label='best decision predecessor')
        if hashlib.sha256(predecessor_payload).hexdigest() \
                != previous['sha256']:
            raise FormalTrainingError('best decision chain mismatch')
    observed, observed_stat = _read_absolute_regular_file_nofollow(
        path, label='best decision')
    if observed != payload or (observed_stat.st_dev, observed_stat.st_ino) != (
            authority.st_dev, authority.st_ino):
        raise FormalTrainingError('best decision authority changed')
    return document, path, hashlib.sha256(payload).hexdigest()


def _parse_best_pointer(
        payload: bytes, output: Path, *, completed_epoch: int
        ) -> tuple[Mapping[str, Any], Path, int]:
    try:
        document = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalTrainingError(
            'best checkpoint pointer is malformed') from error
    if not isinstance(document, Mapping) or set(document) != {
            'schema_version', 'decision'} or document['schema_version'] != 1 \
            or not isinstance(document['decision'], Mapping) \
            or set(document['decision']) != {'path', 'sha256'} \
            or not isinstance(document['decision']['path'], str) \
            or not _is_sha(document['decision']['sha256']):
        raise FormalTrainingError('best checkpoint pointer fields are invalid')
    match = re.fullmatch(
        r'best-lineages/epoch_([1-9][0-9]*)\.json',
        document['decision']['path'])
    if match is None or int(match.group(1)) > completed_epoch:
        raise FormalTrainingError('best checkpoint pointer authority mismatch')
    decision = output / document['decision']['path']
    return document, decision, int(match.group(1))


def _read_optional_output_child(
        output: Path, name: str, *, label: str
        ) -> tuple[bytes, os.stat_result] | None:
    output_fd, _ = _open_absolute_directory_nofollow(
        output, label='formal output root')
    try:
        if _stat_child_nofollow(output_fd, name) is None:
            return None
        return _read_regular_file_at(output_fd, name, label=label)
    finally:
        os.close(output_fd)


def _replace_output_child(
        output: Path, name: str, payload: bytes | None) -> None:
    output_fd, output_stat = _open_absolute_directory_nofollow(
        output, label='formal output root')
    try:
        if payload is None:
            observed = _stat_child_nofollow(output_fd, name)
            if observed is not None:
                if not stat.S_ISREG(observed.st_mode):
                    raise FormalTrainingError(
                        'best checkpoint pointer is unsafe')
                os.unlink(name, dir_fd=output_fd)
                os.fsync(output_fd)
        else:
            _write_atomic_file_at(
                output_fd, name, payload,
                temporary_name=(
                    f'.{name}.tmp-{os.getpid()}-{time.time_ns()}'))
    finally:
        os.close(output_fd)
    reopened, reopened_stat = _open_absolute_directory_nofollow(
        output, label='formal output root')
    os.close(reopened)
    if (reopened_stat.st_dev, reopened_stat.st_ino) != (
            output_stat.st_dev, output_stat.st_ino):
        raise FormalTrainingError('formal output root authority changed')


def _best_epoch_authority(
        output: Path, init: FormalRunInit, epoch: int,
        decision: Path) -> dict[str, Any]:
    supplied = Path(decision).absolute()
    try:
        relative = supplied.relative_to(output.absolute()).as_posix()
    except ValueError as error:
        raise FormalTrainingError(
            'best decision path is not canonical') from error
    match = re.fullmatch(
        r'best-lineages/epoch_([1-9][0-9]*)\.json', relative)
    if match is None or int(match.group(1)) > epoch:
        raise FormalTrainingError('best decision path is not canonical')
    decision_epoch = int(match.group(1))
    expected_path = _best_decision_path(output, decision_epoch)
    if supplied != expected_path.absolute():
        raise FormalTrainingError('best decision path is not canonical')
    document, path, decision_sha = _load_best_decision(
        output, init, decision_epoch)
    return {
        'decision': {
            'path': path.relative_to(output).as_posix(),
            'sha256': decision_sha,
        },
        'checkpoint': dict(document['checkpoint']),
    }


def _validate_epoch_best_authority(
        output: Path, init: FormalRunInit, epoch: int,
        authority: object) -> None:
    if not isinstance(authority, Mapping) or set(authority) != {
            'decision', 'checkpoint'}:
        raise FormalTrainingError('epoch best authority fields are invalid')
    decision = authority['decision']
    if not isinstance(decision, Mapping) or set(decision) != {
            'path', 'sha256'} or not isinstance(decision['path'], str):
        raise FormalTrainingError('epoch best decision authority is invalid')
    expected = _best_epoch_authority(
        output, init, epoch, output / decision['path'])
    if authority != expected:
        raise FormalTrainingError('epoch best authority mismatch')


def _best_authority_decision_epoch(authority: Mapping[str, Any]) -> int:
    try:
        path = authority['decision']['path']
    except (KeyError, TypeError) as error:
        raise FormalTrainingError(
            'epoch best decision authority is invalid') from error
    match = re.fullmatch(
        r'best-lineages/epoch_([1-9][0-9]*)\.json', path)
    if match is None:
        raise FormalTrainingError('epoch best decision path is invalid')
    return int(match.group(1))


def _write_best_decision(
        root: Path, init: FormalRunInit, *, completed_epoch: int,
        metric: float, checkpoint: Path) -> Path:
    output = root / init.output_root
    decisions = output / 'best-lineages'
    destination = _best_decision_path(output, completed_epoch)
    previous = None
    pointer = output / 'best-lineage.json'
    output_fd, output_stat = _open_absolute_directory_nofollow(
        output, label='formal output root')
    pointer_payload = None
    pointer_stat = None
    try:
        observed_pointer = _stat_child_nofollow(
            output_fd, 'best-lineage.json')
        if observed_pointer is not None:
            pointer_payload, pointer_stat = _read_regular_file_at(
                output_fd, 'best-lineage.json',
                label='best checkpoint pointer')
            pointer_document, predecessor, predecessor_epoch = \
                _parse_best_pointer(
                    pointer_payload, output,
                    completed_epoch=completed_epoch - 1)
            _document, _path, predecessor_sha = _load_best_decision(
                output, init, predecessor_epoch)
            if pointer_document['decision']['sha256'] != predecessor_sha:
                raise FormalTrainingError(
                    'best checkpoint pointer authority mismatch')
            previous = dict(pointer_document['decision'])
        observed_decisions = _stat_child_nofollow(
            output_fd, 'best-lineages')
        if observed_decisions is None:
            os.mkdir('best-lineages', 0o700, dir_fd=output_fd)
            os.fsync(output_fd)
        elif not stat.S_ISDIR(observed_decisions.st_mode):
            raise FormalTrainingError('best decision root is unsafe')
        decision_fd = os.open(
            'best-lineages', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | os.O_CLOEXEC, dir_fd=output_fd)
        try:
            opened_decisions = os.fstat(decision_fd)
            current_decisions = os.stat(
                'best-lineages', dir_fd=output_fd, follow_symlinks=False)
            if (opened_decisions.st_dev, opened_decisions.st_ino) != (
                    current_decisions.st_dev, current_decisions.st_ino):
                raise FormalTrainingError('best decision root changed')
            if _stat_child_nofollow(decision_fd, destination.name) is not None:
                raise FormalTrainingError('best decision is immutable')
        finally:
            os.close(decision_fd)
    except Exception:
        os.close(output_fd)
        raise
    payload = _canonical_json_bytes({
        'schema_version': 1,
        'identity': {
            'run_id': init.run_id, 'role': init.role, 'seed': init.seed},
        'completed_epoch': completed_epoch,
        'metric': metric,
        'checkpoint': {
            'path': checkpoint.name, 'sha256': _sha256_file(checkpoint)},
        'previous_decision': previous,
    })
    try:
        decision_fd = os.open(
            'best-lineages', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | os.O_CLOEXEC, dir_fd=output_fd)
        try:
            _write_atomic_file_at(
                decision_fd, destination.name, payload,
                temporary_name=(
                    f'.{destination.name}.tmp-{os.getpid()}-'
                    f'{time.time_ns()}'))
        finally:
            os.close(decision_fd)
        current_pointer = _stat_child_nofollow(
            output_fd, 'best-lineage.json')
        if pointer_stat is None:
            if current_pointer is not None:
                raise FormalTrainingError(
                    'best checkpoint pointer authority changed')
        else:
            current_payload, current_stat = _read_regular_file_at(
                output_fd, 'best-lineage.json',
                label='best checkpoint pointer')
            if current_payload != pointer_payload or (
                    current_stat.st_dev, current_stat.st_ino) != (
                        pointer_stat.st_dev, pointer_stat.st_ino):
                raise FormalTrainingError(
                    'best checkpoint pointer authority changed')
        _write_atomic_file_at(
            output_fd, 'best-lineage.json',
            _best_pointer_bytes(output, destination),
            temporary_name=(
                f'.best-lineage.json.tmp-{os.getpid()}-{time.time_ns()}'))
        reopened_fd, reopened_stat = _open_absolute_directory_nofollow(
            output, label='formal output root')
        os.close(reopened_fd)
        if (reopened_stat.st_dev, reopened_stat.st_ino) != (
                output_stat.st_dev, output_stat.st_ino):
            raise FormalTrainingError('formal output root authority changed')
    finally:
        os.close(output_fd)
    return destination


def publish_best_checkpoint(
        root: Path, init: FormalRunInit, *, completed_epoch: int,
        metric: float, tensors: Mapping[str, torch.Tensor]) -> Path:
    output = Path(root).absolute() / init.output_root
    prior_metric, _ = _load_best_lineage(
        Path(root).absolute(), init, completed_epoch=completed_epoch - 1)
    if not math.isfinite(metric) or metric <= prior_metric:
        raise FormalTrainingError('new best metric is not an improvement')
    checkpoint = output / f'best_coco_AP_epoch_{completed_epoch}.pth'
    if checkpoint.exists():
        raise FormalTrainingError('versioned best checkpoint is immutable')
    _atomic_save_tensor_mapping(tensors, checkpoint)
    return _write_best_decision(
        Path(root).absolute(), init, completed_epoch=completed_epoch,
        metric=metric, checkpoint=checkpoint)


def inherit_best_checkpoint(
        root: Path, init: FormalRunInit, *, completed_epoch: int) -> Path:
    _metric, checkpoint = _load_best_lineage(
        Path(root).absolute(), init, completed_epoch=completed_epoch - 1)
    if checkpoint is None:
        raise FormalTrainingError('best checkpoint cannot be inherited')
    output = Path(root).absolute() / init.output_root
    captured = _read_optional_output_child(
        output, 'best-lineage.json', label='best checkpoint pointer')
    if captured is None:
        raise FormalTrainingError('best checkpoint lineage is unavailable')
    pointer_payload, pointer_stat = captured
    _document, decision, _epoch = _parse_best_pointer(
        pointer_payload, output, completed_epoch=completed_epoch - 1)
    observed = _read_optional_output_child(
        output, 'best-lineage.json', label='best checkpoint pointer')
    if observed is None or observed[0] != pointer_payload or (
            observed[1].st_dev, observed[1].st_ino) != (
                pointer_stat.st_dev, pointer_stat.st_ino):
        raise FormalTrainingError('best checkpoint pointer authority changed')
    return decision


def _load_best_lineage(
        root: Path, init: FormalRunInit, *, completed_epoch: int
        ) -> tuple[float, Path | None]:
    output = root / init.output_root
    pointer = output / 'best-lineage.json'
    captured_pointer = _read_optional_output_child(
        output, 'best-lineage.json', label='best checkpoint pointer')
    if captured_pointer is None:
        return -math.inf, None
    pointer_payload, pointer_stat = captured_pointer
    document, expected_decision, decision_epoch = _parse_best_pointer(
        pointer_payload, output, completed_epoch=completed_epoch)
    decision, _, decision_sha = _load_best_decision(
        output, init, decision_epoch)
    if document['decision']['sha256'] != decision_sha:
        raise FormalTrainingError('best checkpoint lineage authority drift')
    checkpoint = output / decision['checkpoint']['path']
    if checkpoint.is_symlink() or not checkpoint.is_file() \
            or _sha256_file(checkpoint) != decision['checkpoint']['sha256']:
        raise FormalTrainingError('best checkpoint authority drift')
    observed_pointer = _read_optional_output_child(
        output, 'best-lineage.json', label='best checkpoint pointer')
    if observed_pointer is None or observed_pointer[0] != pointer_payload or (
            observed_pointer[1].st_dev, observed_pointer[1].st_ino) != (
                pointer_stat.st_dev, pointer_stat.st_ino):
        raise FormalTrainingError('best checkpoint pointer authority changed')
    return float(decision['metric']), checkpoint


def _recover_best_lineage_state(
        root: Path, init: FormalRunInit,
        chain: _EpochCommitChain) -> tuple[float, Path | None]:
    output = root / init.output_root
    committed = len(chain.documents)
    committed_best = [document['best'] for document in chain.documents]
    has_best = [item is not None for item in committed_best]
    if any(has_best):
        first_best = has_best.index(True)
        if not all(has_best[first_best:]):
            raise FormalTrainingError('epoch best authority is not continuous')

    decision_root = output / 'best-lineages'
    decision_epochs: list[int] = []
    if decision_root.exists():
        if decision_root.is_symlink() or not decision_root.is_dir():
            raise FormalTrainingError('best decision root is unsafe')
        for child in decision_root.iterdir():
            match = re.fullmatch(r'epoch_([1-9][0-9]*)\.json', child.name)
            if match is None or child.is_symlink() or not child.is_file():
                raise FormalTrainingError('best decision inventory is invalid')
            decision_epochs.append(int(match.group(1)))
    decision_epochs.sort()
    expected_committed = []
    for authority in committed_best:
        if authority is None:
            continue
        epoch = _best_authority_decision_epoch(authority)
        if not expected_committed or expected_committed[-1] != epoch:
            expected_committed.append(epoch)
    if decision_epochs[:len(expected_committed)] != expected_committed:
        raise FormalTrainingError('committed best decision sequence is incomplete')
    extras = decision_epochs[len(expected_committed):]
    if extras not in ([], [committed + 1]):
        raise FormalTrainingError(
            'best inventory contains non-next uncommitted decisions')

    decisions: dict[int, Mapping[str, Any]] = {}
    for epoch in decision_epochs:
        document, _, _ = _load_best_decision(output, init, epoch)
        decisions[epoch] = document

    tensor_paths: dict[str, Path] = {}
    for child in output.iterdir():
        if not child.name.startswith('best_coco_AP_epoch_'):
            continue
        if re.fullmatch(
                r'best_coco_AP_epoch_[1-9][0-9]*\.pth', child.name) is None \
                or child.is_symlink() or not child.is_file():
            raise FormalTrainingError('best checkpoint inventory is invalid')
        tensor_paths[child.name] = child
    referenced_names = {
        document['checkpoint']['path'] for document in decisions.values()}
    unknown = set(tensor_paths) - referenced_names
    allowed_orphan = f'best_coco_AP_epoch_{committed + 1}.pth'
    if unknown not in (set(), {allowed_orphan}):
        raise FormalTrainingError(
            'best inventory contains non-next uncommitted checkpoints')

    pointer = output / 'best-lineage.json'
    latest_checkpoint: Path | None = None
    if committed_best and committed_best[-1] is not None:
        latest = committed_best[-1]
        latest_decision = output / latest['decision']['path']
        expected_pointer = _best_pointer_bytes(output, latest_decision)
        captured_pointer = _read_optional_output_child(
            output, 'best-lineage.json', label='best checkpoint pointer')
        if captured_pointer is None or captured_pointer[0] != expected_pointer:
            _replace_output_child(
                output, 'best-lineage.json', expected_pointer)
        latest_checkpoint = output / latest['checkpoint']['path']
        if latest_checkpoint.is_symlink() or not latest_checkpoint.is_file() \
                or _sha256_file(latest_checkpoint) \
                != latest['checkpoint']['sha256']:
            raise FormalTrainingError('latest committed best checkpoint drift')
    elif _read_optional_output_child(
            output, 'best-lineage.json',
            label='best checkpoint pointer') is not None:
        _replace_output_child(output, 'best-lineage.json', None)

    if extras:
        extra_document = decisions[extras[0]]
        extra_checkpoint = output / extra_document['checkpoint']['path']
        _durable_unlink(_best_decision_path(output, extras[0]))
        if latest_checkpoint is None or extra_checkpoint != latest_checkpoint:
            if extra_checkpoint.exists():
                _durable_unlink(extra_checkpoint)
    orphan = output / allowed_orphan
    if orphan.exists():
        _durable_unlink(orphan)

    for name, checkpoint in tensor_paths.items():
        if checkpoint.exists() and checkpoint != latest_checkpoint:
            _durable_unlink(checkpoint)
    if latest_checkpoint is None:
        return -math.inf, None
    return _load_best_lineage(root, init, completed_epoch=committed)


def _validate_training_log(
        path: Path, init: FormalRunInit, order_hashes: Sequence[str]) -> None:
    if path.is_symlink() or not path.is_file():
        raise FormalTrainingError('resume training log is unavailable')
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise FormalTrainingError('resume training log is malformed') from error
    if len(records) != len(order_hashes):
        raise FormalTrainingError('resume training log length mismatch')
    for epoch, (record, order_hash) in enumerate(
            zip(records, order_hashes), start=1):
        if not isinstance(record, Mapping) or set(record) != {
                'schema_version', 'run_id', 'role', 'seed', 'epoch',
                'order_sha256', 'resume_checkpoint', 'resume_sha256'}:
            raise FormalTrainingError('resume training log fields are invalid')
        checkpoint = path.parent / f'epoch_{epoch}.pth'
        # Older retained checkpoints may have been pruned, but their log hash
        # remains immutable evidence.  The latest two must still revalidate.
        expected_checkpoint = f'epoch_{epoch}.pth'
        if record != {
                'schema_version': 1, 'run_id': init.run_id,
                'role': init.role, 'seed': init.seed, 'epoch': epoch,
                'order_sha256': order_hash,
                'resume_checkpoint': expected_checkpoint,
                'resume_sha256': record['resume_sha256']} \
                or not _is_sha(record['resume_sha256']):
            raise FormalTrainingError('resume training log identity mismatch')
        if checkpoint.exists() and _sha256_file(checkpoint) \
                != record['resume_sha256']:
            raise FormalTrainingError('resume training log checkpoint drift')


def _build_training_hook(
        *, root: Path, init: FormalRunInit, run_init_path: Path,
        resume: ResumeState | None):
    from mmengine.hooks import Hook
    from .formal_determinism import build_seeded_sampler, hash_sample_order

    class FormalTrainingHook(Hook):
        priority = 'LOWEST'

        def __init__(self) -> None:
            self.order_hashes = list(
                () if resume is None else resume.order_hashes)
            self.pending_order: str | None = None
            self.last_checkpoint = (
                None if resume is None else
                root / init.output_root /
                f'epoch_{resume.completed_epoch}.pth')
            completed = 0 if resume is None else resume.completed_epoch
            self.best_metric, best_path = _load_best_lineage(
                root, init, completed_epoch=completed)
            self.best_path = best_path
            self.log_path = root / init.output_root / 'training.jsonl'
            self.pending_checkpoint: dict[str, Any] | None = None

        def before_train_epoch(self, runner) -> None:
            sampler = runner.train_dataloader.sampler
            runtime = tuple(sampler)
            observed = hash_sample_order(runtime)
            expected = hash_sample_order(build_seeded_sampler(
                range(len(runner.train_dataloader.dataset)),
                init.seed, runner.epoch))
            if observed != expected:
                raise FormalTrainingError(
                    'runtime sampler order differs from deterministic rule')
            self.pending_order = observed

        def after_train_iter(self, runner, batch_idx: int,
                             data_batch=None, outputs=None) -> None:
            control = root / init.output_root / 'stop-request.json'
            if not control.exists():
                return
            torch.cuda.synchronize()
            request = poll_cooperative_stop(
                control, stage_output_root=root / init.output_root,
                safe_boundary=StageSafeBoundary(
                    'training', 'optimizer', batch_idx, True))
            if request is None:
                return
            if self.last_checkpoint is None:
                resume_authority = TrainingResumeAuthority(
                    init.run_id, 0, str(run_init_path),
                    _sha256_file(run_init_path))
            else:
                resume_authority = TrainingResumeAuthority(
                    init.run_id, len(self.order_hashes),
                    str(self.last_checkpoint), _sha256_file(
                        self.last_checkpoint))
            write_stop_acknowledgement(request, resume_authority)
            raise SystemExit(75)

        def after_train_epoch(self, runner) -> None:
            if self.pending_order is None:
                raise FormalTrainingError('epoch order evidence is missing')
            completed = runner.epoch + 1
            self.order_hashes.append(self.pending_order)
            self.pending_order = None
            schedulers = {
                str(index): scheduler.state_dict()
                for index, scheduler in enumerate(runner.param_schedulers)}
            scaler = {}
            loss_scaler = getattr(runner.optim_wrapper, 'loss_scaler', None)
            if loss_scaler is not None:
                scaler = loss_scaler.state_dict()
            self.pending_checkpoint = {
                'completed_epoch': completed,
                'model_state': _unwrap_runner_model(runner).state_dict(),
                'optimizer_state': runner.optim_wrapper.state_dict(),
                'scheduler_state': schedulers, 'scaler_state': scaler,
                'dataset_size': len(runner.train_dataloader.dataset),
            }
            train_loop = runner.train_loop
            will_validate = runner.val_loop is not None \
                and completed >= train_loop.val_begin \
                and (completed % train_loop.val_interval == 0
                     or completed == train_loop.max_epochs)
            if not will_validate:
                decision = None
                if self.best_path is not None:
                    decision = inherit_best_checkpoint(
                        root, init, completed_epoch=completed)
                self._commit_pending(runner, decision)

        def _commit_pending(self, runner, decision: Path | None) -> None:
            if self.pending_checkpoint is None:
                raise FormalTrainingError(
                    'epoch checkpoint transaction is unavailable')
            pending = self.pending_checkpoint
            self.last_checkpoint = write_resume_checkpoint(
                repository_root=root, expected=init,
                completed_epoch=pending['completed_epoch'],
                model_state=pending['model_state'],
                optimizer_state=pending['optimizer_state'],
                scheduler_state=pending['scheduler_state'],
                scaler_state=pending['scaler_state'],
                order_hashes=tuple(self.order_hashes),
                dataset_size=pending['dataset_size'],
                best_decision_path=decision)
            self.pending_checkpoint = None
            if decision is not None:
                self.best_metric, self.best_path = _load_best_lineage(
                    root, init, completed_epoch=len(self.order_hashes))

        def after_val_epoch(self, runner, metrics=None) -> None:
            if not isinstance(metrics, Mapping) or 'coco/AP' not in metrics:
                raise FormalTrainingError('formal validation AP is missing')
            metric = float(metrics['coco/AP'])
            if not math.isfinite(metric):
                raise FormalTrainingError('formal validation AP is non-finite')
            completed = runner.epoch
            if metric > self.best_metric:
                decision = publish_best_checkpoint(
                    root, init, completed_epoch=completed, metric=metric,
                    tensors=_unwrap_runner_model(runner).state_dict())
            else:
                decision = inherit_best_checkpoint(
                    root, init, completed_epoch=completed)
            self._commit_pending(runner, decision)

    return FormalTrainingHook()


def _unwrap_runner_model(runner):
    from mmengine.model.wrappers.utils import is_model_wrapper
    return runner.model.module if is_model_wrapper(runner.model) else runner.model


def _prepare_runner_for_authenticated_load(runner):
    """Build runtime objects and initialize once, before safe state injection."""
    model = _unwrap_runner_model(runner)
    if not hasattr(model, 'train_step'):
        raise FormalTrainingError('formal model has no train_step')
    if runner._val_loop is not None and not hasattr(model, 'val_step'):
        raise FormalTrainingError('formal model has no val_step')
    if runner._train_loop is None:
        raise FormalTrainingError('formal runner has no train loop')
    runner._train_loop = runner.build_train_loop(runner._train_loop)
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)
    if runner.param_schedulers is not None:
        runner.param_schedulers = runner.build_param_scheduler(
            runner.param_schedulers)
    if runner._val_loop is not None:
        runner._val_loop = runner.build_val_loop(runner._val_loop)
    runner.call_hook('before_run')
    runner._init_model_weights()
    return _unwrap_runner_model(runner)


def _mark_authenticated_state_loaded(runner) -> None:
    """Close the one-way boundary between trusted state injection and run."""
    if getattr(runner, '_formal_authenticated_state_lifecycle', None) \
            is not None:
        raise FormalTrainingError(
            'formal authenticated state lifecycle was already marked')
    runner._formal_authenticated_state_lifecycle = 'ready'


def _run_authenticated_training_loop(runner):
    """Run after safe init/resume; deliberately omit generic load_or_resume."""
    if getattr(runner, '_formal_authenticated_state_lifecycle', None) \
            != 'ready':
        raise FormalTrainingError(
            'formal authenticated state is not ready for training')
    runner._formal_authenticated_state_lifecycle = 'consumed'
    runner.optim_wrapper.initialize_count_status(
        runner.model, runner.train_loop.iter, runner.train_loop.max_iters)
    runner._maybe_compile('train_step')
    model = runner.train_loop.run()
    runner.call_hook('after_run')
    return model


def train_formal_candidate(
        init_path: Path, resume_path: Path | None) -> FormalTrainResult:
    """Run exactly one canonical 300-epoch MambaPose training member."""
    root = Path.cwd().absolute()
    init = load_formal_run_init(init_path, repository_root=root)
    canonical_init_path = root / init.output_root / 'run-init.json'
    if Path(init_path).absolute() != canonical_init_path:
        raise FormalTrainingError('formal training init path is not canonical')
    environment = EnvironmentAuthority.capture(root)
    if environment.inventory_sha256 != init.environment_inventory_sha256:
        raise FormalTrainingError('formal training environment drift')

    from .formal_determinism import configure_root_determinism
    configure_root_determinism(init.seed)
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    config = load_authenticated_formal_config(init, root)
    _neutralize_pretrained(config)
    if config.train_cfg.max_epochs != 300:
        raise FormalTrainingError('formal trainer config is not 300 epochs')
    if config.get('resume', False) or config.get('load_from', None) is not None:
        raise FormalTrainingError('generic runner resume/load is forbidden')
    config.default_hooks.checkpoint = None
    config.work_dir = str(root / init.output_root)
    init_default_scope(config.get('default_scope', 'mmpose'))
    runner = Runner.from_cfg(config)
    model = _prepare_runner_for_authenticated_load(runner)
    recovered = recover_training_lineage(
        root, init, dataset_size=len(runner.train_dataloader.dataset))
    if resume_path is None and recovered is not None:
        raise FormalTrainingError(
            'existing training lineage requires an explicit resume')
    if resume_path is not None:
        if recovered is None:
            raise FormalTrainingError('resume has no committed lineage')
        canonical_resume = (
            root / init.output_root /
            f'epoch_{recovered.completed_epoch}.pth')
        if Path(resume_path).absolute() != canonical_resume:
            raise FormalTrainingError(
                'resume checkpoint is not the committed latest')
    canonical = canonical_main_root(root)
    asset = init.initialization.asset
    load_formal_backbone_initialization(
        canonical / asset.authority_root / asset.asset_relative_path,
        FileAuthority(
            authority_root=str(canonical / asset.authority_root),
            path=asset.asset_relative_path.as_posix(), sha256=asset.sha256),
        model.backbone, minimum_compatible_tensors=100)

    resume = recovered
    if resume_path is not None:
        _revalidate_resume_state_authority(root, init, resume_path, resume)
        model.load_state_dict(dict(resume.model_state), strict=True)
        runner.optim_wrapper.load_state_dict(dict(resume.optimizer_state))
        for index, scheduler in enumerate(runner.param_schedulers):
            state = resume.scheduler_state.get(str(index))
            if not isinstance(state, Mapping):
                raise FormalTrainingError('resume scheduler state is incomplete')
            scheduler.load_state_dict(dict(state))
        loss_scaler = getattr(runner.optim_wrapper, 'loss_scaler', None)
        if resume.scaler_state:
            if loss_scaler is None:
                raise FormalTrainingError('resume scaler has no runtime consumer')
            loss_scaler.load_state_dict(dict(resume.scaler_state))
        runner.train_loop._epoch = resume.completed_epoch
        _restore_rng_state(resume.rng_state)
        _revalidate_resume_state_authority(root, init, resume_path, resume)
        _validate_training_log(
            root / init.output_root / 'training.jsonl', init,
            resume.order_hashes)
    elif any((root / init.output_root).glob('epoch_*.pth')) \
            or (root / init.output_root / 'training.jsonl').exists() \
            or (root / init.output_root / 'best-lineage.json').exists() \
            or any((root / init.output_root).glob(
                'best_coco_AP_epoch_*.pth')):
        raise FormalTrainingError(
            'existing training lineage requires an explicit resume')
    hook = _build_training_hook(
        root=root, init=init, run_init_path=canonical_init_path,
        resume=resume)
    runner.register_hook(hook, priority='LOWEST')
    _mark_authenticated_state_loaded(runner)
    _run_authenticated_training_loop(runner)
    if len(hook.order_hashes) != 300 or hook.last_checkpoint is None \
            or hook.best_path is None or not hook.best_path.is_file() \
            or not hook.log_path.is_file():
        raise FormalTrainingError('formal training completion is incomplete')
    resume_paths = (
        root / init.output_root / 'epoch_299.pth',
        root / init.output_root / 'epoch_300.pth')
    for path in resume_paths:
        validate_resume_checkpoint(
            path, init, repository_root=root,
            dataset_size=len(runner.train_dataloader.dataset))
    observed_resume_names = tuple(sorted(
        path.name for path in (root / init.output_root).glob('epoch_*.pth')))
    if observed_resume_names != ('epoch_299.pth', 'epoch_300.pth'):
        raise FormalTrainingError('formal resume inventory is not exactly two')
    _load_best_lineage(root, init, completed_epoch=300)
    result = FormalTrainResult(
        run_init_sha256=_sha256_file(canonical_init_path),
        initialization=init.initialization,
        run_id=init.run_id, role=init.role, seed=init.seed,
        output_root=init.output_root,
        best_checkpoint=_file_binding(root, hook.best_path),
        resume_checkpoints=(
            _file_binding(root, resume_paths[0]),
            _file_binding(root, resume_paths[1])),
        structured_log=_file_binding(root, hook.log_path),
        order_hashes=tuple(hook.order_hashes), final_epoch=300,
        status='complete')
    destination = root / init.output_root / 'train-result.json'
    payload = _canonical_json_bytes(_train_result_document(result))
    write_immutable_artifact(destination, payload, root)
    from .formal_schema import load_formal_train_result
    return load_formal_train_result(destination, repository_root=root)
