"""Readiness and checkpoint export for bounded scaled-Binary-Q/K recovery."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
import stat
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
from typing import Any, Mapping

from mmengine.config import Config
from mmengine.logging.history_buffer import HistoryBuffer
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint_to_model
import numpy as np
import torch
from torch import Tensor


READINESS_PATH = Path('optimization/binary_qk_recovery.json')
CONFIG_PATH = Path('configs/optimization/recovery/binary_qk_s_v1.py')
DEPLOY_CONFIG_PATH = Path(
    'configs/optimization/recovery/binary_qk_s_v1_deploy.py')
CHECKPOINT_PATH = Path(
    'work_dirs/reproduction/runs/coco-s-v1/'
    'best_coco_AP_epoch_300.pth')
CHECKPOINT_SHA256 = (
    'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2')


@dataclass(frozen=True)
class BoundCheckpoint:
    path: Path
    sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class BoundFile:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class BinaryRecoveryCompletion:
    """A validated commit-last student export completion."""

    transaction_token: str
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': 2,
            'artifact_kind': 'binary-qk-student-export-completion',
            'status': 'complete',
            'transaction': {
                'lock': '.student-export.lock',
                'token': self.transaction_token,
            },
            'report': dict(self.report),
        }


_SHA256 = re.compile(r'^[0-9a-f]{64}$')


@contextmanager
def _hold_regular_bytes(path: Path, label: str):
    """Yield one regular-file snapshot while retaining its descriptor."""
    candidate = Path(path).absolute()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(
            f'{label} must be a regular non-symlink file') from error
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError(
                f'{label} must be a regular non-symlink file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            data = stream.read()
        yield descriptor, BoundFile(
            path=candidate, data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            device=identity.st_dev, inode=identity.st_ino)
    finally:
        os.close(descriptor)


def _read_regular_bytes_once(path: Path, label: str) -> BoundFile:
    """Read one regular file through a single held descriptor."""
    with _hold_regular_bytes(path, label) as (_, bound):
        return bound


def _config_bases(data: bytes, logical_path: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(data, filename=logical_path)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f'config is invalid Python: {logical_path}') from error
    declarations = []
    for node in tree.body:
        value = None
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == '_base_' for target in node.targets)):
            value = node.value
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)
              and node.target.id == '_base_'):
            value = node.value
        if value is not None:
            declarations.append(value)
    if len(declarations) > 1:
        raise ValueError(f'config has multiple _base_ declarations: {logical_path}')
    if not declarations:
        return ()
    try:
        bases = ast.literal_eval(declarations[0])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'config _base_ is not literal: {logical_path}') from error
    if isinstance(bases, str):
        bases = (bases,)
    if (not isinstance(bases, (list, tuple))
            or any(not isinstance(base, str) or not base for base in bases)):
        raise ValueError(f'config _base_ is invalid: {logical_path}')
    return tuple(bases)


def _resolve_config_base(logical_path: str, base: str) -> str:
    base_path = PurePosixPath(base)
    if base_path.is_absolute():
        raise ValueError(f'config base is absolute: {logical_path}')
    parts: list[str] = []
    for part in PurePosixPath(logical_path).parent.joinpath(base_path).parts:
        if part in {'', '.'}:
            continue
        if part == '..':
            if not parts:
                raise ValueError(f'config base escapes repository: {logical_path}')
            parts.pop()
        else:
            parts.append(part)
    resolved = PurePosixPath(*parts)
    if resolved.suffix != '.py':
        raise ValueError(f'config base is not Python: {logical_path}')
    return resolved.as_posix()


def _capture_config_closure(
        repository_root: Path, entry: Path) -> dict[str, BoundFile]:
    root = Path(repository_root).resolve(strict=True)
    relative = PurePosixPath(entry.as_posix())
    if (relative.is_absolute() or relative.suffix != '.py'
            or any(part in {'', '.', '..'} for part in relative.parts)):
        raise ValueError('config path must be safe and repository-relative')
    captured: dict[str, BoundFile] = {}
    active: set[str] = set()

    def visit(logical_path: str) -> None:
        if logical_path in active:
            raise ValueError('config dependency cycle is invalid')
        if logical_path in captured:
            return
        candidate = _regular_repository_file(
            root, Path(logical_path), 'config dependency')
        active.add(logical_path)
        bound = _read_regular_bytes_once(candidate, 'config dependency')
        for base in _config_bases(bound.data, logical_path):
            visit(_resolve_config_base(logical_path, base))
        active.remove(logical_path)
        captured[logical_path] = bound

    visit(relative.as_posix())
    return captured


def _config_binding(
        entry: str, captured: Mapping[str, BoundFile]) -> dict[str, Any]:
    return {
        'path': entry,
        'sha256': captured[entry].sha256,
        'config_closure': [
            {'path': path, 'sha256': captured[path].sha256}
            for path in sorted(captured)
        ],
    }


def _parse_config_snapshot(
        captured: Mapping[str, BoundFile], entry: str) -> Config:
    with tempfile.TemporaryDirectory(prefix='binary-qk-config-') as directory:
        snapshot_root = Path(directory)
        for logical_path, bound in captured.items():
            destination = snapshot_root / logical_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bound.data)
        return Config.fromfile(snapshot_root / entry)


def build_config_binding(
        repository_root: Path, config_path: Path) -> dict[str, Any]:
    """Bind one config and every inherited base by path and held-byte SHA."""
    entry = PurePosixPath(config_path.as_posix()).as_posix()
    captured = _capture_config_closure(repository_root, Path(entry))
    return _config_binding(entry, captured)


def _load_bound_config_snapshot(
        repository_root: Path, binding: Mapping[str, Any], *, label: str
        ) -> tuple[Config, dict[str, BoundFile]]:
    if (not isinstance(binding, Mapping)
            or set(binding) != {'path', 'sha256', 'config_closure'}
            or not isinstance(binding['path'], str)
            or not binding['path']
            or not isinstance(binding['sha256'], str)
            or not _SHA256.fullmatch(binding['sha256'])
            or not isinstance(binding['config_closure'], list)
            or not binding['config_closure']):
        raise ValueError(f'{label} binding is invalid')
    entry = binding['path']
    captured = _capture_config_closure(repository_root, Path(entry))
    actual = _config_binding(entry, captured)
    if dict(binding) != actual:
        raise ValueError(f'{label} closure disagrees with held bytes')
    return _parse_config_snapshot(captured, entry), captured


def load_bound_config(
        repository_root: Path, binding: Mapping[str, Any], *, label: str
        ) -> Config:
    """Verify and parse exactly one held config-closure snapshot."""
    config, _ = _load_bound_config_snapshot(
        repository_root, binding, label=label)
    return config


def _read_checkpoint_once(path: Path) -> tuple[Path, bytes, str]:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(
            'checkpoint must be a regular non-symlink file') from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                'checkpoint must be a regular non-symlink file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            data = stream.read()
    finally:
        os.close(descriptor)
    return candidate.resolve(), data, hashlib.sha256(data).hexdigest()


def _restricted_checkpoint_payload(data: bytes) -> object:
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
            return torch.load(
                io.BytesIO(data), map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(
            f'checkpoint is not weights-only compatible: {error}') from error


def load_tensor_checkpoint(
        path: Path, expected_sha256: str) -> tuple[dict[str, Tensor], str]:
    """Hash and restricted-load one immutable byte snapshot."""
    _, data, actual_sha256 = _read_checkpoint_once(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            'checkpoint SHA-256 mismatch: '
            f'expected {expected_sha256}, got {actual_sha256}')
    payload = _restricted_checkpoint_payload(data)
    if isinstance(payload, Mapping) and isinstance(
            payload.get('state_dict'), Mapping):
        payload = payload['state_dict']
    if (
            not isinstance(payload, Mapping)
            or not payload
            or not all(
                isinstance(key, str) and isinstance(value, Tensor)
                for key, value in payload.items())):
        raise ValueError('checkpoint must contain a non-empty tensor state_dict')
    return dict(payload), actual_sha256


_RESUME_META_FIELDS = {
    'epoch', 'iter', 'cfg', 'seed', 'experiment_name', 'time',
    'mmengine_version', 'dataset_meta'}
_RESUME_SCHEDULER_FIELDS = (
    {
        'start_factor', 'end_factor', 'total_iters', 'param_name', 'begin',
        'end', 'by_epoch', 'base_values', 'last_step', '_global_step',
        'verbose', '_last_value'},
    {
        'milestones', 'gamma', 'param_name', 'begin', 'end', 'by_epoch',
        'base_values', 'last_step', '_global_step', 'verbose', '_last_value'},
)


def _validate_training_resume_payload(
        payload: object, *, expected_config: str,
        expected_state_dict: Mapping[str, Tensor]) -> Mapping[str, Any]:
    fields = {
        'meta', 'state_dict', 'message_hub', 'optimizer', 'param_schedulers'}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError('training resume checkpoint has invalid top-level fields')
    meta = payload['meta']
    if (
            not isinstance(meta, Mapping) or set(meta) != _RESUME_META_FIELDS
            or isinstance(meta['epoch'], bool) or not isinstance(meta['epoch'], int)
            or meta['epoch'] < 0
            or isinstance(meta['iter'], bool) or not isinstance(meta['iter'], int)
            or meta['iter'] < 0
            or isinstance(meta['seed'], bool) or not isinstance(meta['seed'], int)
            or not all(isinstance(meta[name], str) and meta[name]
                       for name in (
                           'cfg', 'experiment_name', 'time',
                           'mmengine_version'))
            or meta['cfg'] != expected_config
            or not isinstance(meta['dataset_meta'], Mapping)):
        raise ValueError('training resume checkpoint meta schema is invalid')
    state = payload['state_dict']
    if (not isinstance(expected_state_dict, Mapping)
            or not expected_state_dict
            or not all(isinstance(name, str) and name
                       and isinstance(value, Tensor)
                       for name, value in expected_state_dict.items())):
        raise ValueError('expected distiller state_dict is invalid')
    if (not isinstance(state, Mapping) or set(state) != set(expected_state_dict)
            or not all(
                isinstance(value, Tensor)
                and value.shape == expected_state_dict[name].shape
                and value.dtype == expected_state_dict[name].dtype
                for name, value in state.items())):
        raise ValueError('training resume checkpoint state_dict is invalid')
    message_hub = payload['message_hub']
    if (
            not isinstance(message_hub, Mapping)
            or set(message_hub) != {
                'log_scalars', 'runtime_info', 'resumed_keys'}
            or not all(isinstance(message_hub[name], Mapping)
                       for name in message_hub)):
        raise ValueError('training resume checkpoint message_hub is invalid')
    optimizer = payload['optimizer']
    if (
            not isinstance(optimizer, Mapping)
            or set(optimizer) != {'state', 'param_groups'}
            or not isinstance(optimizer['state'], Mapping)
            or not isinstance(optimizer['param_groups'], list)
            or len(optimizer['param_groups']) != 1
            or not isinstance(optimizer['param_groups'][0], Mapping)
            or set(optimizer['param_groups'][0]) != {
                'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
                'foreach', 'capturable', 'differentiable', 'fused',
                'decoupled_weight_decay', 'initial_lr', 'params'}):
        raise ValueError('training resume checkpoint optimizer schema is invalid')
    schedulers = payload['param_schedulers']
    if (
            not isinstance(schedulers, list) or len(schedulers) != 2
            or any(
                not isinstance(row, Mapping) or set(row) != expected
                for row, expected in zip(
                    schedulers, _RESUME_SCHEDULER_FIELDS))
            or schedulers[0]['start_factor'] != 0.001
            or schedulers[0]['end_factor'] != 1.0
            # MMEngine LinearLR defines total_iters as end - begin - 1.
            or schedulers[0]['total_iters'] != 499
            or schedulers[0]['param_name'] != 'lr'
            or schedulers[0]['begin'] != 0
            or schedulers[0]['end'] != 500
            or schedulers[0]['by_epoch'] is not False
            or schedulers[0]['base_values'] != [1e-4]
            or schedulers[1]['gamma'] != 0.1
            or dict(schedulers[1]['milestones']) != {40: 1, 52: 1}
            or schedulers[1]['param_name'] != 'lr'
            or schedulers[1]['begin'] != 0
            or schedulers[1]['end'] != 60
            or schedulers[1]['by_epoch'] is not True
            or schedulers[1]['base_values'] != [1e-4]):
        raise ValueError('training resume checkpoint scheduler schema is invalid')
    return payload


def load_training_resume_checkpoint(
        path: Path, expected_sha256: str, *,
        expected_config: str,
        expected_state_dict: Mapping[str, Tensor]) -> BoundCheckpoint:
    """Load one complete recovery resume checkpoint from held bytes."""
    resolved, data, actual_sha256 = _read_checkpoint_once(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            'training resume checkpoint SHA-256 mismatch: '
            f'expected {expected_sha256}, got {actual_sha256}')
    payload = _validate_training_resume_payload(
        _restricted_checkpoint_payload(data),
        expected_config=expected_config,
        expected_state_dict=expected_state_dict)
    return BoundCheckpoint(resolved, actual_sha256, payload)


class BoundBinaryRecoveryRunner(Runner):
    """Runner whose resume authority is one already-validated byte snapshot."""

    def bind_resume_checkpoint(self, checkpoint: BoundCheckpoint) -> None:
        if hasattr(self, '_binary_resume_checkpoint'):
            raise RuntimeError('binary recovery resume is already bound')
        self._binary_resume_checkpoint = checkpoint
        self._resume = True
        self._load_from = str(checkpoint.path)

    def load_checkpoint(
            self, filename: str, map_location='cpu', strict: bool = True,
            revise_keys: list = [(r'^module.', '')]):
        checkpoint = getattr(self, '_binary_resume_checkpoint', None)
        if checkpoint is None:
            raise RuntimeError(
                'binary recovery runner refuses an unbound checkpoint load')
        requested = Path(filename).absolute()
        if requested != checkpoint.path:
            raise RuntimeError(
                'binary recovery runner resume path differs from held bytes')
        if strict is not True:
            raise RuntimeError(
                'binary recovery runner requires strict checkpoint loading')
        payload = checkpoint.payload
        self.call_hook('after_load_checkpoint', checkpoint=payload)
        model = self.model.module if is_model_wrapper(
            self.model) else self.model
        _load_checkpoint_to_model(
            model, payload, strict, revise_keys=revise_keys)
        self._has_loaded = True
        self.logger.info(
            'Load SHA-bound recovery checkpoint from held bytes: '
            f'{checkpoint.path} ({checkpoint.sha256})')
        return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably publish bytes through a same-directory atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp',
        dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8'))


def write_binary_recovery_launch_metadata(
        work_dir: Path, *, config_text: str,
        readiness: Mapping[str, Any]) -> None:
    """Publish pre-run metadata; absence of the final manifest is incomplete."""
    directory = Path(work_dir)
    atomic_write_bytes(
        directory / 'resolved-binary-qk-recovery.py',
        config_text.encode('utf-8'))
    atomic_write_json(directory / 'readiness.json', readiness)


@contextmanager
def _completion_lock(work_dir: Path, *, exclusive: bool):
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / '.student-export.lock'
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError('student export lock is invalid') from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('student export lock is invalid')
        fcntl.flock(
            descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_binary_recovery_completion(
        work_dir: Path, *, repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None = None,
        ) -> BinaryRecoveryCompletion | None:
    """Load and live-revalidate the commit-last completion transaction."""
    directory = Path(work_dir)
    with _completion_lock(directory, exclusive=False):
        return _load_binary_recovery_completion_unlocked(
            directory, repository_root=repository_root,
            deployment_binding=deployment_binding,
            smoke_inputs=smoke_inputs)


def _regular_repository_file(
        root: Path, relative: Path, label: str, *, allow_symlink: bool = False
        ) -> Path:
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'{label} must be repository-relative')
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() and not allow_symlink:
            raise ValueError(f'{label} path must not contain symlinks')
    if not candidate.resolve(strict=True).is_file():
        raise ValueError(f'{label} is missing: {relative.as_posix()}')
    return candidate


def _load_binary_recovery_readiness_snapshot(
        repository_root: Path) -> tuple[Mapping[str, Any], str]:
    """Load and verify the checked-in recovery operation/readiness contract."""
    root = repository_root.resolve(strict=True)
    readiness_path = _regular_repository_file(
        root, READINESS_PATH, 'readiness manifest')
    try:
        readiness_file = _read_regular_bytes_once(
            readiness_path, 'readiness manifest')
        value = json.loads(readiness_file.data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('binary recovery readiness is unreadable') from exc
    if not isinstance(value, dict) or set(value) != {
            'schema_version', 'artifact_kind', 'base_training_epochs',
            'config', 'checkpoint', 'operation', 'teacher', 'student', 'loss',
            'schedule', 'optional_extensions', 'deployment',
            'verification_commands'}:
        raise ValueError('binary recovery readiness has invalid top-level fields')
    if value['schema_version'] != 2 or value['artifact_kind'] != (
            'binary-qk-recovery-operation-readiness'):
        raise ValueError('binary recovery readiness identity drifted')
    if (not isinstance(value['config'], Mapping)
            or value['config'].get('path') != CONFIG_PATH.as_posix()):
        raise ValueError('binary recovery config binding drifted')
    checkpoint = value.get('checkpoint')
    if checkpoint != {
            'path': CHECKPOINT_PATH.as_posix(),
            'role': (
                'strict-full-s-v1-initialization-for-teacher-and-student'),
            'sha256': CHECKPOINT_SHA256}:
        raise ValueError('binary recovery checkpoint binding drifted')

    checkpoint_path = _regular_repository_file(
        root, CHECKPOINT_PATH, 'S-V1 checkpoint', allow_symlink=True)
    if _read_checkpoint_once(checkpoint_path)[2] != CHECKPOINT_SHA256:
        raise ValueError('S-V1 checkpoint SHA-256 disagrees with readiness')

    config, config_snapshot = _load_bound_config_snapshot(
        root, value['config'], label='recovery config')
    model = config.model
    if (
            value['base_training_epochs'] != 300
            or dict(model) != {
                'type': 'BinaryQKSelfDistiller',
                'base_model_config': (
                    'configs/optimization/coco_s_v1_deterministic.py'),
                'checkpoint': CHECKPOINT_PATH.as_posix(),
                'checkpoint_sha256': CHECKPOINT_SHA256,
                'distill_weight': 0.25}
            or config.randomness != dict(seed=0, deterministic=True)
            or dict(config.train_cfg) != {
                'by_epoch': True, 'max_epochs': 60, 'val_interval': 5}
            or config.load_from is not None
            or config.resume is not False
            or dict(config.binary_qk_recovery) != {
                'schema_version': 1,
                'qk_mode': 'binary_scaled',
                'teacher_qk_mode': 'float',
                'teacher_frozen': True,
                'supervised_loss': 'heatmap-mse-with-target-weight',
                'distillation_target': 'final-heatmap-mse',
                'schedule': 'bounded-60-epoch-recovery',
                'schedule_extensible': True,
                'learnable_attention_bias': False}):
        raise ValueError('binary recovery config disagrees with readiness')
    if model.base_model_config not in config_snapshot:
        raise ValueError('binary recovery base config is outside its closure')
    base_config = _parse_config_snapshot(
        config_snapshot, model.base_model_config)
    if dict(base_config.model.head.loss) != {
            'type': 'KeypointMSELoss', 'use_target_weight': True}:
        raise ValueError('binary recovery supervised loss config drifted')
    if value['loss'] != {
            'distillation_target': 'final-heatmap-mse',
            'distillation_weight': 0.25,
            'supervised': 'heatmap-mse-with-target-weight'}:
        raise ValueError('binary recovery readiness loss contract drifted')
    if value['schedule'] != {
            'deterministic': True,
            'epochs': 60,
            'fraction_of_base_training': 0.2,
            'schedule_extensible': True,
            'seed': 0}:
        raise ValueError('binary recovery readiness schedule contract drifted')
    if value['optional_extensions'] != {
            'relative_attention_bias': {
                'enabled': False,
                'reason': (
                    'DeiT 2-D relative bias does not transfer directly to '
                    'the mixed 17 keypoint and spatial token sequence')}}:
        raise ValueError(
            'binary recovery readiness optional extensions drifted')
    if (
            list(config.param_scheduler) != [
                {
                    'type': 'LinearLR', 'begin': 0, 'end': 500,
                    'start_factor': 0.001, 'by_epoch': False},
                {
                    'type': 'MultiStepLR', 'begin': 0, 'end': 60,
                    'milestones': [40, 52], 'gamma': 0.1,
                    'by_epoch': True}]
            or dict(config.optim_wrapper) != {
                'optimizer': {'type': 'Adam', 'lr': 1e-4}}):
        raise ValueError('binary recovery config schedule drifted')
    if value['operation'] != {
            'qk_mode': 'binary_scaled',
            'q_scale': 'abs(q).mean(token).mean(channel) per batch/head',
            'k_scale': 'abs(k).mean(token).mean(channel) per batch/head',
            'head_dim_scale': (
                'original head_dim^-0.5 after scaled signed dot'),
            'runtime_operations': [
                'q_abs', 'q_token_mean', 'q_channel_mean',
                'k_abs', 'k_token_mean', 'k_channel_mean',
                'signed_dot_scale_multiply_q',
                'signed_dot_scale_multiply_k',
                'original_head_dim_scale_multiply'],
            'signed_dot_backend': 'torch-einsum-float-proxy',
            'ste': 'identity-gradient-sign-with-zero-positive',
            'softmax': 'floating',
            'value_and_output_projection': 'floating',
            'bitwise_kernel_present': False,
            'pure_bitwise_cost_claim': False}:
        raise ValueError('binary recovery operation disclosure drifted')
    if value['teacher'] != {
            'qk_mode': 'float', 'frozen': True,
            'uses_same_checkpoint_as_student': True}:
        raise ValueError('binary recovery teacher policy drifted')
    if value['student'] != {
            'qk_mode': 'binary_scaled', 'learnable_attention_bias': False,
            'new_checkpoint_parameters': []}:
        raise ValueError('binary recovery student policy drifted')
    deploy = value['deployment']
    if (not isinstance(deploy, Mapping)
            or set(deploy) != {'cpu_smoke_scope', 'config', 'qk_mode'}
            or deploy['cpu_smoke_scope'] != 'binary-qk-transformer'
            or deploy['qk_mode'] != 'binary_scaled'
            or not isinstance(deploy['config'], Mapping)
            or deploy['config'].get('path') != DEPLOY_CONFIG_PATH.as_posix()):
        raise ValueError('binary recovery deployment binding drifted')
    deploy_config = load_bound_config(
        root, deploy['config'], label='binary deployment config')
    if (
            deploy_config.model.type != 'TopdownPoseEstimator'
            or deploy_config.model.head.tokenpose_cfg.qk_mode != (
                'binary_scaled')
            or deploy_config.model.backbone.pretrained is not None
            or deploy_config.load_from is not None
            or deploy_config.resume is not False):
        raise ValueError('binary recovery deployment config drifted')
    if value['verification_commands'] != {
            'prepare': (
                'PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
                '.venv/bin/python tools/optimization/'
                'train_binary_qk_recovery.py --prepare-only'),
            'cpu_tests': (
                'PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
                ".venv/bin/python -m pytest -o addopts='' -q "
                'tests/test_optimization/test_binary_qk.py '
                'tests/test_optimization/test_binary_qk_recovery.py '
                'tests/test_optimization/test_binary_readiness.py')}:
        raise ValueError('binary recovery verification commands drifted')
    return value, readiness_file.sha256


def load_binary_recovery_readiness(repository_root: Path) -> Mapping[str, Any]:
    """Load and verify the checked-in recovery operation/readiness contract."""
    return _load_binary_recovery_readiness_snapshot(repository_root)[0]


def build_binary_recovery_readiness(
        repository_root: Path, work_dir: Path) -> dict[str, Any]:
    manifest, manifest_sha256 = _load_binary_recovery_readiness_snapshot(
        repository_root)
    root = repository_root.resolve(strict=True)
    return {
        'schema_version': 2,
        'status': 'ready',
        'operation_manifest': {
            'path': READINESS_PATH.as_posix(),
            'sha256': manifest_sha256,
        },
        'config': dict(manifest['config']),
        'checkpoint': dict(manifest['checkpoint']),
        'deployment': dict(manifest['deployment']),
        'launch': {
            'epochs': manifest['schedule']['epochs'],
            'work_dir': str(work_dir.resolve()),
            'student_export': 'binary_qk_s_v1_student.pth',
            'completion_manifest': 'student-export.json',
            'commit_policy': 'commit-last',
        },
    }


def _state_schema_sha256(state: Mapping[str, Tensor]) -> str:
    schema = [
        {
            'dtype': str(state[name].dtype),
            'name': name,
            'shape': list(state[name].shape),
        }
        for name in sorted(state)
    ]
    encoded = json.dumps(
        schema, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _deployment_runtime(
        repository_root: Path, deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None,
        *, held_config: Config | None = None,
        ) -> tuple[Path, torch.nn.Module, Any, str]:
    root = Path(repository_root).resolve(strict=True)
    config = held_config if held_config is not None else load_bound_config(
        root, deployment_binding, label='deployment config')
    config_path = root / deployment_binding['path']
    try:
        qk_mode = config.model.head.tokenpose_cfg.qk_mode
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(
            'deployment config has no direct Q/K mode binding') from error
    if qk_mode != 'binary_scaled':
        raise ValueError('deployment config must bind binary_scaled Q/K')
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS

    init_default_scope(config.get('default_scope', 'mmpose'))
    model = MODELS.build(config.model)
    model.cpu().eval()
    if smoke_inputs is None:
        try:
            transformer = model.head.tokenpose.transformer
            layers = transformer.layers
            modes = [layer[0].fn.fn.qk_mode for layer in layers]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(
                'deployment model has no Binary Q/K transformer') from error
        if modes != ['binary_scaled'] * 6:
            raise ValueError(
                'deployment model does not bind all six Binary Q/K layers')
        inputs = torch.zeros(1, 65, 256)
        position = torch.zeros(1, 48, 256)
        smoke_scope = 'binary-qk-transformer'

        def execute_smoke():
            return transformer(inputs, pos=position)[0]
    else:
        inputs = smoke_inputs
        smoke_scope = 'full-model-test-double'

        def execute_smoke():
            return model(inputs, data_samples=None, mode='tensor')
    if not isinstance(inputs, Tensor) or inputs.device.type != 'cpu':
        raise ValueError('deployment smoke input must be a CPU tensor')
    return config_path.absolute(), model, execute_smoke, smoke_scope


def _validate_live_student_export(
        bound: BoundFile, report: Mapping[str, Any], *,
        work_dir: Path, repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None,
        ) -> None:
    fields = {
        'path', 'sha256', 'source_sha256', 'tensors', 'checkpoint',
        'deployment_config', 'state_schema_sha256', 'transaction_token',
        'validation'}
    if (not isinstance(report, Mapping) or set(report) != fields
            or not isinstance(report['path'], str)
            or not isinstance(report['sha256'], str)
            or not _SHA256.fullmatch(report['sha256'])
            or not isinstance(report['source_sha256'], str)
            or not _SHA256.fullmatch(report['source_sha256'])
            or not isinstance(report['state_schema_sha256'], str)
            or not _SHA256.fullmatch(report['state_schema_sha256'])
            or not isinstance(report['transaction_token'], str)
            or not _SHA256.fullmatch(report['transaction_token'])
            or isinstance(report['tensors'], bool)
            or not isinstance(report['tensors'], int)
            or report['tensors'] <= 0):
        raise ValueError('student export report schema is invalid')
    checkpoint = report['checkpoint']
    expected_checkpoint = {
        'path': report['path'],
        'sha256': report['sha256'],
    }
    if checkpoint != expected_checkpoint:
        raise ValueError('student export checkpoint report is invalid')
    checkpoint_path = Path(report['path']).absolute()
    if (checkpoint_path.parent != Path(work_dir).resolve()
            or checkpoint_path != bound.path
            or bound.sha256 != report['sha256']):
        raise ValueError('student export checkpoint identity drifted')
    config_path, model, execute_smoke, smoke_scope = _deployment_runtime(
        repository_root, deployment_binding, smoke_inputs)
    expected_deployment = {
        'path': str(config_path),
        'binding': dict(deployment_binding),
    }
    if report['deployment_config'] != expected_deployment:
        raise ValueError('student export deployment authority drifted')
    payload = _restricted_checkpoint_payload(bound.data)
    if (not isinstance(payload, Mapping)
            or set(payload) != {'meta', 'state_dict'}
            or payload['meta'] != {
                'source_checkpoint_sha256': report['source_sha256']}
            or not isinstance(payload['state_dict'], Mapping)):
        raise ValueError('student export checkpoint schema is invalid')
    state = payload['state_dict']
    expected_state = model.state_dict()
    if (set(state) != set(expected_state)
            or not all(
                isinstance(value, Tensor)
                and value.shape == expected_state[name].shape
                and value.dtype == expected_state[name].dtype
                for name, value in state.items())):
        raise ValueError('student export checkpoint state_dict is invalid')
    if (deployment_binding.get('path') == DEPLOY_CONFIG_PATH.as_posix()
            and len(expected_state) != 230):
        raise ValueError('production student export must contain 230 tensors')
    if (report['tensors'] != len(expected_state)
            or report['state_schema_sha256'] !=
            _state_schema_sha256(expected_state)):
        raise ValueError('student export state authority drifted')
    model.load_state_dict(dict(state), strict=True)
    with torch.inference_mode():
        output = execute_smoke()
    if not isinstance(output, Tensor) or not torch.isfinite(output).all():
        raise ValueError('student export CPU smoke is invalid')
    if report['validation'] != {
            'qk_mode': 'binary_scaled',
            'strict_state_load': True,
            'cpu_smoke': True,
            'cpu_smoke_scope': smoke_scope,
            'output_shape': list(output.shape)}:
        raise ValueError('student export validation report drifted')


def export_student_checkpoint(
        source: Path, destination: Path, *, deployment_config: Path,
        smoke_inputs: Tensor | None = None,
        deployment_repository_root: Path | None = None,
        deployment_binding: Mapping[str, Any] | None = None,
        transaction_token: str | None = None,
        ) -> dict[str, Any]:
    """Export and execute a SHA-bound Binary-Q/K pose-estimator artifact."""
    _, source_bytes, source_sha256 = _read_checkpoint_once(source)
    payload_value = _restricted_checkpoint_payload(source_bytes)
    if (
            not isinstance(payload_value, Mapping)
            or not isinstance(payload_value.get('state_dict'), Mapping)):
        raise ValueError('wrapper checkpoint has no state_dict')
    state = payload_value['state_dict']
    if not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state.items()):
        raise ValueError('wrapper checkpoint state_dict is not tensor-only')
    foreign = [
        key for key in state
        if not key.startswith('student.') and not key.startswith('teacher.')]
    if foreign:
        raise ValueError('wrapper checkpoint contains foreign state keys')
    student = OrderedDict(
        (key.removeprefix('student.'), value)
        for key, value in state.items() if key.startswith('student.'))
    if not student:
        raise ValueError('wrapper checkpoint has no student state')
    payload = {
        'meta': {'source_checkpoint_sha256': source_sha256},
        'state_dict': student,
    }

    config_path = Path(deployment_config).absolute()
    held_config = None
    if deployment_binding is None:
        config_root = config_path.parent
        config_entry = Path(config_path.name)
        captured = _capture_config_closure(config_root, config_entry)
        config_authority = _config_binding(config_entry.as_posix(), captured)
        held_config = _parse_config_snapshot(captured, config_entry.as_posix())
    else:
        if deployment_repository_root is None:
            raise ValueError(
                'deployment repository root is required with a binding')
        config_root = Path(deployment_repository_root).resolve(strict=True)
        try:
            config_entry = config_path.relative_to(config_root)
        except ValueError as error:
            raise ValueError('deployment config escapes repository') from error
        if deployment_binding.get('path') != config_entry.as_posix():
            raise ValueError('deployment config path disagrees with binding')
        config_authority = dict(deployment_binding)
    config_path, model, execute_smoke, smoke_scope = _deployment_runtime(
        config_root, config_authority, smoke_inputs,
        held_config=held_config)
    if transaction_token is None:
        transaction_token = secrets.token_hex(32)
    if not _SHA256.fullmatch(transaction_token):
        raise ValueError('student export transaction token is invalid')

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w+b', closefd=True) as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            exported_bytes = stream.read()
            destination_sha256 = hashlib.sha256(exported_bytes).hexdigest()
            exported_payload = _restricted_checkpoint_payload(exported_bytes)
            if (
                    not isinstance(exported_payload, Mapping)
                    or set(exported_payload) != {'meta', 'state_dict'}
                    or exported_payload['meta'] != {
                        'source_checkpoint_sha256': source_sha256}
                    or not isinstance(exported_payload['state_dict'], Mapping)
                    or set(exported_payload['state_dict']) != set(student)
                    or not all(
                        isinstance(value, Tensor)
                        for value in exported_payload['state_dict'].values())):
                raise ValueError('student export round-trip schema is invalid')
            model.load_state_dict(
                dict(exported_payload['state_dict']), strict=True)
            with torch.inference_mode():
                smoke_output = execute_smoke()
            if not isinstance(smoke_output, Tensor) or not torch.isfinite(
                    smoke_output).all():
                raise ValueError(
                    'deployment CPU smoke output is missing or non-finite')
            held = os.fstat(stream.fileno())
            try:
                named = os.stat(temporary, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    'student export changed after validation') from error
            if (not stat.S_ISREG(named.st_mode)
                    or (held.st_dev, held.st_ino) !=
                    (named.st_dev, named.st_ino)):
                raise ValueError('student export changed after validation')
            os.replace(temporary, destination)
            published = os.stat(destination, follow_symlinks=False)
            if (not stat.S_ISREG(published.st_mode)
                    or (held.st_dev, held.st_ino) !=
                    (published.st_dev, published.st_ino)):
                destination.unlink(missing_ok=True)
                raise ValueError('student export changed during publication')
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        'path': str(destination.resolve()),
        'sha256': destination_sha256,
        'source_sha256': source_sha256,
        'tensors': len(student),
        'state_schema_sha256': _state_schema_sha256(student),
        'transaction_token': transaction_token,
        'checkpoint': {
            'path': str(destination.resolve()),
            'sha256': destination_sha256},
        'deployment_config': {
            'path': str(config_path),
            'binding': config_authority},
        'validation': {
            'qk_mode': 'binary_scaled',
            'strict_state_load': True,
            'cpu_smoke': True,
            'cpu_smoke_scope': smoke_scope,
            'output_shape': list(smoke_output.shape)},
    }


def _parse_binary_recovery_completion(
        value: object) -> BinaryRecoveryCompletion:
    if (not isinstance(value, Mapping)
            or set(value) != {
                'schema_version', 'artifact_kind', 'status',
                'transaction', 'report'}
            or value['schema_version'] != 2
            or value['artifact_kind'] !=
            'binary-qk-student-export-completion'
            or value['status'] != 'complete'
            or not isinstance(value['transaction'], Mapping)
            or set(value['transaction']) != {'lock', 'token'}
            or value['transaction']['lock'] != '.student-export.lock'
            or not isinstance(value['transaction']['token'], str)
            or not _SHA256.fullmatch(value['transaction']['token'])
            or not isinstance(value['report'], Mapping)
            or value['report'].get('transaction_token') !=
            value['transaction']['token']):
        raise ValueError('student export completion schema is invalid')
    return BinaryRecoveryCompletion(
        transaction_token=value['transaction']['token'],
        report=dict(value['report']))


def _load_binary_recovery_completion_unlocked(
        work_dir: Path, *, repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None,
        ) -> BinaryRecoveryCompletion | None:
    completion_path = Path(work_dir) / 'student-export.json'
    if not completion_path.exists():
        return None
    completion_file = _read_regular_bytes_once(
        completion_path, 'student export completion')
    try:
        value = json.loads(completion_file.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('student export completion is invalid') from error
    completion = _parse_binary_recovery_completion(value)
    raw_checkpoint_path = completion.report.get('path')
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise ValueError('student export report schema is invalid')
    checkpoint_path = Path(raw_checkpoint_path)
    try:
        with _hold_regular_bytes(
                checkpoint_path, 'student export checkpoint') as (_, bound):
            _validate_live_student_export(
                bound, completion.report, work_dir=Path(work_dir),
                repository_root=repository_root,
                deployment_binding=deployment_binding,
                smoke_inputs=smoke_inputs)
            identity = os.stat(checkpoint_path, follow_symlinks=False)
            if ((identity.st_dev, identity.st_ino) !=
                    (bound.device, bound.inode)):
                raise ValueError(
                    'student export checkpoint changed during validation')
    except ValueError:
        raise
    except OSError as error:
        raise ValueError('student export checkpoint is missing') from error
    return completion


def _publish_binary_recovery_completion_unlocked(
        work_dir: Path, report: Mapping[str, Any], *,
        repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None,
        ) -> BinaryRecoveryCompletion:
    directory = Path(work_dir).resolve()
    token = report.get('transaction_token')
    if not isinstance(token, str) or not _SHA256.fullmatch(token):
        raise ValueError('student export transaction token is invalid')
    raw_checkpoint_path = report.get('path')
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise ValueError('student export report schema is invalid')
    checkpoint_path = Path(raw_checkpoint_path)
    try:
        with _hold_regular_bytes(
                checkpoint_path, 'student export checkpoint') as (_, bound):
            _validate_live_student_export(
                bound, report, work_dir=directory,
                repository_root=repository_root,
                deployment_binding=deployment_binding,
                smoke_inputs=smoke_inputs)
            completion = BinaryRecoveryCompletion(
                transaction_token=token, report=dict(report))
            completion_path = directory / 'student-export.json'
            atomic_write_json(completion_path, completion.to_dict())
            identity = os.stat(checkpoint_path, follow_symlinks=False)
            if ((identity.st_dev, identity.st_ino) !=
                    (bound.device, bound.inode)):
                completion_path.unlink(missing_ok=True)
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                raise ValueError(
                    'student export checkpoint changed at completion commit')
    except ValueError:
        raise
    except OSError as error:
        raise ValueError('student export checkpoint is missing') from error
    return completion


def publish_binary_recovery_completion(
        work_dir: Path, report: Mapping[str, Any], *,
        repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None = None,
        ) -> BinaryRecoveryCompletion:
    """Live-validate and atomically commit one typed completion."""
    with _completion_lock(work_dir, exclusive=True):
        return _publish_binary_recovery_completion_unlocked(
            work_dir, report, repository_root=repository_root,
            deployment_binding=deployment_binding,
            smoke_inputs=smoke_inputs)


def export_and_publish_binary_recovery_completion(
        source: Path, destination: Path, *, deployment_config: Path,
        repository_root: Path,
        deployment_binding: Mapping[str, Any],
        smoke_inputs: Tensor | None = None,
        ) -> BinaryRecoveryCompletion:
    """Export and commit under one lock and one transaction token."""
    work_dir = Path(destination).parent.resolve()
    with _completion_lock(work_dir, exclusive=True):
        try:
            existing = _load_binary_recovery_completion_unlocked(
                work_dir, repository_root=repository_root,
                deployment_binding=deployment_binding,
                smoke_inputs=smoke_inputs)
        except ValueError:
            existing = None
        if existing is not None:
            return existing
        token = secrets.token_hex(32)
        report = export_student_checkpoint(
            source, destination, deployment_config=deployment_config,
            smoke_inputs=smoke_inputs,
            deployment_repository_root=repository_root,
            deployment_binding=deployment_binding,
            transaction_token=token)
        return _publish_binary_recovery_completion_unlocked(
            work_dir, report, repository_root=repository_root,
            deployment_binding=deployment_binding,
            smoke_inputs=smoke_inputs)
