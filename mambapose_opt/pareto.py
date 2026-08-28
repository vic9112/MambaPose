"""Artifact-derived paired accuracy and Binary Q/K Pareto rules."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Mapping

from .binary_operation import (
    canonical_json_sha256, validate_binary_operation_manifest)
from .evaluation import CandidateResult


_T_CRITICAL_95 = {3: 4.302652729911275, 5: 2.7764451051977987}
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')
_SOURCE_FIELDS = {
    'git_commit', 'manifest_path', 'manifest_sha256', 'config_path',
    'config_sha256', 'authority_path', 'authority_sha256'}
_IDENTITY_FIELDS = {
    'candidate_id', 'candidate_kind', 'route', 'seed', 'role'}
_RUN_INIT_FIELDS = {
    'schema_version', 'artifact_kind', 'candidate', 'source',
    'initialization', 'config', 'protocol'}
_TRAIN_RESULT_FIELDS = {
    'schema_version', 'artifact_kind', 'candidate', 'run_init', 'status',
    'final_epoch', 'best_checkpoint', 'resume_checkpoints',
    'structured_log', 'order_hashes'}
_FORMAL_AUTHORITY_FIELDS = {
    'schema_version', 'artifact_kind', 'candidate', 'run_init',
    'train_result', 'evaluation'}
_CANONICAL_ARTIFACT_ROLES = {
    'evaluation': Path('evaluate/evaluate.json'),
    'profile': Path('profile/profile.json'),
    'latency': Path('latency/latency.json'),
    'smoke': Path('smoke-stage-a/smoke.json'),
    'formal_authority': Path('formal/formal-authority.json'),
}
_PUBLIC_EVALUATION_PROTOCOL_FIELDS = {
    'dataset', 'split', 'complete_split', 'batch_size',
    'authority_path', 'authority_sha256', 'authority_image_count',
    'authority_annotation_count', 'authority_detection_count',
    'inventory_authority_sha256', 'annotation_authority_sha256',
    'detection_authority_sha256', 'image_corpus_digest_algorithm',
    'image_corpus_authority_sha256', 'image_corpus_sha256',
    'inventory_annotation_archive_sha256',
    'inventory_image_archive_sha256', 'annotation_sha256',
    'detection_sha256', 'inventory_detection_sha256',
    'annotation_image_count', 'annotation_record_count',
    'detection_record_count', 'verified_image_count', 'source_config',
    'checkpoint', 'data_inventory', 'inventory_projection',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact_root(
        repository_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty relative path')
    raw = value
    relative = Path(raw)
    if (
            relative.is_absolute()
            or any(part in {'', '.', '..'} for part in raw.split('/'))
            or relative.parts[:2] != ('work_dirs', 'optimization')):
        raise ValueError(f'{label} must be a safe optimization artifact root')
    root = Path(repository_root).resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} must not contain symlinks')
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root / 'work_dirs/optimization')
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes authority') from error
    if not resolved.is_dir():
        raise ValueError(f'{label} must be a directory')
    return resolved


def _roots(value: Mapping[int, str], *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} roots must be a mapping')
    seeds = tuple(sorted(value))
    if seeds not in ((0, 1, 2), (0, 1, 2, 3, 4)):
        raise ValueError(f'{label} roots require exact seeds 0..2 or 0..4')
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f'{label} root seed keys are invalid')
    return seeds


def _artifact_hashes(result: Any, artifact_root: Path) -> dict[str, str]:
    paths = getattr(result, 'artifact_paths', None)
    if not isinstance(paths, Mapping) or set(paths) < {
            'evaluation', 'profile', 'latency'}:
        raise ValueError('public CandidateResult artifact paths are incomplete')
    normalized_paths = dict(paths)
    formal_path = artifact_root / _CANONICAL_ARTIFACT_ROLES[
        'formal_authority']
    if formal_path.exists() or formal_path.is_symlink():
        normalized_paths.setdefault('formal_authority', formal_path)
    hashes = {}
    for role, value in sorted(normalized_paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError('public CandidateResult artifact role is invalid')
        if role not in _CANONICAL_ARTIFACT_ROLES:
            raise ValueError('public CandidateResult artifact role is not canonical')
        raw = str(value)
        lexical = Path(raw)
        if any(part in {'.', '..'} for part in lexical.parts):
            raise ValueError(
                'public CandidateResult artifact path has a lexical alias')
        if not lexical.is_absolute():
            lexical = artifact_root / lexical
        try:
            relative = lexical.absolute().relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                'public CandidateResult artifact escapes its root') from error
        if relative != _CANONICAL_ARTIFACT_ROLES[role]:
            raise ValueError(
                'public CandidateResult artifact path is not canonical')
        cursor = artifact_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(
                    'public CandidateResult artifact path contains a symlink')
        try:
            path = cursor.resolve(strict=True)
            path.relative_to(artifact_root)
        except (OSError, ValueError) as error:
            raise ValueError(
                'public CandidateResult artifact is missing or escapes') from error
        if not path.is_file():
            raise ValueError('public CandidateResult artifact is not regular')
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f'{label} must be a lowercase SHA-256')
    return value


def _json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'{label} is invalid JSON: {error}') from error
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _tracked_bytes(
        repository_root: Path, commit: str, relative: str, *,
        label: str) -> bytes:
    try:
        value = subprocess.run(
            ['git', 'show', f'{commit}:{relative}'], cwd=repository_root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f'{label} is not tracked at the recorded commit') \
            from error
    return value


def _repository_file(
        repository_root: Path, value: object, *, label: str,
        expected: Path | None = None) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} path is invalid')
    raw = value
    relative = Path(raw)
    if (
            relative.is_absolute()
            or any(part in {'', '.', '..'} for part in raw.split('/'))
            or (expected is not None and relative != expected)):
        raise ValueError(f'{label} path is not canonical')
    cursor = repository_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path contains a symlink')
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} is missing or escapes authority') from error
    if not resolved.is_file():
        raise ValueError(f'{label} is not a regular file')
    return resolved, relative.as_posix()


def _file_binding(
        value: object, *, repository_root: Path, label: str,
        expected: Path | None = None) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {'path', 'sha256'}:
        raise ValueError(f'{label} binding is invalid')
    path, relative = _repository_file(
        repository_root, value.get('path'), label=label, expected=expected)
    digest = _digest(value.get('sha256'), label=f'{label} hash')
    if _sha256(path) != digest:
        raise ValueError(f'{label} hash mismatch')
    return path, {'path': relative, 'sha256': digest}


def _candidate_row_sha256(manifest: Path, candidate_id: str, seed: int) -> str:
    value = _json_file(manifest, label='formal source candidate manifest')
    rows = value.get('candidates')
    matches = [
        row for row in rows if isinstance(row, Mapping)
        and row.get('id') == candidate_id and row.get('seed') == seed
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise ValueError('formal source candidate row is not unique')
    return canonical_json_sha256(matches[0])


def _strict_public_stage_c_run_authority(
        *, repository_root: Path, manifest_path: Path, artifact_root: Path,
        role: str, seed: int) -> dict[str, Any]:
    """Consume the one strict public Stage-C v2 schema or fail closed.

    Binary Q/K is a derivative of the approved ``baseline``/``no_pif``
    experiment.  It must not invent a third role in that public manifest.  The
    derivative will remain inadmissible until its own strict authority schema
    is integrated and binds one of these authenticated parent runs.
    """
    try:
        from .formal_schema import (
            FormalManifestError, load_formal_manifest,
            load_formal_run_init, load_formal_train_result)
    except ImportError as error:
        raise ValueError(
            'strict public Stage-C schema v2 integration dependency is '
            'unavailable') from error

    root = Path(repository_root).resolve(strict=True)
    try:
        manifest = load_formal_manifest(
            manifest_path, repository_root=root)
    except (FormalManifestError, OSError, ValueError) as error:
        raise ValueError(
            f'strict public Stage-C schema v2 integration rejected the '
            f'manifest: {error}') from error
    if role not in {'baseline', 'no_pif'}:
        raise ValueError(
            'strict public Stage-C schema v2 integration has no Binary '
            'candidate role; an exact derivative authority is required')
    matches = [
        run for run in manifest.runs
        if run.role == role and run.seed == seed]
    if len(matches) != 1:
        raise ValueError(
            'strict public Stage-C schema v2 run authority is missing')
    run = matches[0]
    if root / run.output_root != artifact_root:
        raise ValueError(
            'strict public Stage-C schema v2 output authority mismatch')
    run_init_path = artifact_root / 'run-init.json'
    train_result_path = artifact_root / 'train-result.json'
    try:
        run_init = load_formal_run_init(
            run_init_path, repository_root=root)
        train_result = load_formal_train_result(
            train_result_path, repository_root=root, verify_files=True)
    except (FormalManifestError, OSError, ValueError) as error:
        raise ValueError(
            f'strict public Stage-C run-init/train-result relationship is '
            f'invalid: {error}') from error
    if (
            run_init.run_id != run.run_id
            or run_init.role != run.role
            or run_init.seed != run.seed
            or run_init.config != run.config
            or run_init.config_closure_sha256 != run.config_sha256
            or run_init.output_root != run.output_root
            or train_result.run_id != run.run_id
            or train_result.role != run.role
            or train_result.seed != run.seed
            or train_result.output_root != run.output_root):
        raise ValueError(
            'strict public Stage-C run/init/result authority differs')
    return {
        'run_id': run.run_id,
        'role': run.role,
        'seed': run.seed,
        'conditional': run.conditional,
        'config': run.config.as_posix(),
        'config_sha256': run.config_sha256,
        'initialization_id': run.initialization_id,
        'output_root': run.output_root.as_posix(),
    }


def _resolved_protocol_semantics(
        value: Mapping[str, Any], protocol: Mapping[str, Any], *,
        seed: int = 0) -> dict[str, Any]:
    canonical_protocol = {
        'epochs': 300,
        'effective_batch_size': 128,
        'per_device_batch_size': 128,
        'world_size': 1,
        'accumulation_steps': 1,
        'worker_count': 2,
        'persistent_workers': False,
        'deterministic': True,
        'evaluator': 'mmpose.CocoMetric',
        'tta_modes': {'flip': True, 'no_flip': False},
    }
    if any(
            protocol.get(name) != expected
            for name, expected in canonical_protocol.items()):
        raise ValueError(
            'formal sidecar contradicts the canonical formal protocol')
    train_cfg = value.get('train_cfg')
    dataloader = value.get('train_dataloader')
    randomness = value.get('randomness')
    optimizer = value.get('optim_wrapper')
    schedulers = value.get('param_scheduler')
    val_loader = value.get('val_dataloader')
    test_loader = value.get('test_dataloader')
    val_evaluator = value.get('val_evaluator')
    test_evaluator = value.get('test_evaluator')
    model = value.get('model')
    expected_detection = (
        'data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json')
    expected_dataset = {
        'ann_file': 'annotations/person_keypoints_val2017.json',
        'bbox_file': expected_detection,
    }
    expected_evaluator = {
        'type': 'CocoMetric',
        'ann_file': 'data/coco/annotations/person_keypoints_val2017.json',
    }
    expected_schedulers = [
        {
            'type': 'LinearLR', 'begin': 0, 'end': 500,
            'start_factor': 0.001, 'by_epoch': False,
        },
        {
            'type': 'MultiStepLR', 'begin': 0, 'end': 300,
            'milestones': [200, 260], 'gamma': 0.1,
            'by_epoch': True,
        },
    ]
    val_dataset = val_loader.get('dataset') \
        if isinstance(val_loader, Mapping) else None
    test_dataset = test_loader.get('dataset') \
        if isinstance(test_loader, Mapping) else None
    optimizer_value = optimizer.get('optimizer') \
        if isinstance(optimizer, Mapping) else None
    test_cfg = model.get('test_cfg') if isinstance(model, Mapping) else None
    expected_worker_init = {'type': 'mambapose_seed_worker'}
    expected_train_sampler = {
        'type': 'DefaultSampler', 'shuffle': True,
        'round_up': True, 'seed': seed,
    }
    expected_eval_sampler = {
        'type': 'DefaultSampler', 'shuffle': False,
        'round_up': False, 'seed': seed,
    }
    if (
            not isinstance(train_cfg, Mapping)
            or {
                'max_epochs': train_cfg.get('max_epochs'),
                'val_interval': train_cfg.get('val_interval'),
            } != {'max_epochs': 300, 'val_interval': 5}
            or not isinstance(dataloader, Mapping)
            or dataloader.get('batch_size') != 128
            or dataloader.get('num_workers') != 2
            or dataloader.get('persistent_workers') is not False
            or _plain(dataloader.get('worker_init_fn')) !=
            expected_worker_init
            or _plain(dataloader.get('sampler')) != expected_train_sampler
            or not isinstance(randomness, Mapping)
            or randomness.get('seed') != seed
            or randomness.get('deterministic') is not True
            or optimizer_value != {'type': 'Adam', 'lr': 1e-3}
            or (
                isinstance(optimizer, Mapping)
                and 'accumulative_counts' in optimizer)
            or value.get('auto_scale_lr') is not None
            or _plain(schedulers) != expected_schedulers
            or value.get('data_root') != 'data/coco/'
            or not isinstance(val_loader, Mapping)
            or val_loader.get('batch_size') != 64
            or val_loader.get('num_workers') != 2
            or val_loader.get('persistent_workers') is not False
            or val_loader.get('drop_last') is not False
            or _plain(val_loader.get('worker_init_fn')) != expected_worker_init
            or _plain(val_loader.get('sampler')) != expected_eval_sampler
            or not isinstance(val_dataset, Mapping)
            or {
                'ann_file': val_dataset.get('ann_file'),
                'bbox_file': val_dataset.get('bbox_file'),
            } != expected_dataset
            or not isinstance(test_loader, Mapping)
            or test_loader.get('batch_size') != 64
            or test_loader.get('num_workers') != 2
            or test_loader.get('persistent_workers') is not False
            or test_loader.get('drop_last') is not False
            or _plain(test_loader.get('worker_init_fn')) != expected_worker_init
            or _plain(test_loader.get('sampler')) != expected_eval_sampler
            or not isinstance(test_dataset, Mapping)
            or {
                'ann_file': test_dataset.get('ann_file'),
                'bbox_file': test_dataset.get('bbox_file'),
            } != expected_dataset
            or _plain(val_evaluator) != expected_evaluator
            or _plain(test_evaluator) != expected_evaluator
            or _plain(test_cfg) != {
                'flip_test': True, 'flip_mode': 'heatmap',
                'shift_heatmap': True,
            }):
        raise ValueError(
            'formal resolved config contradicts the canonical formal protocol')
    return {
        **canonical_protocol,
        'optimizer': {'type': 'Adam', 'lr': 1e-3},
        'param_scheduler': expected_schedulers,
        'auto_scale_lr': False,
        'evaluation': {
            'config_evaluator': expected_evaluator,
            'validation_annotations': expected_dataset['ann_file'],
            'detections': expected_detection,
            'test_cfg': _plain(test_cfg),
        },
    }


def _paired_resolved_config(value: Mapping[str, Any]) -> dict[str, Any]:
    paired = _plain(value)
    for name in (
            'experiment_id', 'formal_role', 'formal_run_id',
            'numeric_optimization', 'work_dir'):
        paired.pop(name, None)
    try:
        tokenpose = paired['model']['head']['tokenpose_cfg']
    except (KeyError, TypeError):
        tokenpose = None
    if isinstance(tokenpose, dict):
        tokenpose.pop('pif_mode', None)
        tokenpose.pop('qk_mode', None)
    return paired


def _identity(
        value: object, *, result: Any, role: str,
        label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise ValueError(f'{label} candidate identity is invalid')
    expected = {
        'candidate_id': result.candidate_id,
        'candidate_kind': result.candidate_kind,
        'route': result.route,
        'seed': result.seed,
        'role': role,
    }
    if dict(value) != expected:
        raise ValueError(f'{label} candidate identity mismatch')
    return expected


def _result_mode_authority(result: Any, *, mode: str) -> dict[str, str]:
    protocol = getattr(result, 'protocol', None)
    determinism = getattr(result, 'determinism', None)
    if not isinstance(protocol, Mapping) or not isinstance(determinism, Mapping):
        raise ValueError(f'formal {mode} evaluation authority is incomplete')
    if (
            getattr(result, 'flip_test', None) is not (mode == 'flip')
            or set(protocol) != _PUBLIC_EVALUATION_PROTOCOL_FIELDS
            or protocol.get('dataset') != 'coco'
            or protocol.get('split') != 'val2017'
            or protocol.get('complete_split') is not True):
        raise ValueError(
            f'formal {mode} public evaluation protocol is invalid')
    return {
        'protocol_sha256': canonical_json_sha256(_plain(protocol)),
        'determinism_sha256': canonical_json_sha256(_plain(determinism)),
    }


def _validate_formal_authority(
        *, repository_root: Path, artifact_root: Path, flip: Any,
        no_flip: Any, role: str) -> dict[str, Any]:
    """Consume strict public Stage-C run/evaluation authority without weights."""
    from mmengine.config import Config
    from .numeric_source import validate_numeric_config_closure

    root = Path(repository_root).resolve(strict=True)
    authority_relative = artifact_root.relative_to(root) / (
        'formal/formal-authority.json')
    authority_path, _ = _repository_file(
        root, authority_relative.as_posix(), label='formal Pareto authority',
        expected=authority_relative)
    authority_binding = {
        'path': authority_relative.as_posix(),
        'sha256': _sha256(authority_path),
    }
    authority = _json_file(authority_path, label='formal Pareto authority')
    if (
            set(authority) != _FORMAL_AUTHORITY_FIELDS
            or authority.get('schema_version') != 1
            or authority.get('artifact_kind') !=
            'mambapose-formal-stage-c-pareto-authority'):
        raise ValueError('formal Pareto authority identity is invalid')
    identity = _identity(
        authority.get('candidate'), result=flip, role=role,
        label='formal Pareto authority')
    if (
            flip.candidate_id != no_flip.candidate_id
            or flip.candidate_kind != no_flip.candidate_kind
            or flip.route != no_flip.route or flip.seed != no_flip.seed):
        raise ValueError('formal dual-mode CandidateResult identity differs')

    run_expected = artifact_root.relative_to(root) / 'formal/run-init.json'
    train_expected = artifact_root.relative_to(root) / 'formal/train-result.json'
    run_path, run_binding = _file_binding(
        authority.get('run_init'), repository_root=root,
        label='formal run-init', expected=run_expected)
    train_path, train_binding = _file_binding(
        authority.get('train_result'), repository_root=root,
        label='formal train result', expected=train_expected)
    run = _json_file(run_path, label='formal run-init')
    if (
            set(run) != _RUN_INIT_FIELDS or run.get('schema_version') != 1
            or run.get('artifact_kind') !=
            'mambapose-formal-stage-c-pareto-run-init'):
        raise ValueError('formal run-init identity is invalid')
    _identity(run.get('candidate'), result=flip, role=role,
              label='formal run-init')

    source = run.get('source')
    source_fields = _SOURCE_FIELDS | {
        'candidate_row_sha256', 'formal_manifest_path',
        'formal_manifest_sha256'}
    result_source = getattr(flip, 'source', None)
    no_flip_source = getattr(no_flip, 'source', None)
    if (
            not isinstance(source, Mapping) or set(source) != source_fields
            or not isinstance(result_source, Mapping)
            or set(result_source) != _SOURCE_FIELDS
            or not isinstance(no_flip_source, Mapping)
            or _plain(no_flip_source) != _plain(result_source)
            or {name: source[name] for name in _SOURCE_FIELDS} !=
            _plain(result_source)):
        raise ValueError('formal run-init source authority mismatch')
    commit = source.get('git_commit')
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError('formal source commit is invalid')
    coco_authority, _ = _repository_file(
        root, source.get('authority_path'), label='formal COCO authority')
    if (
            _sha256(coco_authority) != _digest(
                source.get('authority_sha256'),
                label='formal COCO authority hash')
            or coco_authority.read_bytes() != _tracked_bytes(
                root, commit, source['authority_path'],
                label='formal COCO authority')):
        raise ValueError('formal COCO source authority mismatch')
    manifest, _ = _repository_file(
        root, source.get('manifest_path'), label='formal candidate manifest')
    if (
            _sha256(manifest) != _digest(
                source.get('manifest_sha256'), label='formal manifest hash')
            or manifest.read_bytes() != _tracked_bytes(
                root, commit, source['manifest_path'],
                label='formal candidate manifest')
            or source.get('candidate_row_sha256') !=
            _candidate_row_sha256(manifest, flip.candidate_id, flip.seed)):
        raise ValueError('formal source manifest/candidate row mismatch')
    formal_manifest, _ = _repository_file(
        root, source.get('formal_manifest_path'),
        label='formal Stage-C manifest',
        expected=Path('optimization/formal_stage_c.json'))
    formal_manifest_value = _json_file(
        formal_manifest, label='formal Stage-C manifest')
    if (
            _sha256(formal_manifest) != _digest(
                source.get('formal_manifest_sha256'),
                label='formal Stage-C manifest hash')
            or formal_manifest.read_bytes() != _tracked_bytes(
                root, commit, source['formal_manifest_path'],
                label='formal Stage-C manifest')
            or formal_manifest_value.get('schema_version') != 2
            or formal_manifest_value.get('experiment_id') !=
            'mambapose-formal-stage-c'):
        raise ValueError('formal Stage-C manifest authority mismatch')

    initialization = run.get('initialization')
    if not isinstance(initialization, Mapping) or set(initialization) != {
            'id', 'path', 'sha256'}:
        raise ValueError('formal initialization authority is invalid')
    init_path, _ = _repository_file(
        root, initialization.get('path'), label='formal initialization')
    if (
            initialization.get('id') != 'vmamba-t-imagenet-262'
            or initialization.get('path') !=
            'pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'
            or _sha256(init_path) != _digest(
                initialization.get('sha256'), label='formal initialization')):
        raise ValueError('formal initialization authority mismatch')

    config = run.get('config')
    config_fields = {
        'path', 'config_closure', 'resolved_config_sha256',
        'paired_base_config_path', 'paired_base_config_closure'}
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ValueError('formal config authority is invalid')
    config_path, config_relative = _repository_file(
        root, config.get('path'), label='formal config')
    if (
            config_relative != result_source['config_path']
            or _sha256(config_path) != result_source['config_sha256']
            or config_path.read_bytes() != _tracked_bytes(
                root, commit, config_relative, label='formal config')):
        raise ValueError('formal config source binding mismatch')
    try:
        config_closure = list(validate_numeric_config_closure(
            root, Path(config_relative), git_commit=commit))
        base_path, base_relative = _repository_file(
            root, config.get('paired_base_config_path'),
            label='formal paired base config')
        del base_path
        base_closure = list(validate_numeric_config_closure(
            root, Path(base_relative), git_commit=commit))
        resolved_config = Config.fromfile(config_path)
        resolved = resolved_config.dump()
        resolved_value = _plain(resolved_config.to_dict())
    except ValueError as error:
        raise ValueError(f'formal config closure is invalid: {error}') from error
    if (
            config.get('config_closure') != config_closure
            or config.get('paired_base_config_closure') != base_closure
            or config.get('resolved_config_sha256') != hashlib.sha256(
                    resolved.encode('utf-8')).hexdigest()):
        raise ValueError('formal config closure/resolved identity mismatch')
    formal_run = _strict_public_stage_c_run_authority(
        repository_root=root, manifest_path=formal_manifest,
        artifact_root=artifact_root, role=role, seed=flip.seed)

    protocol = run.get('protocol')
    protocol_fields = {
        'epochs', 'effective_batch_size', 'per_device_batch_size',
        'world_size', 'accumulation_steps', 'worker_count',
        'persistent_workers', 'deterministic',
        'environment_inventory_sha256', 'evaluator', 'tta_modes',
        'data_authority'}
    data = protocol.get('data_authority') \
        if isinstance(protocol, Mapping) else None
    if (
            not isinstance(protocol, Mapping) or set(protocol) != protocol_fields
            or protocol.get('epochs') != 300
            or protocol.get('effective_batch_size') != 128
            or protocol.get('per_device_batch_size') != 128
            or protocol.get('world_size') != 1
            or protocol.get('accumulation_steps') != 1
            or protocol.get('worker_count') != 2
            or protocol.get('persistent_workers') is not False
            or protocol.get('deterministic') is not True
            or protocol.get('evaluator') != 'mmpose.CocoMetric'
            or protocol.get('tta_modes') != {
                'flip': True, 'no_flip': False}
            or not isinstance(data, Mapping) or set(data) != {
                'authority_path', 'authority_sha256',
                'data_inventory_sha256', 'detections_sha256'}
            or not all(
                isinstance(protocol.get(name), str)
                and _SHA256.fullmatch(protocol[name])
                for name in ('environment_inventory_sha256',))):
        raise ValueError('formal 300-epoch protocol authority is invalid')
    resolved_protocol = _resolved_protocol_semantics(
        resolved_value, protocol, seed=flip.seed)
    paired_config_sha256 = canonical_json_sha256(
        _paired_resolved_config(resolved_value))
    flip_protocol = getattr(flip, 'protocol', None)
    no_flip_protocol = getattr(no_flip, 'protocol', None)
    flip_provenance = getattr(flip, 'provenance', None)
    no_flip_provenance = getattr(no_flip, 'provenance', None)
    paired_provenance_fields = {
        'checkpoint_sha256', 'data_inventory_sha256', 'git_commit'}
    if (
            not isinstance(flip_protocol, Mapping)
            or not isinstance(no_flip_protocol, Mapping)
            or not isinstance(flip_provenance, Mapping)
            or not isinstance(no_flip_provenance, Mapping)
            or _plain(no_flip_protocol) != _plain(flip_protocol)
            or any(
                no_flip_provenance.get(name) != flip_provenance.get(name)
                for name in paired_provenance_fields)
            or data.get('authority_path') != flip_protocol.get('authority_path')
            or data.get('authority_sha256') !=
            flip_protocol.get('authority_sha256')
            or data.get('data_inventory_sha256') !=
            flip_provenance.get('data_inventory_sha256')
            or data.get('detections_sha256') !=
            flip_protocol.get('detection_sha256')):
        raise ValueError('formal COCO/detections authority mismatch')
    for mode, result in (('flip', flip), ('no_flip', no_flip)):
        determinism = getattr(result, 'determinism', None)
        if (
                not isinstance(determinism, Mapping)
                or any(determinism.get(name) != flip.seed for name in (
                    'python_seed', 'numpy_seed', 'torch_seed'))
                or determinism.get('worker_count') != protocol['worker_count']
                or determinism.get('persistent_workers') is not False):
            raise ValueError(f'formal {mode} determinism authority mismatch')

    train = _json_file(train_path, label='formal train result')
    if (
            set(train) != _TRAIN_RESULT_FIELDS
            or train.get('schema_version') != 1
            or train.get('artifact_kind') !=
            'mambapose-formal-stage-c-pareto-train-result'):
        raise ValueError('formal train result identity is invalid')
    _identity(train.get('candidate'), result=flip, role=role,
              label='formal train result')
    _, recorded_run = _file_binding(
        train.get('run_init'), repository_root=root,
        label='formal train-result run-init', expected=run_expected)
    order_hashes = train.get('order_hashes')
    if (
            recorded_run != run_binding
            or train.get('status') != 'complete'
            or train.get('final_epoch') != 300
            or not isinstance(order_hashes, list) or len(order_hashes) != 300
            or any(not isinstance(item, str) or not _SHA256.fullmatch(item)
                   for item in order_hashes)
            or not isinstance(train.get('resume_checkpoints'), list)
            or len(train['resume_checkpoints']) != 2):
        raise ValueError('formal completed 300-epoch train authority is invalid')
    best_path, best = _file_binding(
        train.get('best_checkpoint'), repository_root=root,
        label='formal best checkpoint')
    try:
        best_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError('formal best checkpoint escapes run root') from error
    for index, binding in enumerate(train['resume_checkpoints']):
        resume_path, _ = _file_binding(
            binding, repository_root=root,
            label=f'formal resume checkpoint {index}')
        try:
            resume_path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError('formal resume checkpoint escapes run root') from error
    log_path, _ = _file_binding(
        train.get('structured_log'), repository_root=root,
        label='formal structured log')
    try:
        log_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError('formal structured log escapes run root') from error
    provenance = getattr(flip, 'provenance', None)
    if (
            not isinstance(provenance, Mapping)
            or provenance.get('checkpoint_sha256') != best['sha256']
            or flip_protocol.get('checkpoint') != best['path']):
        raise ValueError('formal checkpoint/evaluation lineage mismatch')

    evaluation = authority.get('evaluation')
    evaluation_fields = {'artifact', 'evaluator', 'modes'}
    evaluation_expected = artifact_root.relative_to(root) / (
        'evaluate/evaluate.json')
    _, evaluation_binding = _file_binding(
        evaluation.get('artifact') if isinstance(evaluation, Mapping) else None,
        repository_root=root, label='formal evaluation',
        expected=evaluation_expected)
    expected_modes = {
        mode: _result_mode_authority(result, mode=mode)
        for mode, result in (('flip', flip), ('no_flip', no_flip))}
    expected_authority = {
        'schema_version': 1,
        'artifact_kind': 'mambapose-formal-stage-c-pareto-authority',
        'candidate': identity,
        'run_init': run_binding,
        'train_result': train_binding,
        'evaluation': {
            'artifact': evaluation_binding,
            'evaluator': 'mmpose.CocoMetric',
            'modes': expected_modes,
        },
    }
    if (
            not isinstance(evaluation, Mapping)
            or set(evaluation) != evaluation_fields
            or _plain(authority) != expected_authority):
        raise ValueError(
            'formal Pareto authority disagrees with public Stage-C evidence')
    return {
        'authority': authority_binding,
        'source_pair': {
            name: source[name] for name in (
                'git_commit', 'manifest_path', 'manifest_sha256',
                'authority_path', 'authority_sha256',
                'formal_manifest_path', 'formal_manifest_sha256')},
        'initialization': _plain(initialization),
        'paired_base_config_path': config['paired_base_config_path'],
        'paired_base_config_closure': _plain(
            config['paired_base_config_closure']),
        'formal_manifest_run': formal_run,
        'resolved_protocol_semantics': resolved_protocol,
        'paired_resolved_config_sha256': paired_config_sha256,
        'protocol': _plain(protocol),
        'order_hashes': list(order_hashes),
        'best_checkpoint': best,
    }


def _finite_ap(value: object, *, label: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0):
        raise ValueError(f'{label} AP is invalid')
    return float(value)


def _load_seed_rows(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> tuple[list[dict[str, Any]], dict]:
    seeds = _roots(baseline_roots, label='baseline')
    if _roots(candidate_roots, label='candidate') != seeds:
        raise ValueError('baseline and candidate seed roots disagree')
    rows = []
    common_operation = None
    common_formal_experiment = None
    for seed in seeds:
        baseline_root = _artifact_root(
            repository_root, baseline_roots[seed],
            label=f'seed {seed} baseline root')
        candidate_root = _artifact_root(
            repository_root, candidate_roots[seed],
            label=f'seed {seed} candidate root')
        baseline = CandidateResult.from_artifacts(baseline_root, mode='flip')
        candidate = CandidateResult.from_artifacts(candidate_root, mode='flip')
        baseline_no_flip = CandidateResult.from_artifacts(
            baseline_root, mode='no_flip')
        candidate_no_flip = CandidateResult.from_artifacts(
            candidate_root, mode='no_flip')
        if (
                baseline.candidate_id != 'full-s-v1'
                or baseline.candidate_kind != 'float'
                or baseline.route != 'baseline'
                or baseline.seed != seed
                or baseline.flip_test is not True):
            raise ValueError('public baseline CandidateResult identity is invalid')
        if (
                candidate.candidate_id != candidate_id
                or candidate.candidate_kind != 'binary-qk'
                or candidate.route != 'ssm-quant-pwl'
                or candidate.seed != seed
                or candidate.flip_test is not True):
            raise ValueError('public Binary CandidateResult identity is invalid')
        baseline_source = getattr(baseline, 'source', None)
        candidate_source = getattr(candidate, 'source', None)
        if (
                not isinstance(baseline_source, Mapping)
                or not isinstance(candidate_source, Mapping)
                or baseline_source.get('git_commit') !=
                candidate_source.get('git_commit')
                or baseline_source.get('authority_sha256') !=
                candidate_source.get('authority_sha256')):
            raise ValueError('paired CandidateResult source authority differs')
        baseline_formal = _validate_formal_authority(
            repository_root=repository_root, artifact_root=baseline_root,
            flip=baseline, no_flip=baseline_no_flip, role='baseline')
        candidate_formal = _validate_formal_authority(
            repository_root=repository_root, artifact_root=candidate_root,
            flip=candidate, no_flip=candidate_no_flip, role='candidate')
        paired_fields = {
            'source_pair', 'initialization', 'paired_base_config_path',
            'paired_base_config_closure', 'resolved_protocol_semantics',
            'paired_resolved_config_sha256', 'protocol', 'order_hashes'}
        if any(
                _plain(baseline_formal[name]) !=
                _plain(candidate_formal[name])
                for name in paired_fields):
            raise ValueError(
                'formal paired Stage-C initialization/protocol authority differs')
        paired_formal_sha = canonical_json_sha256({
            'baseline': baseline_formal['authority'],
            'candidate': candidate_formal['authority'],
        })
        formal_experiment = {
            'git_commit': baseline_formal['source_pair']['git_commit'],
            'authority_path': baseline_formal['source_pair']['authority_path'],
            'authority_sha256': baseline_formal[
                'source_pair']['authority_sha256'],
            'formal_manifest_path': baseline_formal[
                'source_pair']['formal_manifest_path'],
            'formal_manifest_sha256': baseline_formal[
                'source_pair']['formal_manifest_sha256'],
            'initialization': baseline_formal['initialization'],
            'paired_base_config_path': baseline_formal[
                'paired_base_config_path'],
            'paired_base_config_closure': baseline_formal[
                'paired_base_config_closure'],
            'resolved_protocol_semantics': baseline_formal[
                'resolved_protocol_semantics'],
            'protocol': baseline_formal['protocol'],
        }
        if common_formal_experiment is None:
            common_formal_experiment = _plain(formal_experiment)
        elif _plain(formal_experiment) != common_formal_experiment:
            raise ValueError(
                'formal Stage-C experiment authority differs across seeds')
        operation = validate_binary_operation_manifest(
            candidate.binary_operation)
        if common_operation is None:
            common_operation = dict(operation)
        elif dict(operation) != common_operation:
            raise ValueError('Binary operation manifest differs across seeds')
        baseline_ap = _finite_ap(
            baseline.metrics.ap, label=f'seed {seed} baseline')
        candidate_ap = _finite_ap(
            candidate.metrics.ap, label=f'seed {seed} candidate')
        rows.append({
            'seed': seed,
            'baseline_root': baseline_roots[seed],
            'candidate_root': candidate_roots[seed],
            'baseline_artifact_sha256': _artifact_hashes(
                baseline, baseline_root),
            'candidate_artifact_sha256': _artifact_hashes(
                candidate, candidate_root),
            'baseline_source_sha256': canonical_json_sha256(baseline_source),
            'candidate_source_sha256': canonical_json_sha256(candidate_source),
            'baseline_formal_authority': baseline_formal['authority'],
            'candidate_formal_authority': candidate_formal['authority'],
            'baseline_formal_manifest_run': baseline_formal[
                'formal_manifest_run'],
            'candidate_formal_manifest_run': candidate_formal[
                'formal_manifest_run'],
            'paired_resolved_config_sha256': baseline_formal[
                'paired_resolved_config_sha256'],
            'paired_formal_authority_sha256': paired_formal_sha,
            'baseline_ap_points': baseline_ap,
            'candidate_ap_points': candidate_ap,
            'ap_drop_points': baseline_ap - candidate_ap,
            'operation_manifest_sha256': canonical_json_sha256(operation),
        })
    assert common_operation is not None
    return rows, common_operation


def _statistics(drops: list[float]) -> dict[str, Any]:
    count = len(drops)
    mean = statistics.fmean(drops)
    stddev = statistics.stdev(drops)
    margin = _T_CRITICAL_95[count] * stddev / math.sqrt(count)
    return {
        'seed_count': count,
        'paired_sample_stddev_points': stddev,
        'paired_95_ci_points': [mean - margin, mean + margin],
        'confidence_method': f'paired-student-t-df-{count - 1}',
    }


def _build_record(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> dict[str, Any]:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('Pareto candidate id must be non-empty')
    rows, operation = _load_seed_rows(
        candidate_id=candidate_id, repository_root=repository_root,
        baseline_roots=baseline_roots, candidate_roots=candidate_roots)
    drops = [float(row['ap_drop_points']) for row in rows]
    statistics_record = _statistics(drops)
    initial_statistics = _statistics(drops[:3])
    initial_ci = initial_statistics['paired_95_ci_points']
    initial_intersects = initial_ci[0] <= 0.1 <= initial_ci[1]
    if len(rows) == 5 and not initial_intersects:
        raise ValueError('seeds 3-4 are only admitted after CI intersection')
    mean_drop = statistics.fmean(drops)
    max_drop = max(drops)
    point_rules = mean_drop < 0.1 and max_drop < 0.3
    requires_more = len(rows) == 3 and initial_intersects
    accuracy_pass = point_rules and not requires_more
    operation_sha = canonical_json_sha256(operation)
    hardware_pass = (
        operation['theoretical_qk_multiplications_replaced'] == 6_489_600
        and operation['bitwise_kernel_present'] is False
        and operation['measured_integer_latency'] is False
        and operation['hardware_claim'] == 'none-software-proxy'
        and operation['binaryattention_reproduction'] is False
        and operation['learnable_attention_bias'] is False)
    if requires_more:
        decision = 'requires-seeds-3-4'
    elif accuracy_pass and hardware_pass:
        decision = 'pareto-eligible'
    else:
        decision = 'rejected'
    return {
        'schema_version': 3,
        'artifact_kind': 'mambapose-final-pareto-record',
        'candidate_id': candidate_id,
        'candidate_kind': 'binary-qk',
        'seeds': rows,
        'statistics': statistics_record,
        'accuracy_gate': {
            'mean_ap_drop_points': mean_drop,
            'max_ap_drop_points': max_drop,
            'mean_limit_exclusive': 0.1,
            'max_limit_exclusive': 0.3,
            'point_rules_passed': point_rules,
            'three_seed_ci_intersects_mean_limit': initial_intersects,
            'passed': accuracy_pass,
        },
        'hardware_evidence': {
            'kind': 'binary-qk-theoretical-operation-replacement',
            'theoretical_qk_multiplications_replaced': 6_489_600,
            'operation_manifest_sha256': operation_sha,
            'reference_scope': operation['reference_scope'],
            'learnable_attention_bias': False,
            'binaryattention_reproduction': False,
            'bitwise_kernel_present': False,
            'measured_integer_latency': False,
            'speedup_claim': 'none-software-proxy',
            'passed': hardware_pass,
        },
        'decision': decision,
    }


def build_final_pareto_record(
        *, candidate_id: str, repository_root: Path,
        baseline_roots: Mapping[int, str],
        candidate_roots: Mapping[int, str]) -> dict[str, Any]:
    """Build Pareto evidence solely from public-valid artifact roots.

    Admission intentionally fails closed until the strict public Stage-C v2
    schema and a separate Binary derivative authority are both integrated.
    """
    record = _build_record(
        candidate_id=candidate_id, repository_root=repository_root,
        baseline_roots=baseline_roots, candidate_roots=candidate_roots)
    validate_final_pareto_record(record, repository_root=repository_root)
    return record


def validate_final_pareto_record(
        value: object, *, repository_root: Path) -> Mapping[str, Any]:
    """Reload every artifact and reject any caller-reported derived value."""
    required = {
        'schema_version', 'artifact_kind', 'candidate_id', 'candidate_kind',
        'seeds', 'statistics', 'accuracy_gate', 'hardware_evidence',
        'decision'}
    if (
            not isinstance(value, Mapping) or set(value) != required
            or value.get('schema_version') != 3
            or value.get('artifact_kind') !=
            'mambapose-final-pareto-record'
            or value.get('candidate_kind') != 'binary-qk'
            or value.get('decision') not in {
                'pareto-eligible', 'requires-seeds-3-4', 'rejected'}
            or not isinstance(value.get('seeds'), list)):
        raise ValueError('final Pareto record identity is invalid')
    try:
        baseline_roots = {
            int(row['seed']): row['baseline_root'] for row in value['seeds']}
        candidate_roots = {
            int(row['seed']): row['candidate_root'] for row in value['seeds']}
        expected = _build_record(
            candidate_id=value['candidate_id'],
            repository_root=repository_root,
            baseline_roots=baseline_roots,
            candidate_roots=candidate_roots)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f'final Pareto public evidence is invalid: {error}') from error
    if dict(value) != expected:
        raise ValueError(
            'final Pareto record disagrees with recomputed public evidence')
    return value
