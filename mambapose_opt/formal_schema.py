"""Strict schemas for formal paired Stage C experiments.

This module intentionally uses only the Python standard library.  Parsing an
experiment contract never imports Torch and never deserializes a checkpoint.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Literal, Mapping


INITIALIZATION_ID = 'vmamba-t-imagenet-262'
INITIALIZATION_PATH = Path('pretrained/vssm_tiny_0230_ckpt_epoch_262.pth')
INITIALIZATION_SHA256 = (
    '09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201')
PRIMARY_SEEDS = (0, 1, 2)
CONDITIONAL_SEEDS = (3, 4)
ALL_SEEDS = PRIMARY_SEEDS + CONDITIONAL_SEEDS
ROLES = ('baseline', 'no_pif')

_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')
_IDENTIFIER = re.compile(r'^[a-z0-9][a-z0-9-]{0,127}$')
_MANIFEST_FIELDS = frozenset({
    'schema_version', 'experiment_id', 'initialization', 'protocol',
    'data_authority', 'prior_artifact', 'runs',
})
_INITIALIZATION_FIELDS = frozenset({'id', 'kind', 'asset'})
_PROTOCOL_FIELDS = frozenset({
    'epochs', 'worker_count', 'persistent_workers', 'effective_batch_size',
    'per_device_batch_size', 'world_size', 'accumulation_steps',
    'primary_seeds', 'conditional_seeds',
})
_RUN_FIELDS = frozenset({
    'run_id', 'role', 'seed', 'conditional', 'config', 'config_sha256',
    'initialization_id', 'output_root',
})
_DATA_FIELDS = frozenset({
    'inventory', 'train_annotations', 'validation_annotations', 'detections',
})
_FILE_BINDING_FIELDS = frozenset({'path', 'sha256'})
_ASSET_BINDING_FIELDS = frozenset({
    'authority_root', 'target_root', 'asset_relative_path', 'sha256'})
_ASSET_AUTHORITIES = MappingProxyType({
    'pretrained': 'canonical-main/pretrained',
    'data': 'canonical-main/data',
    'work_dirs/reproduction': 'canonical-main/work_dirs/reproduction',
})
_PRIOR_FIELDS = frozenset({
    'link_root', 'target_root', 'bundle_manifest_sha256', 'source_commit',
    'unpruned_parent_checkpoint_sha256', 'audit_sha256', 'artifacts',
})
_PRIOR_LINK_ROOT = Path('work_dirs/optimization/prior-stage-b')
_PRIOR_TARGET_ROOT = (
    'canonical-main/work_dirs/optimization/'
    'formal-stage-c-prior/no-pif-seed0')
_PRIOR_SOURCE_COMMIT = 'a6adf6d84f9b51b133b0fca7272c51f728bfb7ac'
_PRIOR_BUNDLE_SHA256 = (
    '522645bdd5612af0a58009b0b30ce1aae22ed4d460c482ba08799c1500b6ed9f')
_PRIOR_PARENT_SHA256 = (
    '28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb')
_PRIOR_AUDIT_SHA256 = (
    'b7382dbaca7294477990c2463bf642f3959762a7b9e54d626e46e45d8bb5f361')
_PRIOR_ARTIFACTS = MappingProxyType({
    'train': (
        'artifacts/train.json',
        'ea276713f214acce1949798445203e7e05026dd4ed7d6e76e77cffde35041819'),
    'runtime_metadata': (
        'artifacts/runtime-metadata.json',
        '99e75ff3a4862f08eba3a7c75e586617aa1438efa8f164a73da093ffe5b672ce'),
    'pruned_checkpoint': (
        'artifacts/pruned-runtime.pth',
        '5797ceaffbf7d369d8eaf8f47b548a79bd43d66dd603933571fa3b88ff28e3db'),
    'profile': (
        'artifacts/profile.json',
        'cb0c933291c28bc341cc5bb62b956c14a76255825321bc0291d4e688fa7aa7a3'),
    'evaluate': (
        'artifacts/evaluate.json',
        '9bcf326417cb0b9ffbf78deb01d19ac09525d9deb491c93189e8b769df0f3bf5'),
    'latency': (
        'artifacts/latency.json',
        '9e7448a6b9ab96eb625c08fe0d6e5e9c3933349da2098cb91ab633faca41b98e'),
    'source_config': (
        'source/no_pif_pruned.py',
        'e2a84676bb489ea4a9bb3f4a68a965afb7d2168efa013880732db6c818139df2'),
    'candidate_manifest': (
        'source/candidates.json',
        '1081b546af0a9f9c95e9ca5e4fffca6f9609f38bf111db6bf47c0c1f5dca3d76'),
    'coco_authority': (
        'source/coco_val2017_authority.json',
        '5d5945c35f9bf59d2ff537c088dad3119ae9d7e5ead0b51a641b20667c8c9edb'),
    'parent_config': (
        'source/coco_s_v1_no_pif.py',
        '706eaa316934d89a446510651d3d7dc3b34278436fa8caa47bdf3c228d1209e1'),
    'independent_audit': (
        'audit/task-4-final-artifact-audit.md',
        _PRIOR_AUDIT_SHA256),
})


class FormalManifestError(ValueError):
    """Raised when formal experiment provenance is incomplete or ambiguous."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class AssetBinding:
    authority_root: str
    target_root: str
    asset_relative_path: Path
    sha256: str

    @property
    def path(self) -> Path:
        return Path(self.authority_root) / self.asset_relative_path

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> AssetBinding:
        document = _require_mapping(value, 'asset binding')
        Path(repository_root).resolve(strict=False)
        _require_exact_fields(
            document, _ASSET_BINDING_FIELDS, 'asset binding')
        authority = document['authority_root']
        if not isinstance(authority, str) or authority not in _ASSET_AUTHORITIES:
            raise FormalManifestError('asset authority_root is not approved')
        target = document['target_root']
        if target != _ASSET_AUTHORITIES[authority]:
            raise FormalManifestError(
                'asset target_root does not match its authority_root')
        relative = _safe_relative_path(
            document['asset_relative_path'], 'asset_relative_path')
        return cls(
            authority_root=authority,
            target_root=target,
            asset_relative_path=relative,
            sha256=_require_sha256(
                document['sha256'], 'asset binding SHA-256'))


@dataclass(frozen=True)
class InitializationAuthority:
    id: str
    kind: Literal['vmamba-backbone']
    asset: AssetBinding

    @property
    def path(self) -> Path:
        return self.asset.path

    @property
    def sha256(self) -> str:
        return self.asset.sha256


@dataclass(frozen=True)
class PriorArtifactAuthority:
    link_root: Path
    target_root: str
    bundle_manifest_sha256: str
    source_commit: str
    unpruned_parent_checkpoint_sha256: str
    audit_sha256: str
    artifacts: Mapping[str, FileBinding]

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> PriorArtifactAuthority:
        document = _require_mapping(value, 'prior artifact authority')
        _require_exact_fields(
            document, _PRIOR_FIELDS, 'prior artifact authority')
        link_root = _safe_relative_path(
            document['link_root'], 'prior artifact link_root')
        if link_root != _PRIOR_LINK_ROOT:
            raise FormalManifestError(
                'prior artifact link_root is not the approved logical root')
        if document['target_root'] != _PRIOR_TARGET_ROOT:
            raise FormalManifestError(
                'prior artifact target_root is not approved')
        if document['source_commit'] != _PRIOR_SOURCE_COMMIT:
            raise FormalManifestError(
                'prior artifact source commit is not approved')
        if (
                document['bundle_manifest_sha256']
                != _PRIOR_BUNDLE_SHA256):
            raise FormalManifestError(
                'prior artifact bundle manifest is not approved')
        if (
                document['unpruned_parent_checkpoint_sha256']
                != _PRIOR_PARENT_SHA256):
            raise FormalManifestError(
                'prior artifact checkpoint roles are invalid')
        if document['audit_sha256'] != _PRIOR_AUDIT_SHA256:
            raise FormalManifestError(
                'prior artifact independent audit is not approved')
        raw_artifacts = _require_mapping(
            document['artifacts'], 'prior artifacts')
        if set(raw_artifacts) != set(_PRIOR_ARTIFACTS):
            raise FormalManifestError(
                'prior artifact roles must be exact')
        artifacts = {
            name: _parse_file_binding(
                item, f'prior artifact {name}',
                repository_root=repository_root)
            for name, item in raw_artifacts.items()}
        for name, binding in artifacts.items():
            expected_path, expected_sha = _PRIOR_ARTIFACTS[name]
            if (
                    binding.path != Path(expected_path)
                    or binding.sha256 != expected_sha):
                if name == 'pruned_checkpoint':
                    raise FormalManifestError(
                        'prior artifact checkpoint roles are invalid')
                raise FormalManifestError(
                    f'prior artifact {name} is not approved')
        if artifacts['independent_audit'].sha256 != document['audit_sha256']:
            raise FormalManifestError(
                'prior artifact audit binding is inconsistent')
        return cls(
            link_root=link_root,
            target_root=_PRIOR_TARGET_ROOT,
            bundle_manifest_sha256=_PRIOR_BUNDLE_SHA256,
            source_commit=_PRIOR_SOURCE_COMMIT,
            unpruned_parent_checkpoint_sha256=_PRIOR_PARENT_SHA256,
            audit_sha256=_PRIOR_AUDIT_SHA256,
            artifacts=MappingProxyType(artifacts),
        )


@dataclass(frozen=True)
class FormalProtocol:
    epochs: int
    worker_count: int
    persistent_workers: bool
    per_device_batch_size: int
    world_size: int
    accumulation_steps: int
    effective_batch_size: int
    primary_seeds: tuple[int, ...]
    conditional_seeds: tuple[int, ...]


@dataclass(frozen=True)
class FormalRunSpec:
    run_id: str
    role: Literal['baseline', 'no_pif']
    seed: int
    conditional: bool
    config: Path
    config_sha256: str
    initialization_id: str
    output_root: Path


@dataclass(frozen=True)
class FormalStageCManifest:
    schema_version: int
    experiment_id: str
    initialization: InitializationAuthority
    protocol: FormalProtocol
    data_authority: Mapping[str, AssetBinding]
    prior_artifact: PriorArtifactAuthority
    runs: tuple[FormalRunSpec, ...]
    repository_root: Path

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> FormalStageCManifest:
        document = _require_mapping(value, 'manifest')
        _require_exact_fields(document, _MANIFEST_FIELDS, 'manifest')
        if document['schema_version'] != 1:
            raise FormalManifestError('schema_version must be 1')
        if document['experiment_id'] != 'mambapose-formal-stage-c':
            raise FormalManifestError(
                'experiment_id must be mambapose-formal-stage-c')

        root = Path(repository_root).resolve(strict=False)
        initialization = _parse_initialization(
            document['initialization'], repository_root=root)
        protocol = _parse_protocol(document['protocol'])
        data_authority = _parse_data_authority(
            document['data_authority'], repository_root=root)
        prior_artifact = PriorArtifactAuthority.from_dict(
            document['prior_artifact'], repository_root=root)

        raw_runs = document['runs']
        if not isinstance(raw_runs, list):
            raise FormalManifestError('runs must be a list')
        runs = tuple(
            _parse_run(item, repository_root=root) for item in raw_runs)
        run_ids = tuple(run.run_id for run in runs)
        if len(run_ids) != len(set(run_ids)):
            raise FormalManifestError('run ids must be unique')
        output_roots = tuple(run.output_root for run in runs)
        if len(output_roots) != len(set(output_roots)):
            raise FormalManifestError('output roots must be unique')
        config_paths = tuple(run.config for run in runs)
        if len(config_paths) != len(set(config_paths)):
            raise FormalManifestError('config paths must be unique')

        expected_matrix = tuple(
            (role, seed) for seed in ALL_SEEDS for role in ROLES)
        actual_matrix = tuple((run.role, run.seed) for run in runs)
        if actual_matrix != expected_matrix:
            raise FormalManifestError(
                'runs must form the exact paired seed matrix')
        for run in runs:
            expected_conditional = run.seed in CONDITIONAL_SEEDS
            if run.conditional is not expected_conditional:
                raise FormalManifestError(
                    f'{run.run_id} has the wrong conditional flag')
            if run.initialization_id != initialization.id:
                raise FormalManifestError(
                    f'{run.run_id} does not use the initialization authority')
            _validate_canonical_run(run)

        return cls(
            schema_version=1,
            experiment_id='mambapose-formal-stage-c',
            initialization=initialization,
            protocol=protocol,
            data_authority=MappingProxyType(data_authority),
            prior_artifact=prior_artifact,
            runs=runs,
            repository_root=root,
        )


@dataclass(frozen=True)
class FormalRunInit:
    manifest_sha256: str
    git_commit: str
    config: Path
    config_closure_sha256: str
    resolved_config_sha256: str
    environment_inventory_sha256: str
    data_authority: Mapping[str, str]
    run_id: str
    role: Literal['baseline', 'no_pif']
    seed: int
    epochs: int
    effective_batch_size: int
    worker_count: int
    persistent_workers: bool
    output_root: Path
    initialization: InitializationAuthority

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> FormalRunInit:
        document = _require_mapping(value, 'run init')
        root = Path(repository_root).resolve(strict=False)
        fields = frozenset({
            'schema_version', 'manifest_sha256', 'source', 'config',
            'environment_inventory_sha256', 'data_authority', 'run',
            'initialization',
        })
        _require_exact_fields(document, fields, 'run init')
        if document['schema_version'] != 1:
            raise FormalManifestError('run init schema_version must be 1')

        source = _require_mapping(document['source'], 'source')
        _require_exact_fields(source, {'git_commit', 'clean_tree'}, 'source')
        commit = source['git_commit']
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise FormalManifestError('source git_commit must be a full commit')
        if source['clean_tree'] is not True:
            raise FormalManifestError('source tree must be clean')

        config = _require_mapping(document['config'], 'config')
        _require_exact_fields(
            config, {'path', 'closure_sha256', 'resolved_sha256'}, 'config')
        config_path = _safe_relative_path(config['path'], 'config path')
        _require_prefix(
            config_path, Path('configs/optimization/formal_stage_c'),
            'config path')
        _validate_internal_relative_path(root, config_path, 'config path')

        data = _require_mapping(document['data_authority'], 'data authority')
        expected_data = frozenset({
            'inventory_sha256', 'train_annotations_sha256',
            'validation_annotations_sha256', 'detections_sha256',
        })
        _require_exact_fields(data, expected_data, 'data authority')
        normalized_data = {
            key: _require_sha256(item, key) for key, item in data.items()}

        run = _require_mapping(document['run'], 'run')
        run_fields = frozenset({
            'run_id', 'role', 'seed', 'epochs', 'effective_batch_size',
            'worker_count', 'persistent_workers', 'output_root',
        })
        _require_exact_fields(run, run_fields, 'run')
        run_id, role, seed = _parse_run_identity(run)
        epochs = _require_int(run['epochs'], 'epochs', minimum=1)
        if epochs != 300:
            raise FormalManifestError('formal training requires 300 epochs')
        worker_count = _require_int(
            run['worker_count'], 'worker_count', minimum=0)
        if run['persistent_workers'] is not False:
            raise FormalManifestError('persistent workers must be disabled')
        output_root = _safe_formal_output_root(
            run['output_root'], 'output_root')
        _validate_internal_relative_path(root, output_root, 'output_root')
        if output_root.name != run_id:
            raise FormalManifestError('output_root must end in the run id')

        return cls(
            manifest_sha256=_require_sha256(
                document['manifest_sha256'], 'manifest_sha256'),
            git_commit=commit,
            config=config_path,
            config_closure_sha256=_require_sha256(
                config['closure_sha256'], 'config closure SHA-256'),
            resolved_config_sha256=_require_sha256(
                config['resolved_sha256'], 'resolved config SHA-256'),
            environment_inventory_sha256=_require_sha256(
                document['environment_inventory_sha256'],
                'environment inventory SHA-256'),
            data_authority=MappingProxyType(normalized_data),
            run_id=run_id,
            role=role,
            seed=seed,
            epochs=epochs,
            effective_batch_size=_require_int(
                run['effective_batch_size'], 'effective_batch_size', minimum=1),
            worker_count=worker_count,
            persistent_workers=False,
            output_root=output_root,
            initialization=_parse_initialization(
                document['initialization'], repository_root=root),
        )


@dataclass(frozen=True)
class FormalTrainResult:
    run_init_sha256: str
    initialization: InitializationAuthority
    run_id: str
    role: Literal['baseline', 'no_pif']
    seed: int
    output_root: Path
    best_checkpoint: FileBinding
    resume_checkpoints: tuple[FileBinding, FileBinding]
    structured_log: FileBinding
    order_hashes: tuple[str, ...]
    final_epoch: int
    status: Literal['complete']

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> FormalTrainResult:
        document = _require_mapping(value, 'train result')
        root = Path(repository_root).resolve(strict=False)
        fields = frozenset({
            'schema_version', 'run_init_sha256', 'initialization', 'run',
            'best_checkpoint', 'resume_checkpoints', 'structured_log',
            'order_hashes', 'final_epoch', 'status',
        })
        _require_exact_fields(document, fields, 'train result')
        if document['schema_version'] != 1:
            raise FormalManifestError('train result schema_version must be 1')
        initialization = _parse_initialization(
            document['initialization'], repository_root=root)

        run = _require_mapping(document['run'], 'run')
        _require_exact_fields(
            run, {'run_id', 'role', 'seed', 'output_root'}, 'run')
        run_id, role, seed = _parse_run_identity(run)
        output_root = _safe_formal_output_root(
            run['output_root'], 'output_root')
        _validate_internal_relative_path(root, output_root, 'output_root')
        if output_root.name != run_id:
            raise FormalManifestError('output_root must end in the run id')

        best = _parse_file_binding(
            document['best_checkpoint'], 'best_checkpoint',
            repository_root=root)
        raw_resume = document['resume_checkpoints']
        if not isinstance(raw_resume, list) or len(raw_resume) != 2:
            raise FormalManifestError(
                'train result requires exactly two resume checkpoints')
        resume = tuple(
            _parse_file_binding(
                item, f'resume_checkpoints[{index}]',
                repository_root=root)
            for index, item in enumerate(raw_resume))
        log = _parse_file_binding(
            document['structured_log'], 'structured_log',
            repository_root=root)
        for binding in (best, *resume, log):
            if binding.path.parent != output_root:
                raise FormalManifestError(
                    'all train outputs must be under the declared output root')

        raw_hashes = document['order_hashes']
        if not isinstance(raw_hashes, list) or len(raw_hashes) != 300:
            raise FormalManifestError('train result requires 300 order hashes')
        order_hashes = tuple(
            _require_sha256(item, f'order_hashes[{index}]')
            for index, item in enumerate(raw_hashes))
        if document['final_epoch'] != 300:
            raise FormalManifestError('final epoch must be 300')
        if document['status'] != 'complete':
            raise FormalManifestError('train result status must be complete')

        return cls(
            run_init_sha256=_require_sha256(
                document['run_init_sha256'], 'run_init_sha256'),
            initialization=initialization,
            run_id=run_id,
            role=role,
            seed=seed,
            output_root=output_root,
            best_checkpoint=best,
            resume_checkpoints=(resume[0], resume[1]),
            structured_log=log,
            order_hashes=order_hashes,
            final_epoch=300,
            status='complete',
        )


def load_formal_manifest(
        path: Path | str, *,
        repository_root: Path | str) -> FormalStageCManifest:
    """Load and authenticate a formal manifest without loading checkpoints."""
    root = Path(repository_root).resolve(strict=True)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    _validate_tracked_path(root, source, 'formal manifest')
    try:
        value = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalManifestError(f'cannot load formal manifest: {error}') from error
    manifest = FormalStageCManifest.from_dict(
        value, repository_root=root)
    for run in manifest.runs:
        try:
            actual = config_closure_sha256(root, run.config)
        except (OSError, SyntaxError, ValueError) as error:
            raise FormalManifestError(
                f'{run.run_id} config closure SHA-256 cannot be verified: '
                f'{error}') from error
        if actual != run.config_sha256:
            raise FormalManifestError(
                f'{run.run_id} config closure SHA-256 mismatch')
    canonical_root = canonical_main_root(root)
    validate_asset_binding(
        manifest.initialization.asset,
        repository_root=root,
        canonical_repository_root=canonical_root,
        field='initialization authority')
    for name, binding in manifest.data_authority.items():
        validate_asset_binding(
            binding,
            repository_root=root,
            canonical_repository_root=canonical_root,
            field=f'data authority {name}')
    validate_prior_artifact_authority(
        manifest.prior_artifact,
        repository_root=root,
        canonical_repository_root=canonical_root)
    return manifest


def config_closure_sha256(
        repository_root: Path | str, config: Path | str) -> str:
    """Hash a config and its complete ``_base_`` inheritance closure."""
    root = Path(repository_root).resolve(strict=False)
    relative = _safe_relative_path(str(config), 'config')
    records: dict[str, str] = {}
    visiting: set[str] = set()

    def visit(item: Path) -> None:
        normalized = _relative_to_root(root, root / item, 'config')
        key = normalized.as_posix()
        if key in visiting:
            raise ValueError(f'config inheritance cycle at {key}')
        if key in records:
            return
        visiting.add(key)
        data = (root / normalized).read_bytes()
        tree = ast.parse(data.decode('utf-8'), filename=key)
        bases: object = []
        for node in tree.body:
            if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == '_base_'):
                bases = ast.literal_eval(node.value)
                break
        if isinstance(bases, str):
            bases = [bases]
        if not isinstance(bases, (list, tuple)) or any(
                not isinstance(base, str) for base in bases):
            raise ValueError(f'{key} has a non-literal _base_ declaration')
        for base in bases:
            base_path = _relative_to_root(
                root, (root / normalized).parent / base, 'config base')
            visit(base_path)
        records[key] = hashlib.sha256(data).hexdigest()
        visiting.remove(key)

    visit(relative)
    payload = json.dumps(
        [{'path': key, 'sha256': records[key]} for key in sorted(records)],
        sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def canonical_main_root(repository_root: Path | str) -> Path:
    """Return the canonical main checkout that owns read-only assets."""
    root = Path(repository_root).resolve(strict=True)
    try:
        common = subprocess.check_output(
            ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
            cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalManifestError(
            'cannot resolve canonical main-checkout asset authority') from error
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = root / common_path
    common_path = common_path.absolute()
    if common_path.name != '.git' or not common_path.is_dir():
        raise FormalManifestError(
            'Git common directory does not identify a canonical main checkout')
    canonical = common_path.parent
    if canonical.is_symlink() or not canonical.is_dir():
        raise FormalManifestError('canonical main checkout root has drifted')
    return canonical


def validate_asset_binding(
        binding: AssetBinding, *, repository_root: Path | str,
        canonical_repository_root: Path | str,
        field: str = 'asset binding') -> None:
    """Authenticate one shared asset through its exact declared link root."""
    if not isinstance(binding, AssetBinding):
        raise FormalManifestError(f'{field} must be an AssetBinding')
    runtime = Path(repository_root).resolve(strict=True)
    canonical = Path(canonical_repository_root).absolute()
    if canonical.is_symlink() or not canonical.is_dir():
        raise FormalManifestError('canonical asset root has drifted')

    authority_parts = Path(binding.authority_root).parts
    expected_target = canonical.joinpath(*authority_parts)
    _reject_symlink_components(
        canonical, Path(binding.authority_root), 'canonical asset root')
    if not expected_target.is_dir():
        raise FormalManifestError(
            f'{field} canonical asset root is missing')

    runtime_link = runtime.joinpath(*authority_parts)
    _reject_parent_symlinks(runtime, Path(binding.authority_root), field)
    if not runtime_link.is_symlink():
        raise FormalManifestError(
            f'{field} authority root must be the declared symlink')
    try:
        actual_target = runtime_link.resolve(strict=True)
        canonical_target = expected_target.resolve(strict=True)
    except OSError as error:
        raise FormalManifestError(f'{field} link target is unavailable') from error
    if actual_target != canonical_target:
        raise FormalManifestError(f'{field} link target drift')

    _reject_symlink_components(
        expected_target, binding.asset_relative_path, field)
    asset = expected_target / binding.asset_relative_path
    if not asset.is_file():
        raise FormalManifestError(f'{field} file is missing')
    if _sha256_file(asset) != binding.sha256:
        raise FormalManifestError(f'{field} SHA-256 mismatch')


def validate_prior_artifact_authority(
        authority: PriorArtifactAuthority, *,
        repository_root: Path | str,
        canonical_repository_root: Path | str) -> None:
    """Validate the one approved Stage-B bundle without loading its model."""
    if not isinstance(authority, PriorArtifactAuthority):
        raise FormalManifestError(
            'prior artifact authority has the wrong type')
    runtime = Path(repository_root).resolve(strict=True)
    canonical = Path(canonical_repository_root).absolute()
    if canonical.is_symlink() or not canonical.is_dir():
        raise FormalManifestError('canonical prior root has drifted')
    target_relative = Path(
        authority.target_root.removeprefix('canonical-main/'))
    expected_target = canonical / target_relative
    _reject_symlink_components(
        canonical, target_relative, 'canonical prior artifact root')
    if not expected_target.is_dir():
        raise FormalManifestError('canonical prior artifact bundle is missing')

    link = runtime / authority.link_root
    _reject_parent_symlinks(runtime, authority.link_root, 'prior artifact')
    if not link.is_symlink():
        raise FormalManifestError(
            'prior artifact link_root must be the declared symlink')
    raw_target = Path(os.readlink(link))
    if not raw_target.is_absolute():
        raw_target = (link.parent / raw_target).absolute()
    if raw_target != expected_target.absolute():
        raise FormalManifestError('prior artifact link target drift')
    try:
        if link.resolve(strict=True) != expected_target.resolve(strict=True):
            raise FormalManifestError('prior artifact link target drift')
    except OSError as error:
        raise FormalManifestError(
            'prior artifact link target is unavailable') from error

    bundle_manifest = expected_target / 'bundle.json'
    if (
            bundle_manifest.is_symlink()
            or _sha256_file(bundle_manifest)
            != authority.bundle_manifest_sha256):
        raise FormalManifestError(
            'prior artifact bundle manifest SHA-256 mismatch')
    try:
        document = json.loads(bundle_manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalManifestError(
            'prior artifact bundle manifest is malformed') from error
    if set(document) != {
            'schema_version', 'kind', 'source_commit',
            'unpruned_parent_checkpoint_sha256', 'entries'}:
        raise FormalManifestError(
            'prior artifact bundle manifest fields are invalid')
    if (
            document['schema_version'] != 1
            or document['kind']
            != 'mambapose-formal-stage-c-prior-no-pif-seed0'
            or document['source_commit'] != authority.source_commit
            or document['unpruned_parent_checkpoint_sha256']
            != authority.unpruned_parent_checkpoint_sha256):
        raise FormalManifestError(
            'prior artifact bundle provenance mismatch')
    entries = _require_mapping(document['entries'], 'prior bundle entries')
    if set(entries) != set(authority.artifacts):
        raise FormalManifestError('prior bundle artifact roles mismatch')

    allowed = {'bundle.json'}
    for name, binding in authority.artifacts.items():
        record = _require_mapping(entries[name], f'prior bundle {name}')
        _require_exact_fields(
            record, {'path', 'sha256', 'bytes'}, f'prior bundle {name}')
        if (
                record['path'] != binding.path.as_posix()
                or record['sha256'] != binding.sha256
                or isinstance(record['bytes'], bool)
                or not isinstance(record['bytes'], int)
                or record['bytes'] <= 0):
            raise FormalManifestError(
                f'prior bundle {name} authority mismatch')
        candidate = expected_target / binding.path
        _reject_symlink_components(expected_target, binding.path, name)
        if (
                not candidate.is_file()
                or candidate.stat().st_size != record['bytes']
                or _sha256_file(candidate) != binding.sha256):
            raise FormalManifestError(
                f'prior bundle {name} SHA-256 mismatch')
        if candidate.stat().st_mode & stat.S_IWUSR:
            raise FormalManifestError(
                f'prior bundle {name} is not read-only')
        allowed.add(binding.path.as_posix())
    if bundle_manifest.stat().st_mode & stat.S_IWUSR:
        raise FormalManifestError(
            'prior artifact bundle manifest is not read-only')
    actual = {
        path.relative_to(expected_target).as_posix()
        for path in expected_target.rglob('*') if path.is_file()}
    if actual != allowed:
        raise FormalManifestError(
            'prior artifact bundle contains unexpected files')


def _validate_tracked_path(root: Path, candidate: Path, field: str) -> None:
    lexical = candidate.absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise FormalManifestError(f'{field} must remain inside the worktree') from error
    if not relative.parts:
        raise FormalManifestError(f'{field} must name a file')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise FormalManifestError(f'{field} must not traverse a symlink')
    if not candidate.is_file():
        raise FormalManifestError(f'{field} file is missing')


def _validate_internal_relative_path(
        root: Path, relative: Path, field: str) -> None:
    """Reject any existing symlink component on an internal logical path."""
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise FormalManifestError(
                f'{field} must not traverse an internal symlink')


def _reject_parent_symlinks(root: Path, relative: Path, field: str) -> None:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise FormalManifestError(
                f'{field} authority parent must not be a symlink')


def _reject_symlink_components(root: Path, relative: Path, field: str) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise FormalManifestError(f'{field} must not traverse a nested symlink')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _parse_initialization(
        value: Any, *, repository_root: Path) -> InitializationAuthority:
    document = _require_mapping(value, 'initialization')
    _require_exact_fields(document, _INITIALIZATION_FIELDS, 'initialization')
    if document['id'] != INITIALIZATION_ID:
        raise FormalManifestError('unexpected initialization authority id')
    if document['kind'] != 'vmamba-backbone':
        raise FormalManifestError('initialization kind must be vmamba-backbone')
    asset = AssetBinding.from_dict(
        document['asset'], repository_root=repository_root)
    if (
            asset.path != INITIALIZATION_PATH
            or asset.sha256 != INITIALIZATION_SHA256):
        raise FormalManifestError('unexpected initialization authority')
    return InitializationAuthority(
        id=INITIALIZATION_ID, kind='vmamba-backbone', asset=asset)


def _parse_protocol(value: Any) -> FormalProtocol:
    document = _require_mapping(value, 'protocol')
    _require_exact_fields(document, _PROTOCOL_FIELDS, 'protocol')
    epochs = _require_int(document['epochs'], 'epochs', minimum=1)
    if epochs != 300:
        raise FormalManifestError('formal protocol requires 300 epochs')
    workers = _require_int(
        document['worker_count'], 'worker_count', minimum=0)
    if document['persistent_workers'] is not False:
        raise FormalManifestError('persistent workers must be disabled')
    effective_batch = _require_int(
        document['effective_batch_size'], 'effective_batch_size', minimum=1)
    per_device_batch = _require_int(
        document['per_device_batch_size'],
        'per_device_batch_size', minimum=1)
    world_size = _require_int(
        document['world_size'], 'world_size', minimum=1)
    accumulation = _require_int(
        document['accumulation_steps'], 'accumulation_steps', minimum=1)
    if per_device_batch != 128:
        raise FormalManifestError('per-device batch size must be 128')
    if world_size != 1:
        raise FormalManifestError('formal world size must be 1')
    if accumulation != 1:
        raise FormalManifestError('formal accumulation steps must be 1')
    if effective_batch != per_device_batch * world_size * accumulation:
        raise FormalManifestError(
            'effective batch must equal per-device batch times world size '
            'times accumulation')
    primary = _parse_seed_list(document['primary_seeds'], 'primary seeds')
    conditional = _parse_seed_list(
        document['conditional_seeds'], 'conditional seeds')
    if primary != PRIMARY_SEEDS:
        raise FormalManifestError('primary seeds must be exactly 0, 1, 2')
    if conditional != CONDITIONAL_SEEDS:
        raise FormalManifestError('conditional seeds must be exactly 3, 4')
    return FormalProtocol(
        epochs=epochs,
        worker_count=workers,
        persistent_workers=False,
        per_device_batch_size=per_device_batch,
        world_size=world_size,
        accumulation_steps=accumulation,
        effective_batch_size=effective_batch,
        primary_seeds=primary,
        conditional_seeds=conditional,
    )


def _parse_data_authority(
        value: Any, *, repository_root: Path) -> dict[str, AssetBinding]:
    document = _require_mapping(value, 'data authority')
    _require_exact_fields(document, _DATA_FIELDS, 'data authority')
    return {
        name: AssetBinding.from_dict(
            item, repository_root=repository_root)
        for name, item in document.items()
    }


def _parse_run(
        value: Any, *, repository_root: Path) -> FormalRunSpec:
    document = _require_mapping(value, 'run')
    _require_exact_fields(document, _RUN_FIELDS, 'run')
    run_id, role, seed = _parse_run_identity(document)
    conditional = document['conditional']
    if not isinstance(conditional, bool):
        raise FormalManifestError('conditional must be a boolean')
    config_path = _safe_relative_path(document['config'], 'config')
    output_root = _safe_formal_output_root(
        document['output_root'], 'output_root')
    _validate_internal_relative_path(
        repository_root, config_path, 'config')
    _validate_internal_relative_path(
        repository_root, output_root, 'output_root')
    return FormalRunSpec(
        run_id=run_id,
        role=role,
        seed=seed,
        conditional=conditional,
        config=config_path,
        config_sha256=_require_sha256(
            document['config_sha256'], 'config SHA-256'),
        initialization_id=_require_identifier(
            document['initialization_id'], 'initialization_id'),
        output_root=output_root,
    )


def _parse_run_identity(
        document: Mapping[str, Any],
        ) -> tuple[str, Literal['baseline', 'no_pif'], int]:
    run_id = _require_identifier(document['run_id'], 'run_id')
    role = document['role']
    if role not in ROLES:
        raise FormalManifestError('role must be baseline or no_pif')
    seed = _require_int(document['seed'], 'seed', minimum=0)
    if seed not in ALL_SEEDS:
        raise FormalManifestError('seed must be one of 0, 1, 2, 3, 4')
    return run_id, role, seed


def _validate_canonical_run(run: FormalRunSpec) -> None:
    stem = 'full' if run.role == 'baseline' else 'no_pif'
    run_stem = stem.replace('_', '-')
    expected_id = f'{run_stem}-seed{run.seed}'
    if run.run_id != expected_id:
        raise FormalManifestError(
            f'run id must be canonical for role and seed: {expected_id}')
    expected_config = Path(
        f'configs/optimization/formal_stage_c/{stem}_seed{run.seed}.py')
    if run.config != expected_config:
        raise FormalManifestError(
            f'{run.run_id} config path must be {expected_config.as_posix()}')
    expected_output = Path(
        f'work_dirs/optimization/formal-stage-c/{expected_id}')
    if run.output_root != expected_output:
        raise FormalManifestError(
            f'{run.run_id} output_root must be {expected_output.as_posix()}')


def _parse_file_binding(
        value: Any, field: str, *,
        repository_root: Path | str) -> FileBinding:
    document = _require_mapping(value, field)
    _require_exact_fields(document, _FILE_BINDING_FIELDS, field)
    path = _safe_relative_path(document['path'], f'{field} path')
    _validate_internal_relative_path(
        Path(repository_root).resolve(strict=False), path, field)
    return FileBinding(
        path=path,
        sha256=_require_sha256(document['sha256'], f'{field} SHA-256'))


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalManifestError(f'{field} must be an object')
    return value


def _require_exact_fields(
        value: Mapping[str, Any], expected: set[str] | frozenset[str],
        field: str) -> None:
    actual = set(value)
    unknown = actual - set(expected)
    missing = set(expected) - actual
    if unknown:
        raise FormalManifestError(
            f'{field} has unknown fields: {sorted(unknown)}')
    if missing:
        raise FormalManifestError(
            f'{field} is missing fields: {sorted(missing)}')


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FormalManifestError(f'{field} must be a lowercase SHA-256')
    return value


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FormalManifestError(
            f'{field} must be a lowercase path-safe identifier')
    return value


def _require_int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FormalManifestError(
            f'{field} must be an integer no smaller than {minimum}')
    return value


def _parse_seed_list(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FormalManifestError(f'{field} must be a list')
    seeds = tuple(_require_int(item, field, minimum=0) for item in value)
    if len(seeds) != len(set(seeds)):
        raise FormalManifestError(f'{field} must not contain duplicate seeds')
    return seeds


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FormalManifestError(f'{field} must be a non-empty relative path')
    path = Path(value)
    if path.is_absolute():
        raise FormalManifestError(f'{field} must be relative')
    if any(part in {'.', '..'} for part in path.parts):
        raise FormalManifestError(f'{field} path traversal is forbidden')
    if any(character in value for character in ('\0', '\n', '\r', ';', '|', '&', '`', '$')):
        raise FormalManifestError(f'{field} contains unsafe characters')
    return path


def _safe_formal_output_root(value: Any, field: str) -> Path:
    path = _safe_relative_path(value, field)
    _require_prefix(
        path, Path('work_dirs/optimization/formal-stage-c'), field)
    if len(path.parts) != 4:
        raise FormalManifestError(
            f'{field} must name one direct formal-stage-c run directory')
    return path


def _require_prefix(path: Path, prefix: Path, field: str) -> None:
    if path.parts[:len(prefix.parts)] != prefix.parts:
        raise FormalManifestError(
            f'{field} must remain under {prefix.as_posix()}')


def _relative_to_root(root: Path, candidate: Path, field: str) -> Path:
    normalized = candidate.resolve(strict=False)
    try:
        return normalized.relative_to(root)
    except ValueError as error:
        raise ValueError(f'{field} escapes repository root') from error


def _verify_file(root: Path, binding: FileBinding, field: str) -> None:
    candidate = root / binding.path
    if not candidate.is_file():
        raise FormalManifestError(f'{field} file is missing: {binding.path}')
    digest = hashlib.sha256()
    with candidate.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != binding.sha256:
        raise FormalManifestError(f'{field} SHA-256 mismatch')
