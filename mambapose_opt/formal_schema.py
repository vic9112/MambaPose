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
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Literal, Mapping
import zipfile


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
_DATA_FILE_FIELDS = frozenset({
    'inventory', 'train_annotations', 'validation_annotations', 'detections',
    'annotation_archive', 'train_image_archive', 'validation_image_archive',
})
_DATA_CORPUS_FIELDS = frozenset({
    'train_image_corpus', 'validation_image_corpus'})
_DATA_FIELDS = _DATA_FILE_FIELDS | _DATA_CORPUS_FIELDS
_FILE_BINDING_FIELDS = frozenset({'path', 'sha256'})
_ASSET_BINDING_FIELDS = frozenset({
    'authority_root', 'target_root', 'asset_relative_path', 'sha256'})
_ASSET_AUTHORITIES = MappingProxyType({
    'pretrained': 'canonical-main/pretrained',
    'data': 'canonical-main/data',
    'work_dirs/reproduction': 'canonical-main/work_dirs/reproduction',
})
_CANONICAL_DATA_FILES = MappingProxyType({
    'inventory': (
        'data', 'inventory.json',
        '2a82ab3cfe05a514d141c17921463e72f22741d67eaf978d93317155bbeaf0ed'),
    'train_annotations': (
        'data', 'coco/annotations/person_keypoints_train2017.json',
        '7fc1549d934547c470384d8a207c38707ca33fc016d00e33b795e408603af83e'),
    'validation_annotations': (
        'data', 'coco/annotations/person_keypoints_val2017.json',
        '788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14'),
    'detections': (
        'data',
        ('coco/person_detection_results/'
         'COCO_val2017_detections_AP_H_56_person.json'),
        '53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59'),
    'annotation_archive': (
        'work_dirs/reproduction', 'downloads/annotations_trainval2017.zip',
        '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'),
    'train_image_archive': (
        'work_dirs/reproduction', 'downloads/train2017.zip',
        '69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929'),
    'validation_image_archive': (
        'work_dirs/reproduction', 'downloads/val2017.zip',
        '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05'),
})
_CORPUS_BINDING_FIELDS = frozenset({
    'authority_root', 'target_root', 'corpus_relative_path', 'archive_role',
    'archive_prefix', 'image_count', 'digest_algorithm', 'sha256',
})
_CORPUS_ALGORITHM = 'sha256-zip-member-and-extracted-content-v1'
_CANONICAL_CORPORA = MappingProxyType({
    'train_image_corpus': MappingProxyType({
        'corpus_relative_path': 'coco/train2017',
        'archive_role': 'train_image_archive',
        'archive_prefix': 'train2017/',
        'image_count': 118287,
        'sha256': 'f552fb95e1f40129d9726146ddfbadcbf4fb931bf6c8631cf6425a6994999695',
    }),
    'validation_image_corpus': MappingProxyType({
        'corpus_relative_path': 'coco/val2017',
        'archive_role': 'validation_image_archive',
        'archive_prefix': 'val2017/',
        'image_count': 5000,
        'sha256': '6bf5c46be73304e0e8d77af5cf5764eed9e598e1f815f844e3366675c23e610e',
    }),
})
_INVENTORY_PROJECTION = MappingProxyType({
    'coco-train2017': MappingProxyType({
        'path': 'work_dirs/reproduction/downloads/train2017.zip',
        'sha256': _CANONICAL_DATA_FILES['train_image_archive'][2],
        'required_paths': ('data/coco/train2017',),
    }),
    'coco-val2017': MappingProxyType({
        'path': 'work_dirs/reproduction/downloads/val2017.zip',
        'sha256': _CANONICAL_DATA_FILES['validation_image_archive'][2],
        'required_paths': ('data/coco/val2017',),
    }),
    'coco-annotations': MappingProxyType({
        'path': 'work_dirs/reproduction/downloads/annotations_trainval2017.zip',
        'sha256': _CANONICAL_DATA_FILES['annotation_archive'][2],
        'required_paths': (
            'data/coco/annotations/person_keypoints_train2017.json',
            'data/coco/annotations/person_keypoints_val2017.json'),
    }),
    'coco-val-detections': MappingProxyType({
        'path': 'data/' + _CANONICAL_DATA_FILES['detections'][1],
        'sha256': _CANONICAL_DATA_FILES['detections'][2],
        'required_paths': (
            'data/' + _CANONICAL_DATA_FILES['detections'][1],),
    }),
})
_CANONICAL_CONFIG_FILES = MappingProxyType({
    'configs/optimization/formal_stage_c/_base_/paired_300ep.py': (
        'ac32c37c4ec3c87a195e5ee6fd705da02212ed04c8c9f865b04af027091ba84b'),
    'configs/reproduction/coco_s_v1.py': (
        '7aa78c2b8394b57b16517380a479adb5c3b3161cadb802c26c2d2f66583a6b34'),
    ('configs/body_2d_keypoint/tokenpose/'
     'mamba_tokenpose_T2_coco_256x192_300ep.py'): (
        'f04bba18bc32e5eec64173f9e1106a3bd9d0f19ed07e9a8db83f66d5ddda905b'),
    'configs/_base_/default_runtime.py': (
        'bb7df18270b1ca337192faae7a416b2dea23b45e798e832caab5781f41189d06'),
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
class CorpusBinding:
    authority_root: Literal['data']
    target_root: str
    corpus_relative_path: Path
    archive_role: Literal['train_image_archive', 'validation_image_archive']
    archive_prefix: str
    image_count: int
    digest_algorithm: str
    sha256: str

    @property
    def path(self) -> Path:
        return Path(self.authority_root) / self.corpus_relative_path

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorpusBinding:
        document = _require_mapping(value, 'image corpus authority')
        _require_exact_fields(
            document, _CORPUS_BINDING_FIELDS, 'image corpus authority')
        if document['authority_root'] != 'data' \
                or document['target_root'] != 'canonical-main/data':
            raise FormalManifestError(
                'image corpus authority root is not canonical')
        relative = _safe_relative_path(
            document['corpus_relative_path'], 'corpus_relative_path')
        archive_role = document['archive_role']
        if archive_role not in {
                'train_image_archive', 'validation_image_archive'}:
            raise FormalManifestError('image corpus archive role is invalid')
        prefix = document['archive_prefix']
        if not isinstance(prefix, str) or not prefix.endswith('/'):
            raise FormalManifestError('image corpus archive prefix is invalid')
        if document['digest_algorithm'] != _CORPUS_ALGORITHM:
            raise FormalManifestError('image corpus digest algorithm is invalid')
        return cls(
            authority_root='data', target_root='canonical-main/data',
            corpus_relative_path=relative, archive_role=archive_role,
            archive_prefix=prefix,
            image_count=_require_int(
                document['image_count'], 'image_count', minimum=1),
            digest_algorithm=_CORPUS_ALGORITHM,
            sha256=_require_sha256(
                document['sha256'], 'image corpus SHA-256'))


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
    data_authority: Mapping[str, AssetBinding | CorpusBinding]
    prior_artifact: PriorArtifactAuthority
    runs: tuple[FormalRunSpec, ...]
    repository_root: Path

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str) -> FormalStageCManifest:
        document = _require_mapping(value, 'manifest')
        _require_exact_fields(document, _MANIFEST_FIELDS, 'manifest')
        if isinstance(document['schema_version'], bool) \
                or document['schema_version'] != 2:
            raise FormalManifestError('schema_version must be 2')
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
            schema_version=2,
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
    output_root_device: int | None = None
    output_root_inode: int | None = None

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any], *,
            repository_root: Path | str,
            manifest: FormalStageCManifest | None = None,
            expected_manifest_sha256: str | None = None) -> FormalRunInit:
        document = _require_mapping(value, 'run init')
        root = Path(repository_root).resolve(strict=False)
        fields = frozenset({
            'schema_version', 'manifest_sha256', 'source', 'config',
            'environment_inventory_sha256', 'data_authority', 'run',
            'initialization',
        })
        _require_exact_fields(document, fields, 'run init')
        if isinstance(document['schema_version'], bool) \
                or document['schema_version'] != 1:
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
        expected_data = frozenset(
            f'{name}_sha256' for name in _DATA_FIELDS)
        _require_exact_fields(data, expected_data, 'data authority')
        normalized_data = {
            key: _require_sha256(item, key) for key, item in data.items()}
        canonical_data = {
            f'{name}_sha256': values[2]
            for name, values in _CANONICAL_DATA_FILES.items()}
        canonical_data.update({
            f'{name}_sha256': values['sha256']
            for name, values in _CANONICAL_CORPORA.items()})
        if normalized_data != canonical_data:
            raise FormalManifestError('run init data authority mismatch')

        run = _require_mapping(document['run'], 'run')
        run_fields = frozenset({
            'run_id', 'role', 'seed', 'epochs', 'effective_batch_size',
            'worker_count', 'persistent_workers', 'output_root',
            'output_root_device', 'output_root_inode',
        })
        _require_exact_fields(run, run_fields, 'run')
        run_id, role, seed = _parse_run_identity(run)
        expected_id, expected_config, expected_output = _canonical_run_identity(
            role, seed)
        if run_id != expected_id:
            raise FormalManifestError('run identity is not canonical')
        if config_path != expected_config:
            raise FormalManifestError('run config is not canonical')
        epochs = _require_int(run['epochs'], 'epochs', minimum=1)
        if epochs != 300:
            raise FormalManifestError('formal training requires 300 epochs')
        worker_count = _require_int(
            run['worker_count'], 'worker_count', minimum=0)
        if worker_count != 2:
            raise FormalManifestError('formal worker count must be 2')
        if run['persistent_workers'] is not False:
            raise FormalManifestError('persistent workers must be disabled')
        output_root = _safe_formal_output_root(
            run['output_root'], 'output_root')
        _validate_internal_relative_path(root, output_root, 'output_root')
        if output_root.name != run_id:
            raise FormalManifestError('output_root must end in the run id')
        if output_root != expected_output:
            raise FormalManifestError('run output_root is not canonical')
        effective_batch_size = _require_int(
            run['effective_batch_size'], 'effective_batch_size', minimum=1)
        if effective_batch_size != 128:
            raise FormalManifestError('formal effective batch must be 128')
        output_root_device = _require_int(
            run['output_root_device'], 'output root device', minimum=1)
        output_root_inode = _require_int(
            run['output_root_inode'], 'output root inode', minimum=1)
        closure_sha256 = _require_sha256(
            config['closure_sha256'], 'config closure SHA-256')
        try:
            observed_closure = config_closure_sha256(root, config_path)
        except (OSError, SyntaxError, ValueError) as error:
            raise FormalManifestError(
                f'run config closure cannot be verified: {error}') from error
        if observed_closure != closure_sha256:
            raise FormalManifestError('run config closure SHA-256 mismatch')
        resolved_sha256 = _require_sha256(
            config['resolved_sha256'], 'resolved config SHA-256')
        expected_resolved = formal_resolved_config_sha256(
            root, role=role, seed=seed)
        if resolved_sha256 != expected_resolved:
            raise FormalManifestError('run resolved config SHA-256 mismatch')

        manifest_sha256 = _require_sha256(
            document['manifest_sha256'], 'manifest_sha256')
        initialization = _parse_initialization(
            document['initialization'], repository_root=root)
        if (manifest is None) != (expected_manifest_sha256 is None):
            raise FormalManifestError(
                'manifest and expected manifest SHA-256 must be supplied together')
        if manifest is not None:
            expected_manifest = _require_sha256(
                expected_manifest_sha256, 'expected manifest SHA-256')
            if manifest_sha256 != expected_manifest:
                raise FormalManifestError('run init manifest SHA-256 mismatch')
            matching = tuple(
                item for item in manifest.runs if item.run_id == run_id)
            if len(matching) != 1:
                raise FormalManifestError(
                    'run init is absent from the manifest authority')
            spec = matching[0]
            if (
                    spec.role != role or spec.seed != seed
                    or spec.config != config_path
                    or spec.config_sha256 != closure_sha256
                    or spec.output_root != output_root
                    or manifest.protocol.epochs != epochs
                    or manifest.protocol.worker_count != worker_count
                    or manifest.protocol.effective_batch_size
                    != effective_batch_size
                    or manifest.initialization != initialization):
                raise FormalManifestError(
                    'run init does not match manifest authority')
            manifest_data = {
                f'{name}_sha256': binding.sha256
                for name, binding in manifest.data_authority.items()}
            if normalized_data != manifest_data:
                raise FormalManifestError(
                    'run init data authority does not match manifest')

        return cls(
            manifest_sha256=manifest_sha256,
            git_commit=commit,
            config=config_path,
            config_closure_sha256=closure_sha256,
            resolved_config_sha256=resolved_sha256,
            environment_inventory_sha256=_require_sha256(
                document['environment_inventory_sha256'],
                'environment inventory SHA-256'),
            data_authority=MappingProxyType(normalized_data),
            run_id=run_id,
            role=role,
            seed=seed,
            epochs=epochs,
            effective_batch_size=effective_batch_size,
            worker_count=worker_count,
            persistent_workers=False,
            output_root=output_root,
            initialization=initialization,
            output_root_device=output_root_device,
            output_root_inode=output_root_inode,
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
            repository_root: Path | str,
            verify_files: bool = False,
            run_init: FormalRunInit | None = None,
            expected_run_init_sha256: str | None = None,
            _final_epoch_commit_payload: bytes | None = None
            ) -> FormalTrainResult:
        document = _require_mapping(value, 'train result')
        root = Path(repository_root).resolve(strict=False)
        fields = frozenset({
            'schema_version', 'run_init_sha256', 'initialization', 'run',
            'best_checkpoint', 'resume_checkpoints', 'structured_log',
            'order_hashes', 'final_epoch', 'status',
        })
        _require_exact_fields(document, fields, 'train result')
        if isinstance(document['schema_version'], bool) \
                or document['schema_version'] != 1:
            raise FormalManifestError('train result schema_version must be 1')
        initialization = _parse_initialization(
            document['initialization'], repository_root=root)

        run = _require_mapping(document['run'], 'run')
        _require_exact_fields(
            run, {'run_id', 'role', 'seed', 'output_root'}, 'run')
        run_id, role, seed = _parse_run_identity(run)
        expected_id, _expected_config, expected_output = _canonical_run_identity(
            role, seed)
        if run_id != expected_id:
            raise FormalManifestError('train result run identity is not canonical')
        output_root = _safe_formal_output_root(
            run['output_root'], 'output_root')
        _validate_internal_relative_path(root, output_root, 'output_root')
        if output_root.name != run_id:
            raise FormalManifestError('output_root must end in the run id')
        if output_root != expected_output:
            raise FormalManifestError(
                'train result output_root is not canonical')

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
        resume_paths = tuple(binding.path for binding in resume)
        resume_hashes = tuple(binding.sha256 for binding in resume)
        if len(set(resume_paths)) != 2 or len(set(resume_hashes)) != 2:
            raise FormalManifestError(
                'resume checkpoints must be distinct')
        expected_resume = (
            output_root / 'epoch_299.pth', output_root / 'epoch_300.pth')
        if resume_paths != expected_resume:
            raise FormalManifestError(
                'resume checkpoints must be the chronological latest two')
        if best.path in set(resume_paths):
            raise FormalManifestError(
                'best checkpoint must not impersonate a resume checkpoint')
        best_match = re.fullmatch(
            r'best_coco_AP_epoch_([1-9][0-9]*)\.pth', best.path.name)
        if best_match is None:
            raise FormalManifestError('best checkpoint path is not canonical')
        best_epoch = int(best_match.group(1))
        if best_epoch < 5 or best_epoch > 300 or best_epoch % 5:
            raise FormalManifestError('best checkpoint path is not canonical')
        if log.path != output_root / 'training.jsonl':
            raise FormalManifestError('structured log path is not canonical')

        raw_hashes = document['order_hashes']
        if not isinstance(raw_hashes, list) or len(raw_hashes) != 300:
            raise FormalManifestError('train result requires 300 order hashes')
        order_hash_values: list[str] = []
        for index, item in enumerate(raw_hashes, start=1):
            record = _require_mapping(
                item, f'order_hashes[{index - 1}]')
            _require_exact_fields(
                record, {'epoch', 'sha256'},
                f'order_hashes[{index - 1}]')
            epoch = _require_int(
                record['epoch'], f'order_hashes[{index - 1}].epoch',
                minimum=1)
            if epoch != index:
                raise FormalManifestError(
                    'train result order hash epoch sequence is not canonical')
            order_hash_values.append(_require_sha256(
                record['sha256'],
                f'order_hashes[{index - 1}].sha256'))
        order_hashes = tuple(order_hash_values)
        if len(set(order_hashes)) != 300:
            raise FormalManifestError(
                'train result order hashes must be distinct')
        if document['final_epoch'] != 300:
            raise FormalManifestError('final epoch must be 300')
        if document['status'] != 'complete':
            raise FormalManifestError('train result status must be complete')
        if verify_files:
            for field, binding in (
                    ('best_checkpoint', best),
                    ('resume_checkpoints[0]', resume[0]),
                    ('resume_checkpoints[1]', resume[1]),
                    ('structured_log', log)):
                _verify_file(root, binding, field)

        run_init_sha256 = _require_sha256(
            document['run_init_sha256'], 'run_init_sha256')
        if (run_init is None) != (expected_run_init_sha256 is None):
            raise FormalManifestError(
                'run init and expected run init SHA-256 must be supplied together')
        if run_init is not None:
            expected_init = _require_sha256(
                expected_run_init_sha256, 'expected run init SHA-256')
            if run_init_sha256 != expected_init:
                raise FormalManifestError(
                    'train result run init SHA-256 mismatch')
            if (
                    run_init.run_id != run_id or run_init.role != role
                    or run_init.seed != seed
                    or run_init.output_root != output_root
                    or run_init.initialization != initialization):
                raise FormalManifestError(
                    'train result does not match run init authority')

        _verify_final_epoch_commit_best(
            root, output_root=output_root, run_id=run_id, role=role,
            seed=seed, run_init_sha256=run_init_sha256,
            best=best, latest_resume=resume[1], structured_log=log,
            final_order_sha256=order_hashes[-1],
            captured_payload=_final_epoch_commit_payload)

        return cls(
            run_init_sha256=run_init_sha256,
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
    validate_canonical_formal_config_closure(root)
    canonical_root = canonical_main_root(root)
    validate_asset_binding(
        manifest.initialization.asset,
        repository_root=root,
        canonical_repository_root=canonical_root,
        field='initialization authority')
    validate_data_authority(
        manifest.data_authority, repository_root=root,
        canonical_repository_root=canonical_root)
    validate_prior_artifact_authority(
        manifest.prior_artifact,
        repository_root=root,
        canonical_repository_root=canonical_root)
    return manifest


def _load_runtime_json(
        path: Path | str, *, repository_root: Path,
        field: str) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path)
    if not source.is_absolute():
        source = repository_root / source
    lexical = source.absolute()
    try:
        relative = lexical.relative_to(repository_root)
    except ValueError as error:
        raise FormalManifestError(
            f'{field} must remain inside the worktree') from error
    if not relative.parts:
        raise FormalManifestError(f'{field} must name a file')
    _reject_symlink_components(repository_root, relative, field)
    if source.is_symlink() or not source.is_file():
        raise FormalManifestError(f'{field} file is missing')
    try:
        document = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalManifestError(f'{field} is malformed: {error}') from error
    return source, _require_mapping(document, field)


def _validate_formal_source_authority(
        repository_root: Path | str, expected_commit: str) -> None:
    root = Path(repository_root).resolve(strict=True)
    commit = _require_commit(expected_commit, 'formal source commit')
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        attached = subprocess.run(
            ['git', 'symbolic-ref', '-q', 'HEAD'], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
        status_output = subprocess.check_output(
            ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
            cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalManifestError(
            'formal source Git authority cannot be verified') from error
    if head != commit:
        raise FormalManifestError('formal source HEAD commit mismatch')
    if attached.returncode == 0:
        raise FormalManifestError('formal source must be detached')
    if attached.returncode != 1:
        raise FormalManifestError('formal source attachment cannot be verified')
    if status_output:
        raise FormalManifestError('formal source tracked tree must be clean')


def load_formal_run_init(
        path: Path | str, *, repository_root: Path | str) -> FormalRunInit:
    """Load one canonical run-init record and bind it to the tracked manifest."""
    root = Path(repository_root).resolve(strict=True)
    source, document = _load_runtime_json(
        path, repository_root=root, field='formal run init')
    manifest_path = root / 'optimization/formal_stage_c.json'
    manifest = load_formal_manifest(manifest_path, repository_root=root)
    manifest_sha256 = _sha256_file(manifest_path)
    run_init = FormalRunInit.from_dict(
        document, repository_root=root, manifest=manifest,
        expected_manifest_sha256=manifest_sha256)
    expected_path = root / run_init.output_root / 'run-init.json'
    if source.absolute() != expected_path.absolute():
        raise FormalManifestError('formal run init path is not canonical')
    _validate_formal_source_authority(root, run_init.git_commit)
    return run_init


def load_formal_train_result(
        path: Path | str, *, repository_root: Path | str,
        verify_files: bool = True) -> FormalTrainResult:
    """Load a canonical train result and bind it to its exact run-init bytes."""
    root = Path(repository_root).resolve(strict=True)
    source, document = _load_runtime_json(
        path, repository_root=root, field='formal train result')
    preliminary = FormalTrainResult.from_dict(
        document, repository_root=root, verify_files=False)
    expected_path = root / preliminary.output_root / 'train-result.json'
    if source.absolute() != expected_path.absolute():
        raise FormalManifestError('formal train result path is not canonical')
    run_init_path = root / preliminary.output_root / 'run-init.json'
    run_init = load_formal_run_init(run_init_path, repository_root=root)
    return FormalTrainResult.from_dict(
        document, repository_root=root, verify_files=verify_files,
        run_init=run_init,
        expected_run_init_sha256=_sha256_file(run_init_path))


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


def validate_canonical_formal_config_closure(
        repository_root: Path | str) -> None:
    """Require the reviewed byte-exact common formal configuration closure."""
    root = Path(repository_root).resolve(strict=False)
    for relative_text, expected_sha256 in _CANONICAL_CONFIG_FILES.items():
        relative = Path(relative_text)
        _validate_internal_relative_path(root, relative, 'formal config closure')
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FormalManifestError(
                f'canonical formal config closure is missing: {relative_text}')
        if _sha256_file(path) != expected_sha256:
            raise FormalManifestError(
                f'canonical formal config closure drift: {relative_text}')
    for seed in ALL_SEEDS:
        for role in ROLES:
            _validate_canonical_formal_leaf(root, role=role, seed=seed)


def _config_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_config_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _config_literal(key): _config_literal(item)
            for key, item in zip(node.keys, node.values)}
    if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'dict' and not node.args):
        return {
            keyword.arg: _config_literal(keyword.value)
            for keyword in node.keywords}
    raise FormalManifestError(
        'formal leaf config uses a non-literal assignment')


def _validate_canonical_formal_leaf(
        root: Path, *, role: str, seed: int) -> None:
    run_id, relative, output_root = _canonical_run_identity(role, seed)
    path = root / relative
    _validate_internal_relative_path(root, relative, 'formal leaf config')
    if not path.is_file() or path.is_symlink():
        raise FormalManifestError('canonical formal leaf config is missing')
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise FormalManifestError('canonical formal leaf config is malformed') \
            from error
    actual: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 \
                or not isinstance(node.targets[0], ast.Name):
            raise FormalManifestError(
                'canonical formal leaf config has executable statements')
        name = node.targets[0].id
        if name in actual:
            raise FormalManifestError(
                'canonical formal leaf config has duplicate assignments')
        actual[name] = _config_literal(node.value)
    mode = 'full' if role == 'baseline' else 'disabled'
    expected = {
        '_base_': ['./_base_/paired_300ep.py'],
        'formal_role': role,
        'formal_seed': seed,
        'experiment_id': f'formal-stage-c-{run_id}',
        'work_dir': output_root.as_posix(),
        'randomness': {'seed': seed, 'deterministic': True},
        'train_dataloader': {'sampler': {'seed': seed}},
        'val_dataloader': {'sampler': {'seed': seed}},
        'test_dataloader': {'sampler': {'seed': seed}},
        'model': {'head': {'tokenpose_cfg': {'pif_mode': mode}}},
    }
    if actual != expected:
        raise FormalManifestError(
            f'canonical formal leaf config drift: {relative.as_posix()}')


def formal_resolved_config_sha256(
        repository_root: Path | str, *, role: str, seed: int) -> str:
    """Hash the exact resolved formal protocol projection for one run."""
    if role not in ROLES or seed not in ALL_SEEDS:
        raise FormalManifestError('resolved config identity is invalid')
    validate_canonical_formal_config_closure(repository_root)
    run_id, config, output_root = _canonical_run_identity(role, seed)
    projection = {
        'run_id': run_id,
        'role': role,
        'seed': seed,
        'config': config.as_posix(),
        'output_root': output_root.as_posix(),
        'pif_mode': 'full' if role == 'baseline' else 'disabled',
        'deterministic': True,
        'epochs': 300,
        'worker_count': 2,
        'persistent_workers': False,
        'per_device_batch_size': 128,
        'world_size': 1,
        'accumulation_steps': 1,
        'effective_batch_size': 128,
        'auto_scale_lr': False,
        'optimizer': {'type': 'Adam', 'lr': '1e-3'},
        'scheduler': {
            'warmup_iterations': 500,
            'milestones': [200, 260],
            'gamma': 0.1,
        },
        'evaluation': {
            'type': 'CocoMetric',
            'detections': (
                'data/coco/person_detection_results/'
                'COCO_val2017_detections_AP_H_56_person.json'),
            'flip_test': True,
            'flip_mode': 'heatmap',
            'shift_heatmap': True,
        },
    }
    return canonical_json_sha256(projection)


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


def _verify_zip_corpus(
        archive: Path | str, image_root: Path | str, *, prefix: str,
        expected_count: int) -> str:
    """Byte-compare an extracted image corpus with its official ZIP members."""
    archive_path = Path(archive)
    corpus_root = Path(image_root)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise FormalManifestError('image corpus archive is missing or unsafe')
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        raise FormalManifestError('live image corpus is missing or unsafe')
    if not isinstance(prefix, str) or not prefix.endswith('/'):
        raise FormalManifestError('image corpus archive prefix is invalid')
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) \
            or expected_count < 1:
        raise FormalManifestError('image corpus expected count is invalid')

    try:
        with zipfile.ZipFile(archive_path) as packed:
            members: dict[str, zipfile.ZipInfo] = {}
            for info in packed.infolist():
                name = info.filename
                pure = PurePosixPath(name)
                if (
                        '\\' in name or pure.is_absolute()
                        or '..' in pure.parts):
                    raise FormalManifestError(
                        'image corpus archive has an unsafe member')
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise FormalManifestError(
                        'image corpus archive has a symlink member')
                if not name.startswith(prefix) or not name.lower().endswith('.jpg'):
                    raise FormalManifestError(
                        'image corpus archive has an unexpected member')
                relative = name[len(prefix):]
                if not relative or '/' in relative:
                    raise FormalManifestError(
                        'image corpus archive layout is not canonical')
                if name in members:
                    raise FormalManifestError(
                        'image corpus archive has duplicate members')
                members[name] = info
            if len(members) != expected_count:
                raise FormalManifestError(
                    'image corpus archive count mismatch')

            live: dict[str, Path] = {}
            for candidate in corpus_root.rglob('*'):
                relative = candidate.relative_to(corpus_root)
                if candidate.is_symlink():
                    raise FormalManifestError(
                        'live image corpus must not contain symlinks')
                if candidate.is_dir():
                    continue
                if not candidate.is_file() or candidate.suffix.lower() != '.jpg':
                    raise FormalManifestError(
                        'live image corpus has an unexpected file')
                name = prefix + relative.as_posix()
                live[name] = candidate
            if set(live) != set(members):
                raise FormalManifestError(
                    'live image corpus inventory differs from archive')

            digest = hashlib.sha256()
            for name in sorted(members):
                info = members[name]
                digest.update(name.encode('utf-8'))
                digest.update(b'\0')
                digest.update(str(info.file_size).encode('ascii'))
                digest.update(b'\0')
                with packed.open(info, 'r') as source, live[name].open('rb') as target:
                    observed_size = 0
                    while True:
                        source_block = source.read(8 * 1024 * 1024)
                        target_block = target.read(8 * 1024 * 1024)
                        if source_block != target_block:
                            raise FormalManifestError(
                                'live image corpus differs from archive')
                        if not source_block:
                            break
                        observed_size += len(source_block)
                        digest.update(source_block)
                    if observed_size != info.file_size:
                        raise FormalManifestError(
                            'image corpus member size mismatch')
            return digest.hexdigest()
    except FormalManifestError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise FormalManifestError(
            f'image corpus archive cannot be verified: {error}') from error


def _verify_archive_member(
        archive: Path, member_name: str, extracted: Path, field: str) -> None:
    if archive.is_symlink() or extracted.is_symlink() \
            or not archive.is_file() or not extracted.is_file():
        raise FormalManifestError(f'{field} archive linkage is unavailable')
    try:
        with zipfile.ZipFile(archive) as packed:
            info = packed.getinfo(member_name)
            with packed.open(info, 'r') as source, extracted.open('rb') as target:
                while True:
                    source_block = source.read(8 * 1024 * 1024)
                    target_block = target.read(8 * 1024 * 1024)
                    if source_block != target_block:
                        raise FormalManifestError(
                            f'{field} differs from its official archive')
                    if not source_block:
                        break
    except FormalManifestError:
        raise
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise FormalManifestError(
            f'{field} archive linkage cannot be verified: {error}') from error


def validate_data_authority(
        authority: Mapping[str, AssetBinding | CorpusBinding], *,
        repository_root: Path | str,
        canonical_repository_root: Path | str) -> None:
    """Authenticate exact COCO files, archives, inventory, and live images."""
    if set(authority) != set(_DATA_FIELDS):
        raise FormalManifestError('data authority roles are not exact')
    runtime = Path(repository_root).resolve(strict=True)
    canonical = Path(canonical_repository_root).absolute()
    for name in _DATA_FILE_FIELDS:
        binding = authority[name]
        if not isinstance(binding, AssetBinding):
            raise FormalManifestError(f'data authority {name} has wrong type')
        validate_asset_binding(
            binding, repository_root=runtime,
            canonical_repository_root=canonical,
            field=f'data authority {name}')

    inventory_binding = authority['inventory']
    assert isinstance(inventory_binding, AssetBinding)
    inventory_path = (
        canonical / inventory_binding.authority_root
        / inventory_binding.asset_relative_path)
    try:
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalManifestError('data inventory is malformed') from error
    if not isinstance(inventory, dict) \
            or isinstance(inventory.get('schema_version'), bool) \
            or inventory.get('schema_version') != 1 \
            or not isinstance(inventory.get('assets'), list):
        raise FormalManifestError('data inventory schema is invalid')
    records: dict[str, Mapping[str, Any]] = {}
    for item in inventory['assets']:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str):
            raise FormalManifestError('data inventory asset is malformed')
        if item['id'] in records:
            raise FormalManifestError('data inventory asset ids are not unique')
        records[item['id']] = item
    for asset_id, expected in _INVENTORY_PROJECTION.items():
        record = records.get(asset_id)
        if record is None or (
                record.get('path') != expected['path']
                or record.get('sha256') != expected['sha256']
                or tuple(record.get('required_paths', ()))
                != expected['required_paths']):
            raise FormalManifestError(
                f'data inventory authority mismatch for {asset_id}')

    annotation_archive = authority['annotation_archive']
    train_annotations = authority['train_annotations']
    validation_annotations = authority['validation_annotations']
    assert isinstance(annotation_archive, AssetBinding)
    assert isinstance(train_annotations, AssetBinding)
    assert isinstance(validation_annotations, AssetBinding)
    annotation_zip = (
        canonical / annotation_archive.authority_root
        / annotation_archive.asset_relative_path)
    _verify_archive_member(
        annotation_zip,
        'annotations/person_keypoints_train2017.json',
        canonical / train_annotations.authority_root
        / train_annotations.asset_relative_path,
        'train annotations')
    _verify_archive_member(
        annotation_zip,
        'annotations/person_keypoints_val2017.json',
        canonical / validation_annotations.authority_root
        / validation_annotations.asset_relative_path,
        'validation annotations')

    for name in _DATA_CORPUS_FIELDS:
        corpus = authority[name]
        if not isinstance(corpus, CorpusBinding):
            raise FormalManifestError(f'data authority {name} has wrong type')
        archive_binding = authority[corpus.archive_role]
        assert isinstance(archive_binding, AssetBinding)
        digest = _verify_zip_corpus(
            canonical / archive_binding.authority_root
            / archive_binding.asset_relative_path,
            canonical / corpus.authority_root / corpus.corpus_relative_path,
            prefix=corpus.archive_prefix,
            expected_count=corpus.image_count)
        if digest != corpus.sha256:
            raise FormalManifestError(
                f'data authority {name} SHA-256 mismatch')


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
    if expected_target.stat().st_mode & stat.S_IWUSR:
        raise FormalManifestError(
            'canonical prior artifact bundle root is not read-only')

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
    allowed_directories: set[str] = set()
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
        parent = binding.path.parent
        while parent != Path('.'):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    if bundle_manifest.stat().st_mode & stat.S_IWUSR:
        raise FormalManifestError(
            'prior artifact bundle manifest is not read-only')
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in expected_target.rglob('*'):
        relative = path.relative_to(expected_target).as_posix()
        if path.is_symlink():
            raise FormalManifestError(
                'prior artifact bundle contains an unexpected symlink')
        if path.is_dir():
            actual_directories.add(relative)
            if path.stat().st_mode & stat.S_IWUSR:
                raise FormalManifestError(
                    'prior artifact bundle directory is not read-only')
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise FormalManifestError(
                'prior artifact bundle contains a non-file entry')
    if actual_files != allowed or actual_directories != allowed_directories:
        raise FormalManifestError(
            'prior artifact bundle contains unexpected entries')


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
    if workers != 2:
        raise FormalManifestError('formal worker count must be 2')
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
        value: Any, *, repository_root: Path,
        ) -> dict[str, AssetBinding | CorpusBinding]:
    document = _require_mapping(value, 'data authority')
    _require_exact_fields(document, _DATA_FIELDS, 'data authority')
    parsed: dict[str, AssetBinding | CorpusBinding] = {}
    for name in _DATA_FILE_FIELDS:
        binding = AssetBinding.from_dict(
            document[name], repository_root=repository_root)
        authority, relative, sha256 = _CANONICAL_DATA_FILES[name]
        if (
                binding.authority_root != authority
                or binding.asset_relative_path != Path(relative)
                or binding.sha256 != sha256):
            raise FormalManifestError(
                f'data authority {name} is not canonical')
        parsed[name] = binding
    for name in _DATA_CORPUS_FIELDS:
        binding = CorpusBinding.from_dict(document[name])
        expected = _CANONICAL_CORPORA[name]
        if (
                binding.corpus_relative_path
                != Path(expected['corpus_relative_path'])
                or binding.archive_role != expected['archive_role']
                or binding.archive_prefix != expected['archive_prefix']
                or binding.image_count != expected['image_count']
                or binding.sha256 != expected['sha256']):
            raise FormalManifestError(
                f'data authority {name} is not canonical')
        parsed[name] = binding
    return parsed


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
    expected_id, expected_config, expected_output = _canonical_run_identity(
        run.role, run.seed)
    if run.run_id != expected_id:
        raise FormalManifestError(
            f'run id must be canonical for role and seed: {expected_id}')
    if run.config != expected_config:
        raise FormalManifestError(
            f'{run.run_id} config path must be {expected_config.as_posix()}')
    if run.output_root != expected_output:
        raise FormalManifestError(
            f'{run.run_id} output_root must be {expected_output.as_posix()}')


def _canonical_run_identity(
        role: str, seed: int) -> tuple[str, Path, Path]:
    stem = 'full' if role == 'baseline' else 'no_pif'
    run_stem = stem.replace('_', '-')
    run_id = f'{run_stem}-seed{seed}'
    config = Path(
        f'configs/optimization/formal_stage_c/{stem}_seed{seed}.py')
    output_root = Path(
        f'work_dirs/optimization/formal-stage-c/{run_id}')
    return run_id, config, output_root


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


def _require_commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise FormalManifestError(f'{field} must be a full lowercase commit')
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
    _validate_internal_relative_path(root, binding.path, field)
    if candidate.is_symlink() or not candidate.is_file():
        raise FormalManifestError(f'{field} file is missing: {binding.path}')
    digest = hashlib.sha256()
    with candidate.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != binding.sha256:
        raise FormalManifestError(f'{field} SHA-256 mismatch')


def _read_internal_regular_nofollow(
        root: Path, relative: Path, field: str) -> bytes:
    if relative.is_absolute() or not relative.parts or any(
            part in {'', '.', '..'} for part in relative.parts):
        raise FormalManifestError(f'{field} path is not canonical')
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, 'O_CLOEXEC', 0))
    descriptor = None
    try:
        descriptor = os.open('/', flags)
        for component in root.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        for component in relative.parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_fd = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0),
            dir_fd=descriptor)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise FormalManifestError(f'{field} is not a regular file')
            chunks: list[bytes] = []
            while True:
                block = os.read(file_fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(file_fd)
            if (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns):
                raise FormalManifestError(f'{field} changed during read')
            return b''.join(chunks)
        finally:
            os.close(file_fd)
    except FormalManifestError:
        raise
    except OSError as error:
        raise FormalManifestError(f'{field} is unavailable') from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_final_epoch_commit_best(
        root: Path, *, output_root: Path, run_id: str,
        role: Literal['baseline', 'no_pif'], seed: int,
        run_init_sha256: str, best: FileBinding,
        latest_resume: FileBinding, structured_log: FileBinding,
        final_order_sha256: str,
        captured_payload: bytes | None = None) -> None:
    if captured_payload is not None and not isinstance(
            captured_payload, bytes):
        raise FormalManifestError(
            'captured final epoch commit must be bytes')
    relative = output_root / 'epoch-commits/epoch_300.json'
    payload = (captured_payload if captured_payload is not None else
               _read_internal_regular_nofollow(
                   root, relative, 'final epoch commit'))
    try:
        document = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalManifestError('final epoch commit is malformed') from error
    fields = {
        'schema_version', 'identity', 'completed_epoch', 'checkpoint',
        'structured_log', 'log_record', 'best', 'previous_commit'}
    if not isinstance(document, Mapping) or set(document) != fields \
            or document['schema_version'] != 1 \
            or document['completed_epoch'] != 300:
        raise FormalManifestError('final epoch commit fields are invalid')
    if document['identity'] != {
            'run_id': run_id, 'role': role, 'seed': seed,
            'run_init_sha256': run_init_sha256}:
        raise FormalManifestError('final epoch commit identity is invalid')
    if document['checkpoint'] != {
            'path': latest_resume.path.name,
            'sha256': latest_resume.sha256}:
        raise FormalManifestError(
            'final epoch commit resume authority is invalid')
    if document['structured_log'] != {
            'path': structured_log.path.name,
            'sha256': structured_log.sha256}:
        raise FormalManifestError(
            'final epoch commit log authority is invalid')
    expected_record = {
        'schema_version': 1, 'run_id': run_id, 'role': role, 'seed': seed,
        'epoch': 300, 'order_sha256': final_order_sha256,
        'resume_checkpoint': latest_resume.path.name,
        'resume_sha256': latest_resume.sha256}
    if document['log_record'] != expected_record:
        raise FormalManifestError(
            'final epoch commit log record is invalid')
    authority = document['best']
    if not isinstance(authority, Mapping) or set(authority) != {
            'decision', 'checkpoint'} \
            or authority['checkpoint'] != {
                'path': best.path.name, 'sha256': best.sha256}:
        raise FormalManifestError(
            'final epoch commit best authority mismatch')
    decision = authority['decision']
    best_epoch = int(re.fullmatch(
        r'best_coco_AP_epoch_([1-9][0-9]*)\.pth', best.path.name).group(1))
    if not isinstance(decision, Mapping) or set(decision) != {
            'path', 'sha256'} or decision['path'] != (
                f'best-lineages/epoch_{best_epoch}.json') \
            or not isinstance(decision['sha256'], str) \
            or _SHA256.fullmatch(decision['sha256']) is None:
        raise FormalManifestError(
            'final epoch commit best decision is invalid')
    previous = document['previous_commit']
    if not isinstance(previous, Mapping) or set(previous) != {
            'path', 'sha256'} or previous['path'] != 'epoch_299.json' \
            or not isinstance(previous['sha256'], str) \
            or _SHA256.fullmatch(previous['sha256']) is None:
        raise FormalManifestError(
            'final epoch commit predecessor is invalid')
