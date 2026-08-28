#!/usr/bin/env python3
"""Build or check the immutable formal Stage C manifest.

Only the Python standard library and the standard-library-only formal schema
are imported.  This tool hashes files as bytes and never deserializes a model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from mambapose_opt.formal_schema import (  # noqa: E402
    AssetBinding,
    CONDITIONAL_SEEDS,
    CorpusBinding,
    INITIALIZATION_ID,
    INITIALIZATION_PATH,
    INITIALIZATION_SHA256,
    PRIMARY_SEEDS,
    PriorArtifactAuthority,
    canonical_main_root,
    config_closure_sha256,
    validate_canonical_formal_config_closure,
    validate_asset_binding,
    validate_data_authority,
    validate_prior_artifact_authority,
)


_DATA_ASSETS = {
    'inventory': ('data', 'inventory.json'),
    'train_annotations': (
        'data', 'coco/annotations/person_keypoints_train2017.json'),
    'validation_annotations': (
        'data', 'coco/annotations/person_keypoints_val2017.json'),
    'detections': (
        'data',
        'coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json'),
    'annotation_archive': (
        'work_dirs/reproduction',
        'downloads/annotations_trainval2017.zip'),
    'train_image_archive': (
        'work_dirs/reproduction', 'downloads/train2017.zip'),
    'validation_image_archive': (
        'work_dirs/reproduction', 'downloads/val2017.zip'),
}
_DATA_CORPORA = {
    'train_image_corpus': {
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'corpus_relative_path': 'coco/train2017',
        'archive_role': 'train_image_archive',
        'archive_prefix': 'train2017/',
        'image_count': 118287,
        'digest_algorithm': 'sha256-zip-member-and-extracted-content-v1',
        'sha256': (
            'f552fb95e1f40129d9726146ddfbadcbf4fb931bf6c8631cf6425a6994999695'),
    },
    'validation_image_corpus': {
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'corpus_relative_path': 'coco/val2017',
        'archive_role': 'validation_image_archive',
        'archive_prefix': 'val2017/',
        'image_count': 5000,
        'digest_algorithm': 'sha256-zip-member-and-extracted-content-v1',
        'sha256': (
            '6bf5c46be73304e0e8d77af5cf5764eed9e598e1f815f844e3366675c23e610e'),
    },
}
_TARGET_ROOTS = {
    'pretrained': 'canonical-main/pretrained',
    'data': 'canonical-main/data',
    'work_dirs/reproduction': 'canonical-main/work_dirs/reproduction',
}
_LEAF_FIELDS = frozenset({
    '_base_', 'formal_role', 'formal_seed', 'experiment_id', 'work_dir',
    'randomness', 'train_dataloader', 'val_dataloader', 'test_dataloader',
    'model',
})


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f'manifest input file is missing: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _asset_binding(
        repository_root: Path, canonical_root: Path, authority_root: str,
        asset_relative_path: str) -> dict[str, str]:
    document = {
        'authority_root': authority_root,
        'target_root': _TARGET_ROOTS[authority_root],
        'asset_relative_path': asset_relative_path,
        'sha256': _sha256_file(
            canonical_root / authority_root / asset_relative_path),
    }
    binding = AssetBinding.from_dict(
        document, repository_root=repository_root)
    validate_asset_binding(
        binding,
        repository_root=repository_root,
        canonical_repository_root=canonical_root)
    return document


def _prior_artifact_binding(
        root: Path, canonical_root: Path) -> dict[str, Any]:
    link_root = Path('work_dirs/optimization/prior-stage-b')
    bundle_path = root / link_root / 'bundle.json'
    if not bundle_path.is_file():
        raise ValueError('formal prior-stage-b bundle is missing')
    try:
        bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('formal prior-stage-b bundle is malformed') from error
    if set(bundle) != {
            'schema_version', 'kind', 'source_commit',
            'unpruned_parent_checkpoint_sha256', 'entries'}:
        raise ValueError('formal prior-stage-b bundle has wrong fields')
    entries = bundle['entries']
    if not isinstance(entries, dict):
        raise ValueError('formal prior-stage-b entries must be an object')
    artifacts = {
        name: {'path': record['path'], 'sha256': record['sha256']}
        for name, record in entries.items()
    }
    audit = artifacts.get('independent_audit')
    if not isinstance(audit, dict):
        raise ValueError('formal prior-stage-b audit is missing')
    document = {
        'link_root': link_root.as_posix(),
        'target_root': (
            'canonical-main/work_dirs/optimization/'
            'formal-stage-c-prior/no-pif-seed0'),
        'bundle_manifest_sha256': _sha256_file(bundle_path),
        'source_commit': bundle['source_commit'],
        'unpruned_parent_checkpoint_sha256': (
            bundle['unpruned_parent_checkpoint_sha256']),
        'audit_sha256': audit['sha256'],
        'artifacts': artifacts,
    }
    authority = PriorArtifactAuthority.from_dict(
        document, repository_root=root)
    validate_prior_artifact_authority(
        authority,
        repository_root=root,
        canonical_repository_root=canonical_root)
    return document


def _ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _ast_value(key): _ast_value(item)
            for key, item in zip(node.keys, node.values)}
    if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'dict'
            and not node.args):
        return {keyword.arg: _ast_value(keyword.value)
                for keyword in node.keywords}
    raise ValueError('paired config uses a non-literal assignment')


def _leaf_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            raise ValueError('paired config symmetry forbids executable statements')
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id in values:
            raise ValueError('paired config symmetry requires unique named assignments')
        values[target.id] = _ast_value(node.value)
    if set(values) != _LEAF_FIELDS:
        raise ValueError(
            'paired config symmetry requires the exact formal leaf fields')
    return values


def _expected_leaf(role: str, seed: int) -> dict[str, Any]:
    if role not in {'baseline', 'no_pif'}:
        raise ValueError('paired config symmetry received an invalid role')
    stem = 'full' if role == 'baseline' else 'no_pif'
    run_stem = stem.replace('_', '-')
    run_id = f'{run_stem}-seed{seed}'
    return {
        '_base_': ['./_base_/paired_300ep.py'],
        'formal_role': role,
        'formal_seed': seed,
        'experiment_id': f'formal-stage-c-{run_id}',
        'work_dir': f'work_dirs/optimization/formal-stage-c/{run_id}',
        'randomness': {'seed': seed, 'deterministic': True},
        'train_dataloader': {'sampler': {'seed': seed}},
        'val_dataloader': {'sampler': {'seed': seed}},
        'test_dataloader': {'sampler': {'seed': seed}},
        'model': {'head': {'tokenpose_cfg': {
            'pif_mode': 'full' if role == 'baseline' else 'disabled'}}},
    }


def validate_paired_config_symmetry(
        repository_root: Path | str, baseline_config: Path | str,
        no_pif_config: Path | str, *, seed: int) -> None:
    """Require the exact paired leaf projection for one fixed seed."""
    root = Path(repository_root).resolve(strict=False)
    baseline = Path(baseline_config)
    no_pif = Path(no_pif_config)
    if not baseline.is_absolute():
        baseline = root / baseline
    if not no_pif.is_absolute():
        no_pif = root / no_pif
    actual_baseline = _leaf_assignments(baseline)
    actual_no_pif = _leaf_assignments(no_pif)
    if (
            actual_baseline != _expected_leaf('baseline', seed)
            or actual_no_pif != _expected_leaf('no_pif', seed)):
        raise ValueError(
            f'paired config symmetry mismatch for seed {seed}')
    try:
        validate_canonical_formal_config_closure(root)
    except ValueError as error:
        raise ValueError(f'paired config canonical closure mismatch: {error}') \
            from error


def build_manifest_document(
        repository_root: Path | str, *,
        canonical_repository_root: Path | str | None = None,
        ) -> dict[str, Any]:
    """Build a commit-independent document from configs and asset bytes."""
    root = Path(repository_root).resolve(strict=True)
    canonical = (
        canonical_main_root(root)
        if canonical_repository_root is None
        else Path(canonical_repository_root).absolute())
    config_root = Path('configs/optimization/formal_stage_c')
    runs: list[dict[str, Any]] = []
    for seed in (*PRIMARY_SEEDS, *CONDITIONAL_SEEDS):
        baseline = config_root / f'full_seed{seed}.py'
        no_pif = config_root / f'no_pif_seed{seed}.py'
        validate_paired_config_symmetry(root, baseline, no_pif, seed=seed)
        for role, config in (('baseline', baseline), ('no_pif', no_pif)):
            stem = 'full' if role == 'baseline' else 'no-pif'
            run_id = f'{stem}-seed{seed}'
            runs.append({
                'run_id': run_id,
                'role': role,
                'seed': seed,
                'conditional': seed in CONDITIONAL_SEEDS,
                'config': config.as_posix(),
                'config_sha256': config_closure_sha256(root, config),
                'initialization_id': INITIALIZATION_ID,
                'output_root': (
                    f'work_dirs/optimization/formal-stage-c/{run_id}'),
            })
    data_authority = {
        name: _asset_binding(root, canonical, authority, relative)
        for name, (authority, relative) in _DATA_ASSETS.items()
    }
    data_authority.update(_DATA_CORPORA)
    parsed_data = {
        name: AssetBinding.from_dict(document, repository_root=root)
        for name, document in data_authority.items()
        if name in _DATA_ASSETS}
    parsed_data.update({
        name: CorpusBinding.from_dict(document)
        for name, document in data_authority.items()
        if name in _DATA_CORPORA})
    validate_data_authority(
        parsed_data, repository_root=root,
        canonical_repository_root=canonical)
    prior_artifact = _prior_artifact_binding(root, canonical)
    initialization_asset = _asset_binding(
        root, canonical, 'pretrained', INITIALIZATION_PATH.name)
    initialization_sha256 = initialization_asset['sha256']
    if initialization_sha256 != INITIALIZATION_SHA256:
        raise ValueError('formal initialization SHA-256 mismatch')
    return {
        'schema_version': 2,
        'experiment_id': 'mambapose-formal-stage-c',
        'initialization': {
            'id': INITIALIZATION_ID,
            'kind': 'vmamba-backbone',
            'asset': initialization_asset,
        },
        'protocol': {
            'epochs': 300,
            'worker_count': 2,
            'persistent_workers': False,
            'per_device_batch_size': 128,
            'world_size': 1,
            'accumulation_steps': 1,
            'effective_batch_size': 128,
            'primary_seeds': list(PRIMARY_SEEDS),
            'conditional_seeds': list(CONDITIONAL_SEEDS),
        },
        'data_authority': data_authority,
        'prior_artifact': prior_artifact,
        'runs': runs,
    }


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(
        document, sort_keys=True, indent=2,
        ensure_ascii=False) + '\n').encode('utf-8')


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--repository-root', type=Path, default=DEFAULT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    destination = args.manifest
    if not destination.is_absolute():
        destination = root / destination
    expected = canonical_bytes(build_manifest_document(root))
    if args.check:
        if not destination.is_file() or destination.read_bytes() != expected:
            print('formal manifest is stale', file=sys.stderr)
            return 1
        status = 'current'
    else:
        _atomic_write(destination, expected)
        status = 'written'
    print(json.dumps({
        'status': status,
        'torch_imported': 'torch' in sys.modules,
        'mmengine_imported': 'mmengine' in sys.modules,
        'mmpose_imported': 'mmpose' in sys.modules,
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
