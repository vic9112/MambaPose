import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from mmengine.config import Config
import pytest

import mambapose_opt.formal_schema as formal_schema
from mambapose_opt.formal_schema import (
    AssetBinding,
    FormalManifestError,
    FormalRunInit,
    FormalStageCManifest,
    FormalTrainResult,
    PriorArtifactAuthority,
    load_formal_manifest,
    validate_asset_binding,
    validate_prior_artifact_authority,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'optimization/formal_stage_c.json'
BUILDER = ROOT / 'tools/optimization/build_formal_manifest.py'
INIT_SHA256 = '09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201'
ROLES = ('baseline', 'no_pif')
SEEDS = tuple(range(5))
PRIOR_BUNDLE_SHA256 = (
    '522645bdd5612af0a58009b0b30ce1aae22ed4d460c482ba08799c1500b6ed9f')


def _prior_artifact() -> dict[str, object]:
    records = {
        'train': ('artifacts/train.json',
                  'ea276713f214acce1949798445203e7e05026dd4ed7d6e76e77cffde35041819'),
        'runtime_metadata': (
            'artifacts/runtime-metadata.json',
            '99e75ff3a4862f08eba3a7c75e586617aa1438efa8f164a73da093ffe5b672ce'),
        'pruned_checkpoint': (
            'artifacts/pruned-runtime.pth',
            '5797ceaffbf7d369d8eaf8f47b548a79bd43d66dd603933571fa3b88ff28e3db'),
        'profile': ('artifacts/profile.json',
                    'cb0c933291c28bc341cc5bb62b956c14a76255825321bc0291d4e688fa7aa7a3'),
        'evaluate': ('artifacts/evaluate.json',
                     '9bcf326417cb0b9ffbf78deb01d19ac09525d9deb491c93189e8b769df0f3bf5'),
        'latency': ('artifacts/latency.json',
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
            'b7382dbaca7294477990c2463bf642f3959762a7b9e54d626e46e45d8bb5f361'),
    }
    return {
        'link_root': 'work_dirs/optimization/prior-stage-b',
        'target_root': (
            'canonical-main/work_dirs/optimization/'
            'formal-stage-c-prior/no-pif-seed0'),
        'bundle_manifest_sha256': PRIOR_BUNDLE_SHA256,
        'source_commit': 'a6adf6d84f9b51b133b0fca7272c51f728bfb7ac',
        'unpruned_parent_checkpoint_sha256': (
            '28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb'),
        'audit_sha256': (
            'b7382dbaca7294477990c2463bf642f3959762a7b9e54d626e46e45d8bb5f361'),
        'artifacts': {
            name: {'path': path, 'sha256': sha256}
            for name, (path, sha256) in records.items()
        },
    }


def _sha(value: bytes = b'fixture') -> str:
    return hashlib.sha256(value).hexdigest()


def _binding(name: str) -> dict[str, str]:
    return {
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'asset_relative_path': f'{name}.json',
        'sha256': _sha(name.encode()),
    }


def _data_authority() -> dict[str, object]:
    files = {
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
            'work_dirs/reproduction',
            'downloads/annotations_trainval2017.zip',
            '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'),
        'train_image_archive': (
            'work_dirs/reproduction', 'downloads/train2017.zip',
            '69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929'),
        'validation_image_archive': (
            'work_dirs/reproduction', 'downloads/val2017.zip',
            '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05'),
    }
    result: dict[str, object] = {
        name: {
            'authority_root': authority,
            'target_root': f'canonical-main/{authority}',
            'asset_relative_path': relative,
            'sha256': sha256,
        }
        for name, (authority, relative, sha256) in files.items()
    }
    result.update({
        'train_image_corpus': {
            'authority_root': 'data',
            'target_root': 'canonical-main/data',
            'corpus_relative_path': 'coco/train2017',
            'archive_role': 'train_image_archive',
            'archive_prefix': 'train2017/',
            'image_count': 118287,
            'digest_algorithm': (
                'sha256-zip-member-and-extracted-content-v1'),
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
            'digest_algorithm': (
                'sha256-zip-member-and-extracted-content-v1'),
            'sha256': (
                '6bf5c46be73304e0e8d77af5cf5764eed9e598e1f815f844e3366675c23e610e'),
        },
    })
    return result


def _run(role: str, seed: int) -> dict[str, object]:
    stem = 'full' if role == 'baseline' else 'no_pif'
    run_id = f'{stem.replace("_", "-")}-seed{seed}'
    return {
        'run_id': run_id,
        'role': role,
        'seed': seed,
        'conditional': seed in (3, 4),
        'config': f'configs/optimization/formal_stage_c/{stem}_seed{seed}.py',
        'config_sha256': _sha(f'{role}:{seed}'.encode()),
        'initialization_id': 'vmamba-t-imagenet-262',
        'output_root': f'work_dirs/optimization/formal-stage-c/{run_id}',
    }


def _manifest_document() -> dict[str, object]:
    return {
        'schema_version': 2,
        'experiment_id': 'mambapose-formal-stage-c',
        'initialization': {
            'id': 'vmamba-t-imagenet-262',
            'kind': 'vmamba-backbone',
            'asset': {
                'authority_root': 'pretrained',
                'target_root': 'canonical-main/pretrained',
                'asset_relative_path': 'vssm_tiny_0230_ckpt_epoch_262.pth',
                'sha256': INIT_SHA256,
            },
        },
        'protocol': {
            'epochs': 300,
            'worker_count': 2,
            'persistent_workers': False,
            'per_device_batch_size': 128,
            'world_size': 1,
            'accumulation_steps': 1,
            'effective_batch_size': 128,
            'primary_seeds': [0, 1, 2],
            'conditional_seeds': [3, 4],
        },
        'data_authority': _data_authority(),
        'prior_artifact': _prior_artifact(),
        'runs': [
            _run(role, seed)
            for seed in SEEDS
            for role in ROLES
        ],
    }


def _run_init_document() -> dict[str, object]:
    closure = formal_schema.config_closure_sha256(
        ROOT, 'configs/optimization/formal_stage_c/full_seed0.py')
    resolved = formal_schema.formal_resolved_config_sha256(
        ROOT, role='baseline', seed=0)
    data = _data_authority()
    return {
        'schema_version': 1,
        'manifest_sha256': _sha(b'manifest'),
        'source': {'git_commit': 'a' * 40, 'clean_tree': True},
        'config': {
            'path': 'configs/optimization/formal_stage_c/full_seed0.py',
            'closure_sha256': closure,
            'resolved_sha256': resolved,
        },
        'environment_inventory_sha256': _sha(b'environment'),
        'data_authority': {
            f'{name}_sha256': record['sha256']
            for name, record in data.items()
        },
        'run': {
            'run_id': 'full-seed0',
            'role': 'baseline',
            'seed': 0,
            'epochs': 300,
            'effective_batch_size': 128,
            'worker_count': 2,
            'persistent_workers': False,
            'output_root': 'work_dirs/optimization/formal-stage-c/full-seed0',
        },
        'initialization': {
            'id': 'vmamba-t-imagenet-262',
            'kind': 'vmamba-backbone',
            'asset': {
                'authority_root': 'pretrained',
                'target_root': 'canonical-main/pretrained',
                'asset_relative_path': 'vssm_tiny_0230_ckpt_epoch_262.pth',
                'sha256': INIT_SHA256,
            },
        },
    }


def _train_result_document() -> dict[str, object]:
    output = 'work_dirs/optimization/formal-stage-c/full-seed0'
    return {
        'schema_version': 1,
        'run_init_sha256': _sha(b'run-init'),
        'initialization': {
            'id': 'vmamba-t-imagenet-262',
            'kind': 'vmamba-backbone',
            'asset': {
                'authority_root': 'pretrained',
                'target_root': 'canonical-main/pretrained',
                'asset_relative_path': 'vssm_tiny_0230_ckpt_epoch_262.pth',
                'sha256': INIT_SHA256,
            },
        },
        'run': {
            'run_id': 'full-seed0',
            'role': 'baseline',
            'seed': 0,
            'output_root': output,
        },
        'best_checkpoint': {
            'path': f'{output}/best_coco_AP_epoch_300.pth',
            'sha256': _sha(b'best'),
        },
        'resume_checkpoints': [
            {'path': f'{output}/epoch_299.pth', 'sha256': _sha(b'299')},
            {'path': f'{output}/epoch_300.pth', 'sha256': _sha(b'300')},
        ],
        'structured_log': {
            'path': f'{output}/training.jsonl',
            'sha256': _sha(b'log'),
        },
        'order_hashes': [
            {'epoch': epoch, 'sha256': _sha(f'epoch:{epoch}'.encode())}
            for epoch in range(1, 301)],
        'final_epoch': 300,
        'status': 'complete',
    }


def test_manifest_requires_exact_paired_seed_matrix(tmp_path):
    document = _manifest_document()
    document['runs'].pop()
    with pytest.raises(FormalManifestError, match='paired seed matrix'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


@pytest.mark.parametrize(
    ('path', 'value', 'match'),
    [
        (('unexpected',), True, 'unknown fields'),
        (('protocol', 'epochs'), 299, '300 epochs'),
        (('protocol', 'persistent_workers'), True, 'disabled'),
        (('protocol', 'world_size'), 2, 'world size'),
        (('protocol', 'accumulation_steps'), 2, 'accumulation'),
        (('protocol', 'effective_batch_size'), 64, 'effective batch'),
        (('protocol', 'primary_seeds'), [0, 2, 1], 'primary seeds'),
        (('protocol', 'conditional_seeds'), [3], 'conditional seeds'),
        (('runs', 0, 'seed'), True, 'seed'),
        (('runs', 0, 'config_sha256'), 'not-a-hash', 'SHA-256'),
        (('runs', 0, 'config'), '/tmp/config.py', 'relative'),
        (('runs', 0, 'output_root'), '../escaped', 'traversal'),
    ],
)
def test_manifest_rejects_malformed_contract(tmp_path, path, value, match):
    document = _manifest_document()
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(FormalManifestError, match=match):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_rejects_duplicate_run_ids(tmp_path):
    document = _manifest_document()
    document['runs'][1]['run_id'] = document['runs'][0]['run_id']
    with pytest.raises(FormalManifestError, match='run ids must be unique'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_requires_prior_artifact_authority(tmp_path):
    document = _manifest_document()
    document.pop('prior_artifact')
    with pytest.raises(FormalManifestError, match='missing fields'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_prior_artifact_rejects_missing_candidate_result_member(tmp_path):
    document = _manifest_document()
    document['prior_artifact']['artifacts'].pop('evaluate')
    with pytest.raises(FormalManifestError, match='artifact roles'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_prior_artifact_rejects_swapped_checkpoint_roles(tmp_path):
    document = _manifest_document()
    prior = document['prior_artifact']
    pruned = prior['artifacts']['pruned_checkpoint']['sha256']
    parent = prior['unpruned_parent_checkpoint_sha256']
    prior['artifacts']['pruned_checkpoint']['sha256'] = parent
    prior['unpruned_parent_checkpoint_sha256'] = pruned
    with pytest.raises(FormalManifestError, match='checkpoint roles'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_prior_artifact_accepts_exact_bundle_authority():
    authority = PriorArtifactAuthority.from_dict(
        _prior_artifact(), repository_root=ROOT)
    validate_prior_artifact_authority(
        authority, repository_root=ROOT,
        canonical_repository_root=Path('/home/vicchen/workspace/MambaPose'))
    assert authority.artifacts['pruned_checkpoint'].sha256.startswith('5797ce')


def test_prior_artifact_rejects_alternate_same_byte_bundle(tmp_path):
    canonical = Path('/home/vicchen/workspace/MambaPose')
    real_bundle = (
        canonical / 'work_dirs/optimization/formal-stage-c-prior/no-pif-seed0')
    alias = tmp_path / 'alias'
    alias.symlink_to(real_bundle)
    runtime = tmp_path / 'runtime'
    link = runtime / 'work_dirs/optimization/prior-stage-b'
    link.parent.mkdir(parents=True)
    link.symlink_to(alias)
    authority = PriorArtifactAuthority.from_dict(
        _prior_artifact(), repository_root=runtime)
    with pytest.raises(FormalManifestError, match='link target drift'):
        validate_prior_artifact_authority(
            authority, repository_root=runtime,
            canonical_repository_root=canonical)


def test_manifest_rejects_duplicate_role_seed(tmp_path):
    document = _manifest_document()
    document['runs'][1]['role'] = 'baseline'
    with pytest.raises(FormalManifestError, match='paired seed matrix'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_rejects_wrong_conditional_flag(tmp_path):
    document = _manifest_document()
    document['runs'][6]['conditional'] = False
    with pytest.raises(FormalManifestError, match='conditional flag'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_requires_exact_worker_count(tmp_path):
    document = _manifest_document()
    document['protocol']['worker_count'] = 99
    with pytest.raises(FormalManifestError, match='worker count'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_rejects_swapped_data_roles(tmp_path):
    document = _manifest_document()
    data = document['data_authority']
    data['train_annotations'], data['validation_annotations'] = (
        data['validation_annotations'], data['train_annotations'])
    with pytest.raises(FormalManifestError, match='data authority'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_tracked_manifest_binds_archives_and_live_image_corpora():
    document = json.loads(MANIFEST.read_text())
    assert document['schema_version'] == 2
    assert set(document['data_authority']) == {
        'inventory', 'train_annotations', 'validation_annotations',
        'detections', 'annotation_archive', 'train_image_archive',
        'validation_image_archive', 'train_image_corpus',
        'validation_image_corpus',
    }
    assert document['data_authority']['train_image_corpus'] == {
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'corpus_relative_path': 'coco/train2017',
        'archive_role': 'train_image_archive',
        'archive_prefix': 'train2017/',
        'image_count': 118287,
        'digest_algorithm': 'sha256-zip-member-and-extracted-content-v1',
        'sha256': 'f552fb95e1f40129d9726146ddfbadcbf4fb931bf6c8631cf6425a6994999695',
    }
    assert document['data_authority']['validation_image_corpus'][
        'image_count'] == 5000


def test_zip_corpus_verifier_binds_member_bytes_count_and_digest(tmp_path):
    archive = tmp_path / 'images.zip'
    image_root = tmp_path / 'images'
    image_root.mkdir()
    members = {'train2017/0002.jpg': b'second',
               'train2017/0001.jpg': b'first'}
    with zipfile.ZipFile(archive, 'w') as stream:
        for name, payload in members.items():
            stream.writestr(name, payload)
            (image_root / Path(name).name).write_bytes(payload)
    digest = formal_schema._verify_zip_corpus(
        archive, image_root, prefix='train2017/', expected_count=2)
    expected = hashlib.sha256()
    for name in sorted(members):
        payload = members[name]
        expected.update(name.encode())
        expected.update(b'\0')
        expected.update(str(len(payload)).encode())
        expected.update(b'\0')
        expected.update(payload)
    assert digest == expected.hexdigest()


def test_zip_corpus_verifier_rejects_live_byte_drift(tmp_path):
    archive = tmp_path / 'images.zip'
    image_root = tmp_path / 'images'
    image_root.mkdir()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('val2017/0001.jpg', b'official')
    (image_root / '0001.jpg').write_bytes(b'changed!')
    with pytest.raises(FormalManifestError, match='differs from archive'):
        formal_schema._verify_zip_corpus(
            archive, image_root, prefix='val2017/', expected_count=1)


def test_manifest_rejects_output_root_alias(tmp_path):
    document = _manifest_document()
    document['runs'][1]['output_root'] = document['runs'][0]['output_root']
    with pytest.raises(FormalManifestError, match='output roots must be unique'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_rejects_duplicate_config_paths(tmp_path):
    document = _manifest_document()
    document['runs'][1]['config'] = document['runs'][0]['config']
    with pytest.raises(FormalManifestError, match='config paths must be unique'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_rejects_initialization_path_in_output_field(tmp_path):
    document = _manifest_document()
    document['runs'][0]['output_root'] = 'pretrained/model.pth'
    with pytest.raises(FormalManifestError, match='output_root'):
        FormalStageCManifest.from_dict(document, repository_root=tmp_path)


def test_manifest_parses_frozen_records(tmp_path):
    manifest = FormalStageCManifest.from_dict(
        _manifest_document(), repository_root=tmp_path)
    assert manifest.protocol.primary_seeds == (0, 1, 2)
    assert manifest.protocol.conditional_seeds == (3, 4)
    assert len(manifest.runs) == 10
    assert manifest.runs[0].role == 'baseline'
    assert manifest.runs[-1].seed == 4
    with pytest.raises(TypeError):
        manifest.data_authority['inventory'] = None


def _asset_layout(tmp_path):
    canonical = tmp_path / 'canonical-main'
    runtime = tmp_path / 'frozen-runtime'
    (canonical / 'pretrained').mkdir(parents=True)
    (canonical / 'data').mkdir()
    (canonical / 'work_dirs/reproduction').mkdir(parents=True)
    runtime.mkdir()
    (runtime / 'work_dirs').mkdir()
    (runtime / 'pretrained').symlink_to(canonical / 'pretrained')
    (runtime / 'data').symlink_to(canonical / 'data')
    (runtime / 'work_dirs/reproduction').symlink_to(
        canonical / 'work_dirs/reproduction')
    return canonical, runtime


@pytest.mark.parametrize(
    ('authority_root', 'target_root', 'relative_path'),
    [
        ('outside', 'canonical-main/outside', 'asset.bin'),
        ('data', 'canonical-main/pretrained', 'asset.bin'),
        ('data', 'canonical-main/data', '../asset.bin'),
        ('data', 'canonical-main/data', '/absolute.bin'),
    ],
)
def test_asset_binding_rejects_unapproved_or_unsafe_paths(
        authority_root, target_root, relative_path):
    with pytest.raises(FormalManifestError):
        AssetBinding.from_dict({
            'authority_root': authority_root,
            'target_root': target_root,
            'asset_relative_path': relative_path,
            'sha256': _sha(),
        }, repository_root=Path.cwd())


def test_asset_authority_accepts_exact_canonical_link(tmp_path):
    canonical, runtime = _asset_layout(tmp_path)
    payload = canonical / 'data/inventory.json'
    payload.write_bytes(b'authority')
    binding = AssetBinding.from_dict({
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'asset_relative_path': 'inventory.json',
        'sha256': _sha(b'authority'),
    }, repository_root=runtime)
    validate_asset_binding(
        binding, repository_root=runtime,
        canonical_repository_root=canonical)
    assert binding.path == Path('data/inventory.json')


def test_asset_authority_rejects_alternate_same_byte_tree(tmp_path):
    canonical, runtime = _asset_layout(tmp_path)
    alternate = tmp_path / 'alternate/data'
    alternate.mkdir(parents=True)
    (canonical / 'data/inventory.json').write_bytes(b'same')
    (alternate / 'inventory.json').write_bytes(b'same')
    (runtime / 'data').unlink()
    (runtime / 'data').symlink_to(alternate)
    binding = AssetBinding.from_dict({
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'asset_relative_path': 'inventory.json',
        'sha256': _sha(b'same'),
    }, repository_root=runtime)
    with pytest.raises(FormalManifestError, match='link target drift'):
        validate_asset_binding(
            binding, repository_root=runtime,
            canonical_repository_root=canonical)


def test_asset_authority_rejects_primary_link_drift(tmp_path):
    canonical, runtime = _asset_layout(tmp_path)
    alternate = tmp_path / 'alternate/data'
    alternate.mkdir(parents=True)
    (alternate / 'inventory.json').write_bytes(b'same')
    (canonical / 'data').rmdir()
    (canonical / 'data').symlink_to(alternate)
    binding = AssetBinding.from_dict({
        'authority_root': 'data',
        'target_root': 'canonical-main/data',
        'asset_relative_path': 'inventory.json',
        'sha256': _sha(b'same'),
    }, repository_root=runtime)
    with pytest.raises(FormalManifestError, match='canonical asset root'):
        validate_asset_binding(
            binding, repository_root=runtime,
            canonical_repository_root=canonical)


def test_asset_authority_rejects_hash_drift(tmp_path):
    canonical, runtime = _asset_layout(tmp_path)
    (canonical / 'pretrained/model.pth').write_bytes(b'changed')
    binding = AssetBinding.from_dict({
        'authority_root': 'pretrained',
        'target_root': 'canonical-main/pretrained',
        'asset_relative_path': 'model.pth',
        'sha256': _sha(b'expected'),
    }, repository_root=runtime)
    with pytest.raises(FormalManifestError, match='SHA-256 mismatch'):
        validate_asset_binding(
            binding, repository_root=runtime,
            canonical_repository_root=canonical)


def test_load_manifest_rejects_changed_config_hash(tmp_path):
    document = _manifest_document()
    path = tmp_path / 'formal.json'
    path.write_text(json.dumps(document))
    with pytest.raises(FormalManifestError, match='config closure SHA-256'):
        load_formal_manifest(path, repository_root=tmp_path)


def test_run_init_is_strict_and_keeps_initialization_separate():
    result = FormalRunInit.from_dict(
        _run_init_document(), repository_root=ROOT)
    assert result.run_id == 'full-seed0'
    assert result.initialization.sha256 == INIT_SHA256
    assert result.output_root.as_posix().endswith('full-seed0')


@pytest.mark.parametrize(
    ('mutation', 'match'),
    [
        (lambda value: value.update(unexpected=True), 'unknown fields'),
        (lambda value: value['source'].update(clean_tree=False), 'clean'),
        (lambda value: value['run'].update(seed=True), 'seed'),
        (lambda value: value['run'].update(epochs=60), '300 epochs'),
        (lambda value: value['run'].update(persistent_workers=True), 'disabled'),
        (lambda value: value['run'].update(output_root='pretrained/model.pth'),
         'output_root'),
    ],
)
def test_run_init_rejects_malformed_lineage(mutation, match):
    document = _run_init_document()
    mutation(document)
    with pytest.raises(FormalManifestError, match=match):
        FormalRunInit.from_dict(document, repository_root=ROOT)


@pytest.mark.parametrize(
    ('mutation', 'match'),
    [
        (lambda value: value['run'].update(effective_batch_size=1),
         'effective batch'),
        (lambda value: value['run'].update(worker_count=99), 'worker count'),
        (lambda value: value['run'].update(
            run_id='no-pif-seed0', role='baseline',
            output_root=(
                'work_dirs/optimization/formal-stage-c/no-pif-seed0')),
         'canonical'),
        (lambda value: value['config'].update(
            path='configs/optimization/formal_stage_c/no_pif_seed0.py'),
         'config'),
    ],
)
def test_run_init_rejects_noncanonical_manifest_identity(mutation, match):
    document = _run_init_document()
    mutation(document)
    with pytest.raises(FormalManifestError, match=match):
        FormalRunInit.from_dict(document, repository_root=ROOT)


def test_run_init_rejects_manifest_or_data_authority_rebinding():
    manifest_document = _manifest_document()
    manifest_document['runs'][0]['config_sha256'] = (
        formal_schema.config_closure_sha256(
            ROOT, 'configs/optimization/formal_stage_c/full_seed0.py'))
    manifest = FormalStageCManifest.from_dict(
        manifest_document, repository_root=ROOT)
    manifest_sha256 = formal_schema.canonical_json_sha256(manifest_document)
    document = _run_init_document()
    document['manifest_sha256'] = manifest_sha256
    result = FormalRunInit.from_dict(
        document, repository_root=ROOT, manifest=manifest,
        expected_manifest_sha256=manifest_sha256)
    assert result.run_id == 'full-seed0'

    document['data_authority']['train_image_corpus_sha256'] = '0' * 64
    with pytest.raises(FormalManifestError, match='data authority'):
        FormalRunInit.from_dict(
            document, repository_root=ROOT, manifest=manifest,
            expected_manifest_sha256=manifest_sha256)


def test_run_init_requires_exact_manifest_sha256_when_authority_is_given():
    manifest_document = _manifest_document()
    manifest = FormalStageCManifest.from_dict(
        manifest_document, repository_root=ROOT)
    document = _run_init_document()
    with pytest.raises(FormalManifestError, match='manifest SHA-256'):
        FormalRunInit.from_dict(
            document, repository_root=ROOT, manifest=manifest,
            expected_manifest_sha256='f' * 64)


def test_train_result_cannot_rebind_initialization():
    document = _train_result_document()
    document['initialization']['asset']['sha256'] = '0' * 64
    with pytest.raises(FormalManifestError, match='initialization authority'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


@pytest.mark.parametrize(
    ('mutation', 'match'),
    [
        (lambda value: value.update(final_epoch=299), 'final epoch'),
        (lambda value: value.update(status='running'), 'complete'),
        (lambda value: value.update(order_hashes=value['order_hashes'][:-1]),
         '300 order hashes'),
        (lambda value: value.update(resume_checkpoints=value['resume_checkpoints'][:1]),
         'exactly two'),
        (lambda value: value['best_checkpoint'].update(path='/tmp/model.pth'),
         'relative'),
    ],
)
def test_train_result_rejects_incomplete_output(mutation, match):
    document = _train_result_document()
    mutation(document)
    with pytest.raises(FormalManifestError, match=match):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_requires_outputs_under_declared_root():
    document = _train_result_document()
    document['structured_log']['path'] = 'work_dirs/optimization/other/log.jsonl'
    with pytest.raises(FormalManifestError, match='declared output root'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_role_run_id_mismatch():
    document = _train_result_document()
    document['run'].update(
        run_id='no-pif-seed0', role='baseline',
        output_root='work_dirs/optimization/formal-stage-c/no-pif-seed0')
    output = document['run']['output_root']
    document['best_checkpoint']['path'] = f'{output}/best_coco_AP_epoch_300.pth'
    document['resume_checkpoints'][0]['path'] = f'{output}/epoch_299.pth'
    document['resume_checkpoints'][1]['path'] = f'{output}/epoch_300.pth'
    document['structured_log']['path'] = f'{output}/training.jsonl'
    with pytest.raises(FormalManifestError, match='canonical'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_duplicate_resume_binding():
    document = _train_result_document()
    document['resume_checkpoints'][1] = copy.deepcopy(
        document['resume_checkpoints'][0])
    with pytest.raises(FormalManifestError, match='distinct'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_reversed_or_nonlatest_resume_sequence():
    document = _train_result_document()
    document['resume_checkpoints'].reverse()
    with pytest.raises(FormalManifestError, match='chronological|latest'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_best_checkpoint_as_resume():
    document = _train_result_document()
    document['best_checkpoint'] = copy.deepcopy(
        document['resume_checkpoints'][1])
    with pytest.raises(FormalManifestError, match='best checkpoint'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_duplicate_order_hashes():
    document = _train_result_document()
    document['order_hashes'][1]['sha256'] = (
        document['order_hashes'][0]['sha256'])
    with pytest.raises(FormalManifestError, match='order hashes.*distinct'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_noncanonical_order_hash_sequence():
    document = _train_result_document()
    document['order_hashes'].reverse()
    with pytest.raises(FormalManifestError, match='order hash.*sequence'):
        FormalTrainResult.from_dict(document, repository_root=ROOT)


def test_train_result_rejects_nonexistent_or_unhashed_outputs():
    with pytest.raises(FormalManifestError, match='file is missing'):
        FormalTrainResult.from_dict(
            _train_result_document(), repository_root=ROOT,
            verify_files=True)


def test_train_result_rejects_run_init_identity_rebinding():
    run_init_document = _run_init_document()
    run_init = FormalRunInit.from_dict(
        run_init_document, repository_root=ROOT)
    run_init_sha256 = formal_schema.canonical_json_sha256(run_init_document)
    document = _train_result_document()
    document['run_init_sha256'] = run_init_sha256
    result = FormalTrainResult.from_dict(
        document, repository_root=ROOT, run_init=run_init,
        expected_run_init_sha256=run_init_sha256)
    assert result.run_id == run_init.run_id

    document['run']['role'] = 'no_pif'
    with pytest.raises(FormalManifestError, match='run init|canonical'):
        FormalTrainResult.from_dict(
            document, repository_root=ROOT, run_init=run_init,
            expected_run_init_sha256=run_init_sha256)


def test_train_result_requires_exact_run_init_sha256_when_authority_is_given():
    run_init = FormalRunInit.from_dict(
        _run_init_document(), repository_root=ROOT)
    with pytest.raises(FormalManifestError, match='run init SHA-256'):
        FormalTrainResult.from_dict(
            _train_result_document(), repository_root=ROOT,
            run_init=run_init, expected_run_init_sha256='f' * 64)


@pytest.mark.parametrize('factory', [
    _manifest_document, _run_init_document, _train_result_document])
def test_formal_schema_rejects_boolean_schema_version(factory):
    document = factory()
    document['schema_version'] = True
    parser = {
        _manifest_document: FormalStageCManifest,
        _run_init_document: FormalRunInit,
        _train_result_document: FormalTrainResult,
    }[factory]
    with pytest.raises(FormalManifestError, match='schema_version'):
        parser.from_dict(document, repository_root=ROOT)


def test_tracked_manifest_and_configs_form_exact_pair_matrix():
    manifest = load_formal_manifest(MANIFEST, repository_root=ROOT)
    assert [(run.role, run.seed) for run in manifest.runs] == [
        (role, seed) for seed in SEEDS for role in ROLES]

    resolved = {run.run_id: Config.fromfile(ROOT / run.config) for run in manifest.runs}
    for seed in SEEDS:
        baseline = resolved[f'full-seed{seed}']
        no_pif = resolved[f'no-pif-seed{seed}']
        for config, role, mode in (
                (baseline, 'baseline', 'full'),
                (no_pif, 'no_pif', 'disabled')):
            assert config.formal_role == role
            assert config.formal_seed == seed
            assert config.randomness == {'seed': seed, 'deterministic': True}
            assert config.train_cfg.max_epochs == 300
            assert config.train_dataloader.batch_size == 128
            assert config.formal_protocol.per_device_batch_size == 128
            assert config.formal_protocol.world_size == 1
            assert config.formal_protocol.accumulation_steps == 1
            assert config.formal_protocol.effective_batch_size == 128
            assert 'auto_scale_lr' not in config
            for loader_name in ('train_dataloader', 'val_dataloader',
                                'test_dataloader'):
                loader = config[loader_name]
                assert loader.num_workers == 2
                assert loader.persistent_workers is False
                assert loader.worker_init_fn.type == 'mambapose_seed_worker'
                assert loader.sampler.seed == seed
            assert config.model.head.tokenpose_cfg.pif_mode == mode


def test_manifest_builder_check_is_read_only_and_standard_library_only(tmp_path):
    before = MANIFEST.read_bytes()
    command = [
        sys.executable, '-B', str(BUILDER), '--check', str(MANIFEST),
        '--repository-root', str(ROOT),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        env={'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
             'PATH': '/usr/bin:/bin'})
    assert completed.returncode == 0, completed.stderr
    assert MANIFEST.read_bytes() == before
    imported = json.loads(completed.stdout)
    assert imported['torch_imported'] is False
    assert imported['mmengine_imported'] is False
    assert imported['mmpose_imported'] is False
    assert imported['status'] == 'current'


def test_manifest_builder_has_no_containing_commit_self_reference():
    spec = importlib.util.spec_from_file_location('formal_builder_cycle', BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    first = module.build_manifest_document(ROOT)
    second = module.build_manifest_document(ROOT)
    assert first == second
    assert 'source_commit' not in first
    assert 'git_commit' not in json.dumps(first, sort_keys=True)


def test_manifest_builder_detects_config_pair_asymmetry(tmp_path):
    spec = importlib.util.spec_from_file_location('formal_builder', BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = tmp_path / 'repo'
    config_dir = root / 'configs/optimization/formal_stage_c'
    config_dir.mkdir(parents=True)
    (config_dir / 'full_seed0.py').write_text(
        "_base_ = ['./_base_/paired_300ep.py']\n"
        "formal_role = 'baseline'\nformal_seed = 0\n"
        "experiment_id = 'formal-stage-c-full-seed0'\n"
        "work_dir = 'work_dirs/optimization/formal-stage-c/full-seed0'\n"
        "randomness = dict(seed=0, deterministic=True)\n"
        "model = dict(head=dict(tokenpose_cfg=dict(pif_mode='full')))\n")
    (config_dir / 'no_pif_seed0.py').write_text(
        (config_dir / 'full_seed0.py').read_text().replace(
            "formal_role = 'baseline'", "formal_role = 'no_pif'").replace(
            'full-seed0', 'no-pif-seed0').replace(
            "pif_mode='full'", "pif_mode='disabled'")
        + "optim_wrapper = dict(optimizer=dict(lr=0.5))\n")
    with pytest.raises(ValueError, match='paired config symmetry'):
        module.validate_paired_config_symmetry(
            root, config_dir / 'full_seed0.py',
            config_dir / 'no_pif_seed0.py', seed=0)


def _copy_formal_config_closure(destination):
    relative_paths = (
        'configs/optimization/formal_stage_c/_base_/paired_300ep.py',
        'configs/optimization/formal_stage_c/full_seed0.py',
        'configs/optimization/formal_stage_c/no_pif_seed0.py',
        'configs/reproduction/coco_s_v1.py',
        ('configs/body_2d_keypoint/tokenpose/'
         'mamba_tokenpose_T2_coco_256x192_300ep.py'),
        'configs/_base_/default_runtime.py',
    )
    for relative in relative_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())


@pytest.mark.parametrize(
    ('relative', 'old', 'new'),
    [
        ('configs/optimization/formal_stage_c/_base_/paired_300ep.py',
         "_base_ = ['../../../reproduction/coco_s_v1.py']",
         "_base_ = ['../../../reproduction/coco_s_v1.py', "
         "'../../../reproduction/coco_s_v2.py']"),
        ('configs/optimization/formal_stage_c/_base_/paired_300ep.py',
         'formal_protocol = dict(',
         'auto_scale_lr = dict(base_batch_size=64)\nformal_protocol = dict('),
        ('configs/optimization/formal_stage_c/_base_/paired_300ep.py',
         'accumulation_steps=1', 'accumulation_steps=2'),
        ('configs/optimization/formal_stage_c/_base_/paired_300ep.py',
         'batch_size=128', 'batch_size=64'),
        ('configs/optimization/formal_stage_c/_base_/paired_300ep.py',
         'num_workers=2', 'num_workers=3'),
        ('configs/body_2d_keypoint/tokenpose/'
         'mamba_tokenpose_T2_coco_256x192_300ep.py',
         "type='Adam'", "type='AdamW'"),
        ('configs/body_2d_keypoint/tokenpose/'
         'mamba_tokenpose_T2_coco_256x192_300ep.py',
         'milestones=[200, 260]', 'milestones=[199, 260]'),
        ('configs/body_2d_keypoint/tokenpose/'
         'mamba_tokenpose_T2_coco_256x192_300ep.py',
         'flip_test=True', 'flip_test=False'),
        ('configs/body_2d_keypoint/tokenpose/'
         'mamba_tokenpose_T2_coco_256x192_300ep.py',
         "type='CocoMetric'", "type='OtherMetric'"),
    ],
)
def test_paired_config_validator_rejects_common_closure_drift(
        tmp_path, relative, old, new):
    _copy_formal_config_closure(tmp_path)
    path = tmp_path / relative
    content = path.read_text()
    assert old in content
    path.write_text(content.replace(old, new, 1))
    spec = importlib.util.spec_from_file_location('formal_builder_drift', BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='canonical|symmetry|closure'):
        module.validate_paired_config_symmetry(
            tmp_path,
            'configs/optimization/formal_stage_c/full_seed0.py',
            'configs/optimization/formal_stage_c/no_pif_seed0.py', seed=0)


def test_public_run_init_and_train_result_loaders_exist():
    assert callable(getattr(formal_schema, 'load_formal_run_init'))
    assert callable(getattr(formal_schema, 'load_formal_train_result'))


def test_formal_source_authority_requires_clean_detached_exact_head(tmp_path):
    repository = tmp_path / 'repo'
    repository.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repository, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'formal@example.invalid'],
        cwd=repository, check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Formal Fixture'],
        cwd=repository, check=True)
    tracked = repository / 'tracked.txt'
    tracked.write_text('frozen\n')
    subprocess.run(['git', 'add', 'tracked.txt'], cwd=repository, check=True)
    subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=repository, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    subprocess.run(['git', 'checkout', '--detach', '-q'], cwd=repository, check=True)
    formal_schema._validate_formal_source_authority(repository, commit)

    tracked.write_text('drift\n')
    with pytest.raises(FormalManifestError, match='clean'):
        formal_schema._validate_formal_source_authority(repository, commit)
