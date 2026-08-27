import hashlib
import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest


def _commit_fixture(repository: Path, message: str) -> str:
    if not (repository / '.git').exists():
        subprocess.run(['git', 'init', '-q'], cwd=repository, check=True)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@example.com'],
            cwd=repository, check=True)
        subprocess.run(
            ['git', 'config', 'user.name', 'Test'], cwd=repository, check=True)
    subprocess.run(['git', 'add', '.'], cwd=repository, check=True)
    subprocess.run(
        ['git', 'commit', '-qm', message], cwd=repository, check=True)
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()


def _tensor_calibration_record():
    return {
        'granularity': 'tensor', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 2.0,
        'range': [-2.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 2.0, '0.99': 2.0,
                        '0.999': 2.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }


def _calibration_identity(source, *, checkpoint_sha):
    sha = 'a' * 64
    return {
        'candidate_id': 'full-s-v1',
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': sha, 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': checkpoint_sha,
        'policy': source['policy_path'],
        'policy_sha256': source['policy_sha256'], 'split': 'train2017',
        'git_commit': source['git_commit'],
        'dataset': {
            'annotation':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_sha256': sha,
            'image_prefix': 'data/coco/train2017',
            'inventory': 'data/inventory.json', 'inventory_sha256': sha,
            'train_archive': 'downloads/train2017.zip',
            'train_archive_sha256': sha, 'image_count': 118287,
            'image_content_algorithm': 'sha256-zip-member-bytes-v1',
            'image_content_aggregate_sha256': sha,
            'image_order_algorithm':
                'sha256-zip-central-directory-order-v1',
            'image_order_sha256': sha,
            'annotation_archive': 'downloads/annotations.zip',
            'annotation_archive_sha256': sha,
            'annotation_member':
                'annotations/person_keypoints_train2017.json',
            'annotation_member_sha256': sha,
        },
    }


def _candidate(identifier, kind, features):
    from mambapose_opt.schema import CandidateSpec

    return CandidateSpec.from_dict({
        'id': identifier, 'route': 'ssm-quant-pwl', 'kind': kind,
        'config': 'configs/candidate.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0, 'features': features,
    })


def _write_evaluation_authority(repository):
    inventory = repository / 'data/inventory.json'
    authority = repository / 'optimization/coco_val2017_authority.json'
    inventory.parent.mkdir(parents=True, exist_ok=True)
    authority.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps({
        'schema_version': 1, 'assets': [
            {'id': 'coco-annotations', 'sha256': '3' * 64},
            {'id': 'coco-val2017', 'sha256': '4' * 64},
            {'id': 'coco-val-detections', 'sha256': '2' * 64},
        ],
    }))
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    authority.write_text(json.dumps({
        'schema_version': 1, 'dataset': 'coco', 'split': 'val2017',
        'inventory': {
            'path': 'data/inventory.json', 'sha256': inventory_sha},
        'annotation': {
            'path': 'data/coco/annotations/person_keypoints_val2017.json',
            'sha256': '1' * 64, 'image_count': 5000,
            'annotation_count': 11004,
            'inventory_asset_id': 'coco-annotations',
            'inventory_archive_sha256': '3' * 64},
        'images': {
            'prefix': 'data/coco/val2017', 'image_count': 5000,
            'corpus_digest_algorithm': 'sha256-filename-size-content-v1',
            'corpus_sha256': '6' * 64,
            'inventory_asset_id': 'coco-val2017',
            'inventory_archive_sha256': '4' * 64},
        'detections': {
            'path': ('data/coco/person_detection_results/'
                     'COCO_val2017_detections_AP_H_56_person.json'),
            'sha256': '2' * 64, 'record_count': 104125,
            'inventory_asset_id': 'coco-val-detections'},
    }))
    return inventory_sha


def _formal_evaluation(
        *, candidate, source, source_config, checkpoint, checkpoint_sha,
        mode_config_shas, inventory_sha, ap):
    def row(config_sha):
        provenance = {
            'checkpoint_sha256': checkpoint_sha,
            'config_sha256': config_sha,
            'data_inventory_sha256': inventory_sha,
            'git_commit': source['git_commit'],
        }
        determinism = {
            'python_seed': candidate.seed, 'numpy_seed': candidate.seed,
            'torch_seed': candidate.seed, 'worker_count': 2,
            'workers': [{
                'worker_id': worker_id,
                'torch_seed_source': 'torch.initial_seed()',
                'python_seed_derivation': 'torch_seed % 2**32',
                'numpy_seed_derivation': 'torch_seed % 2**32',
            } for worker_id in range(2)],
            'persistent_workers': False, 'order_hashes': {'0': 'e' * 64},
            'provenance': provenance,
        }
        return {
            'metrics': {
                'unit': 'percentage_points', 'AP': ap, 'AP50': 89.7,
                'AP75': 80.5, 'APM': 69.4, 'APL': 79.2, 'AR': 78.2},
            'provenance': provenance, 'determinism': determinism,
            'protocol': protocol,
        }

    protocol = {
        'dataset': 'coco', 'split': 'val2017', 'complete_split': True,
        'batch_size': 1,
        'authority_path': 'optimization/coco_val2017_authority.json',
        'authority_sha256': source['authority_sha256'],
        'authority_image_count': 5000,
        'authority_annotation_count': 11004,
        'authority_detection_count': 104125,
        'inventory_authority_sha256': inventory_sha,
        'annotation_authority_sha256': '1' * 64,
        'detection_authority_sha256': '2' * 64,
        'image_corpus_digest_algorithm': 'sha256-filename-size-content-v1',
        'image_corpus_authority_sha256': '6' * 64,
        'image_corpus_sha256': '6' * 64,
        'inventory_annotation_archive_sha256': '3' * 64,
        'inventory_image_archive_sha256': '4' * 64,
        'annotation_sha256': '1' * 64,
        'detection_sha256': '2' * 64,
        'inventory_detection_sha256': '2' * 64,
        'annotation_image_count': 5000,
        'annotation_record_count': 11004,
        'detection_record_count': 104125,
        'verified_image_count': 5000,
        'source_config': source_config,
        'checkpoint': checkpoint,
        'data_inventory': 'data/inventory.json',
        'inventory_projection': {
            'inventory_path': 'data/inventory.json',
            'inventory_sha256': inventory_sha,
            'annotation_asset_id': 'coco-annotations',
            'annotation_declared_sha256': '3' * 64,
            'annotation_observed_archive_sha256': '3' * 64,
            'image_asset_id': 'coco-val2017',
            'image_declared_sha256': '4' * 64,
            'image_observed_archive_sha256': '4' * 64,
            'detection_asset_id': 'coco-val-detections',
            'detection_declared_sha256': '2' * 64,
            'detection_observed_sha256': '2' * 64,
        },
    }
    return {
        'schema_version': 1, 'candidate_id': candidate.id,
        'stage': 'evaluate',
        'result': {
            'route': candidate.route,
            'calibration_split': (
                'train2017'
                if candidate.features.get('numeric_kind') == 'w8a8'
                else None),
            'modes': {
                'flip': row(mode_config_shas['flip']),
                'no_flip': row(mode_config_shas['no_flip']),
            },
            'source': source,
        },
    }


def _producer_mode_config_shas(candidate, config_path):
    from tools.optimization.evaluate_candidate import (
        _deterministic_config, _dump_config)

    hashes = {}
    with tempfile.TemporaryDirectory() as directory:
        for mode in ('flip', 'no_flip'):
            resolved = Path(directory) / f'resolved-{mode}.py'
            _dump_config(
                _deterministic_config(
                    candidate, mode == 'flip', config_path), resolved)
            hashes[mode] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return hashes


def test_route3_stage_plans_follow_numeric_admission_order():
    from mambapose_opt.numeric_conversion import numeric_stage_plan

    assert numeric_stage_plan('weight-only', conditional=False) == (
        'convert', 'export', 'profile', 'evaluate', 'latency')
    assert numeric_stage_plan('w8a8', conditional=False) == (
        'calibrate', 'convert', 'profile', 'evaluate', 'latency')
    with pytest.raises(ValueError, match='conditional admission'):
        numeric_stage_plan('binary-qk', conditional=False)
    assert numeric_stage_plan('binary-qk', conditional=True) == (
        'profile', 'evaluate', 'latency')
    assert all(
        'compare' not in numeric_stage_plan(kind, conditional=conditional)
        for kind, conditional in (
            ('observer', False), ('weight-only', False), ('w8a8', False),
            ('pwl', True), ('binary-qk', True)))


def test_numeric_downstream_binding_rehashes_referenced_artifacts(tmp_path):
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, bind_numeric_inputs, verify_numeric_inputs)

    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    policy = tmp_path / 'policy.json'
    config.write_text('config')
    checkpoint.write_bytes(b'checkpoint')
    policy.write_text('{}')
    binding = bind_numeric_inputs({
        'config': config, 'checkpoint': checkpoint, 'policy': policy})

    verify_numeric_inputs(binding)
    policy.write_text('{"drift": true}')
    with pytest.raises(NumericBindingError, match='policy'):
        verify_numeric_inputs(binding)


def test_numeric_source_binds_clean_commit_manifest_row_policy_and_checkpoint(
        tmp_path):
    from mambapose_opt.numeric_source import (
        NumericSourceError, build_numeric_source_binding,
        validate_numeric_source_binding)
    from mambapose_opt.schema import load_candidate_manifest

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'configs/candidate.py').write_text('policy = True\n')
    (tmp_path / 'checkpoints/model.pth').write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(b'checkpoint').hexdigest()
    manifest = {
        'schema_version': 1,
        'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'checkpoints/model.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }
    (tmp_path / 'optimization/candidates.json').write_text(json.dumps(manifest))
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text(
        '{"split":"train2017"}\n')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'], cwd=tmp_path,
        check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    candidate = load_candidate_manifest(
        tmp_path / 'optimization/candidates.json')[0]

    source = build_numeric_source_binding(
        repository_root=tmp_path, candidate=candidate,
        manifest_path=tmp_path / 'optimization/candidates.json',
        policy_path=tmp_path / 'configs/candidate.py')
    assert validate_numeric_source_binding(
        source, repository_root=tmp_path, candidate=candidate,
        manifest_path=tmp_path / 'optimization/candidates.json') == source
    (tmp_path / 'checkpoints/model.pth').write_bytes(b'drift')
    with pytest.raises(NumericSourceError, match='checkpoint'):
        validate_numeric_source_binding(
            source, repository_root=tmp_path, candidate=candidate,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_numeric_source_binds_the_complete_inherited_config_closure(tmp_path):
    from mambapose_opt.numeric_source import (
        NumericSourceError, build_numeric_source_binding,
        validate_numeric_source_binding)
    from mambapose_opt.schema import load_candidate_manifest

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'configs/base.py').write_text('model = dict(type="S-V1")\n')
    (tmp_path / 'configs/candidate.py').write_text(
        "_base_ = ['./base.py']\npolicy = True\n")
    (tmp_path / 'checkpoint.pth').write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(b'checkpoint').hexdigest()
    (tmp_path / 'optimization/candidates.json').write_text(json.dumps({
        'schema_version': 1, 'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'checkpoint.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }))
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text('{}\n')
    _commit_fixture(tmp_path, 'inherited config fixture')
    manifest = tmp_path / 'optimization/candidates.json'
    candidate = load_candidate_manifest(manifest)[0]
    source = build_numeric_source_binding(
        repository_root=tmp_path, candidate=candidate,
        manifest_path=manifest,
        policy_path=tmp_path / 'configs/candidate.py')

    (tmp_path / 'configs/base.py').write_text('model = dict(type="forged")\n')
    with pytest.raises(NumericSourceError, match='dependency.*commit'):
        validate_numeric_source_binding(
            source, repository_root=tmp_path, candidate=candidate,
            manifest_path=manifest)


def test_numeric_source_accepts_only_common_checkout_reproduction_link(tmp_path):
    from mambapose_opt.numeric_source import (
        NumericSourceError, build_numeric_source_binding,
        resolve_numeric_file)
    from mambapose_opt.schema import load_candidate_manifest

    main = tmp_path / 'main'
    main.mkdir()
    (main / 'configs').mkdir()
    (main / 'optimization').mkdir()
    (main / '.gitignore').write_text('work_dirs/\n')
    (main / 'configs/candidate.py').write_text('policy = True\n')
    (main / 'optimization/coco_train2017_authority.json').write_text(
        '{"split":"train2017"}\n')
    checkpoint_bytes = b'checkpoint'
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest = {
        'schema_version': 1,
        'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'work_dirs/reproduction/model.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }
    (main / 'optimization/candidates.json').write_text(json.dumps(manifest))
    subprocess.run(['git', 'init', '-q'], cwd=main, check=True)
    subprocess.run(['git', 'add', '.'], cwd=main, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=main, check=True)
    shared = main / 'work_dirs/reproduction'
    shared.mkdir(parents=True)
    (shared / 'model.pth').write_bytes(checkpoint_bytes)
    linked = tmp_path / 'linked'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '--detach', str(linked), 'HEAD'],
        cwd=main, check=True)
    (linked / 'work_dirs').mkdir()
    exposed = linked / 'work_dirs/reproduction'
    exposed.symlink_to(shared, target_is_directory=True)
    candidate = load_candidate_manifest(
        linked / 'optimization/candidates.json')[0]

    binding = build_numeric_source_binding(
        repository_root=linked, candidate=candidate,
        manifest_path=linked / 'optimization/candidates.json',
        policy_path=linked / 'configs/candidate.py')
    assert binding['checkpoint_path'] == 'work_dirs/reproduction/model.pth'
    assert binding['checkpoint_sha256'] == checkpoint_sha
    assert resolve_numeric_file(
        linked, Path(binding['checkpoint_path']), 'checkpoint').read_bytes() \
        == checkpoint_bytes

    exposed.unlink()
    alternate = tmp_path / 'alternate'
    alternate.mkdir()
    (alternate / 'model.pth').write_bytes(checkpoint_bytes)
    exposed.symlink_to(alternate, target_is_directory=True)
    with pytest.raises(NumericSourceError, match='shared|symlink'):
        build_numeric_source_binding(
            repository_root=linked, candidate=candidate,
            manifest_path=linked / 'optimization/candidates.json',
            policy_path=linked / 'configs/candidate.py')


def _linked_numeric_checkpoint_fixture(tmp_path):
    from mambapose_opt.schema import load_candidate_manifest

    main = tmp_path / 'main'
    main.mkdir()
    (main / 'configs').mkdir()
    (main / 'optimization').mkdir()
    (main / '.gitignore').write_text('/data\nwork_dirs/\n')
    (main / 'configs/candidate.py').write_text('policy = True\n')
    (main / 'optimization/coco_train2017_authority.json').write_text(
        '{"split":"train2017"}\n')
    (main / 'optimization/coco_val2017_authority.json').write_text(
        '{"split":"val2017"}\n')
    checkpoint_bytes = b'checkpoint'
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    (main / 'optimization/candidates.json').write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'work_dirs/reproduction/model.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }))
    _commit_fixture(main, 'linked numeric controller fixture')
    (main / 'data').mkdir()
    checkpoint = main / 'work_dirs/reproduction/model.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
    linked = tmp_path / 'linked'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '--detach', str(linked), 'HEAD'],
        cwd=main, check=True)
    (linked / 'data').symlink_to(main / 'data', target_is_directory=True)
    (linked / 'work_dirs').mkdir()
    (linked / 'work_dirs/reproduction').symlink_to(
        main / 'work_dirs/reproduction', target_is_directory=True)
    manifest = linked / 'optimization/candidates.json'
    candidate = load_candidate_manifest(manifest)[0]
    return {
        'main': main, 'linked': linked, 'manifest': manifest,
        'candidate': candidate, 'checkpoint': checkpoint,
    }


def test_numeric_controller_preflight_accepts_exact_common_checkout_checkpoint(
        tmp_path):
    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.numeric_runtime import resolve_numeric_runtime

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    linked = fixture['linked']
    manifest = fixture['manifest']
    candidate = fixture['candidate']
    controller = OptimizationController(
        linked / 'work_dirs/optimization', candidate,
        lambda *_args: (_ for _ in ()).throw(
            AssertionError('preflight entered runner')),
        repository_root=linked, manifest_path=manifest,
        stages=('profile',))

    assert controller._preflight_error() is None
    runtime = resolve_numeric_runtime(
        candidate, repository_root=linked, manifest_path=manifest,
        downstream_output=(
            linked / 'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
            'profile/profile.json'))
    assert runtime['checkpoint_path'] == fixture['checkpoint'].resolve(strict=True)
    assert runtime['checkpoint_name'] == candidate.checkpoint.as_posix()


def test_runtime_canonicalizes_linked_checkpoint_for_non_numeric_route(tmp_path):
    from mambapose_opt.numeric_runtime import resolve_numeric_runtime
    from mambapose_opt.schema import load_candidate_manifest

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    manifest = fixture['main'] / 'optimization/candidates.json'
    value = json.loads(manifest.read_text())
    value['candidates'][0]['route'] = 'baseline'
    value['candidates'][0]['kind'] = 'float'
    value['candidates'][0]['features'] = {}
    manifest.write_text(json.dumps(value))
    _commit_fixture(fixture['main'], 'use non-numeric route')
    linked_manifest = fixture['linked'] / 'optimization/candidates.json'
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=fixture['main'], text=True).strip()
    subprocess.run(
        ['git', 'reset', '--hard', '-q', commit], cwd=fixture['linked'],
        check=True)
    candidate = load_candidate_manifest(linked_manifest)[0]

    runtime = resolve_numeric_runtime(
        candidate, repository_root=fixture['linked'],
        manifest_path=linked_manifest,
        downstream_output=(
            fixture['linked'] /
            'work_dirs/optimization/baseline/numeric/0/profile/profile.json'))

    assert runtime['checkpoint_path'] == fixture['checkpoint'].resolve(strict=True)
    assert runtime['checkpoint_name'] == candidate.checkpoint.as_posix()


def test_numeric_shared_authorizer_rejects_alternate_same_byte_checkpoint_tree(
        tmp_path):
    from mambapose_opt.checkpoints import authorize_manifest_candidate
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime)

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    linked = fixture['linked']
    exposed = linked / 'work_dirs/reproduction'
    exposed.unlink()
    alternate = tmp_path / 'alternate-reproduction'
    alternate.mkdir()
    (alternate / 'model.pth').write_bytes(fixture['checkpoint'].read_bytes())
    exposed.symlink_to(alternate, target_is_directory=True)

    with pytest.raises(ValueError, match='Git common checkout'):
        authorize_manifest_candidate(
            linked, fixture['manifest'], fixture['candidate'].id)
    with pytest.raises(NumericRuntimeError, match='authorized|symlink'):
        resolve_numeric_runtime(
            fixture['candidate'], repository_root=linked,
            manifest_path=fixture['manifest'],
            downstream_output=(
                linked / 'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
                'profile/profile.json'))


def test_numeric_runtime_rejects_primary_checkpoint_symlink_drift(tmp_path):
    from mambapose_opt.checkpoints import authorize_manifest_candidate
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime)

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    main = fixture['main']
    primary = main / 'work_dirs/reproduction'
    alternate = main / 'work_dirs/alternate-reproduction'
    primary.rename(alternate)
    primary.symlink_to(alternate, target_is_directory=True)

    with pytest.raises(ValueError, match='primary checkout.*symlinks'):
        authorize_manifest_candidate(
            fixture['linked'], fixture['manifest'], fixture['candidate'].id)
    with pytest.raises(NumericRuntimeError, match='authorized|symlink'):
        resolve_numeric_runtime(
            fixture['candidate'], repository_root=fixture['linked'],
            manifest_path=fixture['manifest'],
            downstream_output=(
                fixture['linked'] /
                'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
                'profile/profile.json'))


def test_numeric_authority_rejects_hash_drift_before_runtime(tmp_path):
    from mambapose_opt.checkpoints import authorize_manifest_candidate
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime)

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    fixture['checkpoint'].write_bytes(b'drifted checkpoint')

    with pytest.raises(ValueError, match='checkpoint sha256 mismatch'):
        authorize_manifest_candidate(
            fixture['linked'], fixture['manifest'], fixture['candidate'].id)
    with pytest.raises(NumericRuntimeError, match='sha256 mismatch'):
        resolve_numeric_runtime(
            fixture['candidate'], repository_root=fixture['linked'],
            manifest_path=fixture['manifest'],
            downstream_output=(
                fixture['linked'] /
                'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
                'profile/profile.json'))


def test_numeric_controller_rejects_unapproved_checkpoint_root(tmp_path):
    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.schema import load_candidate_manifest

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    main = fixture['main']
    unapproved = main / 'models/model.pth'
    unapproved.parent.mkdir()
    unapproved.write_bytes(fixture['checkpoint'].read_bytes())
    manifest = main / 'optimization/candidates.json'
    value = json.loads(manifest.read_text())
    value['candidates'][0]['checkpoint'] = 'models/model.pth'
    manifest.write_text(json.dumps(value))
    _commit_fixture(main, 'move numeric checkpoint to unapproved root')
    candidate = load_candidate_manifest(manifest)[0]
    controller = OptimizationController(
        main / 'work_dirs/optimization', candidate, lambda *_args: None,
        repository_root=main, manifest_path=manifest, stages=('profile',))

    error = controller._preflight_error()

    assert error is not None
    assert 'approved shared asset root' in error


def test_numeric_profile_consumes_canonical_checkpoint_and_records_logical_name(
        tmp_path, monkeypatch):
    import torch
    from torch import nn

    from tools.optimization import profile_model

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    captured = {}

    class FixtureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(3, 1, 1)

        def forward(self, inputs, **_kwargs):
            return self.layer(inputs)

    def initialize(config, checkpoint, *, device):
        captured.update(
            config=Path(config), checkpoint=Path(checkpoint), device=device)
        return FixtureModel()

    monkeypatch.setattr(profile_model, 'REPOSITORY_ROOT', fixture['linked'])
    monkeypatch.setattr('mmpose.apis.init_model', initialize)

    result = profile_model.profile(
        fixture['candidate'], (1, 3, 4, 4),
        manifest_path=fixture['manifest'],
        output=(fixture['linked'] /
                'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
                'profile/profile.json'))

    assert captured['checkpoint'] == fixture['checkpoint'].resolve(strict=True)
    assert captured['device'] == 'cpu'
    assert result['checkpoint'] == fixture['candidate'].checkpoint.as_posix()
    assert result['checkpoint_sha256'] == (
        fixture['candidate'].checkpoint_sha256)


def test_numeric_convert_rejects_alternate_same_byte_tree_before_model_load(
        tmp_path, monkeypatch):
    from tools.optimization import convert_numeric

    fixture = _linked_numeric_checkpoint_fixture(tmp_path)
    exposed = fixture['linked'] / 'work_dirs/reproduction'
    exposed.unlink()
    alternate = tmp_path / 'alternate-reproduction'
    alternate.mkdir()
    (alternate / 'model.pth').write_bytes(fixture['checkpoint'].read_bytes())
    exposed.symlink_to(alternate, target_is_directory=True)
    entered = []
    monkeypatch.setattr(convert_numeric, 'REPOSITORY_ROOT', fixture['linked'])
    monkeypatch.setattr(
        'mmpose.apis.init_model', lambda *_args, **_kwargs: entered.append(True))

    with pytest.raises(ValueError, match='Git common checkout'):
        convert_numeric.convert(
            fixture['candidate'], stage='convert',
            output=(fixture['linked'] /
                    'work_dirs/optimization/ssm-quant-pwl/numeric/0/'
                    'convert/convert.json'),
            manifest_path=fixture['manifest'])

    assert entered == []


@pytest.mark.parametrize(
    ('numeric_kind', 'stage'),
    (('weight-only', 'convert'), ('weight-only', 'export'),
     ('w8a8', 'convert')),
)
def test_real_numeric_producer_round_trips_public_and_controller_validation(
        tmp_path, monkeypatch, numeric_kind, stage):
    from torch import nn

    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_numeric_convert_artifact)
    from mambapose_opt.schema import load_candidate_manifest
    from tools.optimization import convert_numeric

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'work_dirs/optimization').mkdir(parents=True)
    (tmp_path / '.gitignore').write_text('work_dirs/\n')
    checkpoint = tmp_path / 'work_dirs/reproduction/checkpoint.pth'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    activation = (
        "        activation_observers={'layer': 'layer.input'},\n"
        if numeric_kind == 'w8a8' else '')
    activation_bits = (
        "activation_bits=8, activation_scale="
        "'runtime-calibration-artifact'"
        if numeric_kind == 'w8a8' else 'activation_bits=None')
    config = tmp_path / 'configs/candidate.py'
    config.write_text(
        'numeric_optimization = dict(\n'
        f'    candidate_kind={numeric_kind!r},\n'
        '    quant_policy=dict(\n'
        "        allow=('layer',), deny=(),\n"
        f'{activation}'
        '        spec=dict(enabled=True, weight_bits=8, '
        f'{activation_bits}, per_output_channel=True, symmetric=True)),\n'
        '    precision_invariants=dict(\n'
        "        selective_scan_state_accumulation='fp32',\n"
        "        attention_softmax='floating', norms='floating'))\n")
    candidate_id = f'{numeric_kind}-{stage}'
    manifest = tmp_path / 'optimization/candidates.json'
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': candidate_id, 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'work_dirs/reproduction/checkpoint.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': numeric_kind, 'auto_run': False},
        }],
    }))
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text('{}\n')
    (tmp_path / 'optimization/coco_val2017_authority.json').write_text('{}\n')
    (tmp_path / 'data').mkdir()
    _commit_fixture(tmp_path, 'producer fixture')
    candidate = load_candidate_manifest(manifest)[0]
    stage_dir = (
        tmp_path / 'work_dirs/optimization' / candidate.route /
        candidate.id / str(candidate.seed) / stage)
    stage_dir.mkdir(parents=True)
    output = stage_dir / f'{stage}.json'
    calibration_path = None
    if numeric_kind == 'w8a8':
        calibration_path = stage_dir.parent / 'calibrate/calibrate.json'
        calibration_path.parent.mkdir()
        identity = _calibration_identity(
            {'policy_path': candidate.config.as_posix(),
             'policy_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
             'git_commit': subprocess.check_output(
                 ['git', 'rev-parse', 'HEAD'], cwd=tmp_path,
                 text=True).strip()},
            checkpoint_sha=checkpoint_sha)
        calibration_path.write_text(json.dumps({
            'schema_version': 1, 'candidate_id': candidate.id,
            'stage': 'calibrate', 'source': {}, 'identity': identity,
            'protocol': {
                'model_mode': 'eval', 'grad_enabled': False,
                'shuffle': False, 'worker_count': 0, 'sample_count': 2,
                'sample_order_sha256': 'b' * 64},
            'hooks': {
                'records': {'layer.input': _tensor_calibration_record()},
                'required_records': ['layer.input'],
                'unsupported_internals': [],
                'activation_scales': {
                    'layer': {'source_record': 'layer.input',
                              'granularity': 'tensor', 'scale': 2 / 127}},
            },
        }))

    model = nn.Module()
    model.layer = nn.Linear(2, 2)
    monkeypatch.setattr(convert_numeric, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr('mmpose.apis.init_model', lambda *_args, **_kwargs: model)
    if numeric_kind == 'w8a8':
        monkeypatch.setattr(
            convert_numeric, 'validate_calibration_provenance',
            lambda value, **_kwargs: value)
        monkeypatch.setattr(
            'mambapose_opt.numeric_calibration.validate_calibration_provenance',
            lambda value, **_kwargs: value)

    produced = convert_numeric.convert(
        candidate, stage=stage, output=output, manifest_path=manifest,
        calibration_artifact=calibration_path)
    output.write_text(json.dumps(produced, allow_nan=False))
    round_tripped = json.loads(output.read_text())
    assert round_tripped['result']['conversion']['simulation_only'] is True
    assert round_tripped['result']['conversion'][
        'integer_kernel_latency_claimed'] is False
    assert validate_numeric_convert_artifact(
        round_tripped, candidate=candidate, repository_root=tmp_path,
        manifest_path=manifest, artifact_path=output) == round_tripped

    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *_args: None, repository_root=tmp_path,
        manifest_path=manifest, stages=(stage,))
    assert controller._artifact_schema(stage, output) == (
        'optimization-stage-envelope-v1')

    if numeric_kind == 'weight-only' and stage == 'convert':
        invalid_claims = []
        for field, value in (
                ('simulation_only', False), ('simulation_only', 1),
                ('integer_kernel_latency_claimed', True),
                ('integer_kernel_latency_claimed', 0)):
            invalid = deepcopy(round_tripped)
            invalid['result']['conversion'][field] = value
            invalid_claims.append(invalid)
        unknown = deepcopy(round_tripped)
        unknown['result']['conversion']['hardware_latency_measured'] = False
        invalid_claims.append(unknown)
        for invalid in invalid_claims:
            with pytest.raises(NumericRuntimeError, match='conversion report'):
                validate_numeric_convert_artifact(
                    invalid, candidate=candidate, repository_root=tmp_path,
                    manifest_path=manifest, artifact_path=output)


def test_controller_rehashes_numeric_nested_bindings_and_export(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    config = tmp_path / 'configs/candidate.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    policy = tmp_path / 'configs/policy.py'
    export = tmp_path / 'work_dirs/optimization/packed.int8.pt'
    for path, content in ((config, b'config'), (checkpoint, b'checkpoint'),
                          (policy, b'policy'), (export, b'int8')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    candidate = _candidate(
        'weight', 'fake-quant',
        {'numeric_kind': 'weight-only', 'auto_run': False})
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, lambda *_args: None,
        repository_root=tmp_path, stages=('export',))
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.validate_numeric_convert_artifact',
        lambda *_args, **_kwargs: {})
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = tmp_path / 'export.json'
    artifact.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': 'weight', 'stage': 'export',
        'result': {
            'runtime_bindings': {
                'config': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'checkpoint': {'path': 'checkpoint.pth',
                               'sha256': digest(checkpoint)},
                'policy': {'path': 'configs/policy.py',
                           'sha256': digest(policy)},
            },
            'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
            'export': {
                'path': 'work_dirs/optimization/packed.int8.pt',
                'sha256': digest(export), 'bytes': export.stat().st_size,
                'format': 'symmetric-int8-per-output-channel-v1',
            },
        },
    }))

    value = json.loads(artifact.read_text())
    value['result']['source'] = {}
    artifact.write_text(json.dumps(value))
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})

    assert controller._artifact_schema('export', artifact) == (
        'optimization-stage-envelope-v1')
    policy.write_bytes(b'drift')
    with pytest.raises(ArtifactValidationError, match='policy.*hash'):
        controller._artifact_schema('export', artifact)
    policy.write_bytes(b'policy')
    export.write_bytes(b'drift')
    with pytest.raises(ArtifactValidationError, match='export.*hash'):
        controller._artifact_schema('export', artifact)


def test_controller_rejects_self_authorized_w8a8_runtime_config(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    config = tmp_path / 'configs/candidate.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    calibration = tmp_path / 'work_dirs/optimization/w8a8/calibrate/calibrate.json'
    injected = tmp_path / 'work_dirs/optimization/w8a8/convert/injected.py'
    for path, content in (
            (config, b'numeric_optimization = dict(candidate_kind="w8a8")\n'),
            (checkpoint, b'checkpoint'), (calibration, b'{}'),
            (injected, b'numeric_optimization = dict(candidate_kind="baseline")\n')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = _candidate(
        'w8a8', 'fake-quant', {'numeric_kind': 'w8a8', 'auto_run': False})
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, lambda *_args: None,
        repository_root=tmp_path, stages=('convert',))
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})
    artifact = tmp_path / 'work_dirs/optimization/w8a8/convert/convert.json'
    artifact.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': candidate.id, 'stage': 'convert',
        'result': {
            'source': {},
            'runtime_bindings': {
                'config': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'checkpoint': {'path': 'checkpoint.pth',
                               'sha256': digest(checkpoint)},
                'policy': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'calibration': {
                    'path': calibration.relative_to(tmp_path).as_posix(),
                    'sha256': digest(calibration)},
            },
            'runtime_config': {
                'path': injected.relative_to(tmp_path).as_posix(),
                'sha256': digest(injected)},
            'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
        },
    }))

    with pytest.raises(ArtifactValidationError, match='canonical|calibration|fields'):
        controller._artifact_schema('convert', artifact)


def test_train_artifact_rejects_hash_matching_empty_recovery_admission(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_numeric_train_artifact)

    checkpoint = tmp_path / 'checkpoint.pth'
    config = tmp_path / 'configs/candidate.py'
    admission = tmp_path / 'work_dirs/optimization/c/train/recovery-admission.json'
    runtime_config = tmp_path / 'work_dirs/optimization/c/train/resolved-train.py'
    runtime_checkpoint = tmp_path / 'work_dirs/optimization/c/train/best_numeric.pth'
    metadata = tmp_path / 'work_dirs/optimization/c/train/runtime-metadata.json'
    for path, content in (
            (checkpoint, b'parent'), (config, b'config'), (admission, b'{}'),
            (runtime_config, b'runtime'), (runtime_checkpoint, b'trained')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = _candidate(
        'recovery', 'fake-quant', {
            'numeric_kind': 'pwl', 'recovery_candidate': True,
            'runtime_checkpoint': runtime_checkpoint.relative_to(
                tmp_path).as_posix(),
        })
    metadata.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': candidate.id,
        'route': candidate.route, 'numeric_kind': 'pwl',
        'parent_checkpoint_sha256': candidate.checkpoint_sha256,
        'runtime_checkpoint_sha256': digest(runtime_checkpoint),
        'recovery_admission_sha256': digest(admission),
    }))
    reference = lambda path: {
        'path': path.relative_to(tmp_path).as_posix(), 'sha256': digest(path)}
    artifact = {
        'schema_version': 1, 'candidate_id': candidate.id, 'stage': 'train',
        'result': {
            'route': 'ssm-quant-pwl', 'source': {},
            'parent': {
                'config': candidate.config.as_posix(),
                'checkpoint': candidate.checkpoint.as_posix(),
                'checkpoint_sha256': candidate.checkpoint_sha256},
            'dependency': {'recovery_admission': reference(admission)},
            'protocol': {
                'seed': 0, 'operation': 'one-bounded-numeric-recovery',
                'attributed_error': 'fabricated',
                'preliminary_ap_drop': 0.31},
            'runtime': {
                'config': reference(runtime_config),
                'checkpoint': reference(runtime_checkpoint),
                'metadata': reference(metadata),
                'transform': 'bounded-numeric-recovery-v1'},
        },
    }
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})

    with pytest.raises(NumericRuntimeError, match='admission'):
        validate_numeric_train_artifact(
            artifact, candidate=candidate, repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def _w8a8_recovery_fixture(tmp_path, monkeypatch):
    from mmengine.config import Config

    from mambapose_opt.evaluation import build_source_binding
    from mambapose_opt.numeric_source import (
        build_numeric_source_binding, file_sha256)
    from mambapose_opt.schema import load_candidate_manifest

    (tmp_path / 'configs/reproduction').mkdir(parents=True)
    (tmp_path / 'configs/numeric').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'work_dirs/optimization').mkdir(parents=True)
    (tmp_path / '.gitignore').write_text('work_dirs/\n')
    student_checkpoint = tmp_path / 'work_dirs/reproduction/student.pth'
    teacher_checkpoint = tmp_path / 'work_dirs/reproduction/teacher.pth'
    student_checkpoint.parent.mkdir()
    student_checkpoint.write_bytes(b'S-V1 parent')
    teacher_checkpoint.write_bytes(b'B teacher')
    student_sha = file_sha256(student_checkpoint)
    teacher_sha = file_sha256(teacher_checkpoint)
    baseline_config = tmp_path / 'configs/reproduction/coco_s_v1.py'
    teacher_config = tmp_path / 'configs/reproduction/coco_b.py'
    evaluation_config = (
        'model = dict(type="S-V1", test_cfg=dict(flip_test=True))\n'
        'train_dataloader = dict(batch_size=1, num_workers=0, '
        'persistent_workers=False)\n'
        'val_dataloader = dict(batch_size=1, num_workers=0, '
        'persistent_workers=False)\n'
        'test_dataloader = dict(batch_size=1, num_workers=0, '
        'persistent_workers=False)\n')
    baseline_config.write_text(evaluation_config)
    teacher_config.write_text('model = dict(type="B")\n')
    policy = tmp_path / 'configs/numeric/w8a8.py'
    policy.write_text(
        evaluation_config +
        'custom_imports = dict(\n'
        "    imports=['mambapose_opt.numeric_conversion'],\n"
        '    allow_failed_imports=False)\n'
        "custom_hooks = [dict(type='NumericRuntimeHook', priority='VERY_HIGH')]\n"
        'numeric_optimization = dict(\n'
        "    candidate_kind='w8a8',\n"
        '    calibration=dict(source_candidate="full-s-v1"),\n'
        '    quant_policy=dict(\n'
        "        allow=('layer',), deny=(),\n"
        "        activation_observers={'layer': 'layer.input'},\n"
        '        spec=dict(enabled=True, weight_bits=8, activation_bits=8,\n'
        "                  activation_scale='runtime-calibration-artifact',\n"
        '                  per_output_channel=True, symmetric=True)),\n'
        '    precision_invariants=dict(\n'
        "        selective_scan_state_accumulation='fp32',\n"
        "        attention_softmax='floating', norms='floating'),\n"
        '    train_envelope=dict(\n'
        "        recovery='one-bounded-qat-or-distillation-run',\n"
        '        requires_attributed_error=True, max_preliminary_ap_drop=0.3,\n'
        '        resume_checkpoints=2, student_candidate="full-s-v1",\n'
        '        student_config="configs/reproduction/coco_s_v1.py",\n'
        '        student_checkpoint="work_dirs/reproduction/student.pth",\n'
        f'        student_checkpoint_sha256="{student_sha}",\n'
        '        teacher_candidate="coco-b-teacher",\n'
        '        teacher_config="configs/reproduction/coco_b.py",\n'
        '        teacher_checkpoint="work_dirs/reproduction/teacher.pth",\n'
        f'        teacher_checkpoint_sha256="{teacher_sha}"))\n')
    inventory_sha = _write_evaluation_authority(tmp_path)
    sha = 'a' * 64
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text(
        json.dumps({
            'schema_version': 1, 'dataset': 'coco', 'split': 'train2017',
            'image_count': 118287, 'image_prefix': 'data/coco/train2017',
            'annotation_path':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_archive_member':
                'annotations/person_keypoints_train2017.json',
            'image_archive_asset_id': 'coco-train2017',
            'image_archive_sha256': sha,
            'annotation_archive_asset_id': 'coco-annotations',
            'annotation_archive_sha256': sha,
        }))
    screen_manifest = tmp_path / 'optimization/candidates.json'
    baseline_row = {
        'id': 'full-s-v1', 'route': 'baseline', 'kind': 'float',
        'config': baseline_config.relative_to(tmp_path).as_posix(),
        'checkpoint': student_checkpoint.relative_to(tmp_path).as_posix(),
        'checkpoint_sha256': student_sha, 'seed': 0, 'features': {},
    }
    teacher_row = {
        'id': 'coco-b-teacher', 'route': 'baseline', 'kind': 'float',
        'config': teacher_config.relative_to(tmp_path).as_posix(),
        'checkpoint': teacher_checkpoint.relative_to(tmp_path).as_posix(),
        'checkpoint_sha256': teacher_sha, 'seed': 0,
        'features': {'role': 'teacher'},
    }
    screen_row = {
        'id': 'w8a8-screen', 'route': 'ssm-quant-pwl',
        'kind': 'fake-quant', 'config': policy.relative_to(tmp_path).as_posix(),
        'checkpoint': student_checkpoint.relative_to(tmp_path).as_posix(),
        'checkpoint_sha256': student_sha, 'seed': 0,
        'features': {'numeric_kind': 'w8a8'},
    }
    screen_manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [baseline_row, teacher_row, screen_row]}))
    screen_commit = _commit_fixture(tmp_path, 'screen manifest')
    baseline, teacher, screen = load_candidate_manifest(screen_manifest)
    screen_numeric_source = build_numeric_source_binding(
        repository_root=tmp_path, candidate=screen,
        manifest_path=screen_manifest, policy_path=policy,
        git_commit=screen_commit)
    identity = _calibration_identity(
        screen_numeric_source, checkpoint_sha=student_sha)
    monkeypatch.setattr(
        'mambapose_opt.numeric_calibration.calibration_identity',
        lambda **_kwargs: identity)
    calibration_path = (
        tmp_path / 'work_dirs/optimization/ssm-quant-pwl' /
        screen.id / str(screen.seed) / 'calibrate/calibrate.json')
    calibration_path.parent.mkdir(parents=True)
    calibration = {
        'schema_version': 1, 'candidate_id': screen.id,
        'stage': 'calibrate', 'source': screen_numeric_source,
        'identity': identity,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 2,
            'sample_order_sha256': 'b' * 64},
        'hooks': {
            'records': {'layer.input': _tensor_calibration_record()},
            'required_records': ['layer.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'layer': {'source_record': 'layer.input',
                          'granularity': 'tensor', 'scale': 2 / 127}},
        },
    }
    calibration_path.write_text(json.dumps(calibration))
    calibration_reference = {
        'path': calibration_path.relative_to(tmp_path).as_posix(),
        'sha256': file_sha256(calibration_path)}

    convert_dir = calibration_path.parent.parent / 'convert'
    convert_dir.mkdir()
    runtime_config = convert_dir / 'resolved-runtime.py'
    resolved_screen = Config.fromfile(policy)
    resolved_screen.numeric_optimization.quant_policy.calibration_artifact = (
        calibration_reference)
    resolved_screen.dump(runtime_config)
    ref = lambda path: {
        'path': path.relative_to(tmp_path).as_posix(),
        'sha256': file_sha256(path)}
    conversion = {
        'schema_version': 1, 'candidate_id': screen.id, 'stage': 'convert',
        'result': {
            'source': screen_numeric_source,
            'runtime_bindings': {
                'config': ref(policy), 'checkpoint': ref(student_checkpoint),
                'policy': ref(policy),
                'calibration': calibration_reference},
            'conversion': {
                'converted': ['layer'], 'skipped': [],
                'original_weight_bytes': 4,
                'simulated_weight_bytes': 1,
                'simulated_coverage': 1.0,
                'simulation_only': True,
                'integer_kernel_latency_claimed': False},
            'precision_invariants': dict(
                resolved_screen.numeric_optimization.precision_invariants),
            'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
            'runtime_config': ref(runtime_config),
        },
    }
    (convert_dir / 'convert.json').write_text(json.dumps(conversion))

    baseline_source = build_source_binding(
        repository_root=tmp_path, candidate=baseline,
        manifest_path=screen_manifest, git_commit=screen_commit)
    screen_source = build_source_binding(
        repository_root=tmp_path, candidate=screen,
        manifest_path=screen_manifest, git_commit=screen_commit)
    baseline_evaluation_path = (
        tmp_path / 'work_dirs/optimization/baseline/full-s-v1/0/'
        'evaluate/evaluate.json')
    screen_evaluation_path = (
        calibration_path.parent.parent / 'evaluate/evaluate.json')
    baseline_evaluation_path.parent.mkdir(parents=True)
    screen_evaluation_path.parent.mkdir()
    baseline_mode_config_shas = _producer_mode_config_shas(
        baseline, baseline_config)
    screen_mode_config_shas = _producer_mode_config_shas(
        screen, runtime_config)
    baseline_evaluation_path.write_text(json.dumps(_formal_evaluation(
        candidate=baseline, source=baseline_source,
        source_config=baseline.config.as_posix(),
        checkpoint=baseline.checkpoint.as_posix(),
        checkpoint_sha=baseline.checkpoint_sha256,
        mode_config_shas=baseline_mode_config_shas,
        inventory_sha=inventory_sha, ap=72.8)))
    screen_evaluation_path.write_text(json.dumps(_formal_evaluation(
        candidate=screen, source=screen_source,
        source_config=runtime_config.relative_to(tmp_path).as_posix(),
        checkpoint=screen.checkpoint.as_posix(),
        checkpoint_sha=screen.checkpoint_sha256,
        mode_config_shas=screen_mode_config_shas,
        inventory_sha=inventory_sha, ap=72.49)))

    recovery_stage = (
        tmp_path / 'work_dirs/optimization/ssm-quant-pwl' /
        'w8a8-recovery/0/train')
    recovery_checkpoint = recovery_stage / 'best_numeric.pth'
    recovery_manifest = tmp_path / 'optimization/recovery-candidates.json'
    recovery_row = {
        'id': 'w8a8-recovery', 'route': 'ssm-quant-pwl',
        'kind': 'fake-quant', 'config': screen_row['config'],
        'checkpoint': screen_row['checkpoint'],
        'checkpoint_sha256': student_sha, 'seed': 0,
        'features': {
            'numeric_kind': 'w8a8', 'conditional': True,
            'recovery_candidate': True,
            'recovery_screen_candidate': screen.id,
            'recovery_screen_manifest':
                screen_manifest.relative_to(tmp_path).as_posix(),
            'recovery_screen_manifest_sha256': file_sha256(screen_manifest),
            'recovery_calibration_artifact':
                calibration_reference['path'],
            'recovery_calibration_sha256':
                calibration_reference['sha256'],
            'recovery_baseline_evaluation':
                baseline_evaluation_path.relative_to(tmp_path).as_posix(),
            'recovery_baseline_evaluation_sha256':
                file_sha256(baseline_evaluation_path),
            'recovery_candidate_evaluation':
                screen_evaluation_path.relative_to(tmp_path).as_posix(),
            'recovery_candidate_evaluation_sha256':
                file_sha256(screen_evaluation_path),
            'runtime_checkpoint':
                recovery_checkpoint.relative_to(tmp_path).as_posix(),
        },
    }
    recovery_manifest.write_text(json.dumps({
        'schema_version': 1, 'candidates': [recovery_row]}))
    recovery_commit = _commit_fixture(tmp_path, 'recovery manifest')
    recovery = load_candidate_manifest(recovery_manifest)[0]
    recovery_source = build_numeric_source_binding(
        repository_root=tmp_path, candidate=recovery,
        manifest_path=recovery_manifest, policy_path=policy,
        git_commit=recovery_commit)

    recovery_stage.mkdir(parents=True)
    admission_path = recovery_stage / 'recovery-admission.json'
    admission = {
        'schema_version': 1, 'candidate_id': recovery.id,
        'decision': 'admit-one-bounded-recovery',
        'attributed_error': 'activation quantization',
        'threshold_ap': 0.3, 'preliminary_ap_drop': 0.31,
        'baseline_evaluation': ref(baseline_evaluation_path),
        'candidate_evaluation': ref(screen_evaluation_path),
    }
    admission_path.write_text(json.dumps(admission))
    resolved_train = recovery_stage / 'resolved-train.py'
    recovery_config = Config.fromfile(policy)
    recovery_config.numeric_optimization.quant_policy.calibration_artifact = (
        calibration_reference)
    recovery_config.work_dir = str(recovery_stage / 'mmpose')
    recovery_config.load_from = str(tmp_path / recovery.checkpoint)
    recovery_config.resume = False
    recovery_config.randomness = dict(seed=recovery.seed, deterministic=True)
    recovery_config.dump(resolved_train)
    recovery_checkpoint.write_bytes(b'trained')
    metadata_path = recovery_stage / 'runtime-metadata.json'
    metadata_path.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': recovery.id,
        'route': recovery.route, 'numeric_kind': 'w8a8',
        'parent_checkpoint_sha256': student_sha,
        'runtime_checkpoint_sha256': file_sha256(recovery_checkpoint),
        'recovery_admission_sha256': file_sha256(admission_path),
        'screen_calibration_candidate_id': screen.id,
        'screen_calibration_sha256': calibration_reference['sha256'],
    }))
    train = {
        'schema_version': 1, 'candidate_id': recovery.id, 'stage': 'train',
        'result': {
            'route': recovery.route, 'source': recovery_source,
            'parent': {
                'config': recovery.config.as_posix(),
                'checkpoint': recovery.checkpoint.as_posix(),
                'checkpoint_sha256': recovery.checkpoint_sha256},
            'dependency': {
                'recovery_admission': ref(admission_path),
                'screen_calibration': calibration_reference},
            'protocol': {
                'seed': 0, 'operation': 'one-bounded-numeric-recovery',
                'attributed_error': 'activation quantization',
                'preliminary_ap_drop': 0.31},
            'runtime': {
                'config': ref(resolved_train),
                'checkpoint': ref(recovery_checkpoint),
                'metadata': ref(metadata_path),
                'transform': 'bounded-numeric-recovery-v1'},
        },
    }
    train_path = recovery_stage / 'train.json'
    train_path.write_text(json.dumps(train))
    return {
        'baseline': baseline, 'teacher': teacher, 'screen': screen,
        'recovery': recovery, 'screen_manifest': screen_manifest,
        'screen_commit': screen_commit,
        'baseline_mode_config_shas': baseline_mode_config_shas,
        'screen_mode_config_shas': screen_mode_config_shas,
        'screen_runtime_config': runtime_config,
        'recovery_manifest': recovery_manifest,
        'calibration_path': calibration_path,
        'calibration_reference': calibration_reference,
        'baseline_evaluation_path': baseline_evaluation_path,
        'screen_evaluation_path': screen_evaluation_path,
        'recovery_stage': recovery_stage, 'admission_path': admission_path,
        'resolved_train': resolved_train,
        'recovery_checkpoint': recovery_checkpoint,
        'metadata_path': metadata_path, 'train': train,
        'train_path': train_path,
    }


def test_recovery_train_runner_forwards_follow_on_manifest_to_child_lookup(
        tmp_path, monkeypatch):
    from tools.optimization import run_campaign, train_candidate

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    recovery = fixture['recovery']
    manifest = fixture['recovery_manifest']
    artifact = fixture['recovery_stage'] / 'train.json'
    monkeypatch.setattr(run_campaign, 'REPO_ROOT', tmp_path)
    runner = run_campaign.SubprocessStageRunner(
        tmp_path / 'work_dirs/optimization', manifest)

    command = runner._command(recovery, 'train', artifact)

    assert command == [
        str(tmp_path / '.venv/bin/python'),
        str(tmp_path / 'tools/optimization/train_candidate.py'),
        recovery.id, '--manifest', str(manifest),
        '--output', artifact.relative_to(tmp_path).as_posix(),
    ]
    assert train_candidate._candidate(manifest, recovery.id) == recovery
    with pytest.raises(ValueError, match='resolve exactly once'):
        train_candidate._candidate(fixture['screen_manifest'], recovery.id)


def test_recovery_admission_authenticates_evaluator_mode_config_hashes(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_runtime import validate_recovery_admission
    from mambapose_opt.numeric_source import file_sha256

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    runtime_sha = file_sha256(fixture['screen_runtime_config'])
    mode_shas = fixture['screen_mode_config_shas']
    assert mode_shas['flip'] != runtime_sha
    assert mode_shas['no_flip'] != runtime_sha
    assert mode_shas['flip'] != mode_shas['no_flip']

    admitted = validate_recovery_admission(
        fixture['admission_path'], candidate=fixture['recovery'],
        repository_root=tmp_path,
        manifest_path=fixture['recovery_manifest'])

    assert admitted['preliminary_ap_drop'] == pytest.approx(0.31)


def test_recovery_admission_rejects_runtime_input_hash_as_mode_provenance(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_recovery_admission)
    from mambapose_opt.numeric_source import file_sha256
    from mambapose_opt.schema import load_candidate_manifest

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    evaluations = {
        'recovery_baseline_evaluation_sha256': (
            fixture['baseline_evaluation_path'],
            file_sha256(tmp_path / fixture['baseline'].config)),
        'recovery_candidate_evaluation_sha256': (
            fixture['screen_evaluation_path'],
            file_sha256(fixture['screen_runtime_config'])),
    }
    for evaluation_path, runtime_sha in evaluations.values():
        evaluation = json.loads(evaluation_path.read_text())
        for row in evaluation['result']['modes'].values():
            row['provenance']['config_sha256'] = runtime_sha
            row['determinism']['provenance']['config_sha256'] = runtime_sha
        evaluation_path.write_text(json.dumps(evaluation))
    recovery_manifest = json.loads(fixture['recovery_manifest'].read_text())
    for feature, (evaluation_path, _runtime_sha) in evaluations.items():
        recovery_manifest['candidates'][0]['features'][feature] = file_sha256(
            evaluation_path)
    fixture['recovery_manifest'].write_text(json.dumps(recovery_manifest))
    admission = json.loads(fixture['admission_path'].read_text())
    admission['baseline_evaluation']['sha256'] = file_sha256(
        fixture['baseline_evaluation_path'])
    admission['candidate_evaluation']['sha256'] = file_sha256(
        fixture['screen_evaluation_path'])
    fixture['admission_path'].write_text(json.dumps(admission))
    _commit_fixture(tmp_path, 'authorize wrong runtime-input mode hashes')
    recovery = load_candidate_manifest(fixture['recovery_manifest'])[0]

    with pytest.raises(NumericRuntimeError, match='config hash'):
        validate_recovery_admission(
            fixture['admission_path'], candidate=recovery,
            repository_root=tmp_path,
            manifest_path=fixture['recovery_manifest'])


def test_w8a8_recovery_uses_screen_manifest_calibration_and_trained_checkpoint(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.numeric_runtime import (
        resolve_numeric_runtime, validate_numeric_train_artifact,
        validate_recovery_calibration_dependency)
    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    recovery = fixture['recovery']

    selected, reference, validated = validate_recovery_calibration_dependency(
        recovery, repository_root=tmp_path,
        recovery_manifest_path=fixture['recovery_manifest'],
        recovery_stage_dir=fixture['recovery_stage'])
    assert selected == fixture['screen']
    assert reference == fixture['calibration_reference']
    assert validated == json.loads(fixture['calibration_path'].read_text())

    validated_runtime = validate_numeric_train_artifact(
        fixture['train'], candidate=recovery, repository_root=tmp_path,
        manifest_path=fixture['recovery_manifest'])
    assert validated_runtime['checkpoint_path'] == fixture['recovery_checkpoint']
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', recovery, lambda *_args: None,
        repository_root=tmp_path, manifest_path=fixture['recovery_manifest'],
        stages=('train',))
    assert controller._artifact_schema(
        'train', fixture['train_path']) == 'numeric-train-v1'
    downstream = fixture['recovery_stage'].parent / 'profile/profile.json'
    downstream.parent.mkdir()
    assert resolve_numeric_runtime(
        recovery, repository_root=tmp_path,
        manifest_path=fixture['recovery_manifest'],
        downstream_output=downstream)['checkpoint_path'] == (
            fixture['recovery_checkpoint'])

    fixture['calibration_path'].write_text('{"forged":true}\n')
    with pytest.raises(ValueError, match='calibration.*hash'):
        validate_recovery_calibration_dependency(
            recovery, repository_root=tmp_path,
            recovery_manifest_path=fixture['recovery_manifest'],
            recovery_stage_dir=fixture['recovery_stage'])


def test_w8a8_train_handoff_rejects_hash_matching_non_numeric_runtime_config(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime,
        validate_numeric_train_artifact)
    from mambapose_opt.numeric_source import file_sha256

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    fixture['resolved_train'].write_text('runtime = True\n')
    forged = deepcopy(fixture['train'])
    forged['result']['runtime']['config']['sha256'] = file_sha256(
        fixture['resolved_train'])
    fixture['train_path'].write_text(json.dumps(forged))

    with pytest.raises(NumericRuntimeError, match='config|W8A8|policy'):
        validate_numeric_train_artifact(
            forged, candidate=fixture['recovery'],
            repository_root=tmp_path,
            manifest_path=fixture['recovery_manifest'])
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', fixture['recovery'],
        lambda *_args: None, repository_root=tmp_path,
        manifest_path=fixture['recovery_manifest'], stages=('train',))
    with pytest.raises(ArtifactValidationError, match='config|W8A8|policy'):
        controller._artifact_schema('train', fixture['train_path'])
    downstream = fixture['recovery_stage'].parent / 'profile/profile.json'
    downstream.parent.mkdir()
    with pytest.raises(NumericRuntimeError, match='config|W8A8|policy'):
        resolve_numeric_runtime(
            fixture['recovery'], repository_root=tmp_path,
            manifest_path=fixture['recovery_manifest'],
            downstream_output=downstream)


@pytest.mark.parametrize(
    'mutation', ('missing-calibration', 'changed-calibration', 'missing-hook'))
def test_w8a8_train_handoff_rejects_mutated_runtime_policy(
        tmp_path, monkeypatch, mutation):
    from mmengine.config import Config

    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_numeric_train_artifact)
    from mambapose_opt.numeric_source import file_sha256

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    policy = Config.fromfile(tmp_path / fixture['recovery'].config)
    if mutation != 'missing-calibration':
        policy.numeric_optimization.quant_policy.calibration_artifact = dict(
            fixture['calibration_reference'])
    if mutation == 'changed-calibration':
        policy.numeric_optimization.quant_policy.calibration_artifact.sha256 = (
            'f' * 64)
    if mutation == 'missing-hook':
        policy.custom_hooks = []
    policy.work_dir = str(fixture['recovery_stage'] / 'mmpose')
    policy.load_from = str(tmp_path / fixture['recovery'].checkpoint)
    policy.resume = False
    policy.randomness = dict(seed=fixture['recovery'].seed, deterministic=True)
    policy.dump(fixture['resolved_train'])
    forged = deepcopy(fixture['train'])
    forged['result']['runtime']['config']['sha256'] = file_sha256(
        fixture['resolved_train'])
    fixture['train_path'].write_text(json.dumps(forged))

    with pytest.raises(NumericRuntimeError, match='config|W8A8|policy'):
        validate_numeric_train_artifact(
            forged, candidate=fixture['recovery'],
            repository_root=tmp_path,
            manifest_path=fixture['recovery_manifest'])
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', fixture['recovery'],
        lambda *_args: None, repository_root=tmp_path,
        manifest_path=fixture['recovery_manifest'], stages=('train',))
    with pytest.raises(ArtifactValidationError, match='config|W8A8|policy'):
        controller._artifact_schema('train', fixture['train_path'])


@pytest.mark.parametrize(
    'mutation', ('hash', 'source', 'candidate', 'runtime-config'))
def test_recovery_admission_rejects_mismatched_screen_evaluation_authority(
        tmp_path, monkeypatch, mutation):
    from mambapose_opt.evaluation import build_source_binding
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_recovery_admission)
    from mambapose_opt.numeric_source import file_sha256
    from mambapose_opt.schema import load_candidate_manifest

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    evaluation_path = fixture['screen_evaluation_path']
    if mutation == 'hash':
        evaluation_path.write_text(evaluation_path.read_text() + '\n')
    else:
        value = json.loads(evaluation_path.read_text())
        if mutation == 'source':
            value['result']['source']['manifest_path'] = (
                fixture['recovery_manifest'].relative_to(tmp_path).as_posix())
        elif mutation == 'candidate':
            teacher = fixture['teacher']
            teacher_source = build_source_binding(
                repository_root=tmp_path, candidate=teacher,
                manifest_path=fixture['screen_manifest'],
                git_commit=fixture['screen_commit'])
            value['candidate_id'] = teacher.id
            value['result']['route'] = teacher.route
            value['result']['calibration_split'] = None
            value['result']['source'] = teacher_source
            for row in value['result']['modes'].values():
                row['provenance']['checkpoint_sha256'] = (
                    teacher.checkpoint_sha256)
                row['determinism']['provenance']['checkpoint_sha256'] = (
                    teacher.checkpoint_sha256)
                row['protocol']['source_config'] = teacher.config.as_posix()
                row['protocol']['checkpoint'] = teacher.checkpoint.as_posix()
        else:
            for row in value['result']['modes'].values():
                row['provenance']['config_sha256'] = 'f' * 64
                row['determinism']['provenance']['config_sha256'] = 'f' * 64
        evaluation_path.write_text(json.dumps(value))
        recovery_manifest = json.loads(
            fixture['recovery_manifest'].read_text())
        recovery_manifest['candidates'][0]['features'][
            'recovery_candidate_evaluation_sha256'] = file_sha256(
                evaluation_path)
        fixture['recovery_manifest'].write_text(json.dumps(recovery_manifest))
        admission = json.loads(fixture['admission_path'].read_text())
        admission['candidate_evaluation']['sha256'] = file_sha256(
            evaluation_path)
        fixture['admission_path'].write_text(json.dumps(admission))
        _commit_fixture(tmp_path, f'authorize {mutation} evaluation')
        fixture['recovery'] = load_candidate_manifest(
            fixture['recovery_manifest'])[0]

    with pytest.raises(
            NumericRuntimeError, match='hash|source|manifest|candidate|identity'):
        validate_recovery_admission(
            fixture['admission_path'], candidate=fixture['recovery'],
            repository_root=tmp_path,
            manifest_path=fixture['recovery_manifest'])


def test_w8a8_recovery_rejects_same_manifest_authority_cycle(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_recovery_calibration_dependency)

    fixture = _w8a8_recovery_fixture(tmp_path, monkeypatch)
    with pytest.raises(NumericRuntimeError, match='follow-on|cyclic'):
        validate_recovery_calibration_dependency(
            fixture['recovery'], repository_root=tmp_path,
            recovery_manifest_path=fixture['screen_manifest'],
            recovery_stage_dir=fixture['recovery_stage'])


def test_campaign_does_not_auto_select_opt_in_numeric_candidates():
    from tools.optimization.run_campaign import _select

    baseline = _candidate('baseline', 'fake-quant', {'numeric_kind': 'observer'})
    opt_in = _candidate(
        'binary', 'binary-qk',
        {'numeric_kind': 'binary-qk', 'auto_run': False, 'conditional': True})

    assert _select((baseline, opt_in), (), admit_conditional=False) == (baseline,)
    with pytest.raises(ValueError, match='conditional'):
        _select((baseline, opt_in), ('binary',), admit_conditional=False)
    assert _select(
        (baseline, opt_in), ('binary',), admit_conditional=True) == (opt_in,)


def test_campaign_uses_route_specific_numeric_stage_plan():
    from tools.optimization.run_campaign import _stages_for_candidate

    candidate = _candidate(
        'weight', 'fake-quant',
        {'numeric_kind': 'weight-only', 'auto_run': False})
    assert _stages_for_candidate(candidate)[:3] == ('convert', 'export', 'profile')


def test_initial_conditional_candidates_do_not_depend_on_recovery_training():
    from pathlib import Path

    from tools.optimization.run_campaign import _stages_for_candidate

    w8a8 = _candidate(
        'w8a8', 'fake-quant',
        {'numeric_kind': 'w8a8', 'auto_run': False, 'conditional': True})
    pwl = _candidate(
        'pwl', 'pwl',
        {'numeric_kind': 'pwl', 'auto_run': False, 'conditional': True})
    assert _stages_for_candidate(w8a8) == (
        'calibrate', 'convert', 'profile', 'evaluate', 'latency')
    assert _stages_for_candidate(pwl) == ('profile', 'evaluate', 'latency')
    assert Path('tools/optimization/train_candidate.py').is_file()


def test_explicit_recovery_candidate_trains_before_derived_runtime_stages():
    from tools.optimization.run_campaign import _stages_for_candidate

    recovery = _candidate(
        'w8a8-recovery', 'fake-quant', {
            'numeric_kind': 'w8a8', 'auto_run': False, 'conditional': True,
            'recovery_candidate': True,
            'runtime_checkpoint':
                'work_dirs/optimization/w8a8-recovery/train/best_numeric.pth',
            'recovery_screen_candidate': 'w8a8-s-v1',
            'recovery_baseline_evaluation':
                'work_dirs/optimization/full-s-v1/evaluate/evaluate.json',
            'recovery_candidate_evaluation':
                'work_dirs/optimization/w8a8-s-v1/evaluate/evaluate.json',
        })

    assert _stages_for_candidate(recovery) == (
        'train', 'profile', 'evaluate', 'latency')


def test_policy_loader_rejects_unmeasured_activation_placeholder():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, quant_policy_from_config)

    with pytest.raises(NumericBindingError, match='calibration'):
        quant_policy_from_config({
            'allow': ('head',), 'deny': (),
            'spec': {
                'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
                'activation_scale': 'runtime-calibration-artifact',
                'per_output_channel': True, 'symmetric': True,
            },
        })


def test_runtime_conversion_applies_config_policy_once_and_w8a8_fails_closed():
    from torch import nn

    from mambapose_opt.numeric_conversion import (
        NumericBindingError, apply_numeric_runtime)
    from mmpose.models.utils.hardware_friendly import FakeQuantLinear

    model = nn.Sequential(nn.Linear(4, 3))
    numeric = {
        'candidate_kind': 'weight-only',
        'quant_policy': {
            'allow': ('0',), 'deny': (),
            'spec': {
                'enabled': True, 'weight_bits': 8,
                'activation_bits': None, 'activation_scale': None,
                'per_output_channel': True, 'symmetric': True,
            },
        },
    }
    first = apply_numeric_runtime(model, numeric)
    second = apply_numeric_runtime(model, numeric)

    assert first is second
    assert isinstance(model[0], FakeQuantLinear)
    numeric['candidate_kind'] = 'w8a8'
    numeric['quant_policy']['spec']['activation_bits'] = 8
    numeric['quant_policy']['spec']['activation_scale'] = (
        'runtime-calibration-artifact')
    with pytest.raises(NumericBindingError, match='calibration'):
        apply_numeric_runtime(nn.Sequential(nn.Linear(4, 3)), numeric)


def test_policy_loader_consumes_verified_per_role_calibration_scales():
    from mambapose_opt.numeric_conversion import quant_policy_from_config

    record = {
        'granularity': 'channel', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0,
        'max_abs': [1.0, 2.0, 3.0],
        'range': [-3.0, 3.0],
        'percentiles': {'0.5': 1.0, '0.9': 2.0, '0.99': 3.0,
                        '0.999': 3.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [3],
    }
    wide_record = dict(record)
    wide_record['max_abs'] = [float(index) for index in range(1, 8)]
    wide_record['range'] = [-7.0, 7.0]
    wide_record['observed_shape'] = [7]
    sha = 'a' * 64
    identity = {
        'candidate_id': 'full-s-v1',
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': sha, 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': sha, 'policy': 'policy.py',
        'policy_sha256': sha, 'split': 'train2017',
        'git_commit': 'c' * 40,
        'dataset': {
            'annotation':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_sha256': sha,
            'image_prefix': 'data/coco/train2017',
            'inventory': 'data/inventory.json', 'inventory_sha256': sha,
            'train_archive': 'downloads/train2017.zip',
            'train_archive_sha256': sha, 'image_count': 118287,
            'image_content_algorithm': 'sha256-zip-member-bytes-v1',
            'image_content_aggregate_sha256': sha,
            'image_order_algorithm':
                'sha256-zip-central-directory-order-v1',
            'image_order_sha256': sha,
            'annotation_archive': 'downloads/annotations.zip',
            'annotation_archive_sha256': sha,
            'annotation_member':
                'annotations/person_keypoints_train2017.json',
            'annotation_member_sha256': sha,
        },
    }
    artifact = {
        'schema_version': 1, 'candidate_id': 'w8a8', 'stage': 'calibrate',
        'source': {},
        'identity': identity,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 2,
            'sample_order_sha256': 'b' * 64,
        },
        'hooks': {
            'records': {'narrow.input': record, 'wide.input': wide_record},
            'required_records': ['narrow.input', 'wide.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'narrow': {
                    'source_record': 'narrow.input', 'granularity': 'channel',
                    'scale': [1 / 127, 2 / 127, 3 / 127]},
                'wide': {
                    'source_record': 'wide.input', 'granularity': 'channel',
                    'scale': [index / 127 for index in range(1, 8)]},
            },
        },
    }
    policy = quant_policy_from_config({
        'allow': ('narrow', 'wide'), 'deny': (),
        'activation_observers': {
            'narrow': 'narrow.input', 'wide': 'wide.input'},
        'spec': {
            'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
            'activation_scale': 'runtime-calibration-artifact',
            'per_output_channel': True, 'symmetric': True,
        },
    }, calibration_artifact=artifact)

    assert dict(policy.role_specs)['narrow'].activation_scale == (
        1 / 127, 2 / 127, 3 / 127)
    assert len(dict(policy.role_specs)['wide'].activation_scale) == 7


def test_policy_loader_rejects_scale_not_derived_from_observer_record():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, quant_policy_from_config)

    record = {
        'granularity': 'tensor', 'sample_count': 4, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 2.0,
        'range': [-2.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 2.0, '0.99': 2.0,
                        '0.999': 2.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0,
        'token_ids': None, 'observed_shape': [],
    }
    identity = _calibration_identity_fixture()
    artifact = {
        'schema_version': 1, 'candidate_id': 'w8a8', 'stage': 'calibrate',
        'source': {}, 'identity': identity,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 1,
            'sample_order_sha256': 'b' * 64,
        },
        'hooks': {
            'records': {'layer.input': record},
            'required_records': ['layer.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'layer': {'source_record': 'layer.input',
                          'granularity': 'tensor', 'scale': 0.5},
            },
        },
    }
    policy = {
        'allow': ('layer',), 'deny': (),
        'activation_observers': {'layer': 'layer.input'},
        'spec': {
            'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
            'activation_scale': 'runtime-calibration-artifact',
            'per_output_channel': True, 'symmetric': True,
        },
    }

    with pytest.raises(NumericBindingError, match='derived|measured'):
        quant_policy_from_config(policy, calibration_artifact=artifact)


def _calibration_identity_fixture():
    sha = 'a' * 64
    return {
        'candidate_id': 'full-s-v1',
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': sha, 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': sha, 'policy': 'policy.py',
        'policy_sha256': sha, 'split': 'train2017',
        'git_commit': 'c' * 40,
        'dataset': {
            'annotation':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_sha256': sha,
            'image_prefix': 'data/coco/train2017',
            'inventory': 'data/inventory.json', 'inventory_sha256': sha,
            'train_archive': 'downloads/train2017.zip',
            'train_archive_sha256': sha, 'image_count': 118287,
            'image_content_algorithm': 'sha256-zip-member-bytes-v1',
            'image_content_aggregate_sha256': sha,
            'image_order_algorithm':
                'sha256-zip-central-directory-order-v1',
            'image_order_sha256': sha,
            'annotation_archive': 'downloads/annotations.zip',
            'annotation_archive_sha256': sha,
            'annotation_member':
                'annotations/person_keypoints_train2017.json',
            'annotation_member_sha256': sha,
        },
    }
