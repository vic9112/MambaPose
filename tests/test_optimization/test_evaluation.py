import json
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest


RAW_FRACTION_METRICS = {
    'coco/AP': 0.728,
    'coco/AP .5': 0.897,
    'coco/AP .75': 0.805,
    'coco/AP (M)': 0.694,
    'coco/AP (L)': 0.792,
    'coco/AR': 0.782,
}


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def _provenance():
    return {
        'checkpoint_sha256': 'a' * 64,
        'config_sha256': 'b' * 64,
        'data_inventory_sha256': 'c' * 64,
        'git_commit': 'd' * 40,
    }


def _determinism(provenance=None):
    return {
        'python_seed': 0,
        'numpy_seed': 0,
        'torch_seed': 0,
        'worker_count': 2,
        'workers': [
            {'worker_id': 0, 'torch_seed_source': 'torch.initial_seed()',
             'python_seed_derivation': 'torch_seed % 2**32',
             'numpy_seed_derivation': 'torch_seed % 2**32'},
            {'worker_id': 1, 'torch_seed_source': 'torch.initial_seed()',
             'python_seed_derivation': 'torch_seed % 2**32',
             'numpy_seed_derivation': 'torch_seed % 2**32'},
        ],
        'persistent_workers': False,
        'order_hashes': {'0': 'e' * 64},
        'provenance': dict(provenance or _provenance()),
    }


def _profile(candidate_id='full-s-v1'):
    return {
        'schema_version': 1,
        'git_commit': 'd' * 40,
        'candidate': candidate_id,
        'config': 'configs/reproduction/coco_s_v1.py',
        'checkpoint': 'work_dirs/reproduction/runs/coco-s-v1/best.pth',
        'checkpoint_sha256': 'a' * 64,
        'input_shapes': [1, 3, 256, 192],
        'output_shapes': [1, 17, 64, 48],
        'parameters': {
            'total': 10, 'trainable': 9,
            'bytes_by_dtype': {'torch.float32': 40},
            'by_prefix': {'backbone': 7, 'head': 3},
        },
        'modules': [
            {'name': '', 'kind': 'Model', 'parameters': 10, 'hazard': None},
        ],
    }


def _latency(candidate_id='full-s-v1'):
    summary = {
        'median_ms': 1.0, 'p90_ms': 1.2, 'p95_ms': 1.3,
        'sample_count': 200,
    }
    return {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'stage': 'latency',
        'result': {
            'route': 'baseline',
            'provenance': {**_provenance(), 'config_sha256': 'f' * 64},
            'protocol': {
                'batch_size': 1, 'warmup': 50, 'iterations': 200,
                'timer': 'torch.cuda.Event', 'synchronize': True,
                'scope': 'full_topdown_model',
                'lease_max_age_seconds': 300,
                'lease_max_future_skew_seconds': 30,
                'source_config': 'configs/reproduction/coco_s_v1.py',
                'checkpoint': (
                    'work_dirs/reproduction/runs/coco-s-v1/best.pth'),
                'data_inventory': 'data/inventory.json',
                'data': {
                    'dataset': 'coco', 'split': 'val2017',
                    'complete_split': True,
                    'authority_path': (
                        'optimization/coco_val2017_authority.json'),
                    'authority_sha256': '5' * 64,
                    'authority_image_count': 5000,
                    'authority_annotation_count': 11004,
                    'authority_detection_count': 104125,
                    'annotation_authority_sha256': '1' * 64,
                    'detection_authority_sha256': '2' * 64,
                    'image_corpus_digest_algorithm': (
                        'sha256-filename-size-content-v1'),
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
                },
            },
            'modes': {'flip': summary, 'no_flip': summary},
            'gpu_lease': {
                'stage_id': 'full-s-v1:latency', 'pid': 123,
                'boot_id': '11111111-1111-1111-1111-111111111111',
                'timestamp': '2026-08-27T00:00:00+00:00',
                'device_index': 0, 'allowed_pids': [123],
            },
        },
    }


def _evaluation(candidate_id='full-s-v1'):
    metrics = {
        'unit': 'percentage_points', 'AP': 72.8, 'AP50': 89.7,
        'AP75': 80.5, 'APM': 69.4, 'APL': 79.2, 'AR': 78.2,
    }
    def mode(config_hash):
        provenance = {**_provenance(), 'config_sha256': config_hash}
        return {
            'metrics': metrics,
            'provenance': provenance,
            'determinism': _determinism(provenance),
            'protocol': {
                'dataset': 'coco', 'split': 'val2017',
                'batch_size': 1, 'complete_split': True,
                'authority_path': 'optimization/coco_val2017_authority.json',
                'authority_sha256': '5' * 64,
                'authority_image_count': 5000,
                'authority_annotation_count': 11004,
                'authority_detection_count': 104125,
                'annotation_authority_sha256': '1' * 64,
                'detection_authority_sha256': '2' * 64,
                'image_corpus_digest_algorithm': (
                    'sha256-filename-size-content-v1'),
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
                'source_config': 'configs/reproduction/coco_s_v1.py',
                'checkpoint': (
                    'work_dirs/reproduction/runs/coco-s-v1/best.pth'),
                'data_inventory': 'data/inventory.json',
            },
        }
    return {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'stage': 'evaluate',
        'result': {
            'route': 'baseline', 'calibration_split': None,
            'modes': {'flip': mode('b' * 64), 'no_flip': mode('9' * 64)},
        },
    }


def _bound_artifacts(
        tmp_path, monkeypatch, *, write_profile=True, write_latency=True):
    import hashlib

    repo = tmp_path / 'repo'
    import mambapose_opt.evaluation as evaluation_module
    monkeypatch.setattr(evaluation_module, '_TRUSTED_REPOSITORY_ROOT', repo)
    config = repo / 'configs/reproduction/coco_s_v1.py'
    checkpoint = repo / 'work_dirs/reproduction/runs/coco-s-v1/best.pth'
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text('model = dict(type="Fixture")\n')
    checkpoint.write_bytes(b'checkpoint is represented by manifest hash')
    manifest = _write_json(repo / 'optimization/candidates.json', {
        'schema_version': 1,
        'candidates': [{
            'id': 'full-s-v1', 'route': 'baseline', 'kind': 'float',
            'config': 'configs/reproduction/coco_s_v1.py',
            'checkpoint': 'work_dirs/reproduction/runs/coco-s-v1/best.pth',
            'checkpoint_sha256': 'a' * 64, 'seed': 0, 'features': {},
        }],
    })
    authority = _write_json(repo / 'optimization/coco_val2017_authority.json', {
        'schema_version': 1, 'dataset': 'coco', 'split': 'val2017',
        'annotation': {
            'path': 'data/coco/annotations/person_keypoints_val2017.json',
            'sha256': '1' * 64, 'image_count': 5000,
            'annotation_count': 11004,
            'inventory_asset_id': 'coco-annotations',
            'inventory_archive_sha256': '3' * 64,
        },
        'images': {
            'prefix': 'data/coco/val2017', 'image_count': 5000,
            'corpus_digest_algorithm': 'sha256-filename-size-content-v1',
            'corpus_sha256': '6' * 64,
            'inventory_asset_id': 'coco-val2017',
            'inventory_archive_sha256': '4' * 64,
        },
        'detections': {
            'path': ('data/coco/person_detection_results/'
                     'COCO_val2017_detections_AP_H_56_person.json'),
            'sha256': '2' * 64, 'record_count': 104125,
            'inventory_asset_id': 'coco-val-detections',
        },
    })
    inventory = _write_json(repo / 'data/inventory.json', {
        'schema_version': 1, 'assets': [
            {'id': 'coco-annotations', 'sha256': '3' * 64},
            {'id': 'coco-val2017', 'sha256': '4' * 64},
            {'id': 'coco-val-detections', 'sha256': '2' * 64},
        ],
    })
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(
        ['git', 'add', 'configs/reproduction/coco_s_v1.py',
         'optimization/candidates.json',
         'optimization/coco_val2017_authority.json', 'data/inventory.json'],
        cwd=repo, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com',
         'commit', '-qm', 'fixture authority'], cwd=repo, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = {
        'git_commit': commit,
        'manifest_path': 'optimization/candidates.json',
        'manifest_sha256': digest(manifest),
        'config_path': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': digest(config),
        'authority_path': 'optimization/coco_val2017_authority.json',
        'authority_sha256': digest(authority),
        'data_inventory_path': 'data/inventory.json',
        'data_inventory_sha256': digest(inventory),
    }
    evaluation = _evaluation()
    evaluation['result']['source'] = dict(source)
    for row in evaluation['result']['modes'].values():
        row['provenance']['git_commit'] = commit
        row['provenance']['data_inventory_sha256'] = digest(inventory)
        row['determinism']['provenance'] = dict(row['provenance'])
        row['protocol']['authority_sha256'] = digest(authority)
    profile = _profile()
    profile['git_commit'] = commit
    latency = _latency()
    latency['result']['source'] = dict(source)
    latency['result']['provenance'].update({
        'git_commit': commit,
        'config_sha256': digest(config),
        'data_inventory_sha256': digest(inventory),
    })
    latency['result']['protocol']['data']['authority_sha256'] = digest(authority)
    root = repo / 'work_dirs/optimization/candidates/full-s-v1'
    _write_json(root / 'evaluate/evaluate.json', evaluation)
    if write_profile:
        _write_json(root / 'profile/profile.json', profile)
    if write_latency:
        _write_json(root / 'latency/latency.json', latency)
    return root


def test_metric_loader_requires_all_primary_fields_and_exact_hashes(tmp_path):
    from mambapose_opt.evaluation import MetricError, load_coco_metrics

    path = _write_json(tmp_path / 'metrics.json', {'coco/AP': 0.728})
    with pytest.raises(MetricError, match='AP50'):
        load_coco_metrics(path, provenance=_provenance())

    path = _write_json(tmp_path / 'metrics.json', RAW_FRACTION_METRICS)
    with pytest.raises(MetricError, match='checkpoint_sha256'):
        load_coco_metrics(
            path, provenance={**_provenance(), 'checkpoint_sha256': 'A' * 64})


def test_metric_loader_single_path_fails_closed_without_provenance(tmp_path):
    from mambapose_opt.evaluation import MetricError, load_coco_metrics

    path = _write_json(tmp_path / 'metrics.json', RAW_FRACTION_METRICS)
    with pytest.raises(MetricError, match='provenance'):
        load_coco_metrics(path)


def test_metric_loader_normalizes_mmpose_fractions_once_to_ap_points(tmp_path):
    from mambapose_opt.evaluation import load_coco_metrics

    path = _write_json(tmp_path / 'metrics.json', RAW_FRACTION_METRICS)
    metrics = load_coco_metrics(path, provenance=_provenance())

    assert metrics.unit == 'percentage_points'
    assert metrics.ap == pytest.approx(72.8)
    assert metrics.ap50 == pytest.approx(89.7)
    assert metrics.ap75 == pytest.approx(80.5)
    assert metrics.apm == pytest.approx(69.4)
    assert metrics.apl == pytest.approx(79.2)
    assert metrics.ar == pytest.approx(78.2)


def test_metric_loader_rejects_mixed_units_nonfinite_and_out_of_range(tmp_path):
    from mambapose_opt.evaluation import MetricError, load_coco_metrics

    mixed = {**RAW_FRACTION_METRICS, 'coco/AP .5': 89.7}
    with pytest.raises(MetricError, match='mixed|fraction'):
        load_coco_metrics(
            _write_json(tmp_path / 'mixed.json', mixed),
            provenance=_provenance())
    invalid = {**RAW_FRACTION_METRICS, 'coco/AR': float('nan')}
    with pytest.raises(MetricError, match='finite'):
        load_coco_metrics(
            _write_json(tmp_path / 'nan.json', invalid),
            provenance=_provenance())
    invalid = {**RAW_FRACTION_METRICS, 'coco/AR': -0.1}
    with pytest.raises(MetricError, match='range'):
        load_coco_metrics(
            _write_json(tmp_path / 'negative.json', invalid),
            provenance=_provenance())


def test_candidate_result_validates_nested_evaluation_and_artifact_identity(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, CocoMetrics

    root = _bound_artifacts(tmp_path, monkeypatch)
    result = CandidateResult.from_artifacts(root)

    assert result.candidate_id == 'full-s-v1'
    assert result.metrics.ap == pytest.approx(72.8)
    assert result.flip_test is True
    assert result.provenance['checkpoint_sha256'] == 'a' * 64
    assert result.profile['parameters']['total'] == 10
    assert result.latency['modes']['flip']['sample_count'] == 200
    assert result.gpu_lease['stage_id'] == 'full-s-v1:latency'
    assert result.calibration_split is None
    assert set(result.artifact_paths) == {'evaluation', 'profile', 'latency'}
    with pytest.raises(FrozenInstanceError):
        result.candidate_id = 'changed'
    with pytest.raises(TypeError):
        result.provenance['checkpoint_sha256'] = 'f' * 64

    no_flip = CandidateResult.from_artifacts(root, mode='no_flip')
    assert no_flip.flip_test is False
    assert no_flip.provenance['config_sha256'] == '9' * 64


def test_candidate_result_rejects_self_attested_candidate_without_manifest(
        tmp_path):
    """Removing external manifest binding must make this counterfeit pass."""
    from mambapose_opt.evaluation import CandidateResult, MetricError

    _write_json(tmp_path / 'evaluate/evaluate.json', _evaluation())
    _write_json(tmp_path / 'profile/profile.json', _profile())
    _write_json(tmp_path / 'latency/latency.json', _latency())

    with pytest.raises(MetricError, match='manifest|source|Git'):
        CandidateResult.from_artifacts(tmp_path)


def test_candidate_result_rejects_impostor_absent_from_recorded_manifest(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(tmp_path, monkeypatch)
    evaluation_path = root / 'evaluate/evaluate.json'
    evaluation = json.loads(evaluation_path.read_text())
    evaluation['candidate_id'] = 'impostor'
    _write_json(evaluation_path, evaluation)

    with pytest.raises(MetricError, match='absent.*manifest|candidate'):
        CandidateResult.from_artifacts(root)


def test_candidate_result_rejects_protocol_hashes_outside_recorded_authority(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(tmp_path, monkeypatch)
    evaluation_path = root / 'evaluate/evaluate.json'
    evaluation = json.loads(evaluation_path.read_text())
    for row in evaluation['result']['modes'].values():
        row['protocol']['image_corpus_authority_sha256'] = '7' * 64
        row['protocol']['image_corpus_sha256'] = '7' * 64
    _write_json(evaluation_path, evaluation)

    with pytest.raises(MetricError, match='authority blob'):
        CandidateResult.from_artifacts(root)


def test_candidate_result_uses_recorded_commit_after_head_advances(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult

    root = _bound_artifacts(tmp_path, monkeypatch)
    repo = root.parents[3]
    config = repo / 'configs/reproduction/coco_s_v1.py'
    config.write_text('model = dict(type="LaterHead")\n')
    subprocess.run(['git', 'add', config.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com',
         'commit', '-qm', 'advance head'], cwd=repo, check=True)

    assert CandidateResult.from_artifacts(root).candidate_id == 'full-s-v1'


def test_source_binding_hashes_tracked_manifest_and_rejects_worktree_drift(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import build_source_binding
    from mambapose_opt.schema import load_candidate_manifest

    root = _bound_artifacts(tmp_path, monkeypatch)
    repo = root.parents[3]
    evaluation = json.loads((root / 'evaluate/evaluate.json').read_text())
    expected = evaluation['result']['source']
    candidate = load_candidate_manifest(repo / expected['manifest_path'])[0]

    assert build_source_binding(
        repository_root=repo, candidate=candidate,
        manifest_path=repo / expected['manifest_path'],
        git_commit=expected['git_commit']) == expected

    (repo / candidate.config).write_text('model = dict(type="Drift")\n')
    with pytest.raises(ValueError, match='differs.*commit'):
        build_source_binding(
            repository_root=repo, candidate=candidate,
            manifest_path=repo / expected['manifest_path'],
            git_commit=expected['git_commit'])


def test_candidate_result_requires_directory_and_both_evaluation_modes(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(tmp_path, monkeypatch)
    path = root / 'evaluate/evaluate.json'
    evaluation = json.loads(path.read_text())
    with pytest.raises(MetricError, match='directory'):
        CandidateResult.from_artifacts(path)

    del evaluation['result']['modes']['no_flip']
    _write_json(path, evaluation)
    with pytest.raises(MetricError, match='both|modes'):
        CandidateResult.from_artifacts(root)


def test_candidate_result_requires_strict_profile_latency_and_lease(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(
        tmp_path, monkeypatch, write_profile=False, write_latency=False)
    with pytest.raises(MetricError, match='profile artifact'):
        CandidateResult.from_artifacts(root)

    profile = _profile()
    profile['modules'][0]['parameters'] = -1
    profile['git_commit'] = json.loads(
        (root / 'evaluate/evaluate.json').read_text())['result']['source'][
            'git_commit']
    _write_json(root / 'profile/profile.json', profile)
    source = json.loads(
        (root / 'evaluate/evaluate.json').read_text())['result']['source']
    latency = _latency()
    latency['result']['source'] = dict(source)
    latency['result']['provenance'].update({
        'git_commit': source['git_commit'],
        'config_sha256': source['config_sha256'],
        'data_inventory_sha256': source['data_inventory_sha256'],
    })
    latency['result']['protocol']['data']['authority_sha256'] = (
        source['authority_sha256'])
    _write_json(root / 'latency/latency.json', latency)
    with pytest.raises(MetricError, match='module inventory'):
        CandidateResult.from_artifacts(root)

    valid_profile = _profile()
    valid_profile['git_commit'] = source['git_commit']
    _write_json(root / 'profile/profile.json', valid_profile)
    latency['result']['gpu_lease']['allowed_pids'] = [999]
    _write_json(root / 'latency/latency.json', latency)
    with pytest.raises(MetricError, match='lease provenance'):
        CandidateResult.from_artifacts(root)


@pytest.mark.parametrize('warmup, iterations', [(0, 200), (50, 1)])
def test_candidate_result_requires_formal_50_by_200_latency_protocol(
        tmp_path, monkeypatch, warmup, iterations):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(tmp_path, monkeypatch)
    latency = json.loads((root / 'latency/latency.json').read_text())
    latency['result']['protocol']['warmup'] = warmup
    latency['result']['protocol']['iterations'] = iterations
    for summary in latency['result']['modes'].values():
        summary['sample_count'] = iterations
    _write_json(root / 'latency/latency.json', latency)

    with pytest.raises(MetricError, match='50|200|protocol'):
        CandidateResult.from_artifacts(root)


def test_recorded_evaluation_requires_authority_fields_and_nonzero_annotations():
    from mambapose_opt.evaluation import (
        MetricError, validate_evaluation_envelope)

    missing = _evaluation()
    del missing['result']['modes']['flip']['protocol']['authority_sha256']
    with pytest.raises(MetricError, match='authority'):
        validate_evaluation_envelope(missing)

    empty = _evaluation()
    empty['result']['modes']['flip']['protocol']['annotation_record_count'] = 0
    with pytest.raises(MetricError, match='counts|annotation'):
        validate_evaluation_envelope(empty)


def test_recorded_evaluation_rejects_self_attested_one_by_one_counts():
    from mambapose_opt.evaluation import (
        MetricError, validate_evaluation_envelope)

    counterfeit = _evaluation()
    for row in counterfeit['result']['modes'].values():
        protocol = row['protocol']
        protocol['authority_annotation_count'] = 1
        protocol['authority_detection_count'] = 1
        protocol['annotation_record_count'] = 1
        protocol['detection_record_count'] = 1

    with pytest.raises(MetricError, match='counts|authority'):
        validate_evaluation_envelope(counterfeit)

def test_formal_evaluation_runs_flip_and_no_flip_before_envelope(
        tmp_path, monkeypatch):
    import tools.optimization.evaluate_candidate as tool
    from mambapose_opt.schema import CandidateSpec

    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'checkpoint')
    import hashlib
    candidate = CandidateSpec.from_dict({
        'id': 'fixture', 'route': 'accuracy-first', 'kind': 'float',
        'config': 'config.py', 'checkpoint': 'model.pth',
        'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(),
        'seed': 0, 'features': {},
    })
    monkeypatch.setattr(tool, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(tool, '_git_commit', lambda: 'd' * 40)
    monkeypatch.setattr(
        tool, 'build_source_binding',
        lambda **unused: {'manifest_path': 'optimization/candidates.json'})
    seen = []

    def fake_mode(candidate, output, *, flip_test, **unused):
        seen.append(flip_test)
        return _evaluation()['result']['modes'][
            'flip' if flip_test else 'no_flip']

    monkeypatch.setattr(tool, '_evaluate_mode', fake_mode)
    envelope = tool.evaluate(candidate, tmp_path / 'evaluate.json')

    assert seen == [True, False]
    assert set(envelope['result']['modes']) == {'flip', 'no_flip'}


def _coco_fixture(tmp_path):
    annotation = tmp_path / 'data/coco/annotations/person_keypoints_val2017.json'
    detection = tmp_path / (
        'data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json')
    image = tmp_path / 'data/coco/val2017/0001.jpg'
    annotation.parent.mkdir(parents=True)
    detection.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    image.write_bytes(b'image')
    _write_json(annotation, {
        'images': [{'id': 1, 'file_name': '0001.jpg'}],
        'annotations': [{'id': 7, 'image_id': 1}], 'categories': [],
    })
    _write_json(detection, [{'image_id': 1, 'bbox': [0, 0, 1, 1]}])
    import hashlib
    annotation_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    detection_hash = hashlib.sha256(detection.read_bytes()).hexdigest()
    _write_json(tmp_path / 'data/inventory.json', {
        'schema_version': 1, 'assets': [
            {
                'id': 'coco-annotations', 'sha256': '3' * 64,
                'path': 'missing/annotations.zip',
                'required_paths': [
                    'data/coco/annotations/person_keypoints_val2017.json'],
            },
            {
                'id': 'coco-val2017', 'sha256': '4' * 64,
                'path': 'missing/val2017.zip',
                'required_paths': ['data/coco/val2017'],
            },
            {
                'id': 'coco-val-detections',
                'path': ('data/coco/person_detection_results/'
                         'COCO_val2017_detections_AP_H_56_person.json'),
                'sha256': detection_hash,
            },
        ],
    })
    image_corpus = hashlib.sha256()
    image_corpus.update(b'0001.jpg\0')
    image_corpus.update(str(len(b'image')).encode('ascii'))
    image_corpus.update(b'\0image\0')
    _write_json(tmp_path / 'optimization/coco_val2017_authority.json', {
        'schema_version': 1, 'dataset': 'coco', 'split': 'val2017',
        'annotation': {
            'path': 'data/coco/annotations/person_keypoints_val2017.json',
            'sha256': annotation_hash, 'image_count': 1,
            'annotation_count': 1,
            'inventory_asset_id': 'coco-annotations',
            'inventory_archive_sha256': '3' * 64,
        },
        'images': {
            'prefix': 'data/coco/val2017', 'image_count': 1,
            'corpus_digest_algorithm': 'sha256-filename-size-content-v1',
            'corpus_sha256': image_corpus.hexdigest(),
            'inventory_asset_id': 'coco-val2017',
            'inventory_archive_sha256': '4' * 64,
        },
        'detections': {
            'path': ('data/coco/person_detection_results/'
                     'COCO_val2017_detections_AP_H_56_person.json'),
            'sha256': detection_hash, 'record_count': 1,
            'inventory_asset_id': 'coco-val-detections',
        },
    })
    config = {
        'test_dataloader': {
            'batch_size': 1, 'drop_last': False,
            'sampler': {'type': 'DefaultSampler', 'shuffle': False,
                        'round_up': False},
            'dataset': {
                'type': 'CocoDataset', 'data_root': 'data/coco/',
                'data_mode': 'topdown',
                'ann_file': 'annotations/person_keypoints_val2017.json',
                'bbox_file': ('data/coco/person_detection_results/'
                              'COCO_val2017_detections_AP_H_56_person.json'),
                'data_prefix': {'img': 'val2017/'}, 'test_mode': True,
            },
        },
        'test_evaluator': {
            'type': 'CocoMetric',
            'ann_file': 'data/coco/annotations/person_keypoints_val2017.json',
        },
    }
    return config, detection


def test_coco_protocol_preflight_verifies_config_assets_and_inventory(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, detection = _coco_fixture(tmp_path)
    protocol = validate_coco_val_protocol(
        config, repository_root=tmp_path, expected_image_count=1)

    assert protocol['annotation_image_count'] == 1
    assert protocol['annotation_record_count'] == 1
    assert protocol['detection_record_count'] == 1
    assert protocol['verified_image_count'] == 1
    assert protocol['annotation_sha256'] != protocol['detection_sha256']
    assert protocol['complete_split'] is True

    detection.write_text(json.dumps([
        {'image_id': 1, 'bbox': [0, 0, 2, 2]}]))
    with pytest.raises(ValueError, match='inventory.*hash'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_rejects_mutated_image_bytes_without_archive(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    image = tmp_path / 'data/coco/val2017/0001.jpg'
    image.write_bytes(b'mutated image bytes')

    with pytest.raises(ValueError, match='image.*hash|corpus'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_hashes_inventory_archive_when_present(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    archive = tmp_path / 'missing/val2017.zip'
    archive.parent.mkdir()
    archive.write_bytes(b'counterfeit archive')

    with pytest.raises(ValueError, match='archive hash'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_preflight_rejects_altered_dataset_before_claim(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    config['test_dataloader']['dataset']['type'] = 'CrowdPoseDataset'
    with pytest.raises(ValueError, match='CocoDataset'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_preflight_rejects_annotation_outside_authority(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    annotation = tmp_path / (
        'data/coco/annotations/person_keypoints_val2017.json')
    value = json.loads(annotation.read_text())
    value['annotations'].append({'id': 8, 'image_id': 1})
    _write_json(annotation, value)

    with pytest.raises(ValueError, match='annotation.*authority|hash'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_rejects_malformed_authority_identity(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    authority_path = tmp_path / 'optimization/coco_val2017_authority.json'
    authority = json.loads(authority_path.read_text())
    authority['annotation']['inventory_asset_id'] = []
    _write_json(authority_path, authority)

    with pytest.raises(ValueError, match='authority.*invalid|malformed'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=1)


def test_coco_protocol_rejects_unique_ids_aliasing_one_image_file(tmp_path):
    from mambapose_opt.evaluation import validate_coco_val_protocol

    config, _ = _coco_fixture(tmp_path)
    annotation = tmp_path / (
        'data/coco/annotations/person_keypoints_val2017.json')
    value = json.loads(annotation.read_text())
    value['images'] = [
        {'id': image_id, 'file_name': '0001.jpg'}
        for image_id in range(2)]
    value['annotations'] = [{'id': 1, 'image_id': 0}]
    _write_json(annotation, value)
    import hashlib
    authority_path = tmp_path / 'optimization/coco_val2017_authority.json'
    authority = json.loads(authority_path.read_text())
    authority['annotation']['sha256'] = hashlib.sha256(
        annotation.read_bytes()).hexdigest()
    authority['annotation']['image_count'] = 2
    authority['images']['image_count'] = 2
    _write_json(authority_path, authority)

    with pytest.raises(ValueError, match='unique|identity|name'):
        validate_coco_val_protocol(
            config, repository_root=tmp_path, expected_image_count=2)


def test_candidate_result_rejects_envelope_identity_and_provenance_disagreement(
        tmp_path, monkeypatch):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    root = _bound_artifacts(tmp_path, monkeypatch)
    payload = json.loads((root / 'evaluate/evaluate.json').read_text())
    payload['stage'] = 'latency'
    _write_json(root / 'evaluate' / 'evaluate.json', payload)
    with pytest.raises(MetricError, match='stage'):
        CandidateResult.from_artifacts(root)

    payload['stage'] = 'evaluate'
    payload['result']['modes']['flip']['determinism']['provenance'][
        'config_sha256'] = 'f' * 64
    _write_json(root / 'evaluate' / 'evaluate.json', payload)
    with pytest.raises(MetricError, match='provenance'):
        CandidateResult.from_artifacts(root)


@pytest.mark.parametrize('tool', ['evaluate_candidate.py', 'measure_latency.py'])
def test_stage_cli_help_preserves_positional_candidate_and_output(tool):
    completed = subprocess.run(
        [sys.executable, f'tools/optimization/{tool}', '--help'],
        env={'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'},
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'candidate_id' in completed.stdout
    assert '--output' in completed.stdout
