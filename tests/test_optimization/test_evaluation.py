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


def _determinism():
    return {
        'python_seed': 0,
        'numpy_seed': 0,
        'torch_seed': 0,
        'worker_count': 2,
        'workers': [
            {'worker_id': 0, 'python_seed': 0, 'numpy_seed': 0,
             'torch_seed': 0},
            {'worker_id': 1, 'python_seed': 1, 'numpy_seed': 1,
             'torch_seed': 1},
        ],
        'persistent_workers': False,
        'order_hashes': {'0': 'e' * 64},
        'provenance': _provenance(),
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
            },
            'modes': {'flip': summary, 'no_flip': summary},
            'gpu_lease': {
                'stage_id': 'full-s-v1:latency', 'pid': 123,
                'boot_id': 'boot',
                'timestamp': '2026-08-27T00:00:00+00:00',
                'device_index': 0, 'allowed_pids': [123],
            },
        },
    }


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
        tmp_path):
    from mambapose_opt.evaluation import CandidateResult, CocoMetrics

    metrics = CocoMetrics(72.8, 89.7, 80.5, 69.4, 79.2, 78.2)
    evaluation = {
        'schema_version': 1,
        'candidate_id': 'full-s-v1',
        'stage': 'evaluate',
        'result': {
            'route': 'baseline',
            'flip_test': True,
            'metrics': metrics.to_dict(),
            'provenance': _provenance(),
            'determinism': _determinism(),
            'calibration_split': None,
            'protocol': {
                'dataset': 'coco', 'split': 'val2017',
                'batch_size': 1, 'complete_split': True,
            },
        },
    }
    _write_json(tmp_path / 'evaluate' / 'evaluate.json', evaluation)
    _write_json(tmp_path / 'profile' / 'profile.json', _profile())
    _write_json(tmp_path / 'latency' / 'latency.json', _latency())
    result = CandidateResult.from_artifacts(tmp_path)

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


def test_candidate_result_rejects_envelope_identity_and_provenance_disagreement(
        tmp_path):
    from mambapose_opt.evaluation import CandidateResult, MetricError

    metrics = {
        'unit': 'percentage_points', 'AP': 72.8, 'AP50': 89.7,
        'AP75': 80.5, 'APM': 69.4, 'APL': 79.2, 'AR': 78.2,
    }
    payload = {
        'schema_version': 1, 'candidate_id': 'full-s-v1', 'stage': 'latency',
        'result': {
            'route': 'baseline', 'flip_test': True, 'metrics': metrics,
            'provenance': _provenance(), 'determinism': _determinism(),
            'calibration_split': None,
            'protocol': {'dataset': 'coco', 'split': 'val2017',
                         'batch_size': 1, 'complete_split': True},
        },
    }
    _write_json(tmp_path / 'evaluate' / 'evaluate.json', payload)
    with pytest.raises(MetricError, match='stage'):
        CandidateResult.from_artifacts(tmp_path)

    payload['stage'] = 'evaluate'
    payload['result']['determinism']['provenance']['config_sha256'] = 'f' * 64
    _write_json(tmp_path / 'evaluate' / 'evaluate.json', payload)
    with pytest.raises(MetricError, match='provenance'):
        CandidateResult.from_artifacts(tmp_path)


@pytest.mark.parametrize('tool', ['evaluate_candidate.py', 'measure_latency.py'])
def test_stage_cli_help_preserves_positional_candidate_and_output(tool):
    completed = subprocess.run(
        [sys.executable, f'tools/optimization/{tool}', '--help'],
        env={'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'},
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'candidate_id' in completed.stdout
    assert '--output' in completed.stdout
