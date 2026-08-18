import json
from pathlib import Path

from mmengine.config import Config
import pytest


def test_batch_resolution_uses_largest_stable_exact_divisor():
    from mambapose_repro.calibration import resolve_batch

    assert resolve_batch(128, 100) == (64, 2)
    assert resolve_batch(128, 128) == (128, 1)
    assert resolve_batch(64, 7) == (4, 16)


def test_batch_resolution_rejects_invalid_capacity():
    from mambapose_repro.calibration import resolve_batch

    with pytest.raises(ValueError):
        resolve_batch(128, 0)


def test_materialized_configs_preserve_effective_batch_and_lr(tmp_path):
    from mambapose_repro.calibration import materialize_resolved_configs

    calibration = {
        'schema_version': 1,
        'dtype': 'fp32',
        'runs': {
            'coco-s-v1': {'largest_stable_batch': 100},
            'coco-s-v2': {'largest_stable_batch': 70},
            'coco-b': {'largest_stable_batch': 40},
            'crowdpose-s-v1': {'largest_stable_batch': 20},
            'crowdpose-s-v2': {'largest_stable_batch': 35},
            'coco-s-v1-no-pif': {'largest_stable_batch': 100},
            'crowdpose-s-v1-no-pif': {'largest_stable_batch': 20},
            'crowdpose-s-v1-no-prior': {'largest_stable_batch': 20},
            'crowdpose-s-v1-no-cycling': {'largest_stable_batch': 20},
        },
    }
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration))

    report = materialize_resolved_configs(
        Path('reproduction/manifest.json'), calibration_path, tmp_path)

    assert len(report['configs']) == 11
    for entry in report['configs']:
        config = Config.fromfile(entry['path'])
        if entry['kind'] == 'train':
            assert (config.train_dataloader.batch_size
                    * config.optim_wrapper.accumulative_counts
                    == entry['effective_batch'])
            assert config.optim_wrapper.optimizer.lr == pytest.approx(1e-3)
            assert config.reproduction_resolution.dtype == 'fp32'
