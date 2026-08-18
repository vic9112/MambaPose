import json
from pathlib import Path

import pytest
import torch


def _checkpoint(path: Path, epoch: int, config_sha='config'):
    torch.save({
        'state_dict': {'weight': torch.tensor([epoch], dtype=torch.float32)},
        'meta': {'epoch': epoch, 'iter': epoch * 10},
        'optimizer': {'state': {}, 'param_groups': []},
        'param_schedulers': [],
    }, path)
    (path.parent / 'provenance.json').write_text(json.dumps({
        'config_sha256': config_sha,
        'repo_commit': 'commit',
        'data_inventory_sha256': 'data',
        'environment_sha256': 'environment',
    }))


def test_corrupt_latest_falls_back_to_valid_predecessor(tmp_path):
    from mambapose_repro.checkpoint import select_resume

    _checkpoint(tmp_path / 'epoch_1.pth', 1)
    (tmp_path / 'epoch_2.pth').write_bytes(b'corrupt')
    expected = {
        'config_sha256': 'config',
        'repo_commit': 'commit',
        'data_inventory_sha256': 'data',
        'environment_sha256': 'environment',
    }
    selected = select_resume(tmp_path, expected)
    assert selected is not None
    assert selected.path.name == 'epoch_1.pth'
    assert selected.epoch == 1


def test_changed_config_forbids_resume(tmp_path):
    from mambapose_repro.checkpoint import PermanentCheckpointError, select_resume

    _checkpoint(tmp_path / 'epoch_1.pth', 1)
    expected = {
        'config_sha256': 'different',
        'repo_commit': 'commit',
        'data_inventory_sha256': 'data',
        'environment_sha256': 'environment',
    }
    with pytest.raises(PermanentCheckpointError, match='provenance'):
        select_resume(tmp_path, expected)


def test_checkpoint_requires_training_state_for_resume(tmp_path):
    from mambapose_repro.checkpoint import validate_checkpoint

    path = tmp_path / 'epoch_1.pth'
    torch.save({'state_dict': {'weight': torch.ones(1)}, 'meta': {'epoch': 1}},
               path)
    result = validate_checkpoint(path, require_training_state=True)
    assert result.valid is False
    assert 'optimizer' in result.error


def test_best_checkpoint_is_selected_for_evaluation(tmp_path):
    from mambapose_repro.checkpoint import select_evaluation_checkpoint

    _checkpoint(tmp_path / 'epoch_3.pth', 3)
    _checkpoint(tmp_path / 'best_coco_AP_epoch_2.pth', 2)
    assert select_evaluation_checkpoint(tmp_path).name == (
        'best_coco_AP_epoch_2.pth')

