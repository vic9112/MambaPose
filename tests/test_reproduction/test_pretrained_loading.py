from pathlib import Path

import pytest
import torch

from mmengine.config import Config
from mmpose.models.backbones.Vmamba.vmamba import Backbone_VSSM


class _TinyLoader(torch.nn.Module):

    def __init__(self, *, strict: bool, minimum: int = 1):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(1))
        self.second = torch.nn.Parameter(torch.zeros(1))
        self.pretrained_strict = strict
        self.minimum_pretrained_tensors = minimum
        self.pretrained_load_report = None


def _save(path: Path, model: dict[str, torch.Tensor]) -> None:
    torch.save({'model': model}, path)


def test_strict_pretrained_loading_rejects_missing_checkpoint(tmp_path):
    loader = _TinyLoader(strict=True)

    with pytest.raises(FileNotFoundError):
        Backbone_VSSM.load_pretrained(loader, tmp_path / 'missing.pth')


def test_strict_pretrained_loading_rejects_too_few_compatible_tensors(
        tmp_path):
    checkpoint = tmp_path / 'partial.pth'
    _save(checkpoint, {'first': torch.ones(1)})
    loader = _TinyLoader(strict=True, minimum=2)

    with pytest.raises(RuntimeError, match='compatible tensors'):
        Backbone_VSSM.load_pretrained(loader, checkpoint)


def test_strict_pretrained_loading_records_compatible_tensor_count(tmp_path):
    checkpoint = tmp_path / 'complete.pth'
    _save(checkpoint, {
        'first': torch.ones(1),
        'second': torch.ones(1),
        'unused': torch.ones(1),
    })
    loader = _TinyLoader(strict=True, minimum=2)

    report = Backbone_VSSM.load_pretrained(loader, checkpoint)

    assert report['status'] == 'loaded'
    assert report['compatible_tensors'] == 2
    assert loader.pretrained_load_report == report


@pytest.mark.parametrize('config_name', [
    'coco_s_v1.py',
    'coco_s_v2.py',
    'coco_b.py',
    'crowdpose_s_v1.py',
    'crowdpose_s_v2.py',
])
def test_formal_configs_fail_closed_on_pretrained_weight(config_name):
    config = Config.fromfile(
        Path('configs/reproduction') / config_name)

    assert config.model.backbone.pretrained_strict is True
    assert config.model.backbone.minimum_pretrained_tensors == 100
