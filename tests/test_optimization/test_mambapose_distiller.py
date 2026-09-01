from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _ToyHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 3, 1)
        self.loss_module = self._weighted_mse

    def forward(self, features):
        return self.projection(features[-1])

    @staticmethod
    def _weighted_mse(prediction, target, weights):
        per_keypoint = (prediction - target).square().mean(dim=(-1, -2))
        return (per_keypoint * weights).mean()


class _ToyPoseEstimator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.initializations = 0
        self.backbone = nn.Conv2d(3, 2, 1)
        self.head = _ToyHead()
        self.train_cfg = {'compute_acc': False}
        self.test_cfg = {}
        self.metainfo = None
        self.data_preprocessor = None

    def init_weights(self):
        self.initializations += 1

    def extract_feat(self, inputs):
        return (self.backbone(inputs), )

    def predict(self, inputs, data_samples):
        return self.head(self.extract_feat(inputs))

    def _forward(self, inputs):
        return self.head(self.extract_feat(inputs))


def _samples(batch: int):
    return [
        SimpleNamespace(
            gt_fields=SimpleNamespace(heatmaps=torch.zeros(3, 4, 4)),
            gt_instance_labels=SimpleNamespace(
                keypoint_weights=torch.ones(1, 3))) for _ in range(batch)
    ]


def test_teacher_is_frozen_and_stays_eval_only():
    from mmpose.models.distillers import MambaPoseHeatmapDistiller

    distiller = MambaPoseHeatmapDistiller(
        teacher=_ToyPoseEstimator(),
        student=_ToyPoseEstimator(),
        heatmap_loss_weight=0.5)

    distiller.train()

    assert distiller.teacher.training is False
    assert all(not parameter.requires_grad
               for parameter in distiller.teacher.parameters())
    assert distiller.student.training is True


def test_init_weights_uses_mmengine_one_shot_lifecycle():
    from mmpose.models.distillers import MambaPoseHeatmapDistiller

    distiller = MambaPoseHeatmapDistiller(
        teacher=_ToyPoseEstimator(),
        student=_ToyPoseEstimator(),
        heatmap_loss_weight=0.5)

    distiller.init_weights()
    assert distiller.is_init is True
    assert distiller.teacher.initializations == 1
    assert distiller.student.initializations == 1

    distiller.init_weights()
    assert distiller.teacher.initializations == 1
    assert distiller.student.initializations == 1


def test_loss_uses_frozen_teacher_and_backpropagates_only_to_student():
    from mmpose.models.distillers import MambaPoseHeatmapDistiller

    teacher = _ToyPoseEstimator()
    student = _ToyPoseEstimator()
    distiller = MambaPoseHeatmapDistiller(
        teacher=teacher, student=student, heatmap_loss_weight=0.5)
    losses = distiller.loss(torch.randn(2, 3, 4, 4), _samples(2))

    assert set(losses) == {
        'loss_kpt', 'loss_heatmap_distill', 'heatmap_distill_mse'
    }
    assert losses['loss_heatmap_distill'].item() == pytest.approx(
        losses['heatmap_distill_mse'].item() * 0.5)
    (losses['loss_kpt'] + losses['loss_heatmap_distill']).backward()

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_export_contains_no_teacher_and_round_trips_student(tmp_path):
    from mmpose.models.distillers import MambaPoseHeatmapDistiller
    from mmpose.models.distillers.mambapose_heatmap_distiller import \
        export_student_checkpoint

    student = _ToyPoseEstimator()
    distiller = MambaPoseHeatmapDistiller(
        teacher=_ToyPoseEstimator(), student=student, heatmap_loss_weight=1.0)
    verification_input = torch.randn(1, 3, 4, 4)
    expected = distiller.student._forward(verification_input)
    path = export_student_checkpoint(distiller, tmp_path / 'student.pth')
    payload = torch.load(path, map_location='cpu', weights_only=False)

    assert payload['state_dict']
    assert not any(key.startswith(('teacher.', 'student.'))
                   for key in payload['state_dict'])

    restored = _ToyPoseEstimator()
    restored.load_state_dict(payload['state_dict'], strict=True)
    actual = restored._forward(verification_input)
    torch.testing.assert_close(actual, expected)


def test_teacher_checkpoint_hash_is_checked_before_loading(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import \
        load_hash_validated_checkpoint

    checkpoint = tmp_path / 'teacher.pth'
    torch.save({'state_dict': _ToyPoseEstimator().state_dict()}, checkpoint)

    with pytest.raises(ValueError, match='sha256 mismatch'):
        load_hash_validated_checkpoint(
            _ToyPoseEstimator(), checkpoint, expected_sha256='0' * 64)


def test_production_config_binds_reproduced_teacher_and_student():
    from mmengine.config import Config

    config = Config.fromfile(
        'configs/optimization/accuracy_first/distill_s_v1_from_b.py')

    assert config.model.teacher_checkpoint.endswith(
        'coco-b/best_coco_AP_epoch_290.pth')
    assert config.model.teacher_checkpoint_sha256 == (
        '38b5e5b1bccfdf7b8b153d91837f7f1bfe371a14f12f6e417a52ebb893efb9b2')
    assert config.model.student_checkpoint.endswith(
        'coco-s-v1/best_coco_AP_epoch_300.pth')
    assert config.model.student_checkpoint_sha256 == (
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2')
    assert config.model.heatmap_loss_weight == 1.0
    assert config.train_cfg.max_epochs == 60
    assert config.optim_wrapper.optimizer.lr == 1e-5
    assert config.param_scheduler == []
