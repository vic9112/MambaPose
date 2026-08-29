"""Bounded self-distillation for scaled Binary Q/K recovery."""

from __future__ import annotations

import copy
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.model import BaseModel
from torch import Tensor

from mambapose_opt.binary_recovery import load_tensor_checkpoint
from mmpose.models import build_pose_estimator
from mmpose.registry import MODELS
from mmpose.utils.typing import ForwardResults, OptSampleList


def _with_qk_mode(model_config: dict, qk_mode: str) -> dict:
    config = copy.deepcopy(model_config)
    backbone = config.get('backbone')
    if isinstance(backbone, Mapping) and 'pretrained' in backbone:
        backbone['pretrained'] = None
    try:
        config['head']['tokenpose_cfg']['qk_mode'] = qk_mode
    except (KeyError, TypeError) as exc:
        raise ValueError(
            'base model must define head.tokenpose_cfg.qk_mode') from exc
    return config


@MODELS.register_module()
class BinaryQKSelfDistiller(BaseModel):
    """Recover a scaled-binary Q/K student from one full float checkpoint.

    Both models use the exact same base pose-estimator config and checkpoint.
    Only the head-local ``qk_mode`` differs.  The teacher remains frozen and
    the student receives its regular supervised loss plus MSE on the final
    heatmaps.
    """

    def __init__(self,
                 base_model_config: str,
                 checkpoint: str,
                 checkpoint_sha256: str,
                 distill_weight: float = 1.0,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if not isinstance(distill_weight, (int, float)) \
                or isinstance(distill_weight, bool) or distill_weight <= 0:
            raise ValueError('distill_weight must be positive')

        resolved = Config.fromfile(base_model_config)
        base_model = copy.deepcopy(resolved.model)
        if data_preprocessor is None:
            data_preprocessor = copy.deepcopy(
                base_model.get('data_preprocessor', None))
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.base_model_config = base_model_config
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.distill_weight = float(distill_weight)
        self.teacher = build_pose_estimator(
            _with_qk_mode(base_model, 'float'))
        self.student = build_pose_estimator(
            _with_qk_mode(base_model, 'binary_scaled'))
        self.train_cfg = getattr(self.student, 'train_cfg', {})
        self.test_cfg = getattr(self.student, 'test_cfg', {})
        self.metainfo = getattr(self.student, 'metainfo', None)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def init_weights(self) -> None:
        # Existing S-V1 checkpoints contain legacy NumPy/MMEngine metadata.
        # Load only their tensor state through the repository's restricted
        # safe-globals path, then enforce exact architecture compatibility.
        state, _ = load_tensor_checkpoint(
            self.checkpoint, self.checkpoint_sha256)
        self.teacher.load_state_dict(state, strict=True)
        self.student.load_state_dict(state, strict=True)
        self._freeze_teacher()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.student(inputs, data_samples, mode='predict')
        if mode == 'tensor':
            return self.student(inputs, data_samples, mode='tensor')
        raise RuntimeError(
            f'Invalid mode "{mode}". Only supports loss, predict and tensor.')

    def loss(self, inputs: Tensor, data_samples: OptSampleList) -> dict:
        student_heatmaps = self.student(inputs, data_samples, mode='tensor')
        loss_from_output = getattr(self.student, 'loss_from_output', None)
        if not callable(loss_from_output):
            raise TypeError(
                'binary recovery student must support loss_from_output')
        losses = loss_from_output(student_heatmaps, data_samples)
        with torch.no_grad():
            teacher_heatmaps = self.teacher(
                inputs, data_samples, mode='tensor')
        if not isinstance(student_heatmaps, Tensor) \
                or not isinstance(teacher_heatmaps, Tensor):
            raise TypeError('final heatmap distillation requires tensor outputs')
        if student_heatmaps.shape != teacher_heatmaps.shape:
            raise ValueError('teacher and student heatmap shapes disagree')
        losses['loss_binary_qk_distill'] = (
            F.mse_loss(student_heatmaps, teacher_heatmaps)
            * self.distill_weight)
        return losses

    def student_state_dict(self):
        """Return a normal pose-estimator state dict for deployment/export."""
        return self.student.state_dict()
