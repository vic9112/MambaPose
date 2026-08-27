"""Accuracy-first heatmap distillation for MambaPose models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.model import BaseModel
from torch import Tensor, nn

from mambapose_opt.evaluation import CandidateResult, MetricError
from mmpose.models.builder import build_pose_estimator
from mmpose.registry import MODELS
from mmpose.utils.typing import ForwardResults, OptSampleList, SampleList

_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_STRUCTURAL_FEATURES = frozenset({
    'no-pif-s-v1',
    'static-neighbor-s-v1',
    'static-graph-lite-s-v1',
})
_NUMERIC_FEATURES = frozenset({
    'w8-weight-only-s-v1',
    'w8a8-s-v1',
    'pwl-silu-s-v1',
    'pwl-gelu-s-v1',
    'pwl-softplus-s-v1',
    'pwl-exp-s-v1',
    'binary-qk-s-v1',
})
_KNOWN_FEATURES = _STRUCTURAL_FEATURES | _NUMERIC_FEATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_validated_checkpoint(
        model: nn.Module,
        checkpoint: Path | str,
        *,
        expected_sha256: str,
        strict: bool = True) -> None:
    """Hash a local checkpoint before deserializing and loading it."""
    path = Path(checkpoint)
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError('expected_sha256 must be a lowercase sha256')
    if not path.is_file():
        raise FileNotFoundError(f'teacher checkpoint does not exist: {path}')
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f'checkpoint sha256 mismatch: expected '
            f'{expected_sha256}, got {actual}')

    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError('teacher checkpoint must contain a mapping')
    state_dict = payload.get('state_dict', payload.get('model', payload))
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError('teacher checkpoint state_dict must be non-empty')
    model.load_state_dict(state_dict, strict=strict)


def _build_model(value) -> nn.Module:
    if isinstance(value, nn.Module):
        return value
    if isinstance(value, (str, Path)):
        value = Config.fromfile(str(value)).model
    elif isinstance(value, Config):
        value = value.model
    elif isinstance(value, Mapping) and 'model' in value:
        value = value['model']
    if not isinstance(value, Mapping):
        raise TypeError('teacher and student must be modules or model configs')
    return build_pose_estimator(value)


@dataclass(frozen=True)
class ParentResult:
    """Hash-bound Stage B result used by an integrated candidate."""

    candidate_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class IntegrationSpec:
    """Features and isolated parent evidence for one integration attempt."""

    features: tuple[str, ...]
    parent_results: tuple[ParentResult, ...]


def validate_integration_features(spec: IntegrationSpec) -> None:
    """Fail closed unless isolated route evidence exactly binds each feature."""
    if not isinstance(spec, IntegrationSpec):
        raise TypeError('spec must be an IntegrationSpec')
    if not spec.features:
        raise ValueError('integration features must not be empty')
    if len(spec.features) != len(set(spec.features)):
        raise ValueError('integration features must be unique')
    unknown = set(spec.features) - _KNOWN_FEATURES
    if unknown:
        raise ValueError(f'unknown integration feature: {sorted(unknown)}')

    structural = set(spec.features) & _STRUCTURAL_FEATURES
    numeric = set(spec.features) & _NUMERIC_FEATURES
    if len(structural) > 1:
        raise ValueError('at most one structural feature is allowed')
    if len(numeric) > 1:
        raise ValueError('at most one numeric feature is allowed')

    parent_ids = [parent.candidate_id for parent in spec.parent_results]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError('parent candidate ids must be unique')
    if set(parent_ids) != set(spec.features):
        raise ValueError('each feature must have exactly one parent result')

    for parent in spec.parent_results:
        if not _SHA256.fullmatch(parent.sha256):
            raise ValueError(
                f'invalid parent sha256 for {parent.candidate_id}')
        path = Path(parent.path)
        if not path.is_file():
            raise ValueError(
                f'parent result does not exist for {parent.candidate_id}')
        actual = _sha256(path)
        if actual != parent.sha256:
            raise ValueError(
                f'parent result sha256 mismatch for {parent.candidate_id}')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f'parent result is not valid JSON for {parent.candidate_id}') \
                from error
        if (
                not isinstance(value, Mapping)
                or value.get('candidate_id') != parent.candidate_id):
            raise ValueError(
                f'parent result candidate mismatch for {parent.candidate_id}')
        expected_suffix = Path('evaluate/evaluate.json')
        try:
            artifact_root = path.parent.parent
            if path.relative_to(artifact_root) != expected_suffix:
                raise ValueError
        except ValueError as error:
            raise ValueError(
                f'parent result must be the canonical Stage B evaluation '
                f'artifact for {parent.candidate_id}') from error
        try:
            validated = CandidateResult.from_artifacts(artifact_root)
        except (MetricError, OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f'parent result is not canonical Stage B evidence for '
                f'{parent.candidate_id}') from error
        if (
                validated.candidate_id != parent.candidate_id
                or validated.evaluation_artifact != path.resolve()):
            raise ValueError(
                f'parent Stage B evidence identity mismatch for '
                f'{parent.candidate_id}')
        expected_route = (
            'structural-pif' if parent.candidate_id in _STRUCTURAL_FEATURES
            else 'ssm-quant-pwl')
        if validated.route != expected_route:
            raise ValueError(
                f'parent Stage B route mismatch for {parent.candidate_id}')


@MODELS.register_module()
class MambaPoseHeatmapDistiller(BaseModel):
    """Distill B-model heatmaps into an S-V1 or optimized student."""

    def __init__(self,
                 teacher,
                 student,
                 heatmap_loss_weight: float,
                 teacher_checkpoint: str | Path | None = None,
                 teacher_checkpoint_sha256: str | None = None,
                 student_checkpoint: str | Path | None = None,
                 student_checkpoint_sha256: str | None = None,
                 data_preprocessor=None,
                 init_cfg=None):
        teacher_model = _build_model(teacher)
        student_model = _build_model(student)
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if (
                isinstance(heatmap_loss_weight, bool)
                or not isinstance(heatmap_loss_weight, (int, float))
                or not 0 < float(heatmap_loss_weight)):
            raise ValueError('heatmap_loss_weight must be finite and positive')
        if not torch.isfinite(torch.tensor(float(heatmap_loss_weight))):
            raise ValueError('heatmap_loss_weight must be finite and positive')
        if (teacher_checkpoint is None) != (
                teacher_checkpoint_sha256 is None):
            raise ValueError('teacher checkpoint path and sha256 are both required')
        if (student_checkpoint is None) != (
                student_checkpoint_sha256 is None):
            raise ValueError('student checkpoint path and sha256 are both required')

        self.teacher = teacher_model
        self.student = student_model
        self.heatmap_loss_weight = float(heatmap_loss_weight)
        self.teacher_checkpoint = (
            None if teacher_checkpoint is None else Path(teacher_checkpoint))
        self.teacher_checkpoint_sha256 = teacher_checkpoint_sha256
        self.student_checkpoint = (
            None if student_checkpoint is None else Path(student_checkpoint))
        self.student_checkpoint_sha256 = student_checkpoint_sha256
        self.train_cfg = getattr(self.student, 'train_cfg', {})
        self.test_cfg = getattr(self.student, 'test_cfg', {})
        self.metainfo = getattr(self.student, 'metainfo', None)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def init_weights(self) -> None:
        already_initialized = self.is_init
        super().init_weights()
        if already_initialized:
            return
        try:
            if self.student_checkpoint is not None:
                load_hash_validated_checkpoint(
                    self.student,
                    self.student_checkpoint,
                    expected_sha256=self.student_checkpoint_sha256,
                    strict=True)
            if self.teacher_checkpoint is not None:
                load_hash_validated_checkpoint(
                    self.teacher,
                    self.teacher_checkpoint,
                    expected_sha256=self.teacher_checkpoint_sha256,
                    strict=True)
            self._freeze_teacher()
        except Exception:
            # Allow an operator to correct a failed checkpoint read and retry;
            # never leave a partly loaded wrapper marked initialized.
            self._is_init = False
            raise

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def _heatmaps(model: nn.Module, inputs: Tensor) -> Tensor:
        features = model.extract_feat(inputs)
        heatmaps = model.head(features)
        if not isinstance(heatmaps, Tensor) or heatmaps.ndim != 4:
            raise RuntimeError('MambaPose head must return NCHW heatmaps')
        return heatmaps

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        if mode == 'tensor':
            return self._forward(inputs)
        raise RuntimeError(
            f'Invalid mode {mode!r}. Only loss, predict and tensor are supported.')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        if not data_samples:
            raise ValueError('data_samples must not be empty')
        with torch.no_grad():
            teacher_heatmaps = self._heatmaps(self.teacher, inputs)
        student_heatmaps = self._heatmaps(self.student, inputs)
        if student_heatmaps.shape != teacher_heatmaps.shape:
            raise RuntimeError(
                'teacher and student heatmap shapes must match exactly')

        gt_heatmaps = torch.stack(
            [sample.gt_fields.heatmaps for sample in data_samples])
        keypoint_weights = torch.cat([
            sample.gt_instance_labels.keypoint_weights
            for sample in data_samples
        ])
        supervised = self.student.head.loss_module(
            student_heatmaps, gt_heatmaps, keypoint_weights)
        distill_mse = F.mse_loss(student_heatmaps, teacher_heatmaps)
        return {
            'loss_kpt': supervised,
            'loss_heatmap_distill': distill_mse * self.heatmap_loss_weight,
            'heatmap_distill_mse': distill_mse.detach(),
        }

    def predict(self, inputs: Tensor, data_samples: OptSampleList):
        return self.student.predict(inputs, data_samples)

    def _forward(self, inputs: Tensor):
        return self.student._forward(inputs)


def export_student_checkpoint(
        distiller: MambaPoseHeatmapDistiller,
        path: Path | str) -> Path:
    """Atomically export a plain student checkpoint without teacher weights."""
    if not isinstance(distiller, MambaPoseHeatmapDistiller):
        raise TypeError('distiller must be a MambaPoseHeatmapDistiller')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': distiller.student.state_dict(),
        'meta': {
            'schema_version': 1,
            'source': 'MambaPoseHeatmapDistiller.student',
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
