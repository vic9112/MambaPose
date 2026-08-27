"""Typed YOLO-track to MMPose adapter; trajectory history stays downstream."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np


class YoloAdapterError(ValueError):
    """Raised when tracked-box or top-down pose geometry is invalid."""


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class TrackedPose:
    track_id: int
    detector_score: float
    pose_result: Any


def _validate_frame(frame: np.ndarray) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise YoloAdapterError('frame must be a NumPy image array')
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise YoloAdapterError('frame geometry must be non-empty')
    return height, width


def _validate_box(box: TrackedBox, height: int, width: int) -> None:
    if not isinstance(box.track_id, int) or isinstance(box.track_id, bool):
        raise YoloAdapterError('tracked box track_id must be an integer')
    if len(box.xyxy) != 4 or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in box.xyxy):
        raise YoloAdapterError('tracked box coordinates must be finite xyxy')
    x1, y1, x2, y2 = (float(value) for value in box.xyxy)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise YoloAdapterError('tracked box must be ordered and inside frame')
    if (
            isinstance(box.score, bool)
            or not isinstance(box.score, (int, float))
            or not math.isfinite(float(box.score))
            or not 0.0 <= float(box.score) <= 1.0):
        raise YoloAdapterError('tracked box score must be finite in range 0..1')


def _keypoints(result: Any) -> np.ndarray:
    try:
        keypoints = np.asarray(result.pred_instances.keypoints)
    except (AttributeError, TypeError, ValueError) as error:
        raise YoloAdapterError('pose result lacks keypoint geometry') from error
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 2 or keypoints.shape[0] == 0:
        raise YoloAdapterError('pose keypoint geometry has invalid shape')
    return keypoints


def infer_tracked_poses(
        model: Any, frame: np.ndarray, boxes: Sequence[TrackedBox], *,
        inference_fn: Callable[..., Sequence[Any]] | None = None
        ) -> list[TrackedPose]:
    """Infer one ordered pose per tracked box with no trajectory state."""
    height, width = _validate_frame(frame)
    if not boxes:
        return []
    for box in boxes:
        if not isinstance(box, TrackedBox):
            raise YoloAdapterError('boxes must contain TrackedBox values')
        _validate_box(box, height, width)
    ordered = np.asarray([box.xyxy for box in boxes], dtype=np.float32)
    if inference_fn is None:
        from mmpose.apis import inference_topdown

        inference_fn = inference_topdown
    results = list(inference_fn(model, frame, ordered, bbox_format='xyxy'))
    if len(results) != len(boxes):
        raise YoloAdapterError(
            'inference_topdown must return exactly one pose per box')
    tracked: list[TrackedPose] = []
    for box, result in zip(boxes, results):
        keypoints = _keypoints(result)
        coordinates = keypoints[:, :2]
        if not np.isfinite(coordinates).all():
            raise YoloAdapterError('pose coordinates must be finite')
        if (
                (coordinates[:, 0] < 0).any()
                or (coordinates[:, 0] >= width).any()
                or (coordinates[:, 1] < 0).any()
                or (coordinates[:, 1] >= height).any()):
            raise YoloAdapterError('pose coordinates must map inside the frame')
        try:
            scores = np.asarray(result.pred_instances.keypoint_scores)
        except (AttributeError, TypeError, ValueError) as error:
            raise YoloAdapterError('pose result lacks keypoint scores') from error
        scores = np.squeeze(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if (
                scores.ndim != 1 or scores.size == 0
                or scores.size != keypoints.shape[0]
                or not np.isfinite(scores).all()):
            raise YoloAdapterError(
                'pose keypoint scores must be finite with one per keypoint')
        tracked.append(TrackedPose(
            track_id=box.track_id,
            detector_score=float(box.score),
            pose_result=result,
        ))
    return tracked
