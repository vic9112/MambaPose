from types import SimpleNamespace

import numpy as np
import pytest


def _pose(points):
    instances = SimpleNamespace(
        keypoints=np.asarray(points, dtype=np.float32)[None, ...],
        keypoint_scores=np.ones((1, len(points)), dtype=np.float32),
    )
    return SimpleNamespace(pred_instances=instances)


def test_yolo_adapter_preserves_order_scores_and_calls_topdown_once():
    from mambapose_opt.yolo_adapter import (
        TrackedBox, infer_tracked_poses)

    boxes = [
        TrackedBox(track_id=9, xyxy=(1., 2., 20., 40.), score=0.91),
        TrackedBox(track_id=4, xyxy=(3., 5., 22., 44.), score=0.73),
    ]
    calls = []

    def fake_inference(model, frame, ordered, bbox_format):
        calls.append((model, frame, ordered.copy(), bbox_format))
        return [_pose([(4., 5.), (10., 12.)]),
                _pose([(6., 8.), (15., 19.)])]

    model = object()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    poses = infer_tracked_poses(
        model, frame, boxes, inference_fn=fake_inference)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][2], np.array([box.xyxy for box in boxes], np.float32))
    assert calls[0][3] == 'xyxy'
    assert [pose.track_id for pose in poses] == [9, 4]
    assert [pose.detector_score for pose in poses] == [0.91, 0.73]
    assert [pose.pose_result for pose in poses]


def test_yolo_adapter_empty_input_does_not_trigger_full_frame_fallback():
    from mambapose_opt.yolo_adapter import infer_tracked_poses

    def forbidden(*args, **kwargs):
        raise AssertionError('inference_topdown must not receive empty boxes')

    result = infer_tracked_poses(
        object(), np.zeros((16, 20, 3), dtype=np.uint8), [],
        inference_fn=forbidden)
    assert result == []


@pytest.mark.parametrize('xyxy', [
    (1., 2., 1., 5.),
    (-1., 2., 5., 5.),
    (1., 2., 21., 5.),
    (1., float('nan'), 5., 5.),
])
def test_yolo_adapter_rejects_invalid_or_out_of_frame_boxes(xyxy):
    from mambapose_opt.yolo_adapter import (
        TrackedBox, YoloAdapterError, infer_tracked_poses)

    with pytest.raises(YoloAdapterError, match='box'):
        infer_tracked_poses(
            object(), np.zeros((16, 20, 3), dtype=np.uint8),
            [TrackedBox(track_id=1, xyxy=xyxy, score=0.8)],
            inference_fn=lambda *args: [])


def test_yolo_adapter_requires_exactly_one_finite_in_frame_pose_per_box():
    from mambapose_opt.yolo_adapter import (
        TrackedBox, YoloAdapterError, infer_tracked_poses)

    frame = np.zeros((16, 20, 3), dtype=np.uint8)
    box = TrackedBox(track_id=1, xyxy=(1., 2., 10., 12.), score=0.8)
    with pytest.raises(YoloAdapterError, match='one pose per box'):
        infer_tracked_poses(
            object(), frame, [box], inference_fn=lambda *args, **kwargs: [])
    with pytest.raises(YoloAdapterError, match='finite'):
        infer_tracked_poses(
            object(), frame, [box],
            inference_fn=lambda *args, **kwargs: [
                _pose([(float('nan'), 3.)])])
    with pytest.raises(YoloAdapterError, match='frame'):
        infer_tracked_poses(
            object(), frame, [box],
            inference_fn=lambda *args, **kwargs: [_pose([(21., 3.)])])
