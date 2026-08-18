"""Dataset, detection-result, and pretrained-weight admission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


@dataclass
class PreflightReport:
    status: str = 'ok'
    errors: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.status = 'permanent_failure'
        self.errors.append(message)


def load_sources(path: Path | str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(raw) != {'schema_version', 'sources'} or raw['schema_version'] != 1:
        raise ValueError('invalid sources manifest root')
    required = {
        'id', 'url', 'source_class', 'license_status', 'terms_url', 'auth',
        'expected_bytes', 'sha256', 'archive', 'required_paths'
    }
    allowed = required | {'download_path', 'extract_to', 'target'}
    ids: set[str] = set()
    for index, source in enumerate(raw['sources']):
        unknown = set(source) - allowed
        missing = required - set(source)
        if unknown or missing:
            raise ValueError(
                f'invalid source {index}: unknown={unknown}, missing={missing}')
        if source['id'] in ids:
            raise ValueError(f'duplicate source id: {source["id"]}')
        ids.add(source['id'])
        if source['archive'] == 'file' and 'target' not in source:
            raise ValueError(f'file source {source["id"]} needs target')
        if source['archive'] != 'file' and not {
                'download_path', 'extract_to'} <= set(source):
            raise ValueError(f'archive source {source["id"]} needs paths')
    return raw['sources']


def _load_json(path: Path, report: PreflightReport) -> Any | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        report.fail(f'cannot load JSON {path}: {error}')
        return None


def _check_annotation(
        root: Path, annotation_path: Path, image_directory: str,
        keypoints: int | None, report: PreflightReport) -> int:
    data = _load_json(annotation_path, report)
    if not isinstance(data, dict) or not isinstance(data.get('images'), list):
        report.fail(f'invalid annotation images at {annotation_path}')
        return 0
    image_ids: set[int] = set()
    for image in data['images']:
        if not isinstance(image, dict) or not {'id', 'file_name'} <= set(image):
            report.fail(f'invalid image record at {annotation_path}')
            continue
        image_ids.add(image['id'])
        image_path = root / image_directory / image['file_name']
        if not image_path.is_file():
            report.fail(f'missing image referenced by annotation: {image_path}')
    if keypoints is not None:
        annotations = data.get('annotations')
        if not isinstance(annotations, list):
            report.fail(f'missing annotations list at {annotation_path}')
        else:
            for annotation in annotations:
                points = annotation.get('keypoints')
                if not isinstance(points, list) or len(points) != keypoints * 3:
                    report.fail(
                        f'invalid keypoint cardinality at {annotation_path}')
                if annotation.get('image_id') not in image_ids:
                    report.fail(f'annotation references unknown image id')
    return len(data['images'])


def _check_detections(
        path: Path, valid_ids: set[int] | None, report: PreflightReport) -> int:
    detections = _load_json(path, report)
    if not isinstance(detections, list):
        report.fail(f'detection result is not a list: {path}')
        return 0
    required = {'image_id', 'category_id', 'bbox', 'score'}
    for detection in detections:
        if not isinstance(detection, dict) or not required <= set(detection):
            report.fail(f'invalid detection record: {path}')
            break
        numbers = [*detection['bbox'], detection['score']]
        if len(detection['bbox']) != 4 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in numbers):
            report.fail(f'non-finite or malformed detection: {path}')
            break
        if valid_ids is not None and detection['image_id'] not in valid_ids:
            report.fail(f'detection references unknown image id: {path}')
            break
    return len(detections)


def preflight_coco(
        root: Path | str,
        expected_counts: dict[str, int] | None = {
            'train_images': 118287,
            'val_images': 5000,
            'testdev_images': 20288,
        }) -> PreflightReport:
    root = Path(root)
    report = PreflightReport()
    report.facts['train_images'] = _check_annotation(
        root, root / 'annotations/person_keypoints_train2017.json',
        'train2017', 17, report)
    report.facts['val_images'] = _check_annotation(
        root, root / 'annotations/person_keypoints_val2017.json',
        'val2017', 17, report)
    report.facts['testdev_images'] = _check_annotation(
        root, root / 'annotations/image_info_test-dev2017.json',
        'test2017', None, report)
    val_info = _load_json(
        root / 'annotations/person_keypoints_val2017.json', report)
    test_info = _load_json(
        root / 'annotations/image_info_test-dev2017.json', report)
    val_ids = ({item['id'] for item in val_info.get('images', [])}
               if isinstance(val_info, dict) else None)
    test_ids = ({item['id'] for item in test_info.get('images', [])}
                if isinstance(test_info, dict) else None)
    report.facts['val_detections'] = _check_detections(
        root / 'person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json', val_ids, report)
    report.facts['testdev_detections'] = _check_detections(
        root / 'person_detection_results/'
        'COCO_test-dev2017_detections_AP_H_609_person.json', test_ids,
        report)
    if expected_counts is not None:
        for key, expected in expected_counts.items():
            if report.facts.get(key) != expected:
                report.fail(
                    f'{key} count mismatch: expected {expected}, '
                    f'got {report.facts.get(key)}')
    return report


def preflight_crowdpose(root: Path | str) -> PreflightReport:
    root = Path(root)
    report = PreflightReport()
    report.facts['trainval_images'] = _check_annotation(
        root, root / 'annotations/mmpose_crowdpose_trainval.json',
        'images', 14, report)
    report.facts['test_images'] = _check_annotation(
        root, root / 'annotations/mmpose_crowdpose_test.json',
        'images', 14, report)
    test_info = _load_json(
        root / 'annotations/mmpose_crowdpose_test.json', report)
    test_ids = ({item['id'] for item in test_info.get('images', [])}
                if isinstance(test_info, dict) else None)
    report.facts['test_detections'] = _check_detections(
        root / 'annotations/det_for_crowd_test_0.1_0.5.json', test_ids,
        report)
    if report.facts['trainval_images'] != 12000:
        report.fail('CrowdPose trainval image count must be 12000')
    if report.facts['test_images'] != 8000:
        report.fail('CrowdPose test image count must be 8000')
    return report


def preflight_pretrained(
        path: Path | str, minimum_tensors: int = 100) -> PreflightReport:
    path = Path(path)
    report = PreflightReport()
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:
        report.fail(f'cannot load pretrained checkpoint {path}: {error}')
        return report
    model = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    tensors = ({key: value for key, value in model.items()
                if isinstance(value, torch.Tensor)}
               if isinstance(model, dict) else {})
    report.facts['model_tensors'] = len(tensors)
    report.facts['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report.facts['bytes'] = path.stat().st_size
    if len(tensors) < minimum_tensors:
        report.fail(
            f'pretrained model has {len(tensors)} tensors; '
            f'minimum is {minimum_tensors}')
    return report

