import json
from pathlib import Path

import torch


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_coco_fixture(root: Path):
    for split, image_id in (('train2017', 1), ('val2017', 2), ('test2017', 3)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{image_id:012d}.jpg').write_bytes(b'jpeg')
    annotation = lambda image_id, name: {
        'images': [{'id': image_id, 'file_name': name}],
        'annotations': [{
            'id': image_id,
            'image_id': image_id,
            'category_id': 1,
            'keypoints': [0, 0, 2] * 17,
            'num_keypoints': 17,
            'bbox': [0, 0, 10, 10],
            'area': 100,
        }],
        'categories': [{'id': 1, 'name': 'person'}],
    }
    _write_json(
        root / 'annotations/person_keypoints_train2017.json',
        annotation(1, '000000000001.jpg'))
    _write_json(
        root / 'annotations/person_keypoints_val2017.json',
        annotation(2, '000000000002.jpg'))
    _write_json(
        root / 'annotations/image_info_test-dev2017.json', {
            'images': [{'id': 3, 'file_name': '000000000003.jpg'}],
            'categories': [{'id': 1, 'name': 'person'}],
        })
    detection = [{'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 10, 10],
                  'score': 0.9}]
    _write_json(
        root / 'person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json', detection)
    detection[0]['image_id'] = 3
    _write_json(
        root / 'person_detection_results/'
        'COCO_test-dev2017_detections_AP_H_609_person.json', detection)


def test_coco_annotation_preflight_rejects_missing_images(tmp_path):
    from mambapose_repro.preflight import preflight_coco

    root = tmp_path / 'coco'
    _make_coco_fixture(root)
    (root / 'val2017/000000000002.jpg').unlink()
    report = preflight_coco(root, expected_counts=None)
    assert report.status == 'permanent_failure'
    assert any('missing image' in error for error in report.errors)


def test_coco_preflight_validates_train_val_testdev_and_detections(tmp_path):
    from mambapose_repro.preflight import preflight_coco

    root = tmp_path / 'coco'
    _make_coco_fixture(root)
    report = preflight_coco(root, expected_counts=None)
    assert report.status == 'ok', report.errors
    assert report.facts['train_images'] == 1
    assert report.facts['val_images'] == 1
    assert report.facts['testdev_images'] == 1


def test_pretrained_checkpoint_requires_model_tensor_mapping(tmp_path):
    from mambapose_repro.preflight import preflight_pretrained

    checkpoint = tmp_path / 'backbone.pth'
    torch.save({'model': {'layer.weight': torch.ones(2, 2)}}, checkpoint)
    report = preflight_pretrained(checkpoint, minimum_tensors=1)
    assert report.status == 'ok'
    assert report.facts['model_tensors'] == 1

    torch.save({'optimizer': {}}, checkpoint)
    report = preflight_pretrained(checkpoint, minimum_tensors=1)
    assert report.status == 'permanent_failure'


def test_source_manifest_has_no_unknown_asset_location():
    from mambapose_repro.preflight import load_sources

    root = Path(__file__).resolve().parents[2]
    sources = load_sources(root / 'reproduction/sources.json')
    required = {
        'coco-train2017', 'coco-val2017', 'coco-test2017',
        'coco-annotations', 'coco-test-info', 'coco-val-detections',
        'coco-testdev-detections', 'crowdpose-images',
        'crowdpose-annotations', 'vmamba-tiny-pretrained'
    }
    assert {source['id'] for source in sources} == required
    assert all(source['url'].startswith(('http://', 'https://'))
               for source in sources)
    assert all(source['source_class'] in {'official', 'derived'}
               for source in sources)
