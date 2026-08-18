import json


def _predictions(path):
    rows = [
        {
            'image_id': 2,
            'category_id': 1,
            'keypoints': [1.0, 2.0, 0.9] * 17,
            'score': 0.8,
        },
        {
            'image_id': 1,
            'category_id': 1,
            'keypoints': [3.0, 4.0, 0.7] * 17,
            'score': 0.9,
        },
    ]
    path.write_text(json.dumps(rows))
    return rows


def test_testdev_predictions_have_coco_keypoint_shape(tmp_path):
    from mambapose_repro.results import validate_testdev

    path = tmp_path / 'submission.json'
    _predictions(path)
    report = validate_testdev(path, expected_image_ids={1, 2})
    assert report.valid, report.errors
    assert all(len(row['keypoints']) == 51 for row in report.rows)
    assert [row['image_id'] for row in report.rows] == [1, 2]


def test_missing_local_ap_is_not_fabricated(tmp_path):
    from mambapose_repro.results import normalize_testdev

    path = tmp_path / 'submission.json'
    _predictions(path)
    normalized = normalize_testdev(path, expected_image_ids={1, 2})
    assert 'AP' not in normalized
    assert normalized['evaluation'] == 'submission_only'
    assert len(normalized['sha256']) == 64


def test_submission_rejects_unknown_ids_wrong_shape_and_nan(tmp_path):
    from mambapose_repro.results import validate_testdev

    path = tmp_path / 'submission.json'
    rows = _predictions(path)
    rows[0]['image_id'] = 99
    rows[0]['keypoints'] = [0.0] * 48
    rows[1]['score'] = float('nan')
    path.write_text(json.dumps(rows))
    report = validate_testdev(path, expected_image_ids={1, 2})
    assert not report.valid
    assert len(report.errors) >= 3


def test_metric_artifact_requires_known_finite_metrics_and_hashes(tmp_path):
    from mambapose_repro.results import validate_metrics

    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps({'coco/AP': 0.728, 'coco/AP50': 0.897}))
    provenance = {
        'checkpoint_sha256': 'a' * 64,
        'config_sha256': 'b' * 64,
        'data_inventory_sha256': 'c' * 64,
    }
    report = validate_metrics(
        path, {'coco/AP', 'coco/AP50'}, provenance=provenance)
    assert report.valid

    path.write_text(json.dumps({'invented': 1.0}))
    report = validate_metrics(path, {'coco/AP'}, provenance=provenance)
    assert not report.valid

    report = validate_metrics(
        path, {'invented'}, provenance={'checkpoint_sha256': 'a' * 64})
    assert not report.valid
    assert any('provenance' in error for error in report.errors)
