import json

import pytest


def _candidate(**overrides):
    candidate = {
        'id': 'full-s-v1',
        'route': 'baseline',
        'kind': 'float',
        'config': 'configs/reproduction/coco_s_v1.py',
        'checkpoint': (
            'work_dirs/reproduction/runs/coco-s-v1/'
            'best_coco_AP_epoch_300.pth'),
        'checkpoint_sha256': 'a' * 64,
        'seed': 0,
        'features': {},
    }
    candidate.update(overrides)
    return candidate


def _write_manifest(path, candidates):
    path.write_text(json.dumps({'schema_version': 1, 'candidates': candidates}))


def test_candidate_manifest_rejects_unknown_fields(tmp_path):
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest

    path = tmp_path / 'candidates.json'
    _write_manifest(path, [_candidate(unexpected=True)])

    with pytest.raises(CandidateManifestError, match='unknown fields'):
        load_candidate_manifest(path)


@pytest.mark.parametrize(
    ('field', 'value', 'match'),
    [
        ('config', '/absolute/config.py', 'relative'),
        ('checkpoint', '../escape.pth', 'traversal'),
        ('checkpoint_sha256', 'not-a-hash', 'sha256'),
        ('route', 'invented-route', 'route'),
        ('route', [], 'route'),
        ('route', {}, 'route'),
        ('kind', [], 'kind'),
        ('kind', {}, 'kind'),
        ('seed', True, 'integer'),
        ('seed', 1.5, 'integer'),
        ('features', {'nested': {'not': 'scalar'}}, 'JSON scalar'),
    ],
)
def test_candidate_manifest_rejects_unsafe_or_unstable_values(
        tmp_path, field, value, match):
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest

    path = tmp_path / 'candidates.json'
    _write_manifest(path, [_candidate(**{field: value})])

    with pytest.raises(CandidateManifestError, match=match):
        load_candidate_manifest(path)


def test_candidate_manifest_rejects_duplicate_ids(tmp_path):
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest

    path = tmp_path / 'candidates.json'
    _write_manifest(path, [_candidate(), _candidate()])

    with pytest.raises(CandidateManifestError, match='duplicate'):
        load_candidate_manifest(path)


def test_candidate_manifest_loads_frozen_scalar_records(tmp_path):
    from mambapose_opt.schema import load_candidate_manifest

    path = tmp_path / 'candidates.json'
    _write_manifest(path, [_candidate(features={'pif': False, 'width': 64})])

    (candidate,) = load_candidate_manifest(path)

    assert candidate.id == 'full-s-v1'
    assert candidate.config.as_posix() == 'configs/reproduction/coco_s_v1.py'
    assert candidate.features == {'pif': False, 'width': 64}
