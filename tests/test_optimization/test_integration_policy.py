from __future__ import annotations

import hashlib
import json

import pytest


def _parent(tmp_path, candidate_id):
    from mmpose.models.distillers.mambapose_heatmap_distiller import \
        ParentResult

    path = tmp_path / f'{candidate_id}.json'
    path.write_text(json.dumps({'candidate_id': candidate_id}))
    return ParentResult(
        candidate_id=candidate_id,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_rejects_more_than_one_structural_feature(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        IntegrationSpec, validate_integration_features)

    features = ('no-pif-s-v1', 'static-neighbor-s-v1')
    spec = IntegrationSpec(
        features=features,
        parent_results=tuple(_parent(tmp_path, item) for item in features))

    with pytest.raises(ValueError, match='one structural'):
        validate_integration_features(spec)


def test_rejects_more_than_one_numeric_feature(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        IntegrationSpec, validate_integration_features)

    features = ('w8-weight-only-s-v1', 'pwl-silu-s-v1')
    spec = IntegrationSpec(
        features=features,
        parent_results=tuple(_parent(tmp_path, item) for item in features))

    with pytest.raises(ValueError, match='one numeric'):
        validate_integration_features(spec)


def test_allows_one_structural_plus_one_numeric_with_bound_parents(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        IntegrationSpec, validate_integration_features)

    features = ('no-pif-s-v1', 'w8-weight-only-s-v1')
    spec = IntegrationSpec(
        features=features,
        parent_results=tuple(_parent(tmp_path, item) for item in features))

    validate_integration_features(spec)


def test_rejects_parent_candidate_id_or_content_hash_mismatch(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        IntegrationSpec, ParentResult, validate_integration_features)

    parent = _parent(tmp_path, 'no-pif-s-v1')
    wrong_id = ParentResult(
        candidate_id='static-neighbor-s-v1',
        path=parent.path,
        sha256=parent.sha256)
    with pytest.raises(ValueError, match='exactly one parent'):
        validate_integration_features(
            IntegrationSpec(features=('no-pif-s-v1', ),
                            parent_results=(wrong_id, )))

    parent.path.write_text('{"mutated": true}')
    with pytest.raises(ValueError, match='sha256 mismatch'):
        validate_integration_features(
            IntegrationSpec(features=('no-pif-s-v1', ),
                            parent_results=(parent, )))


def test_rejects_unknown_or_duplicate_features(tmp_path):
    from mmpose.models.distillers.mambapose_heatmap_distiller import (
        IntegrationSpec, validate_integration_features)

    with pytest.raises(ValueError, match='unknown integration feature'):
        validate_integration_features(
            IntegrationSpec(features=('mystery', ), parent_results=()))

    parent = _parent(tmp_path, 'no-pif-s-v1')
    with pytest.raises(ValueError, match='unique'):
        validate_integration_features(
            IntegrationSpec(
                features=('no-pif-s-v1', 'no-pif-s-v1'),
                parent_results=(parent, parent)))


def test_integrated_config_is_fail_closed_before_isolated_results():
    from mmengine.config import Config

    config = Config.fromfile(
        'configs/optimization/accuracy_first/integrated_candidate.py')

    assert config.auto_run is False
    assert config.model is None
    assert not config.integration_spec.features
    assert 'isolated' in config.blocked_reason
