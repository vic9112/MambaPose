from __future__ import annotations

from dataclasses import replace
import copy

import pytest
import torch

from mambapose_opt.formal_training import (
    FORMAL_TOLERANCES,
    FormalRepeatabilityError,
    build_formal_repeatability_result,
    compare_paired_initial_state,
    compare_repeatability_replay,
    validate_formal_repeatability_result,
    repeatability_result_to_dict,
    formal_repeatability_result_from_dict,
)


SHA = '1' * 64
COMMIT = '2' * 40


def _result(role='baseline', seed=2, *, delta=0.0, custom_scan=True):
    state = {
        'backbone.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
        'head.common': torch.ones(2),
        f'head.{role}': torch.tensor([float(seed)]),
    }
    evidence = {
        'forward': torch.tensor([1.0 + delta]),
        'loss': torch.tensor([2.0 + delta]),
        'backward': torch.tensor([3.0 + delta]),
        'gradient': torch.tensor([4.0 + delta]),
        'optimizer_update': torch.tensor([5.0 + delta]),
    }
    return build_formal_repeatability_result(
        role=role,
        seed=seed,
        config_path=f'configs/{role}-{seed}.py',
        config_closure_sha256=SHA,
        resolved_config_sha256='3' * 64,
        source_commit=COMMIT,
        environment_inventory_sha256='4' * 64,
        initialization_sha256='5' * 64,
        initial_state=state,
        common_state_keys=('backbone.weight', 'head.common'),
        evidence=evidence,
        gradients_finite=True,
        custom_scan_exercised=custom_scan,
        tolerances=FORMAL_TOLERANCES,
    )


def test_repeatability_artifact_has_complete_required_evidence():
    result = _result()
    validate_formal_repeatability_result(result)
    assert result.complete_initial_state_sha256 != result.common_state_sha256
    assert set(result.evidence_sha256) == {
        'forward', 'loss', 'backward', 'gradient', 'optimizer_update'}
    assert result.custom_scan_exercised is True
    assert formal_repeatability_result_from_dict(
        repeatability_result_to_dict(result)) == result


def test_repeatability_public_parser_rejects_unknown_nested_fields():
    document = repeatability_result_to_dict(_result())
    forged = copy.deepcopy(document)
    forged['evidence']['loss']['untracked'] = True
    with pytest.raises(FormalRepeatabilityError, match='record'):
        formal_repeatability_result_from_dict(forged)


def test_paired_initial_state_requires_bit_identical_common_state():
    baseline = _result('baseline')
    no_pif = _result('no_pif')
    paired = compare_paired_initial_state(baseline, no_pif)
    assert paired.seed == 2
    assert paired.common_state_sha256 == baseline.common_state_sha256


@pytest.mark.parametrize('field,value', [
    ('custom_scan_exercised', False),
    ('gradients_finite', False),
    ('environment_inventory_sha256', 'bad'),
    ('source_commit', 'bad'),
    ('tolerances', (('atol', 1.0), ('rtol', 1.0))),
])
def test_repeatability_admission_rejects_missing_or_drifted_identity(field, value):
    result = replace(_result(), **{field: value})
    with pytest.raises(FormalRepeatabilityError):
        validate_formal_repeatability_result(result)


def test_repeatability_replay_requires_same_complete_state_and_bounded_evidence():
    first = _result()
    compare_repeatability_replay(first, _result(delta=1e-8))
    with pytest.raises(FormalRepeatabilityError, match='tolerance'):
        compare_repeatability_replay(first, _result(delta=1e-2))
    with pytest.raises(FormalRepeatabilityError, match='complete initial'):
        compare_repeatability_replay(first, replace(
            _result(), complete_initial_state_sha256='8' * 64))
    with pytest.raises(FormalRepeatabilityError, match='authority'):
        compare_repeatability_replay(first, replace(
            _result(), environment_inventory_sha256='6' * 64))


def test_pair_rejects_wrong_seed_or_common_state():
    with pytest.raises(FormalRepeatabilityError, match='seed'):
        compare_paired_initial_state(_result('baseline'), _result('no_pif', 1))
    with pytest.raises(FormalRepeatabilityError, match='common'):
        compare_paired_initial_state(
            _result('baseline'),
            replace(_result('no_pif'), common_state_sha256='9' * 64))


def test_repeatability_builder_rejects_nonfinite_gradient_evidence():
    with pytest.raises(FormalRepeatabilityError, match='finite'):
        build_formal_repeatability_result(
            role='baseline', seed=0, config_path='configs/baseline-0.py',
            config_closure_sha256=SHA, resolved_config_sha256='3' * 64,
            source_commit=COMMIT, environment_inventory_sha256='4' * 64,
            initialization_sha256='5' * 64,
            initial_state={'weight': torch.ones(1)}, common_state_keys=('weight',),
            evidence={
                'forward': torch.ones(1), 'loss': torch.ones(1),
                'backward': torch.ones(1),
                'gradient': torch.tensor([float('inf')]),
                'optimizer_update': torch.ones(1),
            }, gradients_finite=False, custom_scan_exercised=True,
            tolerances=FORMAL_TOLERANCES)
