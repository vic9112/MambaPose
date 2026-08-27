import hashlib
import json

import pytest


def _candidate(identifier, kind, features):
    from mambapose_opt.schema import CandidateSpec

    return CandidateSpec.from_dict({
        'id': identifier, 'route': 'ssm-quant-pwl', 'kind': kind,
        'config': 'configs/candidate.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0, 'features': features,
    })


def test_route3_stage_plans_follow_numeric_admission_order():
    from mambapose_opt.numeric_conversion import numeric_stage_plan

    assert numeric_stage_plan('weight-only', conditional=False) == (
        'convert', 'export', 'profile', 'evaluate', 'latency', 'compare')
    assert numeric_stage_plan('w8a8', conditional=False) == (
        'calibrate', 'convert', 'train', 'profile', 'evaluate', 'latency',
        'compare')
    with pytest.raises(ValueError, match='conditional admission'):
        numeric_stage_plan('binary-qk', conditional=False)
    assert numeric_stage_plan('binary-qk', conditional=True)[0] == 'train'


def test_numeric_downstream_binding_rehashes_referenced_artifacts(tmp_path):
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, bind_numeric_inputs, verify_numeric_inputs)

    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    policy = tmp_path / 'policy.json'
    config.write_text('config')
    checkpoint.write_bytes(b'checkpoint')
    policy.write_text('{}')
    binding = bind_numeric_inputs({
        'config': config, 'checkpoint': checkpoint, 'policy': policy})

    verify_numeric_inputs(binding)
    policy.write_text('{"drift": true}')
    with pytest.raises(NumericBindingError, match='policy'):
        verify_numeric_inputs(binding)


def test_controller_rehashes_numeric_nested_bindings_and_export(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    config = tmp_path / 'configs/candidate.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    policy = tmp_path / 'configs/policy.py'
    export = tmp_path / 'work_dirs/optimization/packed.int8.pt'
    for path, content in ((config, b'config'), (checkpoint, b'checkpoint'),
                          (policy, b'policy'), (export, b'int8')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    candidate = _candidate(
        'weight', 'fake-quant',
        {'numeric_kind': 'weight-only', 'auto_run': False})
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, lambda *_args: None,
        repository_root=tmp_path, stages=('export',))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = tmp_path / 'export.json'
    artifact.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': 'weight', 'stage': 'export',
        'result': {
            'runtime_bindings': {
                'config': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'checkpoint': {'path': 'checkpoint.pth',
                               'sha256': digest(checkpoint)},
                'policy': {'path': 'configs/policy.py',
                           'sha256': digest(policy)},
            },
            'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
            'export': {
                'path': 'work_dirs/optimization/packed.int8.pt',
                'sha256': digest(export), 'bytes': export.stat().st_size,
                'format': 'symmetric-int8-per-output-channel-v1',
            },
        },
    }))

    assert controller._artifact_schema('export', artifact) == (
        'optimization-stage-envelope-v1')
    policy.write_bytes(b'drift')
    with pytest.raises(ArtifactValidationError, match='policy.*hash'):
        controller._artifact_schema('export', artifact)
    policy.write_bytes(b'policy')
    export.write_bytes(b'drift')
    with pytest.raises(ArtifactValidationError, match='export.*hash'):
        controller._artifact_schema('export', artifact)


def test_campaign_does_not_auto_select_opt_in_numeric_candidates():
    from tools.optimization.run_campaign import _select

    baseline = _candidate('baseline', 'fake-quant', {'numeric_kind': 'observer'})
    opt_in = _candidate(
        'binary', 'binary-qk',
        {'numeric_kind': 'binary-qk', 'auto_run': False, 'conditional': True})

    assert _select((baseline, opt_in), (), admit_conditional=False) == (baseline,)
    with pytest.raises(ValueError, match='conditional'):
        _select((baseline, opt_in), ('binary',), admit_conditional=False)
    assert _select(
        (baseline, opt_in), ('binary',), admit_conditional=True) == (opt_in,)


def test_campaign_uses_route_specific_numeric_stage_plan():
    from tools.optimization.run_campaign import _stages_for_candidate

    candidate = _candidate(
        'weight', 'fake-quant',
        {'numeric_kind': 'weight-only', 'auto_run': False})
    assert _stages_for_candidate(candidate)[:3] == ('convert', 'export', 'profile')


def test_policy_loader_rejects_unmeasured_activation_placeholder():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, quant_policy_from_config)

    with pytest.raises(NumericBindingError, match='calibration'):
        quant_policy_from_config({
            'allow': ('head',), 'deny': (),
            'spec': {
                'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
                'activation_scale': 'runtime-calibration-artifact',
                'per_output_channel': True, 'symmetric': True,
            },
        })


def test_runtime_conversion_applies_config_policy_once_and_w8a8_fails_closed():
    from torch import nn

    from mambapose_opt.numeric_conversion import (
        NumericBindingError, apply_numeric_runtime)
    from mmpose.models.utils.hardware_friendly import FakeQuantLinear

    model = nn.Sequential(nn.Linear(4, 3))
    numeric = {
        'candidate_kind': 'weight-only',
        'quant_policy': {
            'allow': ('0',), 'deny': (),
            'spec': {
                'enabled': True, 'weight_bits': 8,
                'activation_bits': None, 'activation_scale': None,
                'per_output_channel': True, 'symmetric': True,
            },
        },
    }
    first = apply_numeric_runtime(model, numeric)
    second = apply_numeric_runtime(model, numeric)

    assert first is second
    assert isinstance(model[0], FakeQuantLinear)
    numeric['candidate_kind'] = 'w8a8'
    numeric['quant_policy']['spec']['activation_bits'] = 8
    numeric['quant_policy']['spec']['activation_scale'] = (
        'runtime-calibration-artifact')
    with pytest.raises(NumericBindingError, match='calibration'):
        apply_numeric_runtime(nn.Sequential(nn.Linear(4, 3)), numeric)
