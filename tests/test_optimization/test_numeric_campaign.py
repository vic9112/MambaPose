import hashlib
import json
import subprocess

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
        'convert', 'export', 'profile', 'evaluate', 'latency')
    assert numeric_stage_plan('w8a8', conditional=False) == (
        'calibrate', 'convert', 'profile', 'evaluate', 'latency')
    with pytest.raises(ValueError, match='conditional admission'):
        numeric_stage_plan('binary-qk', conditional=False)
    assert numeric_stage_plan('binary-qk', conditional=True) == (
        'profile', 'evaluate', 'latency')
    assert all(
        'compare' not in numeric_stage_plan(kind, conditional=conditional)
        for kind, conditional in (
            ('observer', False), ('weight-only', False), ('w8a8', False),
            ('pwl', True), ('binary-qk', True)))


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


def test_numeric_source_binds_clean_commit_manifest_row_policy_and_checkpoint(
        tmp_path):
    from mambapose_opt.numeric_source import (
        NumericSourceError, build_numeric_source_binding,
        validate_numeric_source_binding)
    from mambapose_opt.schema import load_candidate_manifest

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'configs/candidate.py').write_text('policy = True\n')
    (tmp_path / 'checkpoints/model.pth').write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(b'checkpoint').hexdigest()
    manifest = {
        'schema_version': 1,
        'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'checkpoints/model.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }
    (tmp_path / 'optimization/candidates.json').write_text(json.dumps(manifest))
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text(
        '{"split":"train2017"}\n')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'], cwd=tmp_path,
        check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    candidate = load_candidate_manifest(
        tmp_path / 'optimization/candidates.json')[0]

    source = build_numeric_source_binding(
        repository_root=tmp_path, candidate=candidate,
        manifest_path=tmp_path / 'optimization/candidates.json',
        policy_path=tmp_path / 'configs/candidate.py')
    assert validate_numeric_source_binding(
        source, repository_root=tmp_path, candidate=candidate,
        manifest_path=tmp_path / 'optimization/candidates.json') == source
    (tmp_path / 'checkpoints/model.pth').write_bytes(b'drift')
    with pytest.raises(NumericSourceError, match='checkpoint'):
        validate_numeric_source_binding(
            source, repository_root=tmp_path, candidate=candidate,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_controller_rehashes_numeric_nested_bindings_and_export(
        tmp_path, monkeypatch):
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

    with pytest.raises(ArtifactValidationError, match='source binding'):
        controller._artifact_schema('export', artifact)
    value = json.loads(artifact.read_text())
    value['result']['source'] = {}
    artifact.write_text(json.dumps(value))
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})

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


def test_initial_conditional_candidates_do_not_depend_on_recovery_training():
    from pathlib import Path

    from tools.optimization.run_campaign import _stages_for_candidate

    w8a8 = _candidate(
        'w8a8', 'fake-quant',
        {'numeric_kind': 'w8a8', 'auto_run': False, 'conditional': True})
    pwl = _candidate(
        'pwl', 'pwl',
        {'numeric_kind': 'pwl', 'auto_run': False, 'conditional': True})
    assert _stages_for_candidate(w8a8) == (
        'calibrate', 'convert', 'profile', 'evaluate', 'latency')
    assert _stages_for_candidate(pwl) == ('profile', 'evaluate', 'latency')
    assert Path('tools/optimization/train_candidate.py').is_file()


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


def test_policy_loader_consumes_verified_per_role_calibration_scales():
    from mambapose_opt.numeric_conversion import quant_policy_from_config

    record = {
        'granularity': 'tensor', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 1.0,
        'range': [-1.0, 1.0], 'percentiles': {'0.5': 1.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True, 'relative_error_bound': 0.2,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [],
    }
    sha = 'a' * 64
    identity = {
        'candidate_id': 'full-s-v1',
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': sha, 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': sha, 'policy': 'policy.py',
        'policy_sha256': sha, 'split': 'train2017',
        'git_commit': 'c' * 40,
        'dataset': {
            'annotation':
                'data/coco/annotations/person_keypoints_train2017.json',
            'annotation_sha256': sha,
            'image_prefix': 'data/coco/train2017',
            'inventory': 'data/inventory.json', 'inventory_sha256': sha,
            'train_archive': 'downloads/train2017.zip',
            'train_archive_sha256': sha, 'image_count': 118287,
            'image_content_algorithm': 'sha256-zip-member-bytes-v1',
            'image_content_aggregate_sha256': sha,
            'image_order_algorithm':
                'sha256-zip-central-directory-order-v1',
            'image_order_sha256': sha,
            'annotation_archive': 'downloads/annotations.zip',
            'annotation_archive_sha256': sha,
            'annotation_member':
                'annotations/person_keypoints_train2017.json',
            'annotation_member_sha256': sha,
        },
    }
    artifact = {
        'schema_version': 1, 'candidate_id': 'w8a8', 'stage': 'calibrate',
        'source': {},
        'identity': identity,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 2,
            'sample_order_sha256': 'b' * 64,
        },
        'hooks': {
            'records': {'narrow.input': record, 'wide.input': dict(record)},
            'required_records': ['narrow.input', 'wide.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'narrow': {
                    'source_record': 'narrow.input', 'granularity': 'channel',
                    'scale': [0.1, 0.2, 0.3]},
                'wide': {
                    'source_record': 'wide.input', 'granularity': 'channel',
                    'scale': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]},
            },
        },
    }
    policy = quant_policy_from_config({
        'allow': ('narrow', 'wide'), 'deny': (),
        'activation_observers': {
            'narrow': 'narrow.input', 'wide': 'wide.input'},
        'spec': {
            'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
            'activation_scale': 'runtime-calibration-artifact',
            'per_output_channel': True, 'symmetric': True,
        },
    }, calibration_artifact=artifact)

    assert dict(policy.role_specs)['narrow'].activation_scale == (0.1, 0.2, 0.3)
    assert len(dict(policy.role_specs)['wide'].activation_scale) == 7
