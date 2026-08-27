import hashlib
import json
import subprocess
from pathlib import Path

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


def test_numeric_source_accepts_only_common_checkout_reproduction_link(tmp_path):
    from mambapose_opt.numeric_source import (
        NumericSourceError, build_numeric_source_binding,
        resolve_numeric_file)
    from mambapose_opt.schema import load_candidate_manifest

    main = tmp_path / 'main'
    main.mkdir()
    (main / 'configs').mkdir()
    (main / 'optimization').mkdir()
    (main / '.gitignore').write_text('work_dirs/\n')
    (main / 'configs/candidate.py').write_text('policy = True\n')
    (main / 'optimization/coco_train2017_authority.json').write_text(
        '{"split":"train2017"}\n')
    checkpoint_bytes = b'checkpoint'
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest = {
        'schema_version': 1,
        'candidates': [{
            'id': 'numeric', 'route': 'ssm-quant-pwl',
            'kind': 'fake-quant', 'config': 'configs/candidate.py',
            'checkpoint': 'work_dirs/reproduction/model.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'weight-only'},
        }],
    }
    (main / 'optimization/candidates.json').write_text(json.dumps(manifest))
    subprocess.run(['git', 'init', '-q'], cwd=main, check=True)
    subprocess.run(['git', 'add', '.'], cwd=main, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=main, check=True)
    shared = main / 'work_dirs/reproduction'
    shared.mkdir(parents=True)
    (shared / 'model.pth').write_bytes(checkpoint_bytes)
    linked = tmp_path / 'linked'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '--detach', str(linked), 'HEAD'],
        cwd=main, check=True)
    (linked / 'work_dirs').mkdir()
    exposed = linked / 'work_dirs/reproduction'
    exposed.symlink_to(shared, target_is_directory=True)
    candidate = load_candidate_manifest(
        linked / 'optimization/candidates.json')[0]

    binding = build_numeric_source_binding(
        repository_root=linked, candidate=candidate,
        manifest_path=linked / 'optimization/candidates.json',
        policy_path=linked / 'configs/candidate.py')
    assert binding['checkpoint_path'] == 'work_dirs/reproduction/model.pth'
    assert binding['checkpoint_sha256'] == checkpoint_sha
    assert resolve_numeric_file(
        linked, Path(binding['checkpoint_path']), 'checkpoint').read_bytes() \
        == checkpoint_bytes

    exposed.unlink()
    alternate = tmp_path / 'alternate'
    alternate.mkdir()
    (alternate / 'model.pth').write_bytes(checkpoint_bytes)
    exposed.symlink_to(alternate, target_is_directory=True)
    with pytest.raises(NumericSourceError, match='shared|symlink'):
        build_numeric_source_binding(
            repository_root=linked, candidate=candidate,
            manifest_path=linked / 'optimization/candidates.json',
            policy_path=linked / 'configs/candidate.py')


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
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.validate_numeric_convert_artifact',
        lambda *_args, **_kwargs: {})
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


def test_controller_rejects_self_authorized_w8a8_runtime_config(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    config = tmp_path / 'configs/candidate.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    calibration = tmp_path / 'work_dirs/optimization/w8a8/calibrate/calibrate.json'
    injected = tmp_path / 'work_dirs/optimization/w8a8/convert/injected.py'
    for path, content in (
            (config, b'numeric_optimization = dict(candidate_kind="w8a8")\n'),
            (checkpoint, b'checkpoint'), (calibration, b'{}'),
            (injected, b'numeric_optimization = dict(candidate_kind="baseline")\n')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = _candidate(
        'w8a8', 'fake-quant', {'numeric_kind': 'w8a8', 'auto_run': False})
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, lambda *_args: None,
        repository_root=tmp_path, stages=('convert',))
    monkeypatch.setattr(
        'mambapose_opt.numeric_source.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})
    artifact = tmp_path / 'work_dirs/optimization/w8a8/convert/convert.json'
    artifact.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': candidate.id, 'stage': 'convert',
        'result': {
            'source': {},
            'runtime_bindings': {
                'config': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'checkpoint': {'path': 'checkpoint.pth',
                               'sha256': digest(checkpoint)},
                'policy': {'path': 'configs/candidate.py',
                           'sha256': digest(config)},
                'calibration': {
                    'path': calibration.relative_to(tmp_path).as_posix(),
                    'sha256': digest(calibration)},
            },
            'runtime_config': {
                'path': injected.relative_to(tmp_path).as_posix(),
                'sha256': digest(injected)},
            'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
        },
    }))

    with pytest.raises(ArtifactValidationError, match='canonical|calibration|fields'):
        controller._artifact_schema('convert', artifact)


def test_train_artifact_rejects_hash_matching_empty_recovery_admission(
        tmp_path, monkeypatch):
    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, validate_numeric_train_artifact)

    checkpoint = tmp_path / 'checkpoint.pth'
    config = tmp_path / 'configs/candidate.py'
    admission = tmp_path / 'work_dirs/optimization/c/train/recovery-admission.json'
    runtime_config = tmp_path / 'work_dirs/optimization/c/train/resolved-train.py'
    runtime_checkpoint = tmp_path / 'work_dirs/optimization/c/train/best_numeric.pth'
    metadata = tmp_path / 'work_dirs/optimization/c/train/runtime-metadata.json'
    for path, content in (
            (checkpoint, b'parent'), (config, b'config'), (admission, b'{}'),
            (runtime_config, b'runtime'), (runtime_checkpoint, b'trained')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = _candidate(
        'recovery', 'fake-quant', {
            'numeric_kind': 'w8a8', 'recovery_candidate': True,
            'runtime_checkpoint': runtime_checkpoint.relative_to(
                tmp_path).as_posix(),
        })
    metadata.write_text(json.dumps({
        'schema_version': 1, 'candidate_id': candidate.id,
        'route': candidate.route, 'numeric_kind': 'w8a8',
        'parent_checkpoint_sha256': candidate.checkpoint_sha256,
        'runtime_checkpoint_sha256': digest(runtime_checkpoint),
        'recovery_admission_sha256': digest(admission),
    }))
    reference = lambda path: {
        'path': path.relative_to(tmp_path).as_posix(), 'sha256': digest(path)}
    artifact = {
        'schema_version': 1, 'candidate_id': candidate.id, 'stage': 'train',
        'result': {
            'route': 'ssm-quant-pwl', 'source': {},
            'parent': {
                'config': candidate.config.as_posix(),
                'checkpoint': candidate.checkpoint.as_posix(),
                'checkpoint_sha256': candidate.checkpoint_sha256},
            'dependency': {'recovery_admission': reference(admission)},
            'protocol': {
                'seed': 0, 'operation': 'one-bounded-numeric-recovery',
                'attributed_error': 'fabricated',
                'preliminary_ap_drop': 0.31},
            'runtime': {
                'config': reference(runtime_config),
                'checkpoint': reference(runtime_checkpoint),
                'metadata': reference(metadata),
                'transform': 'bounded-numeric-recovery-v1'},
        },
    }
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.validate_numeric_source_binding',
        lambda *_args, **_kwargs: {})

    with pytest.raises(NumericRuntimeError, match='admission'):
        validate_numeric_train_artifact(
            artifact, candidate=candidate, repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


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


def test_explicit_recovery_candidate_trains_before_derived_runtime_stages():
    from tools.optimization.run_campaign import _stages_for_candidate

    recovery = _candidate(
        'w8a8-recovery', 'fake-quant', {
            'numeric_kind': 'w8a8', 'auto_run': False, 'conditional': True,
            'recovery_candidate': True,
            'runtime_checkpoint':
                'work_dirs/optimization/w8a8-recovery/train/best_numeric.pth',
            'recovery_screen_candidate': 'w8a8-s-v1',
            'recovery_baseline_evaluation':
                'work_dirs/optimization/full-s-v1/evaluate/evaluate.json',
            'recovery_candidate_evaluation':
                'work_dirs/optimization/w8a8-s-v1/evaluate/evaluate.json',
        })

    assert _stages_for_candidate(recovery) == (
        'train', 'profile', 'evaluate', 'latency')


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
        'granularity': 'channel', 'sample_count': 2, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0,
        'max_abs': [1.0, 2.0, 3.0],
        'range': [-3.0, 3.0],
        'percentiles': {'0.5': 1.0, '0.9': 2.0, '0.99': 3.0,
                        '0.999': 3.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0, 'token_ids': None,
        'observed_shape': [3],
    }
    wide_record = dict(record)
    wide_record['max_abs'] = [float(index) for index in range(1, 8)]
    wide_record['range'] = [-7.0, 7.0]
    wide_record['observed_shape'] = [7]
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
            'records': {'narrow.input': record, 'wide.input': wide_record},
            'required_records': ['narrow.input', 'wide.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'narrow': {
                    'source_record': 'narrow.input', 'granularity': 'channel',
                    'scale': [1 / 127, 2 / 127, 3 / 127]},
                'wide': {
                    'source_record': 'wide.input', 'granularity': 'channel',
                    'scale': [index / 127 for index in range(1, 8)]},
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

    assert dict(policy.role_specs)['narrow'].activation_scale == (
        1 / 127, 2 / 127, 3 / 127)
    assert len(dict(policy.role_specs)['wide'].activation_scale) == 7


def test_policy_loader_rejects_scale_not_derived_from_observer_record():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, quant_policy_from_config)

    record = {
        'granularity': 'tensor', 'sample_count': 4, 'zero_count': 0,
        'underflow_count': 0, 'overflow_count': 0, 'max_abs': 2.0,
        'range': [-2.0, 1.0],
        'percentiles': {'0.5': 1.0, '0.9': 2.0, '0.99': 2.0,
                        '0.999': 2.0},
        'algorithm': 'fixed-log2-histogram-v1', 'histogram_bins': 256,
        'histogram_domain': [2 ** -32, 2 ** 32],
        'percentile_bound_valid': True,
        'relative_error_bound': 2 ** 0.25 - 1,
        'outlier_ratio_above_p99_bin': 0.0,
        'token_ids': None, 'observed_shape': [],
    }
    identity = _calibration_identity_fixture()
    artifact = {
        'schema_version': 1, 'candidate_id': 'w8a8', 'stage': 'calibrate',
        'source': {}, 'identity': identity,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': 1,
            'sample_order_sha256': 'b' * 64,
        },
        'hooks': {
            'records': {'layer.input': record},
            'required_records': ['layer.input'],
            'unsupported_internals': [],
            'activation_scales': {
                'layer': {'source_record': 'layer.input',
                          'granularity': 'tensor', 'scale': 0.5},
            },
        },
    }
    policy = {
        'allow': ('layer',), 'deny': (),
        'activation_observers': {'layer': 'layer.input'},
        'spec': {
            'enabled': True, 'weight_bits': 8, 'activation_bits': 8,
            'activation_scale': 'runtime-calibration-artifact',
            'per_output_channel': True, 'symmetric': True,
        },
    }

    with pytest.raises(NumericBindingError, match='derived|measured'):
        quant_policy_from_config(policy, calibration_artifact=artifact)


def _calibration_identity_fixture():
    sha = 'a' * 64
    return {
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
