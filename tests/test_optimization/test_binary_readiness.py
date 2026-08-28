from pathlib import Path
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from types import MappingProxyType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def test_binary_selection_rejects_missing_stage_b_pwl_dependency():
    """A conditional flag alone must never admit the invasive binary route."""
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest
    from tools.optimization.run_campaign import _select

    manifest = REPOSITORY_ROOT / 'optimization/candidates.json'
    candidates = load_candidate_manifest(manifest)

    with pytest.raises(
            CandidateManifestError,
            match='public-valid Stage-B PWL dependency'):
        _select(
            candidates, ['binary-qk-s-v1'], admit_conditional=True)


def test_binary_selection_rejects_unreadable_stage_b_pwl_dependency():
    """A truthy manifest feature must not substitute for validated evidence."""
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest
    from tools.optimization.run_campaign import _select

    candidates = load_candidate_manifest(
        REPOSITORY_ROOT / 'optimization/candidates.json')
    binary = next(item for item in candidates if item.id == 'binary-qk-s-v1')
    claimed = replace(binary, features=MappingProxyType({
        **binary.features,
        'pwl_stage_b_artifact': (
            'work_dirs/optimization/does-not-exist/pwl-stage-b.json'),
        'pwl_stage_b_sha256': '0' * 64,
    }))

    with pytest.raises(
            CandidateManifestError,
            match='public-valid Stage-B PWL dependency'):
        _select((claimed,), [claimed.id], admit_conditional=True)


def _stage_b_fixture(tmp_path: Path, *, candidate_ap: float = 72.6):
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    checkpoint_hash = 'a' * 64
    baseline = CandidateSpec.from_dict({
        'id': 'full-s-v1', 'route': 'baseline', 'kind': 'float',
        'config': 'configs/baseline.py', 'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'seed': 0, 'features': {},
    })
    pwl = CandidateSpec.from_dict({
        'id': 'pwl-gelu-s-v1', 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': 'configs/pwl.py', 'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'seed': 0,
        'features': {'numeric_kind': 'pwl', 'pwl_function': 'gelu'},
    })
    binary_values = {
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'seed': 0,
        'features': {
            'numeric_kind': 'binary-qk', 'conditional': True,
        },
    }
    for path, text in (
            (root / baseline.config, 'model = dict()\n'),
            (root / pwl.config, "pwl_function = 'gelu'\n"),
            (root / binary_values['config'], "qk_mode = 'binary'\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    manifest = root / 'optimization/candidates.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)

    roots = {
        'baseline': root / 'work_dirs/optimization/baseline/full-s-v1/0',
        'candidate': root / (
            'work_dirs/optimization/ssm-quant-pwl/pwl-gelu-s-v1/0'),
    }
    for name, result_root in roots.items():
        _write_json(
            result_root / 'evaluate/evaluate.json',
            {'fixture': name})
    modes = {}
    for mode in ('flip', 'no_flip'):
        modes[mode] = {
            'baseline_root': roots['baseline'].relative_to(root).as_posix(),
            'baseline_evaluation_sha256': _sha256(
                roots['baseline'] / 'evaluate/evaluate.json'),
            'candidate_root': roots['candidate'].relative_to(root).as_posix(),
            'candidate_evaluation_sha256': _sha256(
                roots['candidate'] / 'evaluate/evaluate.json'),
            'ap_drop_points': 72.8 - candidate_ap,
        }
    pwl_authority = {
        'candidate_id': pwl.id,
        'candidate_row_sha256': hashlib.sha256(json.dumps({
            'id': pwl.id, 'route': pwl.route, 'kind': pwl.kind,
            'config': pwl.config.as_posix(),
            'checkpoint': pwl.checkpoint.as_posix(),
            'checkpoint_sha256': pwl.checkpoint_sha256,
            'seed': pwl.seed, 'features': dict(pwl.features),
        }, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True).encode('utf-8')).hexdigest(),
        'git_commit': '2' * 40,
        'manifest_path': 'optimization/candidates.json',
        'manifest_sha256': '3' * 64,
        'config_path': pwl.config.as_posix(),
        'config_sha256': _sha256(root / pwl.config),
        'checkpoint_path': pwl.checkpoint.as_posix(),
        'checkpoint_sha256': pwl.checkpoint_sha256,
        'policy_path': pwl.config.as_posix(),
        'policy_sha256': _sha256(root / pwl.config),
        'authority_path': 'optimization/coco_train2017_authority.json',
        'authority_sha256': '4' * 64,
        'seed': 0,
        'config_closure': [{
            'path': pwl.config.as_posix(),
            'sha256': _sha256(root / pwl.config),
        }],
        'calibration': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-gelu-s-v1/0/calibrate/calibrate.json'),
            'sha256': '5' * 64,
            'schema_version': 3,
            'sample_order_sha256': '6' * 64,
        },
        'selection_policy': 'observed-range-max-then-mean-v1',
        'installation': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-gelu-s-v1/0/convert/pwl-installation.json'),
            'sha256': '7' * 64,
        },
        'operation_manifest_sha256': '8' * 64,
    }
    baseline_source = {
        'git_commit': pwl_authority['git_commit'],
        'manifest_path': pwl_authority['manifest_path'],
        'manifest_sha256': pwl_authority['manifest_sha256'],
        'config_path': baseline.config.as_posix(),
        'config_sha256': _sha256(root / baseline.config),
        'authority_path': pwl_authority['authority_path'],
        'authority_sha256': pwl_authority['authority_sha256'],
    }
    pwl_source = {
        name: pwl_authority[name]
        for name in (
            'git_commit', 'manifest_path', 'manifest_sha256', 'config_path',
            'config_sha256', 'authority_path', 'authority_sha256')
    }
    result_authority = {}
    result_rows = {}
    for mode in ('flip', 'no_flip'):
        baseline_provenance = {
            'checkpoint_sha256': baseline.checkpoint_sha256,
            'config_sha256': baseline_source['config_sha256'],
            'data_inventory_sha256': '9' * 64,
            'git_commit': baseline_source['git_commit'],
        }
        pwl_provenance = {
            **baseline_provenance,
            'config_sha256': pwl_source['config_sha256'],
        }
        shared_protocol = {
            'dataset': 'coco', 'split': 'val2017',
            'complete_split': True,
            'authority_path': baseline_source['authority_path'],
            'authority_sha256': baseline_source['authority_sha256'],
            'data_inventory_sha256': '9' * 64,
            'detections_sha256': 'a' * 64,
            'evaluator': 'mmpose.CocoMetric',
            'tta': {'mode': mode, 'flip_test': mode == 'flip'},
        }
        baseline_protocol = {
            **shared_protocol,
            'source_config': baseline.config.as_posix(),
            'checkpoint': baseline.checkpoint.as_posix(),
        }
        pwl_protocol = {
            **shared_protocol,
            'source_config': pwl.config.as_posix(),
            'checkpoint': pwl.checkpoint.as_posix(),
        }
        baseline_determinism = {
            'python_seed': 0, 'numpy_seed': 0, 'torch_seed': 0,
            'worker_count': 2, 'persistent_workers': False,
            'order_hashes': {'0': 'b' * 64},
            'provenance': baseline_provenance,
        }
        pwl_determinism = {
            **baseline_determinism,
            'provenance': pwl_provenance,
        }
        result_rows[('baseline', mode)] = {
            'source': baseline_source,
            'provenance': baseline_provenance,
            'protocol': baseline_protocol,
            'determinism': baseline_determinism,
        }
        result_rows[('candidate', mode)] = {
            'source': pwl_source,
            'provenance': pwl_provenance,
            'protocol': pwl_protocol,
            'determinism': pwl_determinism,
        }
        result_authority[mode] = {
            'artifact_root': modes[mode]['baseline_root'],
            'evaluation_sha256': modes[mode][
                'baseline_evaluation_sha256'],
            'provenance_sha256': _json_sha256(baseline_provenance),
            'protocol_sha256': _json_sha256(baseline_protocol),
            'determinism_sha256': _json_sha256(baseline_determinism),
        }
    baseline_authority = {
        'candidate_id': baseline.id,
        'candidate_row_sha256': _json_sha256({
            'id': baseline.id, 'route': baseline.route,
            'kind': baseline.kind, 'config': baseline.config.as_posix(),
            'checkpoint': baseline.checkpoint.as_posix(),
            'checkpoint_sha256': baseline.checkpoint_sha256,
            'seed': baseline.seed, 'features': dict(baseline.features),
        }),
        'seed': baseline.seed,
        'source': baseline_source,
        'checkpoint': {
            'path': baseline.checkpoint.as_posix(),
            'sha256': baseline.checkpoint_sha256,
        },
        'config_closure': [{
            'path': baseline.config.as_posix(),
            'sha256': _sha256(root / baseline.config),
        }],
        'modes': result_authority,
    }
    gate = {
        'schema_version': 3,
        'artifact_kind': 'pwl-stage-b-pass',
        'decision': 'passed',
        'ap_drop_limit_points': 0.3,
        'baseline_candidate_id': baseline.id,
        'pwl_candidate_id': pwl.id,
        'pwl_policy': {
            'path': pwl.config.as_posix(),
            'sha256': _sha256(root / pwl.config),
            'function': 'gelu',
        },
        'baseline_authority': baseline_authority,
        'pwl_authority': pwl_authority,
        'modes': modes,
    }
    gate_path = root / 'work_dirs/optimization/gates/pwl-stage-b.json'
    _write_json(gate_path, gate)
    binary_values['features'].update({
        'pwl_stage_b_artifact': gate_path.relative_to(root).as_posix(),
        'pwl_stage_b_sha256': _sha256(gate_path),
    })
    binary = CandidateSpec.from_dict(binary_values)
    _write_json(manifest, {
        'schema_version': 1,
        'candidates': [
            {
                'id': item.id, 'route': item.route, 'kind': item.kind,
                'config': item.config.as_posix(),
                'checkpoint': item.checkpoint.as_posix(),
                'checkpoint_sha256': item.checkpoint_sha256,
                'seed': item.seed, 'features': dict(item.features),
            }
            for item in (baseline, pwl, binary)
        ],
    })
    results = {}
    for mode in ('flip', 'no_flip'):
        baseline_row = result_rows[('baseline', mode)]
        candidate_row = result_rows[('candidate', mode)]
        results[(roots['baseline'].resolve(), mode)] = SimpleNamespace(
            candidate_id=baseline.id, route=baseline.route,
            candidate_kind=baseline.kind, seed=baseline.seed,
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=72.8),
            profile={'parent': {'config': baseline.config.as_posix()}},
            pwl_authority=None, **baseline_row)
        results[(roots['candidate'].resolve(), mode)] = SimpleNamespace(
            candidate_id=pwl.id, route=pwl.route,
            candidate_kind=pwl.kind, seed=pwl.seed,
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=candidate_ap),
            profile={'parent': {'config': pwl.config.as_posix()}},
            pwl_authority=json.loads(json.dumps(pwl_authority)),
            **candidate_row)

    def load_result(result_root: Path, *, mode: str):
        return results[(Path(result_root).resolve(), mode)]

    load_result.results = results

    return root, manifest, binary, gate, load_result


def test_binary_admission_validates_public_stage_b_pwl_results(
        tmp_path, monkeypatch):
    """Both public CandidateResult modes and the 0.3 AP screen are mandatory."""
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    validated = validate_binary_qk_admission(
        binary, repository_root=root, manifest_path=manifest)

    assert validated['pwl_candidate_id'] == 'pwl-gelu-s-v1'
    assert validated['modes']['flip']['ap_drop_points'] == pytest.approx(0.2)
    assert dict(validated) == gate


def test_campaign_selection_admits_only_validated_stage_b_dependency(
        tmp_path, monkeypatch):
    """The production selector must call the public dependency validator."""
    from tools.optimization.run_campaign import _select

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    selected = _select(
        (binary,), (binary.id,), admit_conditional=True,
        repository_root=root, manifest_path=manifest)

    assert selected == (binary,)


def test_binary_admission_normalizes_immutable_public_authority_sequences(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    for (result_root, _mode), result in load_result.results.items():
        if 'pwl-gelu-s-v1' in result_root.as_posix():
            result.pwl_authority['config_closure'] = tuple(
                result.pwl_authority['config_closure'])
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    validated = validate_binary_qk_admission(
        binary, repository_root=root, manifest_path=manifest)

    assert validated['decision'] == 'passed'


def _binary_model_fixture():
    from torch import nn
    from mmpose.models.heads.heatmap_heads.tokenbase import Transformer

    class TokenPoseFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_patches = 48
            self.num_keypoints = 17
            self.transformer = Transformer(
                256, depth=6, heads=8, mlp_dim=768, dropout=0.0,
                num_keypoints=17, scale_with_head=True, qk_mode='binary')

    class ModelFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Module()
            self.head.tokenpose = TokenPoseFixture()
            self.loss_anchor = nn.Parameter(__import__('torch').tensor(1.0))

        def forward(self, inputs, *, mode='tensor', **unused):
            if mode == 'loss':
                return {'loss_pose': self.loss_anchor.square()}
            return self.loss_anchor.expand(inputs.shape[0], 17, 64, 48)

    return ModelFixture()


def test_live_binary_operation_manifest_proves_exact_s_v1_head_contract():
    """A topology drift or floating-QK layer must invalidate hardware claims."""
    import torch

    from mambapose_opt.binary_operation import build_binary_operation_manifest

    manifest = build_binary_operation_manifest(_binary_model_fixture())

    assert manifest == {
        'schema_version': 1,
        'artifact_kind': 'binary-qk-operation-manifest',
        'implementation': 'ste-sign-einsum-software-proxy',
        'reference_scope': 'mechanism-inspired-sign-only-qk-preliminary',
        'learnable_attention_bias': False,
        'binaryattention_reproduction': False,
        'recovery_policy': (
            'bounded-qat-self-distillation-after-stage-b-only'),
        'qk_mode': 'binary',
        'layers': 6,
        'module_names': [
            f'head.tokenpose.transformer.layers.{index}.0.fn.fn'
            for index in range(6)],
        'heads': 8,
        'query_tokens': 65,
        'key_tokens': 65,
        'head_dim': 32,
        'zero_sign': 1,
        'ste_gradient': 'identity',
        'scale': {
            'kind': 'floating-original',
            'formula': '1/sqrt(head_dim)',
            'value': 32 ** -0.5,
        },
        'softmax': 'floating',
        'value': 'floating',
        'attention_accumulation': 'floating',
        'output_projection': 'floating',
        'theoretical_qk_multiplications_replaced': 6_489_600,
        'bitwise_kernel_present': False,
        'measured_integer_latency': False,
        'hardware_claim': 'none-software-proxy',
    }
    assert torch.equal(
        torch.tensor([manifest['zero_sign']]), torch.tensor([1]))


def _valid_operation_manifest() -> dict:
    return {
        'schema_version': 1,
        'artifact_kind': 'binary-qk-operation-manifest',
        'implementation': 'ste-sign-einsum-software-proxy',
        'reference_scope': 'mechanism-inspired-sign-only-qk-preliminary',
        'learnable_attention_bias': False,
        'binaryattention_reproduction': False,
        'recovery_policy': (
            'bounded-qat-self-distillation-after-stage-b-only'),
        'qk_mode': 'binary',
        'layers': 6,
        'module_names': [
            f'head.tokenpose.transformer.layers.{index}.0.fn.fn'
            for index in range(6)],
        'heads': 8,
        'query_tokens': 65,
        'key_tokens': 65,
        'head_dim': 32,
        'zero_sign': 1,
        'ste_gradient': 'identity',
        'scale': {
            'kind': 'floating-original',
            'formula': '1/sqrt(head_dim)',
            'value': 32 ** -0.5,
        },
        'softmax': 'floating',
        'value': 'floating',
        'attention_accumulation': 'floating',
        'output_projection': 'floating',
        'theoretical_qk_multiplications_replaced': 6_489_600,
        'bitwise_kernel_present': False,
        'measured_integer_latency': False,
        'hardware_claim': 'none-software-proxy',
    }


def test_evaluate_and_latency_bind_the_same_profile_operation_manifest(
        tmp_path, monkeypatch):
    """No later result may detach latency or AP from the profiled operation."""
    from mambapose_opt.binary_operation import (
        binary_profile_binding_for_stage, validate_binary_artifact_bundle)

    root = tmp_path / 'repo'
    candidate_root = root / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0')
    profile_path = candidate_root / 'profile/profile.json'
    evaluate_path = candidate_root / 'evaluate/evaluate.json'
    latency_path = candidate_root / 'latency/latency.json'
    smoke_path = candidate_root / 'smoke-stage-a/smoke.json'
    _write_json(smoke_path, {'fixture': True})
    smoke_binding = {
        'path': smoke_path.relative_to(root).as_posix(),
        'sha256': _sha256(smoke_path),
        'operation_sha256': hashlib.sha256(json.dumps(
            _valid_operation_manifest(), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
    }
    monkeypatch.setattr(
        'mambapose_opt.binary_smoke.validate_binary_stage_a_artifact',
        lambda *args, **kwargs: {
            'candidate_id': 'binary-qk-s-v1',
            'execution': {'operation': _valid_operation_manifest()}})
    _write_json(profile_path, {
        'candidate': 'binary-qk-s-v1',
        'binary_qk_operation': _valid_operation_manifest(),
        'binary_qk_smoke': smoke_binding,
    })

    evaluation_binding = binary_profile_binding_for_stage(
        evaluate_path.relative_to(root), repository_root=root)
    latency_binding = binary_profile_binding_for_stage(
        latency_path.relative_to(root), repository_root=root)
    _write_json(evaluate_path, {
        'candidate_id': 'binary-qk-s-v1',
        'stage': 'evaluate',
        'result': {'binary_qk_profile': evaluation_binding},
    })
    _write_json(latency_path, {
        'candidate_id': 'binary-qk-s-v1',
        'stage': 'latency',
        'result': {'binary_qk_profile': latency_binding},
    })

    operation = validate_binary_artifact_bundle(
        profile_path=profile_path.relative_to(root),
        evaluation_path=evaluate_path.relative_to(root),
        latency_path=latency_path.relative_to(root), repository_root=root,
        candidate_id='binary-qk-s-v1')

    assert evaluation_binding == latency_binding
    assert operation['layers'] == 6
    assert operation['theoretical_qk_multiplications_replaced'] == 6_489_600


def test_production_profile_embeds_live_binary_operation_manifest(
        tmp_path, monkeypatch):
    """A binary profile must inventory the live graph, not config prose."""
    import torch

    import tools.optimization.profile_model as tool
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    config_path = root / 'configs/binary.py'
    checkpoint = root / 'approved/model.pth'
    output = root / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0/'
        'profile/profile.json')
    config_path.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config_path.write_text('model = dict(type="Fixture")\n', encoding='utf-8')
    checkpoint.write_bytes(b'checkpoint')
    checkpoint_hash = _sha256(checkpoint)
    candidate = CandidateSpec.from_dict({
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'approved/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'seed': 0,
        'features': {'numeric_kind': 'binary-qk', 'conditional': True},
    })
    runtime = {
        'config_path': config_path,
        'config_sha256': _sha256(config_path),
        'checkpoint_path': checkpoint,
        'checkpoint_name': 'approved/model.pth',
        'checkpoint_sha256': checkpoint_hash,
        'train': None,
    }
    original_zeros = torch.zeros
    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', root)
    monkeypatch.setattr(tool, 'resolve_numeric_runtime', lambda *a, **k: runtime)
    monkeypatch.setattr(tool, 'clean_git_commit', lambda _root: 'a' * 40)
    monkeypatch.setattr(
        tool, 'build_numeric_source_binding', lambda **kwargs: {'bound': True})
    smoke_binding = {
        'path': ('work_dirs/optimization/ssm-quant-pwl/'
                 'binary-qk-s-v1/0/smoke-stage-a/smoke.json'),
        'sha256': '1' * 64,
        'operation_sha256': '2' * 64,
    }
    monkeypatch.setattr(
        'mambapose_opt.binary_operation.binary_smoke_binding_for_profile',
        lambda *args, **kwargs: smoke_binding)
    monkeypatch.setattr(
        'mmpose.apis.init_model', lambda *a, **k: _binary_model_fixture())
    monkeypatch.setattr(
        tool.torch, 'zeros', lambda shape, *, device: original_zeros(shape))
    monkeypatch.setattr(tool.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(tool.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv('MAMBAPOSE_PHYSICAL_DEVICE_INDEX', '0')

    result = tool.profile(
        candidate, (1, 3, 4, 4), manifest_path=root / 'manifest.json',
        output=output)

    assert result['binary_qk_operation'] == _valid_operation_manifest()
    assert result['binary_qk_smoke'] == smoke_binding


def test_binary_campaign_requires_stage_a_smoke_before_measurement(tmp_path):
    from mambapose_opt.controller import CUDA_STAGES
    from mambapose_opt.numeric_conversion import numeric_stage_plan
    from mambapose_opt.schema import CandidateSpec
    from tools.optimization.run_campaign import SubprocessStageRunner

    assert numeric_stage_plan('binary-qk', conditional=True) == (
        'smoke-stage-a', 'profile', 'evaluate', 'latency')
    assert 'smoke-stage-a' not in CUDA_STAGES
    candidate = CandidateSpec.from_dict({
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'binary-qk', 'conditional': True},
    })
    runner = SubprocessStageRunner(
        REPOSITORY_ROOT / 'work_dirs/optimization',
        REPOSITORY_ROOT / 'optimization/candidates.json', device_index=3)
    stage_dir = REPOSITORY_ROOT / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0/'
        'smoke-stage-a')

    command = runner._command(
        candidate, 'smoke-stage-a', stage_dir / 'smoke.json')

    assert Path(command[1]).name == 'smoke_binary_qk.py'
    assert command[command.index('--output-root') + 1] == (
        stage_dir.relative_to(REPOSITORY_ROOT).as_posix())
    assert command[command.index('--device-index') + 1] == '3'


def test_binary_smoke_accepts_only_controller_owned_empty_stage_directory(
        tmp_path):
    from mambapose_opt.binary_smoke import _prepare_smoke_output

    stage = tmp_path / 'smoke-stage-a'
    stage.mkdir()
    (stage / 'attempt-1.log').write_text('', encoding='utf-8')

    assert _prepare_smoke_output(stage) is False
    (stage / 'alternate.json').write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='unexpected existing files'):
        _prepare_smoke_output(stage)


def test_production_evaluation_binds_canonical_binary_profile(
        tmp_path, monkeypatch):
    """Formal COCO metrics must name and hash the profile they measure."""
    import tools.optimization.evaluate_candidate as tool
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    config = root / 'configs/binary.py'
    checkpoint = root / 'weights/model.pth'
    output = root / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0/'
        'evaluate/evaluate.json')
    profile = output.parent.parent / 'profile/profile.json'
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text('model = dict()\n', encoding='utf-8')
    checkpoint.write_bytes(b'checkpoint')
    _write_json(profile, {
        'candidate': 'binary-qk-s-v1',
        'binary_qk_operation': _valid_operation_manifest(),
        'binary_qk_smoke': {'fixture': True},
    })
    checkpoint_hash = _sha256(checkpoint)
    candidate = CandidateSpec.from_dict({
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'seed': 0,
        'features': {'numeric_kind': 'binary-qk', 'conditional': True},
    })
    runtime = {
        'config_path': config, 'config_sha256': _sha256(config),
        'checkpoint_path': checkpoint, 'checkpoint_name': 'weights/model.pth',
        'checkpoint_sha256': checkpoint_hash, 'train': None,
    }
    monkeypatch.setattr(tool, 'REPO_ROOT', root)
    monkeypatch.setattr(tool, 'resolve_numeric_runtime', lambda *a, **k: runtime)
    monkeypatch.setattr(tool, '_git_commit', lambda: 'a' * 40)
    monkeypatch.setattr(tool, 'build_source_binding', lambda **k: {'bound': True})
    monkeypatch.setattr(
        'mambapose_opt.binary_operation._validate_profile_smoke_binding',
        lambda *args, **kwargs: {'fixture': True})
    monkeypatch.setattr(
        tool, '_evaluate_mode',
        lambda *a, **k: {'fixture': True})

    envelope = tool.evaluate(
        candidate, output, manifest_path=root / 'manifest.json')

    assert envelope['result']['binary_qk_profile'] == {
        'path': profile.relative_to(root).as_posix(),
        'sha256': _sha256(profile),
        'operation_sha256': hashlib.sha256(json.dumps(
            _valid_operation_manifest(), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
    }


def test_stage_a_model_core_executes_optimizer_and_exact_round_trip(tmp_path):
    """The real smoke core must prove training and export/load, not infer only."""
    import torch
    from torch import nn

    from mambapose_opt.binary_smoke import execute_binary_stage_a_model

    class TinyPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 2, bias=False)

        def forward(self, inputs, data_samples=None, mode='tensor'):
            output = self.projection(inputs)
            if mode == 'loss':
                target = data_samples
                return {'loss_kpt': ((output - target) ** 2).mean()}
            if mode == 'tensor':
                return output
            raise ValueError(mode)

    torch.manual_seed(7)
    model = TinyPose()
    inputs = torch.tensor([[1.0, -2.0, 0.5]])
    targets = torch.tensor([[0.25, -0.75]])
    initial = model.projection.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    export_path = tmp_path / 'round-trip.pth'

    result = execute_binary_stage_a_model(
        model=model, inputs=inputs, data_samples=targets,
        optimizer=optimizer, model_factory=TinyPose,
        export_path=export_path,
        operation_builder=lambda unused: _valid_operation_manifest(),
        require_binary_targets=False)

    assert result['checks'] == {
        'forward': True,
        'loss': True,
        'backward': True,
        'optimizer_step': True,
        'finite_loss': True,
        'finite_gradients': True,
        'gradient_tensor_count': 1,
        'binary_target_count': 0,
        'state_round_trip_exact': True,
        'output_round_trip_exact': True,
    }
    assert result['binary_targets'] == []
    assert result['losses']['loss_kpt'] >= 0.0
    assert not torch.equal(model.projection.weight.detach(), initial)
    assert result['export'] == {
        'path': str(export_path),
        'sha256': _sha256(export_path),
        'bytes': export_path.stat().st_size,
        'format': 'torch-weights-only-state-dict-v1',
    }
    assert result['operation'] == _valid_operation_manifest()
    assert result['output']['shape'] == [1, 2]


def test_stage_a_rejects_loss_disconnected_from_all_binary_qk_layers(tmp_path):
    import torch

    from mambapose_opt.binary_smoke import execute_binary_stage_a_model

    model = _binary_model_fixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match='Binary Q/K gradient'):
        execute_binary_stage_a_model(
            model=model, inputs=torch.ones(1, 3, 4, 4), data_samples=None,
            optimizer=optimizer, model_factory=_binary_model_fixture,
            export_path=tmp_path / 'disconnected.pth')


def test_stage_a_records_each_binary_qk_gradient_parameter_and_adam_state(
        tmp_path):
    import torch

    from mambapose_opt.binary_smoke import execute_binary_stage_a_model

    class Connected(type(_binary_model_fixture())):
        def forward(self, inputs, *, mode='tensor', **unused):
            value = inputs
            for layer in self.head.tokenpose.transformer.layers:
                value, _, _ = layer[0].fn.fn(value)
            if mode == 'loss':
                return {'loss_pose': value.square().mean()}
            return value

    def factory():
        return Connected()

    torch.manual_seed(31)
    model = factory()
    inputs = torch.randn(1, 65, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = execute_binary_stage_a_model(
        model=model, inputs=inputs, data_samples=None, optimizer=optimizer,
        model_factory=factory, export_path=tmp_path / 'connected.pth')

    assert result['checks']['binary_target_count'] == 6
    assert [row['parameter'] for row in result['binary_targets']] == [
        f'head.tokenpose.transformer.layers.{index}.0.fn.fn.to_qkv.weight'
        for index in range(6)]
    assert all(
        row['q_gradient_norm'] > 0 and row['k_gradient_norm'] > 0
        and row['q_parameter_changed'] and row['k_parameter_changed']
        and row['optimizer_state_finite']
        for row in result['binary_targets'])


def test_binary_model_build_neutralizes_all_implicit_pretrained_initializers(
        monkeypatch):
    import torch
    from mmengine.config import Config

    from mambapose_opt.binary_smoke import _build_binary_model

    captured = {}

    def build(model_config):
        captured['model'] = model_config
        return torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr('mmpose.registry.MODELS.build', build)
    monkeypatch.setattr(
        'mmengine.registry.init_default_scope', lambda *_args, **_kwargs: None)
    config = Config(dict(
        default_scope='mmpose',
        model=dict(
            type='Fixture', init_cfg=dict(type='Pretrained', checkpoint='x'),
            backbone=dict(type='Backbone', pretrained='upstream.pth',
                          init_cfg=dict(type='Pretrained', checkpoint='y'),
                          stages=(dict(pretrained='nested.pth'),)))))

    _build_binary_model(
        config, {'weight': torch.ones(1, 1)}, torch.device('cpu'))

    assert captured['model']['init_cfg'] is None
    assert captured['model']['backbone']['pretrained'] is None
    assert captured['model']['backbone']['init_cfg'] is None
    assert captured['model']['backbone']['stages'][0]['pretrained'] is None


def test_stage_a_optimizer_uses_resolved_paper_config():
    import torch

    from mambapose_opt.binary_smoke import build_stage_a_optimizer

    model = torch.nn.Linear(2, 1)
    optimizer = build_stage_a_optimizer(
        model, {'optimizer': {'type': 'Adam', 'lr': 0.001}})

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == 0.001


def test_stage_a_artifact_contract_binds_full_model_training_evidence(
        tmp_path, monkeypatch):
    """A public smoke artifact must prove every Stage-A action and identity."""
    from mambapose_opt.binary_smoke import validate_binary_stage_a_artifact

    root = tmp_path / 'repo'
    artifact = root / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0/'
        'smoke-stage-a/smoke.json')
    exported = artifact.parent / 'round-trip.pth'
    exported.parent.mkdir(parents=True)
    exported.write_bytes(b'tensor-only-export')
    value = {
        'schema_version': 2,
        'artifact_kind': 'binary-qk-stage-a-full-model-smoke',
        'candidate_id': 'binary-qk-s-v1',
        'source': {
            'git_commit': 'a' * 40,
            'manifest_path': 'optimization/candidates.json',
            'manifest_sha256': '1' * 64,
            'config_path': 'configs/optimization/numeric/binary_qk.py',
            'config_sha256': 'b' * 64,
            'authority_path': 'optimization/coco_val2017_authority.json',
            'authority_sha256': '2' * 64,
        },
        'config': {
            'path': 'configs/optimization/numeric/binary_qk.py',
            'sha256': 'b' * 64,
            'config_closure': [{
                'path': 'configs/optimization/numeric/binary_qk.py',
                'sha256': 'b' * 64,
            }],
            'resolved_config_sha256': '3' * 64,
        },
        'checkpoint': {
            'path': ('work_dirs/reproduction/runs/coco-s-v1/'
                     'best_coco_AP_epoch_300.pth'),
            'sha256': 'c' * 64,
        },
        'pwl_stage_b': {
            'path': 'work_dirs/optimization/gates/pwl-stage-b.json',
            'sha256': 'd' * 64,
        },
        'data': {
            'dataset': 'coco', 'split': 'train2017', 'batch_size': 1,
            'packed_production_pipeline': True,
            'input_shape': [1, 3, 256, 192], 'sample_ids': [42],
        },
        'gpu': {
            'logical': 'cuda:0', 'physical_index': 0,
            'lease': {
                'stage_id': 'binary-smoke:binary-qk-s-v1',
                'pid': 123,
                'boot_id': '00000000-0000-0000-0000-000000000001',
                'timestamp': '2026-08-28T00:00:00+00:00',
                'device_index': 0, 'allowed_pids': [123],
                'lease_id': 'e' * 64,
            },
        },
        'execution': {
            'checks': {
                'forward': True, 'loss': True, 'backward': True,
                'optimizer_step': True, 'finite_loss': True,
                'finite_gradients': True, 'gradient_tensor_count': 17,
                'binary_target_count': 6,
                'state_round_trip_exact': True,
                'output_round_trip_exact': True,
            },
            'losses': {'loss_kpt': 0.125},
            'binary_targets': [
                {
                    'parameter': (
                        f'head.tokenpose.transformer.layers.{index}.'
                        '0.fn.fn.to_qkv.weight'),
                    'gradient_finite': True,
                    'q_gradient_norm': 1.0,
                    'k_gradient_norm': 1.0,
                    'q_parameter_changed': True,
                    'k_parameter_changed': True,
                    'optimizer': 'Adam',
                    'optimizer_step': 1,
                    'optimizer_state_finite': True,
                }
                for index in range(6)
            ],
            'output': {
                'shape': [1, 17, 64, 48], 'dtype': 'torch.float32',
                'sha256': 'f' * 64,
            },
            'operation': _valid_operation_manifest(),
            'export': {
                'path': exported.relative_to(root).as_posix(),
                'sha256': _sha256(exported), 'bytes': exported.stat().st_size,
                'format': 'torch-weights-only-state-dict-v1',
            },
        },
    }
    _write_json(artifact, value)
    authority_calls = []
    monkeypatch.setattr(
        'mambapose_opt.binary_smoke._validate_smoke_authority',
        lambda artifact, **kwargs: authority_calls.append(artifact))

    validated = validate_binary_stage_a_artifact(
        artifact.relative_to(root), repository_root=root)

    assert validated['candidate_id'] == 'binary-qk-s-v1'
    assert validated['execution']['checks']['optimizer_step'] is True
    assert authority_calls == [value]


def test_stage_a_config_authority_binds_inherited_closure_and_resolved_hash(
        tmp_path):
    from mambapose_opt.binary_smoke import _stage_a_config_authority
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    config = root / 'configs/binary.py'
    inherited = root / 'configs/base.py'
    inherited.parent.mkdir(parents=True)
    inherited.write_text('model = dict(type="Canonical")\n', encoding='utf-8')
    config.write_text(
        "_base_ = ['./base.py']\nqk_mode = 'binary'\n", encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.invalid'],
        cwd=root, check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=root, check=True)
    subprocess.run(['git', 'add', 'configs'], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    candidate = CandidateSpec.from_dict({
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'binary-qk', 'conditional': True},
    })

    authority = _stage_a_config_authority(
        root, candidate=candidate, git_commit=commit)

    assert authority['config_closure'] == [
        {'path': 'configs/base.py', 'sha256': _sha256(inherited)},
        {'path': 'configs/binary.py', 'sha256': _sha256(config)},
    ]
    assert len(authority['resolved_config_sha256']) == 64
    inherited.write_text('model = dict(type="Alternate")\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'configs/base.py'], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-qm', 'change base'], cwd=root, check=True)

    with pytest.raises(ValueError, match='recorded commit|closure'):
        _stage_a_config_authority(root, candidate=candidate, git_commit=commit)


def test_stage_a_rejects_config_authority_before_checkpoint_load(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_smoke import run_binary_stage_a_smoke
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    root.mkdir()
    candidate = CandidateSpec.from_dict({
        'id': 'binary-qk-s-v1', 'route': 'ssm-quant-pwl',
        'kind': 'binary-qk', 'config': 'configs/binary.py',
        'checkpoint': 'weights/model.pth',
        'checkpoint_sha256': 'a' * 64, 'seed': 0,
        'features': {'numeric_kind': 'binary-qk', 'conditional': True},
    })
    authorized = SimpleNamespace(
        candidate=candidate, config_path=root / candidate.config,
        checkpoint_path=root / candidate.checkpoint,
        source={'git_commit': 'b' * 40})
    monkeypatch.setattr(
        'mambapose_opt.checkpoints.authorize_manifest_candidate',
        lambda *args, **kwargs: authorized)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.validate_binary_qk_admission',
        lambda *args, **kwargs: {'decision': 'passed'})
    monkeypatch.setattr(
        'mambapose_opt.binary_smoke._stage_a_config_authority',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError('inherited config closure mismatch')))
    checkpoint_called = []
    monkeypatch.setattr(
        'mambapose_opt.checkpoints.tensor_state',
        lambda *args, **kwargs: checkpoint_called.append(True))

    with pytest.raises(ValueError, match='inherited config closure'):
        run_binary_stage_a_smoke(
            repository_root=root, manifest_path=root / 'manifest.json',
            candidate_id=candidate.id,
            output_relative=(
                'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0/'
                'smoke-stage-a'),
            device_index=0)
    assert checkpoint_called == []


def test_binary_stage_a_smoke_cli_is_directly_executable():
    result = subprocess.run(
        [sys.executable, 'tools/optimization/smoke_binary_qk.py', '--help'],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False,
        env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    assert result.returncode == 0, result.stderr
    assert '--candidate' in result.stdout
    assert '--output-root' in result.stdout
    assert '--device-index' in result.stdout


def test_binary_smoke_api_rejects_dot_alias_before_any_model_or_gpu_work(
        tmp_path):
    from mambapose_opt.binary_smoke import run_binary_stage_a_smoke

    with pytest.raises(ValueError, match='smoke-stage-a directory'):
        run_binary_stage_a_smoke(
            repository_root=tmp_path, manifest_path=tmp_path / 'manifest.json',
            candidate_id='binary-qk-s-v1',
            output_relative=(
                'work_dirs/optimization/route/candidate/0/./smoke-stage-a'),
            device_index=0)


def test_binary_smoke_cli_parser_rejects_dot_alias():
    from tools.optimization.smoke_binary_qk import _output_root

    with pytest.raises(__import__('argparse').ArgumentTypeError):
        _output_root(
            'work_dirs/optimization/route/candidate/0/./smoke-stage-a')


def _pareto_fixture(
        tmp_path, monkeypatch, drops, *, baseline_id='full-s-v1'):
    from mmengine.config import Config
    from mambapose_opt.numeric_source import validate_numeric_config_closure

    roots = {'baseline': {}, 'candidate': {}}
    results = {}
    authority_path = tmp_path / 'optimization/coco_val2017_authority.json'
    formal_manifest = tmp_path / 'optimization/formal_stage_c.json'
    initialization = tmp_path / (
        'pretrained/vssm_tiny_0230_ckpt_epoch_262.pth')
    paired_base = tmp_path / 'configs/formal/paired_base.py'
    _write_json(authority_path, {'dataset': 'coco', 'split': 'val2017'})
    _write_json(formal_manifest, {
        'schema_version': 1,
        'experiment_id': 'mambapose-formal-stage-c',
    })
    initialization.parent.mkdir(parents=True)
    initialization.write_bytes(b'vmamba-t-imagenet-initialization')
    paired_base.parent.mkdir(parents=True)
    paired_base.write_text(
        'train_cfg = dict(max_epochs=300)\n', encoding='utf-8')

    for seed, drop in enumerate(drops):
        manifest_path = tmp_path / f'optimization/candidates-seed{seed}.json'
        candidate_rows = []
        for kind, candidate_id, ap in (
                ('baseline', baseline_id, 72.8),
                ('candidate', 'binary-qk-s-v1', 72.8 - drop)):
            relative = Path(
                f'work_dirs/optimization/{kind}/{candidate_id}/{seed}')
            artifact_root = tmp_path / relative
            evaluation = artifact_root / 'evaluate/evaluate.json'
            profile = artifact_root / 'profile/profile.json'
            latency = artifact_root / 'latency/latency.json'
            config = tmp_path / f'configs/formal/{kind}_seed{seed}.py'
            config.write_text(
                "_base_ = ['./paired_base.py']\n"
                f"formal_role = '{kind}'\nseed = {seed}\n",
                encoding='utf-8')
            checkpoint = artifact_root / 'train/best.pth'
            resume_299 = artifact_root / 'train/epoch_299.pth'
            resume_300 = artifact_root / 'train/epoch_300.pth'
            structured_log = artifact_root / 'train/train.jsonl'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f'{kind}-{seed}-best'.encode())
            resume_299.write_bytes(f'{kind}-{seed}-299'.encode())
            resume_300.write_bytes(f'{kind}-{seed}-300'.encode())
            structured_log.write_text(
                json.dumps({'epoch': 300}) + '\n', encoding='utf-8')
            for path, stage in (
                    (evaluation, 'evaluate'), (profile, 'profile'),
                    (latency, 'latency')):
                _write_json(path, {'stage': stage, 'seed': seed})
            roots[kind][seed] = relative.as_posix()
            candidate_kind = (
                'float' if kind == 'baseline' else 'binary-qk')
            route = (
                'baseline' if kind == 'baseline' else 'ssm-quant-pwl')
            checkpoint_relative = checkpoint.relative_to(tmp_path).as_posix()
            row = {
                'id': candidate_id, 'route': route, 'kind': candidate_kind,
                'config': config.relative_to(tmp_path).as_posix(),
                'checkpoint': checkpoint_relative,
                'checkpoint_sha256': _sha256(checkpoint), 'seed': seed,
                'features': {},
            }
            candidate_rows.append(row)
            source = {
                'git_commit': 'a' * 40,
                'manifest_path': manifest_path.relative_to(tmp_path).as_posix(),
                'manifest_sha256': '',
                'config_path': config.relative_to(tmp_path).as_posix(),
                'config_sha256': _sha256(config),
                'authority_path': authority_path.relative_to(tmp_path).as_posix(),
                'authority_sha256': _sha256(authority_path),
            }
            provenance = {
                'checkpoint_sha256': _sha256(checkpoint),
                'config_sha256': _sha256(config),
                'data_inventory_sha256': 'b' * 64,
                'git_commit': source['git_commit'],
            }
            mode_rows = {}
            for mode in ('flip', 'no_flip'):
                protocol = {
                    'dataset': 'coco', 'split': 'val2017',
                    'complete_split': True,
                    'authority_path': source['authority_path'],
                    'authority_sha256': source['authority_sha256'],
                    'data_inventory_sha256': 'b' * 64,
                    'detections_sha256': 'c' * 64,
                    'evaluator': 'mmpose.CocoMetric',
                    'tta': {'mode': mode, 'flip_test': mode == 'flip'},
                    'source_config': source['config_path'],
                    'checkpoint': checkpoint_relative,
                }
                determinism = {
                    'python_seed': seed, 'numpy_seed': seed,
                    'torch_seed': seed, 'worker_count': 2,
                    'persistent_workers': False,
                    'order_hashes': {'0': f'{seed + 1:x}' * 64},
                    'provenance': provenance,
                }
                mode_rows[mode] = (protocol, determinism)
                results[(artifact_root.resolve(), mode)] = SimpleNamespace(
                    candidate_id=candidate_id,
                    candidate_kind=candidate_kind, route=route,
                    seed=seed, flip_test=mode == 'flip',
                    metrics=SimpleNamespace(ap=ap), source=source,
                    provenance=provenance, protocol=protocol,
                    determinism=determinism,
                    binary_operation=(
                        None if kind == 'baseline'
                        else _valid_operation_manifest()),
                    artifact_paths={
                        'evaluation': evaluation.absolute(),
                        'profile': profile.absolute(),
                        'latency': latency.absolute(),
                    })
            formal_dir = artifact_root / 'formal'
            run_init = formal_dir / 'run-init.json'
            train_result = formal_dir / 'train-result.json'
            formal_authority = formal_dir / 'formal-authority.json'
            closure = list(validate_numeric_config_closure(
                tmp_path, config.relative_to(tmp_path)))
            base_closure = list(validate_numeric_config_closure(
                tmp_path, paired_base.relative_to(tmp_path)))
            resolved_sha = hashlib.sha256(
                Config.fromfile(config).dump().encode('utf-8')).hexdigest()
            run_init_value = {
                'schema_version': 1,
                'artifact_kind': 'mambapose-formal-stage-c-pareto-run-init',
                'candidate': {
                    'candidate_id': candidate_id,
                    'candidate_kind': candidate_kind,
                    'route': route, 'seed': seed,
                    'role': 'baseline' if kind == 'baseline' else 'candidate',
                },
                'source': {
                    **source,
                    'candidate_row_sha256': '',
                    'formal_manifest_path': formal_manifest.relative_to(
                        tmp_path).as_posix(),
                    'formal_manifest_sha256': _sha256(formal_manifest),
                },
                'initialization': {
                    'id': 'vmamba-t-imagenet-262',
                    'path': initialization.relative_to(tmp_path).as_posix(),
                    'sha256': _sha256(initialization),
                },
                'config': {
                    'path': source['config_path'],
                    'config_closure': closure,
                    'resolved_config_sha256': resolved_sha,
                    'paired_base_config_path': paired_base.relative_to(
                        tmp_path).as_posix(),
                    'paired_base_config_closure': base_closure,
                },
                'protocol': {
                    'epochs': 300, 'effective_batch_size': 128,
                    'per_device_batch_size': 128, 'world_size': 1,
                    'accumulation_steps': 1, 'worker_count': 2,
                    'persistent_workers': False, 'deterministic': True,
                    'environment_inventory_sha256': 'd' * 64,
                    'evaluator': 'mmpose.CocoMetric',
                    'tta_modes': {'flip': True, 'no_flip': False},
                    'data_authority': {
                        'authority_path': source['authority_path'],
                        'authority_sha256': source['authority_sha256'],
                        'data_inventory_sha256': 'b' * 64,
                        'detections_sha256': 'c' * 64,
                    },
                },
            }
            _write_json(run_init, run_init_value)
            train_value = {
                'schema_version': 1,
                'artifact_kind': 'mambapose-formal-stage-c-pareto-train-result',
                'candidate': dict(run_init_value['candidate']),
                'run_init': {
                    'path': run_init.relative_to(tmp_path).as_posix(),
                    'sha256': _sha256(run_init),
                },
                'status': 'complete', 'final_epoch': 300,
                'best_checkpoint': {
                    'path': checkpoint_relative,
                    'sha256': _sha256(checkpoint),
                },
                'resume_checkpoints': [
                    {'path': item.relative_to(tmp_path).as_posix(),
                     'sha256': _sha256(item)}
                    for item in (resume_299, resume_300)
                ],
                'structured_log': {
                    'path': structured_log.relative_to(tmp_path).as_posix(),
                    'sha256': _sha256(structured_log),
                },
                'order_hashes': [f'{index:064x}' for index in range(300)],
            }
            _write_json(train_result, train_value)
            flip_protocol, flip_determinism = mode_rows['flip']
            no_flip_protocol, no_flip_determinism = mode_rows['no_flip']
            authority_value = {
                'schema_version': 1,
                'artifact_kind': 'mambapose-formal-stage-c-pareto-authority',
                'candidate': dict(run_init_value['candidate']),
                'run_init': {
                    'path': run_init.relative_to(tmp_path).as_posix(),
                    'sha256': _sha256(run_init),
                },
                'train_result': {
                    'path': train_result.relative_to(tmp_path).as_posix(),
                    'sha256': _sha256(train_result),
                },
                'evaluation': {
                    'artifact': {
                        'path': evaluation.relative_to(tmp_path).as_posix(),
                        'sha256': _sha256(evaluation),
                    },
                    'evaluator': 'mmpose.CocoMetric',
                    'modes': {
                        'flip': {
                            'protocol_sha256': _json_sha256(flip_protocol),
                            'determinism_sha256': _json_sha256(
                                flip_determinism),
                        },
                        'no_flip': {
                            'protocol_sha256': _json_sha256(no_flip_protocol),
                            'determinism_sha256': _json_sha256(
                                no_flip_determinism),
                        },
                    },
                },
            }
            _write_json(formal_authority, authority_value)
            for mode in ('flip', 'no_flip'):
                results[(artifact_root.resolve(), mode)].artifact_paths[
                    'formal_authority'] = formal_authority.absolute()

        _write_json(manifest_path, {
            'schema_version': 1, 'candidates': candidate_rows})
        manifest_sha = _sha256(manifest_path)
        for kind in ('baseline', 'candidate'):
            artifact_root = tmp_path / roots[kind][seed]
            result = results[(artifact_root.resolve(), 'flip')]
            for mode in ('flip', 'no_flip'):
                results[(artifact_root.resolve(), mode)].source[
                    'manifest_sha256'] = manifest_sha
            run_init = artifact_root / 'formal/run-init.json'
            run_value = json.loads(run_init.read_text(encoding='utf-8'))
            run_value['source']['manifest_sha256'] = manifest_sha
            row = next(
                item for item in candidate_rows
                if item['id'] == result.candidate_id)
            run_value['source']['candidate_row_sha256'] = _json_sha256(row)
            _write_json(run_init, run_value)
            train_result = artifact_root / 'formal/train-result.json'
            train_value = json.loads(train_result.read_text(encoding='utf-8'))
            train_value['run_init']['sha256'] = _sha256(run_init)
            _write_json(train_result, train_value)
            formal_authority = artifact_root / 'formal/formal-authority.json'
            authority_value = json.loads(
                formal_authority.read_text(encoding='utf-8'))
            authority_value['run_init']['sha256'] = _sha256(run_init)
            authority_value['train_result']['sha256'] = _sha256(train_result)
            _write_json(formal_authority, authority_value)

    def load_result(root, *, mode='flip'):
        return results[(Path(root).resolve(), mode)]

    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.invalid'],
        cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'add', 'configs', 'optimization', 'pretrained'],
        cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '-qm', 'formal source fixture'],
        cwd=tmp_path, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()
    for kind in ('baseline', 'candidate'):
        for seed, relative in roots[kind].items():
            artifact_root = tmp_path / relative
            for mode in ('flip', 'no_flip'):
                result = results[(artifact_root.resolve(), mode)]
                result.source['git_commit'] = commit
                result.provenance['git_commit'] = commit
            run_init = artifact_root / 'formal/run-init.json'
            run_value = json.loads(run_init.read_text(encoding='utf-8'))
            run_value['source']['git_commit'] = commit
            _write_json(run_init, run_value)
            formal_authority = artifact_root / 'formal/formal-authority.json'
            authority_value = json.loads(
                formal_authority.read_text(encoding='utf-8'))
            authority_value['evaluation']['modes'] = {
                mode: {
                    'protocol_sha256': _json_sha256(
                        results[(artifact_root.resolve(), mode)].protocol),
                    'determinism_sha256': _json_sha256(
                        results[(artifact_root.resolve(), mode)].determinism),
                }
                for mode in ('flip', 'no_flip')
            }
            _write_json(formal_authority, authority_value)
            _rebind_formal_fixture(artifact_root)

    monkeypatch.setattr(
        'mambapose_opt.pareto.CandidateResult.from_artifacts', load_result)
    roots['results'] = results
    return roots


def _rebind_formal_fixture(artifact_root: Path) -> None:
    run_init = artifact_root / 'formal/run-init.json'
    train_result = artifact_root / 'formal/train-result.json'
    formal_authority = artifact_root / 'formal/formal-authority.json'
    train_value = json.loads(train_result.read_text(encoding='utf-8'))
    train_value['run_init']['sha256'] = _sha256(run_init)
    _write_json(train_result, train_value)
    authority_value = json.loads(formal_authority.read_text(encoding='utf-8'))
    authority_value['run_init']['sha256'] = _sha256(run_init)
    authority_value['train_result']['sha256'] = _sha256(train_result)
    _write_json(formal_authority, authority_value)


def test_final_pareto_rules_include_binary_without_speedup_overclaim(
        tmp_path, monkeypatch):
    """Binary Q/K may enter Pareto only through public artifact evidence."""
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', repository_root=tmp_path,
        baseline_roots=roots['baseline'], candidate_roots=roots['candidate'])

    assert record['schema_version'] == 3
    assert record['decision'] == 'pareto-eligible'
    assert record['statistics']['paired_sample_stddev_points'] == pytest.approx(
        0.01)
    assert record['statistics']['paired_95_ci_points'][1] < 0.1
    assert record['hardware_evidence']['speedup_claim'] == 'none-software-proxy'
    assert all(
        len(row['paired_formal_authority_sha256']) == 64
        for row in record['seeds'])


@pytest.mark.parametrize(
    ('field_path', 'replacement'),
    [
        (('source', 'manifest_sha256'), '9' * 64),
        (('initialization', 'id'), 'alternate-initialization'),
        (('config', 'resolved_config_sha256'), '9' * 64),
        (('protocol', 'epochs'), 299),
        (('protocol', 'effective_batch_size'), 64),
        (('protocol', 'worker_count'), 3),
        (('protocol', 'persistent_workers'), True),
        (('protocol', 'deterministic'), False),
        (('protocol', 'environment_inventory_sha256'), '9' * 64),
        (('protocol', 'evaluator'), 'alternate.Evaluator'),
        (('protocol', 'tta_modes'), {'flip': False, 'no_flip': False}),
        (('protocol', 'data_authority', 'detections_sha256'), '9' * 64),
    ],
)
def test_final_pareto_rejects_mismatched_formal_stage_c_run_authority(
        tmp_path, monkeypatch, field_path, replacement):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    candidate_root = tmp_path / roots['candidate'][0]
    run_init = candidate_root / 'formal/run-init.json'
    value = json.loads(run_init.read_text(encoding='utf-8'))
    target = value
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    _write_json(run_init, value)
    _rebind_formal_fixture(candidate_root)

    with pytest.raises(ValueError, match='formal|paired'):
        build_final_pareto_record(
            candidate_id='binary-qk-s-v1', repository_root=tmp_path,
            baseline_roots=roots['baseline'],
            candidate_roots=roots['candidate'])


def test_final_pareto_rejects_mismatched_formal_training_order_hash(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    candidate_root = tmp_path / roots['candidate'][0]
    train_result = candidate_root / 'formal/train-result.json'
    value = json.loads(train_result.read_text(encoding='utf-8'))
    value['order_hashes'][42] = '9' * 64
    _write_json(train_result, value)
    _rebind_formal_fixture(candidate_root)

    with pytest.raises(ValueError, match='formal paired'):
        build_final_pareto_record(
            candidate_id='binary-qk-s-v1', repository_root=tmp_path,
            baseline_roots=roots['baseline'],
            candidate_roots=roots['candidate'])


def test_final_pareto_rejects_symlinked_candidate_result_child(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    baseline_root = tmp_path / roots['baseline'][0]
    evaluation = baseline_root / 'evaluate/evaluate.json'
    real = baseline_root / 'evaluate/real-evaluate.json'
    evaluation.rename(real)
    evaluation.symlink_to(real)

    with pytest.raises(ValueError, match='symlink'):
        build_final_pareto_record(
            candidate_id='binary-qk-s-v1', repository_root=tmp_path,
            baseline_roots=roots['baseline'],
            candidate_roots=roots['candidate'])


def test_final_pareto_three_seed_ci_intersection_requires_seeds_three_and_four(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.0, 0.05, 0.2])
    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', repository_root=tmp_path,
        baseline_roots=roots['baseline'], candidate_roots=roots['candidate'])

    assert record['statistics']['paired_95_ci_points'][0] <= 0.1
    assert record['statistics']['paired_95_ci_points'][1] >= 0.1
    assert record['decision'] == 'requires-seeds-3-4'
    assert record['accuracy_gate']['passed'] is False


def test_final_pareto_five_seed_recompute_uses_same_point_rules(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(
        tmp_path, monkeypatch, [0.0, 0.05, 0.2, 0.03, 0.04])
    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', repository_root=tmp_path,
        baseline_roots=roots['baseline'], candidate_roots=roots['candidate'])

    assert record['statistics']['seed_count'] == 5
    assert record['accuracy_gate']['mean_ap_drop_points'] < 0.1
    assert record['accuracy_gate']['max_ap_drop_points'] < 0.3
    assert record['decision'] == 'pareto-eligible'


def test_final_pareto_reloads_public_evidence_and_rejects_tampering(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import (
        build_final_pareto_record, validate_final_pareto_record)

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', repository_root=tmp_path,
        baseline_roots=roots['baseline'], candidate_roots=roots['candidate'])
    record['hardware_evidence']['operation_manifest_sha256'] = '9' * 64

    with pytest.raises(ValueError, match='recomputed public evidence'):
        validate_final_pareto_record(record, repository_root=tmp_path)


def test_final_pareto_api_rejects_caller_reported_ap_and_operation():
    from mambapose_opt.pareto import build_final_pareto_record

    with pytest.raises(TypeError):
        build_final_pareto_record(
            candidate_id='binary-qk-s-v1', repository_root=Path('.'),
            baseline_ap={0: 72.8}, candidate_ap={0: 72.7},
            operation_manifests={0: _valid_operation_manifest()})


def test_final_pareto_rejects_an_alternate_float_baseline(
        tmp_path, monkeypatch):
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(
        tmp_path, monkeypatch, [0.05, 0.06, 0.04],
        baseline_id='alternate-float')

    with pytest.raises(ValueError, match='baseline CandidateResult identity'):
        build_final_pareto_record(
            candidate_id='binary-qk-s-v1', repository_root=tmp_path,
            baseline_roots=roots['baseline'],
            candidate_roots=roots['candidate'])


def _rebind_gate(binary, gate_path: Path, root: Path):
    return replace(binary, features=MappingProxyType({
        **binary.features,
        'pwl_stage_b_artifact': gate_path.relative_to(root).as_posix(),
        'pwl_stage_b_sha256': _sha256(gate_path),
    }))


@pytest.mark.parametrize(
    ('section', 'field', 'replacement'),
    [
        ('source', 'git_commit', '9' * 40),
        ('source', 'manifest_sha256', '9' * 64),
        ('source', 'authority_sha256', '9' * 64),
        ('provenance', 'checkpoint_sha256', '9' * 64),
        ('protocol', 'evaluator', 'alternate.Evaluator'),
        ('determinism', 'order_hashes', {'0': '9' * 64}),
    ],
)
def test_binary_admission_recomputes_full_baseline_result_authority(
        tmp_path, monkeypatch, section, field, replacement):
    """No individually valid but stale baseline may control the PWL AP drop."""
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    baseline_root = next(
        key for key in load_result.results
        if 'full-s-v1' in key[0].as_posix() and key[1] == 'flip')
    getattr(load_result.results[baseline_root], section)[field] = replacement
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    with pytest.raises(ValueError, match='baseline|paired'):
        validate_binary_qk_admission(
            binary, repository_root=root, manifest_path=manifest)


def test_binary_admission_rejects_different_baseline_roots_across_modes(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    original_root = root / gate['modes']['flip']['baseline_root']
    alternate_root = original_root.with_name('0-alternate')
    evaluation = alternate_root / 'evaluate/evaluate.json'
    _write_json(evaluation, {'fixture': 'alternate-baseline'})
    original_result = load_result.results[(original_root.resolve(), 'no_flip')]
    load_result.results[(alternate_root.resolve(), 'no_flip')] = SimpleNamespace(
        **vars(original_result))
    gate['modes']['no_flip']['baseline_root'] = (
        alternate_root.relative_to(root).as_posix())
    gate['modes']['no_flip']['baseline_evaluation_sha256'] = _sha256(evaluation)
    gate_path = root / binary.features['pwl_stage_b_artifact']
    _write_json(gate_path, gate)
    claimed = _rebind_gate(binary, gate_path, root)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    with pytest.raises(ValueError, match='same artifact root'):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest)


def test_binary_admission_rejects_symlinked_manifest_alias(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    alias = manifest.with_name('manifest-alias.json')
    alias.symlink_to(manifest)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    with pytest.raises(ValueError, match='manifest.*symlink'):
        validate_binary_qk_admission(
            binary, repository_root=root, manifest_path=alias)


def test_binary_admission_rejects_relative_path_escape(tmp_path):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    escaped = replace(binary, features=MappingProxyType({
        **binary.features,
        'pwl_stage_b_artifact': '../pwl-stage-b.json',
    }))

    with pytest.raises(ValueError, match='safe relative path'):
        validate_binary_qk_admission(
            escaped, repository_root=root, manifest_path=manifest)


def test_binary_admission_rejects_symlinked_dependency(tmp_path):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    original = root / binary.features['pwl_stage_b_artifact']
    alias = original.with_name('pwl-stage-b-alias.json')
    alias.symlink_to(original)
    claimed = replace(binary, features=MappingProxyType({
        **binary.features,
        'pwl_stage_b_artifact': alias.relative_to(root).as_posix(),
        'pwl_stage_b_sha256': _sha256(original),
    }))

    with pytest.raises(ValueError, match='symlink'):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda gate: gate['pwl_policy'].__setitem__('function', 'silu'),
         'policy differs'),
        (lambda gate: gate.__setitem__('ap_drop_limit_points', 0.31),
         '0.3-point screen'),
    ],
)
def test_binary_admission_rejects_alternate_policy_or_downgraded_gate(
        tmp_path, monkeypatch, mutation, message):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)
    mutation(gate)
    gate_path = root / binary.features['pwl_stage_b_artifact']
    _write_json(gate_path, gate)
    claimed = _rebind_gate(binary, gate_path, root)

    with pytest.raises(ValueError, match=message):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest)


def test_binary_admission_rejects_candidate_result_from_alternate_policy(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)
    for (result_root, _), result in load_result.results.items():
        if 'pwl-gelu-s-v1' in result_root.as_posix():
            result.profile['parent']['config'] = 'configs/pwl-alternate.py'

    with pytest.raises(ValueError, match='alternate PWL policy'):
        validate_binary_qk_admission(
            binary, repository_root=root, manifest_path=manifest)


@pytest.mark.parametrize(
    ('field_path', 'replacement'),
    [
        (('candidate_row_sha256',), '9' * 64),
        (('git_commit',), '9' * 40),
        (('manifest_sha256',), '9' * 64),
        (('config_sha256',), '9' * 64),
        (('checkpoint_sha256',), '9' * 64),
        (('authority_sha256',), '9' * 64),
        (('seed',), 1),
        (('config_closure',), [{
            'path': 'configs/pwl.py', 'sha256': '9' * 64}]),
        (('calibration', 'schema_version'), 2),
        (('calibration', 'sha256'), '9' * 64),
        (('calibration', 'sample_order_sha256'), '9' * 64),
        (('selection_policy',), 'alternate-selection'),
        (('installation', 'sha256'), '9' * 64),
        (('operation_manifest_sha256',), '9' * 64),
    ],
)
def test_binary_admission_rejects_stale_or_alternate_pwl_authority(
        tmp_path, monkeypatch, field_path, replacement):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    target = gate['pwl_authority']
    for name in field_path[:-1]:
        target = target[name]
    target[field_path[-1]] = replacement
    gate_path = root / binary.features['pwl_stage_b_artifact']
    _write_json(gate_path, gate)
    claimed = _rebind_gate(binary, gate_path, root)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)

    with pytest.raises(ValueError, match='PWL authority'):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest)


def test_binary_admission_rejects_changed_inherited_pwl_config(
        tmp_path, monkeypatch):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    config = root / 'configs/pwl.py'
    inherited = root / 'configs/pwl_base.py'
    inherited.write_text("pwl_domain = 'canonical'\n", encoding='utf-8')
    config.write_text(
        "_base_ = ['./pwl_base.py']\npwl_function = 'gelu'\n",
        encoding='utf-8')
    config_hash = _sha256(config)
    authority = gate['pwl_authority']
    gate['pwl_policy']['sha256'] = config_hash
    authority['config_sha256'] = config_hash
    authority['policy_sha256'] = config_hash
    authority['config_closure'] = [
        {'path': 'configs/pwl.py', 'sha256': config_hash},
        {'path': 'configs/pwl_base.py', 'sha256': _sha256(inherited)},
    ]
    for (result_root, _mode), result in load_result.results.items():
        if 'pwl-gelu-s-v1' in result_root.as_posix():
            result.pwl_authority = json.loads(json.dumps(authority))
    gate_path = root / binary.features['pwl_stage_b_artifact']
    _write_json(gate_path, gate)
    claimed = _rebind_gate(binary, gate_path, root)
    monkeypatch.setattr(
        'mambapose_opt.binary_readiness.CandidateResult.from_artifacts',
        load_result)
    inherited.write_text("pwl_domain = 'alternate'\n", encoding='utf-8')

    with pytest.raises(ValueError, match='PWL authority config closure'):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest)


def test_binary_artifact_bundle_rejects_symlinked_profile(tmp_path):
    from mambapose_opt.binary_operation import validate_binary_artifact_bundle

    root = tmp_path / 'repo'
    candidate = root / 'work_dirs/optimization/route/candidate/0'
    real_profile = candidate / 'real/profile.json'
    profile = candidate / 'profile/profile.json'
    evaluation = candidate / 'evaluate/evaluate.json'
    latency = candidate / 'latency/latency.json'
    _write_json(real_profile, {
        'candidate': 'binary-qk-s-v1',
        'binary_qk_operation': _valid_operation_manifest(),
    })
    profile.parent.mkdir(parents=True)
    profile.symlink_to(real_profile)
    _write_json(evaluation, {})
    _write_json(latency, {})

    with pytest.raises(ValueError, match='symlink'):
        validate_binary_artifact_bundle(
            profile_path=profile.relative_to(root),
            evaluation_path=evaluation.relative_to(root),
            latency_path=latency.relative_to(root), repository_root=root,
            candidate_id='binary-qk-s-v1')


def test_binary_operation_rejects_noncanonical_s_v1_module_names():
    from mambapose_opt.binary_operation import (
        validate_binary_operation_manifest)

    operation = _valid_operation_manifest()
    operation['module_names'] = [
        f'forged.module.{index}' for index in range(6)]

    with pytest.raises(ValueError, match='module names'):
        validate_binary_operation_manifest(operation)


@pytest.mark.parametrize('alias_kind', ('absolute', 'parent', 'dot'))
def test_binary_stage_binding_rejects_lexical_path_aliases(
        tmp_path, alias_kind):
    from mambapose_opt.binary_operation import validate_binary_stage_binding

    root = tmp_path / 'repo'
    candidate = root / 'work_dirs/optimization/route/candidate/0'
    profile = candidate / 'profile/profile.json'
    evaluation = candidate / 'evaluate/evaluate.json'
    _write_json(profile, {
        'candidate': 'binary-qk-s-v1',
        'binary_qk_operation': _valid_operation_manifest(),
    })
    binding = {
        'path': profile.relative_to(root).as_posix(),
        'sha256': _sha256(profile),
        'operation_sha256': hashlib.sha256(json.dumps(
            _valid_operation_manifest(), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
    }
    _write_json(evaluation, {
        'candidate_id': 'binary-qk-s-v1', 'stage': 'evaluate',
        'result': {'binary_qk_profile': binding},
    })
    if alias_kind == 'absolute':
        supplied_profile = profile
        supplied_evaluation = evaluation
    elif alias_kind == 'parent':
        alias = candidate / 'alias'
        alias.mkdir()
        supplied_profile = (
            candidate / 'alias/../profile/profile.json').as_posix()
        supplied_evaluation = (
            candidate / 'alias/../evaluate/evaluate.json').as_posix()
    else:
        supplied_profile = (
            'work_dirs/optimization/route/candidate/0/./profile/profile.json')
        supplied_evaluation = (
            'work_dirs/optimization/route/candidate/0/./evaluate/evaluate.json')

    with pytest.raises(ValueError, match='safe repository-relative path'):
        validate_binary_stage_binding(
            profile_path=supplied_profile, stage_path=supplied_evaluation,
            repository_root=root, candidate_id='binary-qk-s-v1',
            stage='evaluate')


def test_binary_profile_binding_syntax_rejects_dot_alias():
    from mambapose_opt.binary_operation import (
        validate_binary_profile_binding)

    binding = {
        'path': (
            'work_dirs/optimization/route/candidate/0/./profile/profile.json'),
        'sha256': '1' * 64,
        'operation_sha256': '2' * 64,
    }

    with pytest.raises(ValueError, match='profile binding path'):
        validate_binary_profile_binding(binding)


def test_binary_config_bounds_recovery_to_post_stage_b_self_distillation():
    from mmengine.config import Config

    config = Config.fromfile(
        REPOSITORY_ROOT / 'configs/optimization/numeric/binary_qk.py')

    envelope = config.numeric_optimization.train_envelope
    assert envelope.recovery == (
        'one-bounded-qat-self-distillation-after-stage-b-only')
    assert envelope.requires_stage_b_pass is True
    assert envelope.requires_attributed_error is True
