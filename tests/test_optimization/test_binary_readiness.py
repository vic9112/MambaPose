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
    gate = {
        'schema_version': 2,
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
        results[(roots['baseline'].resolve(), mode)] = SimpleNamespace(
            candidate_id=baseline.id, route=baseline.route,
            candidate_kind=baseline.kind, seed=baseline.seed,
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=72.8),
            profile={'parent': {'config': baseline.config.as_posix()}},
            pwl_authority=None)
        results[(roots['candidate'].resolve(), mode)] = SimpleNamespace(
            candidate_id=pwl.id, route=pwl.route,
            candidate_kind=pwl.kind, seed=pwl.seed,
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=candidate_ap),
            profile={'parent': {'config': pwl.config.as_posix()}},
            pwl_authority=json.loads(json.dumps(pwl_authority)))

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
        'schema_version': 1,
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
        'config': {'path': 'configs/optimization/numeric/binary_qk.py',
                   'sha256': 'b' * 64},
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
    roots = {'baseline': {}, 'candidate': {}}
    results = {}
    for seed, drop in enumerate(drops):
        for kind, candidate_id, ap in (
                ('baseline', baseline_id, 72.8),
                ('candidate', 'binary-qk-s-v1', 72.8 - drop)):
            relative = Path(
                f'work_dirs/optimization/{kind}/{candidate_id}/{seed}')
            artifact_root = tmp_path / relative
            evaluation = artifact_root / 'evaluate/evaluate.json'
            profile = artifact_root / 'profile/profile.json'
            latency = artifact_root / 'latency/latency.json'
            for path, stage in (
                    (evaluation, 'evaluate'), (profile, 'profile'),
                    (latency, 'latency')):
                _write_json(path, {'stage': stage, 'seed': seed})
            roots[kind][seed] = relative.as_posix()
            results[(artifact_root.resolve(), 'flip')] = SimpleNamespace(
                candidate_id=candidate_id,
                candidate_kind=('float' if kind == 'baseline' else 'binary-qk'),
                route=('baseline' if kind == 'baseline' else 'ssm-quant-pwl'),
                seed=seed, flip_test=True, metrics=SimpleNamespace(ap=ap),
                source={'git_commit': 'a' * 40, 'authority_sha256': 'b' * 64},
                binary_operation=(
                    None if kind == 'baseline' else _valid_operation_manifest()),
                artifact_paths={
                    'evaluation': evaluation.resolve(),
                    'profile': profile.resolve(),
                    'latency': latency.resolve(),
                })

    def load_result(root, *, mode='flip'):
        return results[(Path(root).resolve(), mode)]

    monkeypatch.setattr(
        'mambapose_opt.pareto.CandidateResult.from_artifacts', load_result)
    return roots


def test_final_pareto_rules_include_binary_without_speedup_overclaim(
        tmp_path, monkeypatch):
    """Binary Q/K may enter Pareto only through public artifact evidence."""
    from mambapose_opt.pareto import build_final_pareto_record

    roots = _pareto_fixture(tmp_path, monkeypatch, [0.05, 0.06, 0.04])
    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', repository_root=tmp_path,
        baseline_roots=roots['baseline'], candidate_roots=roots['candidate'])

    assert record['decision'] == 'pareto-eligible'
    assert record['statistics']['paired_sample_stddev_points'] == pytest.approx(
        0.01)
    assert record['statistics']['paired_95_ci_points'][1] < 0.1
    assert record['hardware_evidence']['speedup_claim'] == 'none-software-proxy'


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
