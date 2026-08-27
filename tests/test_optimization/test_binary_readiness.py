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
    gate = {
        'schema_version': 1,
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
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=72.8),
            profile={'parent': {'config': baseline.config.as_posix()}})
        results[(roots['candidate'].resolve(), mode)] = SimpleNamespace(
            candidate_id=pwl.id, route=pwl.route,
            flip_test=mode == 'flip', metrics=SimpleNamespace(ap=candidate_ap),
            profile={'parent': {'config': pwl.config.as_posix()}})

    def load_result(result_root: Path, *, mode: str):
        return results[(Path(result_root).resolve(), mode)]

    load_result.results = results

    return root, manifest, binary, gate, load_result


def test_binary_admission_validates_public_stage_b_pwl_results(tmp_path):
    """Both public CandidateResult modes and the 0.3 AP screen are mandatory."""
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)

    validated = validate_binary_qk_admission(
        binary, repository_root=root, manifest_path=manifest,
        candidate_result_loader=load_result)

    assert validated['pwl_candidate_id'] == 'pwl-gelu-s-v1'
    assert validated['modes']['flip']['ap_drop_points'] == pytest.approx(0.2)
    assert dict(validated) == gate


def test_campaign_selection_admits_only_validated_stage_b_dependency(tmp_path):
    """The production selector must call the public dependency validator."""
    from tools.optimization.run_campaign import _select

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)

    selected = _select(
        (binary,), (binary.id,), admit_conditional=True,
        repository_root=root, manifest_path=manifest,
        candidate_result_loader=load_result)

    assert selected == (binary,)


def _binary_model_fixture():
    from torch import nn
    from mmpose.models.heads.heatmap_heads.tokenbase import Attention

    class TokenPoseFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_patches = 48
            self.num_keypoints = 17
            self.attention = nn.ModuleList([
                Attention(
                    256, heads=8, num_keypoints=17,
                    scale_with_head=True, qk_mode='binary')
                for _ in range(6)
            ])

    class ModelFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Module()
            self.head.tokenpose = TokenPoseFixture()

        def forward(self, inputs, **unused):
            return inputs[:, :1]

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
        'qk_mode': 'binary',
        'layers': 6,
        'module_names': [
            f'head.tokenpose.attention.{index}' for index in range(6)],
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
        'qk_mode': 'binary',
        'layers': 6,
        'module_names': [f'head.attention.{index}' for index in range(6)],
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


def test_evaluate_and_latency_bind_the_same_profile_operation_manifest(tmp_path):
    """No later result may detach latency or AP from the profiled operation."""
    from mambapose_opt.binary_operation import (
        binary_profile_binding_for_stage, validate_binary_artifact_bundle)

    root = tmp_path / 'repo'
    candidate_root = root / (
        'work_dirs/optimization/ssm-quant-pwl/binary-qk-s-v1/0')
    profile_path = candidate_root / 'profile/profile.json'
    evaluate_path = candidate_root / 'evaluate/evaluate.json'
    latency_path = candidate_root / 'latency/latency.json'
    _write_json(profile_path, {
        'candidate': 'binary-qk-s-v1',
        'binary_qk_operation': _valid_operation_manifest(),
    })

    evaluation_binding = binary_profile_binding_for_stage(
        evaluate_path, repository_root=root)
    latency_binding = binary_profile_binding_for_stage(
        latency_path, repository_root=root)
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
        profile_path=profile_path, evaluation_path=evaluate_path,
        latency_path=latency_path, repository_root=root,
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

    assert result['binary_qk_operation'] == _valid_operation_manifest() | {
        'module_names': [
            f'head.tokenpose.attention.{index}' for index in range(6)]}


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
        operation_builder=lambda unused: _valid_operation_manifest())

    assert result['checks'] == {
        'forward': True,
        'loss': True,
        'backward': True,
        'optimizer_step': True,
        'finite_loss': True,
        'finite_gradients': True,
        'gradient_tensor_count': 1,
        'state_round_trip_exact': True,
        'output_round_trip_exact': True,
    }
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


def test_stage_a_optimizer_uses_resolved_paper_config():
    import torch

    from mambapose_opt.binary_smoke import build_stage_a_optimizer

    model = torch.nn.Linear(2, 1)
    optimizer = build_stage_a_optimizer(
        model, {'optimizer': {'type': 'Adam', 'lr': 0.001}})

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == 0.001


def test_stage_a_artifact_contract_binds_full_model_training_evidence(tmp_path):
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
        'source': {'git_commit': 'a' * 40},
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
                'state_round_trip_exact': True,
                'output_round_trip_exact': True,
            },
            'losses': {'loss_kpt': 0.125},
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

    validated = validate_binary_stage_a_artifact(
        artifact, repository_root=root)

    assert validated['candidate_id'] == 'binary-qk-s-v1'
    assert validated['execution']['checks']['optimizer_step'] is True


def test_binary_stage_a_smoke_cli_is_directly_executable():
    result = subprocess.run(
        [sys.executable, 'tools/optimization/smoke_binary_qk.py', '--help'],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False,
        env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    assert result.returncode == 0, result.stderr
    assert '--candidate' in result.stdout
    assert '--output-root' in result.stdout
    assert '--device-index' in result.stdout


def test_final_pareto_rules_include_binary_without_speedup_overclaim():
    """Binary Q/K may enter Pareto only through exact operation/AP evidence."""
    from mambapose_opt.pareto import build_final_pareto_record

    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', candidate_kind='binary-qk',
        baseline_ap={0: 72.80, 1: 72.82, 2: 72.79},
        candidate_ap={0: 72.75, 1: 72.74, 2: 72.73},
        operation_manifests={
            seed: _valid_operation_manifest() for seed in (0, 1, 2)})

    assert record['decision'] == 'pareto-eligible'
    assert record['accuracy_gate'] == {
        'mean_ap_drop_points': pytest.approx(0.06333333333333334),
        'max_ap_drop_points': pytest.approx(0.08),
        'mean_limit_exclusive': 0.1,
        'max_limit_exclusive': 0.3,
        'passed': True,
    }


def test_final_pareto_rejects_tampered_accuracy_summary():
    from mambapose_opt.pareto import (
        build_final_pareto_record, validate_final_pareto_record)

    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', candidate_kind='binary-qk',
        baseline_ap={0: 72.80, 1: 72.82, 2: 72.79},
        candidate_ap={0: 72.75, 1: 72.74, 2: 72.73},
        operation_manifests={
            seed: _valid_operation_manifest() for seed in (0, 1, 2)})
    record['accuracy_gate']['mean_ap_drop_points'] = 0.09

    with pytest.raises(ValueError, match='summary disagrees'):
        validate_final_pareto_record(record)
    assert record['hardware_evidence'] == {
        'kind': 'binary-qk-theoretical-operation-replacement',
        'theoretical_qk_multiplications_replaced': 6_489_600,
        'operation_manifest_sha256': hashlib.sha256(json.dumps(
            _valid_operation_manifest(), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        'bitwise_kernel_present': False,
        'measured_integer_latency': False,
        'speedup_claim': 'none-software-proxy',
        'passed': True,
    }


def test_final_pareto_rejects_malformed_operation_identity():
    from mambapose_opt.pareto import (
        build_final_pareto_record, validate_final_pareto_record)

    record = build_final_pareto_record(
        candidate_id='binary-qk-s-v1', candidate_kind='binary-qk',
        baseline_ap={0: 72.80, 1: 72.82, 2: 72.79},
        candidate_ap={0: 72.75, 1: 72.74, 2: 72.73},
        operation_manifests={
            seed: _valid_operation_manifest() for seed in (0, 1, 2)})
    record['hardware_evidence']['operation_manifest_sha256'] = 'bad'

    with pytest.raises(ValueError, match='hardware evidence'):
        validate_final_pareto_record(record)


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
            escaped, repository_root=root, manifest_path=manifest,
            candidate_result_loader=load_result)


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
            claimed, repository_root=root, manifest_path=manifest,
            candidate_result_loader=load_result)


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
        tmp_path, mutation, message):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, gate, load_result = _stage_b_fixture(tmp_path)
    mutation(gate)
    gate_path = root / binary.features['pwl_stage_b_artifact']
    _write_json(gate_path, gate)
    claimed = _rebind_gate(binary, gate_path, root)

    with pytest.raises(ValueError, match=message):
        validate_binary_qk_admission(
            claimed, repository_root=root, manifest_path=manifest,
            candidate_result_loader=load_result)


def test_binary_admission_rejects_candidate_result_from_alternate_policy(
        tmp_path):
    from mambapose_opt.binary_readiness import validate_binary_qk_admission

    root, manifest, binary, _, load_result = _stage_b_fixture(tmp_path)
    for (result_root, _), result in load_result.results.items():
        if 'pwl-gelu-s-v1' in result_root.as_posix():
            result.profile['parent']['config'] = 'configs/pwl-alternate.py'

    with pytest.raises(ValueError, match='alternate PWL policy'):
        validate_binary_qk_admission(
            binary, repository_root=root, manifest_path=manifest,
            candidate_result_loader=load_result)


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
            profile_path=profile, evaluation_path=evaluation,
            latency_path=latency, repository_root=root,
            candidate_id='binary-qk-s-v1')
