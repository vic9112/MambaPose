from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMBINED_ID = 'no-pif-pwl-softplus-s-v1'
EXPECTED_ROLES = (
    'backbone.layers.0.blocks.0.op',
    'backbone.layers.1.blocks.0.op',
    'backbone.layers.2.blocks.0.op',
    'backbone.layers.2.blocks.1.op',
    'backbone.layers.3.blocks.0.op',
)


def _combined_candidate():
    from mambapose_opt.schema import load_candidate_manifest

    matches = tuple(
        item for item in load_candidate_manifest(
            REPOSITORY_ROOT / 'optimization/candidates.json')
        if item.id == COMBINED_ID)
    assert len(matches) == 1
    return matches[0]


def test_combined_candidate_declares_non_additive_parent_authority_and_plan():
    from mambapose_opt.combined_candidate import (
        load_combined_parent_authority)
    from mambapose_opt.numeric_conversion import numeric_stage_plan

    candidate = _combined_candidate()
    assert candidate.route == 'ssm-quant-pwl'
    assert candidate.checkpoint_sha256 == (
        '28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb')
    assert candidate.features == {
        'numeric_kind': 'pwl-combined',
        'pwl_function': 'softplus',
        'pif_mode': 'disabled',
        'calibration_source_candidate': 'no-pif-s-v1',
        'combined_parent_authority': (
            'optimization/no_pif_softplus_parent_authority.json'),
        'auto_run': False,
        'conditional': True,
        'prune_disabled_pif': True,
    }
    assert numeric_stage_plan('pwl-combined', conditional=True) == (
        'calibrate', 'convert', 'smoke-stage-a', 'profile', 'evaluate',
        'compare', 'latency')

    authority = load_combined_parent_authority(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')
    assert authority['candidate_id'] == COMBINED_ID
    assert authority['parents']['no_pif']['source_commit'] == (
        'd938530bad75fa6a8ac515b2c313792d8150da49')
    assert authority['parents']['tail_aware_softplus_pwl']['source_commit'] == (
        'f71fcff33bebe32241d0825eb2b736605a926183')
    assert authority['comparator']['candidate_id'] == 'full-s-v1'
    assert authority['claim_limits'] == {
        'combined_accuracy_measured': False,
        'fpga_speedup_claimed': False,
        'hardware_latency_claimed': False,
        'isolated_deltas_additive': False,
        'interaction_effect_measured': False,
    }


def test_combined_config_disables_pif_and_replaces_only_five_softplus_roles():
    candidate = _combined_candidate()
    config = Config.fromfile(REPOSITORY_ROOT / candidate.config)
    assert config.model.head.tokenpose_cfg.pif_mode == 'disabled'
    policy = config.numeric_optimization.pwl
    assert policy.candidate_id == COMBINED_ID
    assert policy.enabled_function == 'softplus'
    assert policy.source == 'ss2d-transition'
    assert tuple(policy.roles) == EXPECTED_ROLES
    assert policy.saturation == 'continuous-asymptotic-tail-v1'
    assert policy.qat_form == 'differentiable'
    assert tuple(config.numeric_optimization.stage_order) == (
        'calibrate', 'convert', 'smoke-stage-a', 'profile', 'evaluate',
        'compare', 'latency')
    assert config.numeric_optimization.comparator.candidate_id == 'full-s-v1'
    assert config.numeric_optimization.claim_limits.isolated_deltas_additive \
        is False


def test_combined_parent_authority_rejects_parent_or_claim_tampering():
    from mambapose_opt.combined_candidate import (
        CombinedCandidateAuthorityError, load_combined_parent_authority,
        validate_combined_parent_authority)

    candidate = _combined_candidate()
    value = load_combined_parent_authority(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')
    for mutate in (
            lambda item: item['parents']['no_pif'].__setitem__(
                'source_commit', '0' * 40),
            lambda item: item['parents']['tail_aware_softplus_pwl'][
                'isolated_evidence']['evaluate'].__setitem__(
                    'sha256', '0' * 64),
            lambda item: item['claim_limits'].__setitem__(
                'isolated_deltas_additive', True),
            lambda item: item['comparator'].__setitem__(
                'checkpoint_sha256', '0' * 64)):
        tampered = copy.deepcopy(value)
        mutate(tampered)
        with pytest.raises(CombinedCandidateAuthorityError):
            validate_combined_parent_authority(
                tampered, candidate=candidate,
                repository_root=REPOSITORY_ROOT,
                manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')


def test_combined_admission_is_hash_bound_and_schedules_without_reselection():
    from mambapose_opt.combined_candidate import load_pwl_admission_reference
    from tools.optimization.run_campaign import (
        SubprocessStageRunner, _stages_for_candidate)

    candidate = _combined_candidate()
    authority = REPOSITORY_ROOT / candidate.features[
        'combined_parent_authority']
    reference = {
        'path': authority.relative_to(REPOSITORY_ROOT).as_posix(),
        'sha256': hashlib.sha256(authority.read_bytes()).hexdigest(),
    }
    admitted = load_pwl_admission_reference(
        reference, repository_root=REPOSITORY_ROOT,
        manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')
    assert admitted['decision'] == 'selected'
    assert admitted['selected_candidate_id'] == candidate.id
    assert admitted['admission_kind'] == 'combined-parent-authority-v1'
    with pytest.raises(ValueError, match='hash'):
        load_pwl_admission_reference(
            {**reference, 'sha256': '0' * 64},
            repository_root=REPOSITORY_ROOT,
            manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')

    assert _stages_for_candidate(candidate) == (
        'calibrate', 'convert', 'smoke-stage-a', 'profile', 'evaluate',
        'compare', 'latency')
    runner = SubprocessStageRunner(
        REPOSITORY_ROOT / 'work_dirs/optimization',
        REPOSITORY_ROOT / 'optimization/candidates.json')
    artifact = (
        REPOSITORY_ROOT / 'work_dirs/optimization/ssm-quant-pwl' /
        candidate.id / '0/convert/convert.json')
    command = runner._command(candidate, 'convert', artifact)
    assert '--calibration-artifact' in command
    assert command[command.index('--selection-artifact') + 1] == (
        'optimization/no_pif_softplus_parent_authority.json')
    compare = runner._command(
        candidate, 'compare', artifact.parent.parent / 'compare/compare.json')
    assert compare[1].endswith(
        'tools/optimization/compare_combined_candidate.py')


def test_combined_calibration_uses_exact_no_pif_parent():
    from mambapose_opt.schema import load_candidate_manifest
    from tools.optimization.calibrate_numeric import _calibration_source

    candidates = load_candidate_manifest(
        REPOSITORY_ROOT / 'optimization/candidates.json')
    target = _combined_candidate()
    source = _calibration_source(target, candidates)
    assert source.id == 'no-pif-s-v1'
    assert source.checkpoint == target.checkpoint
    assert source.checkpoint_sha256 == target.checkpoint_sha256


def test_cpu_audit_cli_is_directly_executable():
    result = subprocess.run([
        sys.executable,
        str(REPOSITORY_ROOT /
            'tools/optimization/audit_combined_candidate.py'),
        '--help',
    ], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    assert 'CPU-only structural audit' in result.stdout


def test_parent_authority_binds_and_validates_six_frozen_v6_artifacts():
    from mambapose_opt.combined_candidate import (
        load_combined_parent_authority)

    candidate = _combined_candidate()
    authority = load_combined_parent_authority(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=REPOSITORY_ROOT / 'optimization/candidates.json')
    evidence = authority['parents']['tail_aware_softplus_pwl'][
        'isolated_evidence']
    assert tuple(evidence) == (
        'selection', 'installation', 'smoke', 'evaluate', 'profile',
        'latency')
    assert all(set(reference) == {'path', 'sha256'}
               for reference in evidence.values())
    validated = authority['validated_parent_artifacts']
    assert tuple(validated) == tuple(evidence)
    assert validated['selection']['selected_candidate_id'] == (
        'pwl-softplus-s-v1')
    assert validated['evaluate']['candidate_id'] == 'pwl-softplus-s-v1'


def test_bound_workspace_artifact_rejects_symlink_and_hash_tampering(tmp_path):
    from mambapose_opt.combined_candidate import (
        CombinedCandidateAuthorityError, read_bound_workspace_artifact)

    artifact = tmp_path / 'artifact.json'
    artifact.write_text('{"value": 1}\n', encoding='utf-8')
    reference = {
        'path': 'artifact.json',
        'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert read_bound_workspace_artifact(reference, checkout_root=tmp_path) == {
        'value': 1}
    with pytest.raises(CombinedCandidateAuthorityError, match='hash'):
        read_bound_workspace_artifact(
            {**reference, 'sha256': '0' * 64}, checkout_root=tmp_path)
    alias = tmp_path / 'alias.json'
    alias.symlink_to(artifact)
    with pytest.raises(CombinedCandidateAuthorityError, match='symlink'):
        read_bound_workspace_artifact(
            {**reference, 'path': 'alias.json'}, checkout_root=tmp_path)
    real_directory = tmp_path / 'real'
    real_directory.mkdir()
    nested = real_directory / 'artifact.json'
    nested.write_bytes(artifact.read_bytes())
    directory_alias = tmp_path / 'directory-alias'
    directory_alias.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(CombinedCandidateAuthorityError, match='symlink'):
        read_bound_workspace_artifact(
            {**reference, 'path': 'directory-alias/artifact.json'},
            checkout_root=tmp_path)


def _evaluation(candidate_id, checkpoint, source_config, ap):
    metrics = {
        name: float(ap + index / 10)
        for index, name in enumerate(('AP', 'AP50', 'AP75', 'APM', 'APL', 'AR'))
    }
    metrics['unit'] = 'percentage_points'
    protocol = {
        'dataset': 'coco', 'split': 'val2017', 'complete_split': True,
        'authority_sha256': '1' * 64,
        'annotation_sha256': '2' * 64,
        'detection_sha256': '3' * 64,
        'image_corpus_sha256': '4' * 64,
        'annotation_image_count': 5000,
        'annotation_record_count': 11004,
        'detection_record_count': 104125,
        'source_config': source_config,
        'checkpoint': checkpoint,
    }
    return {
        'schema_version': 1, 'candidate_id': candidate_id,
        'stage': 'evaluate', 'result': {
            'modes': {
                mode: {'metrics': dict(metrics), 'protocol': dict(protocol)}
                for mode in ('flip', 'no_flip')},
        },
    }


def test_combined_comparison_recomputes_full_delta_and_rejects_protocol_drift():
    from mambapose_opt.combined_comparison import (
        CombinedComparisonError, build_combined_comparison)

    full = _evaluation(
        'full-s-v1-frozen-r1',
        'work_dirs/reproduction/runs/coco-s-v1/best_coco_AP_epoch_300.pth',
        'configs/reproduction/coco_s_v1.py', 72.8)
    combined = _evaluation(
        COMBINED_ID,
        'work_dirs/reproduction/runs/coco-s-v1-no-pif/'
        'best_coco_AP_epoch_300.pth',
        'work_dirs/optimization/runtime.py', 72.85)
    result = build_combined_comparison(
        full, combined,
        full_reference={'path': 'full.json', 'sha256': 'a' * 64},
        combined_reference={'path': 'combined.json', 'sha256': 'b' * 64},
        candidate_id=COMBINED_ID)
    assert result['stage'] == 'compare'
    flip = result['result']['modes']['flip']
    assert flip['candidate_minus_full']['AP'] == pytest.approx(0.05)
    assert flip['drop_full_minus_candidate']['AP'] == pytest.approx(-0.05)
    assert set(flip['candidate_minus_full']) == {
        'AP', 'AP50', 'AP75', 'APM', 'APL', 'AR', 'unit'}
    assert result['result']['claim_limits']['formal_paired_pass'] is False
    assert '"isolated_delta":' not in json.dumps(result)
    drifted = copy.deepcopy(combined)
    drifted['result']['modes']['flip']['protocol']['detection_sha256'] = (
        '9' * 64)
    with pytest.raises(CombinedComparisonError, match='protocol'):
        build_combined_comparison(
            full, drifted,
            full_reference={'path': 'full.json', 'sha256': 'a' * 64},
            combined_reference={'path': 'combined.json', 'sha256': 'b' * 64},
            candidate_id=COMBINED_ID)


def test_parent_evidence_snapshots_are_portable_without_sibling_worktrees(
        tmp_path):
    from mambapose_opt.combined_candidate import (
        read_bound_workspace_artifact)

    authority = json.loads((
        REPOSITORY_ROOT /
        'optimization/no_pif_softplus_parent_authority.json').read_text())
    references = list(authority['parents']['tail_aware_softplus_pwl'][
        'isolated_evidence'].values())
    references.append(authority['comparator']['evaluation_artifact'])
    assert all(not reference['path'].startswith('.worktrees/')
               for reference in references)
    for reference in references:
        source = REPOSITORY_ROOT / reference['path']
        target = tmp_path / reference['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert read_bound_workspace_artifact(
            reference, checkout_root=tmp_path)
        original = target.read_bytes()
        target.write_bytes(original + b' ')
        with pytest.raises(ValueError, match='hash'):
            read_bound_workspace_artifact(
                reference, checkout_root=tmp_path)
        target.write_bytes(original)
    assert not (tmp_path / '.worktrees').exists()


def test_full_comparator_snapshot_validates_config_checkpoint_and_source():
    from mambapose_opt.combined_comparison import (
        CombinedComparisonError, validate_comparator_evaluation)

    authority = json.loads((
        REPOSITORY_ROOT /
        'optimization/no_pif_softplus_parent_authority.json').read_text())
    comparator = authority['comparator']
    evaluation = json.loads((
        REPOSITORY_ROOT / comparator['evaluation_artifact']['path']
    ).read_text())
    validated = validate_comparator_evaluation(
        evaluation, comparator=comparator)
    assert validated['candidate_id'] == 'full-s-v1-frozen-r1'
    for mutate in (
            lambda item: item['result']['runtime']['checkpoint'].__setitem__(
                'sha256', '0' * 64),
            lambda item: item['result']['source'].__setitem__(
                'git_commit', '0' * 40),
            lambda item: item['result']['modes']['flip']['protocol'].__setitem__(
                'source_config', 'wrong.py')):
        tampered = copy.deepcopy(evaluation)
        mutate(tampered)
        with pytest.raises(CombinedComparisonError):
            validate_comparator_evaluation(tampered, comparator=comparator)


def test_audit_cli_routes_model_noise_to_stderr(monkeypatch, capsys):
    from tools.optimization import audit_combined_candidate as cli

    def noisy_audit(candidate_id, manifest):
        print('model construction noise')
        return {'candidate_id': candidate_id, 'schema_version': 1}

    monkeypatch.setattr(cli, 'audit', noisy_audit)
    monkeypatch.setattr(sys, 'argv', ['audit_combined_candidate.py'])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        'candidate_id': COMBINED_ID, 'schema_version': 1}
    assert 'model construction noise' in captured.err


class _FakeSS2D(nn.Module):
    def __init__(self):
        super().__init__()
        self._numeric_pwl_function = None

    def install_numeric_pwl(self, function_name, approximation):
        assert self._numeric_pwl_function is None
        self._numeric_pwl_function = function_name
        self.add_module(f'_numeric_pwl_{function_name}', approximation)


class _OpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = _FakeSS2D()


class _Layer(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.blocks = nn.ModuleList([_OpBlock() for _ in range(count)])


class _DisabledPIF(nn.Module):
    mode = 'disabled'

    def __init__(self):
        super().__init__()
        self.MambaBlock = nn.Linear(2, 2)
        self.MambaBlock2 = nn.Linear(2, 2)
        self.Mamba_selfScanBlock = nn.Linear(2, 2)

    def forward(self, value):
        return value


class _CombinedFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([
            _Layer(1), _Layer(1), _Layer(2), _Layer(1)])
        self.head = nn.Module()
        self.head.pose_interaction = _DisabledPIF()


def test_cpu_structure_smoke_removes_pif_modules_and_installs_exact_roles():
    from mambapose_opt.combined_candidate import (
        install_combined_cpu_smoke, prune_disabled_pif)

    model = _CombinedFixture()
    original = model.head.pose_interaction
    tokens = torch.randn(2, 17, 256)
    torch.testing.assert_close(original(tokens), tokens, rtol=0, atol=0)
    removed = prune_disabled_pif(model)
    assert removed == ('head.pose_interaction',)
    pif = model.head.pose_interaction
    assert pif(tokens) is tokens
    assert not tuple(pif.children())
    assert not tuple(pif.parameters())

    summary = install_combined_cpu_smoke(
        model, roles=EXPECTED_ROLES, function_name='softplus',
        domain=(-8.0, 8.0), segments=16,
        saturation='continuous-asymptotic-tail-v1')
    assert summary == {
        'pif_modules_removed': ['head.pose_interaction'],
        'pif_mode': 'disabled',
        'pwl_function': 'softplus',
        'pwl_roles': list(EXPECTED_ROLES),
        'pwl_role_count': 5,
        'tail_policy': 'continuous-asymptotic-tail-v1',
    }
    installed = tuple(
        name for name, module in model.named_modules()
        if getattr(module, '_numeric_pwl_function', None) == 'softplus')
    assert installed == EXPECTED_ROLES
