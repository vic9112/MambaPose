from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
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
        'latency')

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
        'latency')
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
                'isolated_evidence'].__setitem__('flip_ap', 99.0),
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
        'latency')
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
