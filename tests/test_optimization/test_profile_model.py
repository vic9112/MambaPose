import argparse
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
import torch


def test_profile_device_requires_available_controller_cuda(monkeypatch):
    import tools.optimization.profile_model as tool

    monkeypatch.setattr(tool.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('MAMBAPOSE_PHYSICAL_DEVICE_INDEX', '3')

    with pytest.raises(ValueError, match='CUDA'):
        tool._profile_device('cuda:0')


def test_formal_numeric_profile_places_model_and_input_on_logical_cuda(
        tmp_path, monkeypatch):
    import tools.optimization.profile_model as tool
    from mambapose_opt.schema import CandidateSpec

    root = tmp_path / 'repo'
    config_path = root / 'configs/numeric.py'
    checkpoint = root / 'approved/model.pth'
    output = root / 'work_dirs/optimization/candidate/0/profile/profile.json'
    config_path.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config_path.write_text('model = dict(type="Fixture")\n')
    checkpoint.write_bytes(b'checkpoint')
    checkpoint_sha = tool._sha256(checkpoint)
    candidate = CandidateSpec.from_dict({
        'id': 'numeric-fixture', 'route': 'ssm-quant-pwl',
        'kind': 'pwl', 'config': 'configs/numeric.py',
        'checkpoint': 'approved/model.pth',
        'checkpoint_sha256': checkpoint_sha, 'seed': 0,
        'features': {'numeric_kind': 'pwl'},
    })
    runtime = {
        'config_path': config_path,
        'config_sha256': tool._sha256(config_path),
        'checkpoint_path': checkpoint,
        'checkpoint_name': 'approved/model.pth',
        'checkpoint_sha256': checkpoint_sha,
        'train': None,
        'pwl_stage_a': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'numeric-fixture/0/smoke-stage-a/smoke.json'),
            'sha256': '9' * 64,
        },
    }

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, inputs, **unused):
            return inputs[:, :1]

    seen = []
    original_zeros = torch.zeros

    def fake_safe_model(*args, device, **kwargs):
        seen.append(('model', device))
        return Dummy()

    def fake_zeros(shape, *, device):
        seen.append(('input', device))
        return original_zeros(shape)

    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', root)
    monkeypatch.setattr(tool, 'resolve_numeric_runtime', lambda *a, **k: runtime)
    monkeypatch.setattr(tool, 'clean_git_commit', lambda _root: 'a' * 40)
    monkeypatch.setattr(
        tool, 'build_numeric_source_binding', lambda **kwargs: {'bound': True})
    monkeypatch.setattr(tool, 'build_manifest_authorized_model', fake_safe_model)
    monkeypatch.setattr(tool.torch, 'zeros', fake_zeros)
    monkeypatch.setattr(tool.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        tool.torch.cuda, 'synchronize', lambda: seen.append(('sync', 'cuda:0')))
    monkeypatch.setenv('MAMBAPOSE_PHYSICAL_DEVICE_INDEX', '3')

    result = tool.profile(
        candidate, (1, 3, 4, 4), manifest_path=root / 'manifest.json',
        output=output)

    assert seen == [
        ('model', 'cuda:0'), ('input', 'cuda:0'),
        ('sync', 'cuda:0'), ('sync', 'cuda:0')]
    assert result['schema_version'] == 2
    assert result['device'] == {
        'logical': 'cuda:0', 'physical_index': 3, 'kind': 'cuda'}
    assert result['parent'] == {
        'config': 'configs/numeric.py',
        'checkpoint': 'approved/model.pth',
        'checkpoint_sha256': checkpoint_sha,
    }
    assert result['runtime'] == {
        'config': {
            'path': 'configs/numeric.py',
            'sha256': tool._sha256(config_path),
        },
        'checkpoint': {
            'path': 'approved/model.pth',
            'sha256': checkpoint_sha,
        },
    }
    assert result['pwl_stage_a'] == runtime['pwl_stage_a']


def test_profile_cli_is_directly_executable_from_repository_root():
    root = Path(__file__).parents[2]

    result = subprocess.run(
        [sys.executable, 'tools/optimization/profile_model.py', '--help'],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'candidate_id' in result.stdout
    assert '--device' in result.stdout


@pytest.mark.parametrize(
    'output',
    [
        '/tmp/mambapose-profile.json',
        '../mambapose-profile.json',
        'work_dirs/not-optimization/mambapose-profile.json',
    ],
)
def test_profile_cli_rejects_output_outside_optimization_artifacts(output):
    root = Path(__file__).parents[2]

    result = subprocess.run(
        [
            sys.executable,
            'tools/optimization/profile_model.py',
            'full-s-v1',
            '--output',
            output,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert 'work_dirs/optimization' in result.stderr


def test_profile_cli_rejects_symlinked_output_escape(tmp_path):
    root = Path(__file__).parents[2]
    artifact_root = root / 'work_dirs' / 'optimization'
    escape = artifact_root / f'profile-symlink-escape-{uuid4().hex}'
    outside = tmp_path / 'outside'
    outside.mkdir()
    created_work_dirs = not artifact_root.parent.exists()
    created_artifact_root = not artifact_root.exists()
    escape.parent.mkdir(parents=True, exist_ok=True)
    escape.symlink_to(outside, target_is_directory=True)
    output = f'work_dirs/optimization/{escape.name}/profile.json'

    try:
        result = subprocess.run(
            [
                sys.executable,
                'tools/optimization/profile_model.py',
                'full-s-v1',
                '--output',
                output,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        escape.unlink(missing_ok=True)
        if created_artifact_root:
            artifact_root.rmdir()
        if created_work_dirs:
            artifact_root.parent.rmdir()

    assert result.returncode == 2
    assert 'work_dirs/optimization' in result.stderr
    assert not (outside / 'profile.json').exists()


def test_profile_output_rejects_symlinked_optimization_artifact_root(
        tmp_path, monkeypatch):
    import tools.optimization.profile_model as tool

    root = tmp_path / 'repo'
    artifact_root = root / 'work_dirs' / 'optimization'
    outside = tmp_path / 'outside'
    artifact_root.parent.mkdir(parents=True)
    outside.mkdir()
    artifact_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(tool, 'REPOSITORY_ROOT', root)

    with pytest.raises(
            argparse.ArgumentTypeError, match='work_dirs/optimization'):
        tool._output_path('work_dirs/optimization/profile.json')
    assert not (outside / 'profile.json').exists()
