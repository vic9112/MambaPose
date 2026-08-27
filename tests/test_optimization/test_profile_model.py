from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest


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


def test_profile_cli_rejects_symlinked_optimization_artifact_root(tmp_path):
    root = Path(__file__).parents[2]
    artifact_root = root / 'work_dirs' / 'optimization'
    outside = tmp_path / 'outside'
    outside.mkdir()
    assert not artifact_root.exists()
    artifact_root.symlink_to(outside, target_is_directory=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                'tools/optimization/profile_model.py',
                'full-s-v1',
                '--output',
                'work_dirs/optimization/profile.json',
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        artifact_root.unlink(missing_ok=True)

    assert result.returncode == 2
    assert 'work_dirs/optimization' in result.stderr
    assert not (outside / 'profile.json').exists()
