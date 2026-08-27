from pathlib import Path
import subprocess
import sys

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
