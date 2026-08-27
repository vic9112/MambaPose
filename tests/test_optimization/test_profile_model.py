from pathlib import Path
import subprocess
import sys


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
