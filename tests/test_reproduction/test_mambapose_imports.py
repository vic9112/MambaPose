from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _isolated_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True)


def test_registration_does_not_require_unused_mmcv_or_vim_extensions():
    result = _isolated_subprocess(
        'from mmpose.utils import register_all_modules; '
        'register_all_modules()')
    assert result.returncode == 0, result.stdout + result.stderr


def test_setup_metadata_does_not_require_a_missing_readme():
    result = _isolated_subprocess('import runpy, sys; '
                                  'sys.argv=["setup.py", "--name"]; '
                                  'runpy.run_path("setup.py", '
                                  'run_name="__main__")')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith('mmpose')


def test_vim_does_not_use_an_absolute_rope_import():
    source = (
        ROOT / 'mmpose/models/backbones/Vim/vim/models_mamba.py').read_text()
    assert 'from rope import *' not in source
