from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MAMBA_SETUP = (
    ROOT / 'mmpose/models/backbones/Vim/mamba-1p1p1/setup.py')
VMAMBA_SETUP = (
    ROOT / 'mmpose/models/backbones/Vmamba/kernels/selective_scan/setup.py')


def test_blackwell_gencode_targets_sm120_only():
    from tools.reproduction.native_build import blackwell_gencode

    assert blackwell_gencode((12, 8)) == [
        '-gencode', 'arch=compute_120,code=sm_120'
    ]


def test_blackwell_gencode_rejects_cuda_older_than_128():
    from tools.reproduction.native_build import blackwell_gencode

    with pytest.raises(RuntimeError, match='CUDA 12.8'):
        blackwell_gencode((12, 7))


def test_native_setup_scripts_use_the_shared_blackwell_policy():
    for setup_path in (MAMBA_SETUP, VMAMBA_SETUP):
        source = setup_path.read_text()
        assert 'blackwell_gencode' in source
        assert 'arch=compute_70' not in source
        assert 'arch=compute_80' not in source
        assert 'arch=compute_90' not in source


def test_native_build_environment_forces_local_source_builds():
    from tools.reproduction.native_build import build_environment

    environment = build_environment(ROOT / '.venv')
    assert environment['MAMBA_FORCE_BUILD'] == 'TRUE'
    assert environment['PYTHONNOUSERSITE'] == '1'
    assert environment['CUDA_HOME'] == str(ROOT / '.venv')
    target_include = str(ROOT / '.venv/targets/x86_64-linux/include')
    target_lib = str(ROOT / '.venv/targets/x86_64-linux/lib')
    assert target_include in environment['CPATH'].split(os.pathsep)
    assert target_lib in environment['LIBRARY_PATH'].split(os.pathsep)
    assert target_lib in environment['LD_LIBRARY_PATH'].split(os.pathsep)
    nvidia_root = ROOT / '.venv/lib/python3.11/site-packages/nvidia'
    for component in ('cublas', 'cusparse'):
        assert str(nvidia_root / component / 'include') in (
            environment['CPATH'].split(os.pathsep))
    assert 'MAMBA_FORCE_CXX11_ABI' not in environment


def test_native_build_script_runs_from_its_file_path():
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/reproduction/native_build.py'),
         '--show-policy'],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True)
    assert result.returncode == 0, result.stdout + result.stderr
