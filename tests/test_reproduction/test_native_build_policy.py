from pathlib import Path

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
    assert 'MAMBA_FORCE_CXX11_ABI' not in environment
