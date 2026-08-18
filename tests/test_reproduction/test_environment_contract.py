from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_reproduction_requirements_pin_the_blackwell_stack():
    text = (ROOT / 'requirements/reproduction.in').read_text()
    for pin in (
            'torch==2.7.1',
            'torchvision==0.22.1',
            'numpy==1.26.4',
            'mmcv-lite==2.1.0',
            'mmengine==0.10.7',
            'timm==0.9.16'):
        assert pin in text


def test_bootstrap_uses_python311_cuda128_and_disables_user_site():
    text = (ROOT / 'tools/reproduction/bootstrap_env.sh').read_text()
    assert 'PYTHONNOUSERSITE=1' in text
    lock = (ROOT / 'requirements/conda-linux-64.lock').read_text()
    assert 'python-3.11.15-' in lock
    assert 'cuda-nvcc-12.8.93-' in lock
    assert "--no-build-isolation 'chumpy==0.70'" in text


def test_environment_verifier_records_required_evidence_fields():
    text = (ROOT / 'tools/reproduction/verify_environment.py').read_text()
    for field in (
            'python_version',
            'torch_version',
            'torchvision_version',
            'cuda_runtime',
            'nvcc_version',
            'device_capability',
            'cxx11_abi',
            'cuda_smoke_checksum',
            'conda_packages',
            'compiler'):
        assert field in text


def test_bootstrap_cache_is_ignored():
    assert '/.tools/' in (ROOT / '.gitignore').read_text()


def test_isolated_environment_has_no_broken_or_unsupported_packages():
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'],
        capture_output=True,
        text=True)
    assert result.returncode == 0, result.stdout + result.stderr
