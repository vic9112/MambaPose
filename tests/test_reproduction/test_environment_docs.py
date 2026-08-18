from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_environment_docs_name_all_evidence_gates():
    text = (ROOT / 'docs/reproduction/environment.md').read_text()
    for gate in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'):
        assert gate in text
    assert 'RTX 5090' in text
    assert 'sm_120' in text


def test_rebuild_uses_a_temporary_prefix_and_local_native_wheels():
    text = (ROOT / 'tools/reproduction/rebuild_check.sh').read_text()
    assert 'mktemp -d' in text
    assert 'MAMBAPOSE_ENV_PREFIX' in text
    assert 'work_dirs/reproduction/wheelhouse' in text
    assert 'verify_environment.py' in text
    assert 'verify_native.py' in text
    assert 'package_versions_match' in text
    assert 'conda_packages_match' in text
    assert 'dynamic_dependencies_isolated' in text


def test_bootstrap_applies_the_exact_transitive_constraints():
    bootstrap = (ROOT / 'tools/reproduction/bootstrap_env.sh').read_text()
    constraints = (
        ROOT / 'requirements/reproduction-constraints.txt').read_text()
    assert 'reproduction-constraints.txt' in bootstrap
    assert 'conda-linux-64.lock' in bootstrap
    conda_lock = (ROOT / 'requirements/conda-linux-64.lock').read_text()
    assert conda_lock.startswith('@EXPLICIT\n')
    assert 'python-3.11.15-' in conda_lock
    assert 'cuda-nvcc-12.8.93-' in conda_lock
    assert 'cuda-cudart-dev-12.8.90-' in conda_lock
    assert 'patchelf-0.17.0-' in conda_lock
    assert sum(
        len(line.rsplit('#', 1)[-1]) == 64
        for line in conda_lock.splitlines()) >= 60
    assert 'torch==2.7.1+cu128' in constraints
    assert 'nvidia-cudnn-cu12==9.7.1.26' in constraints


def test_environment_verifier_resolves_prefix_local_nvcc():
    verifier = (ROOT / 'tools/reproduction/verify_environment.py').read_text()
    assert "Path(sys.executable).parent / 'nvcc'" in verifier
    assert 'conda_packages' in verifier
    assert 'compiler' in verifier
