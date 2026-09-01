import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module_name', [
    'tools.optimization.evaluate_candidate',
    'tools.optimization.train_candidate',
])
def test_nested_python_environment_prepends_exact_repository_once(
        tmp_path, monkeypatch, module_name):
    module = __import__(module_name, fromlist=['_child_environment'])
    repository = tmp_path / 'linked-worktree'
    existing = tmp_path / 'existing'
    monkeypatch.setattr(module, 'REPO_ROOT', repository)
    monkeypatch.setenv(
        'PYTHONPATH', os.pathsep.join([
            str(existing), str(repository), str(repository), 'relative-entry']))
    monkeypatch.setenv('PYTHONNOUSERSITE', '0')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '0')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    monkeypatch.setenv('MAMBAPOSE_ENV_SENTINEL', 'preserved')

    environment = module._child_environment()

    assert environment['PYTHONPATH'].split(os.pathsep) == [
        str(repository), str(existing), 'relative-entry']
    assert environment['PYTHONNOUSERSITE'] == '1'
    assert environment['PYTHONDONTWRITEBYTECODE'] == '1'
    assert environment['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert environment['MAMBAPOSE_ENV_SENTINEL'] == 'preserved'


@pytest.mark.parametrize('module_name', [
    'tools.optimization.evaluate_candidate',
    'tools.optimization.train_candidate',
])
def test_nested_python_environment_handles_absent_pythonpath(
        tmp_path, monkeypatch, module_name):
    module = __import__(module_name, fromlist=['_child_environment'])
    repository = Path(tmp_path / 'linked-worktree')
    monkeypatch.setattr(module, 'REPO_ROOT', repository)
    monkeypatch.delenv('PYTHONPATH', raising=False)

    environment = module._child_environment()

    assert environment['PYTHONPATH'] == str(repository)


def test_evaluator_child_environment_resolves_repo_local_numeric_import():
    from tools.optimization import evaluate_candidate

    environment = evaluate_candidate._child_environment()
    completed = subprocess.run(
        [sys.executable, '-B', '-c',
         'import mambapose_opt.numeric_conversion'],
        cwd=evaluate_candidate.REPO_ROOT / 'tools',
        env=environment, capture_output=True, text=True, check=False,
        shell=False)

    assert completed.returncode == 0, completed.stderr
