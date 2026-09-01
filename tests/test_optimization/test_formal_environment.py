from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import ast
import json
import sys

import pytest

import mambapose_opt.formal_environment as formal_environment
from mambapose_opt.formal_environment import (
    FormalEnvironmentError,
    apply_required_process_environment,
    validate_environment_authority,
)


ROOT = Path(__file__).resolve().parents[2]
APPROVED_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
STRIPPED_AMBIENT = (
    'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE',
    'LD_PRELOAD', 'LD_LIBRARY_PATH')


def test_environment_authority_captures_and_revalidates_exact_venv():
    from mambapose_opt.formal_environment import EnvironmentAuthority

    authority = EnvironmentAuthority.capture(ROOT)
    assert authority.venv_link == '.venv'
    assert authority.venv_target == str((ROOT / '.venv').resolve())
    assert authority.interpreter_path == '.venv/bin/python'
    assert len(authority.interpreter_sha256) == 64
    assert tuple(item.role for item in authority.requirements) == (
        'requirements', 'runtime', 'reproduction_constraints')
    assert authority.packages
    validate_environment_authority(authority, ROOT)


@pytest.mark.parametrize('field', [
    'venv_target', 'interpreter_sha256', 'packages', 'native_modules',
    'requirements',
])
def test_environment_authority_rejects_every_observed_drift(field):
    from mambapose_opt.formal_environment import EnvironmentAuthority

    authority = EnvironmentAuthority.capture(ROOT)
    if field in {'packages', 'native_modules', 'requirements'}:
        changed = replace(authority, **{field: getattr(authority, field) + (
            getattr(authority, field)[0],)})
    else:
        changed = replace(authority, **{field: '0' * 64})
    with pytest.raises(FormalEnvironmentError):
        validate_environment_authority(changed, ROOT)


@pytest.mark.parametrize('path', ['../.venv', './.venv', '.venv/../.venv'])
def test_environment_authority_rejects_noncanonical_link_alias(path):
    from mambapose_opt.formal_environment import EnvironmentAuthority

    authority = replace(EnvironmentAuthority.capture(ROOT), venv_link=path)
    with pytest.raises(FormalEnvironmentError, match='venv link'):
        validate_environment_authority(authority, ROOT)


def test_environment_document_rejects_unknown_fields():
    from mambapose_opt.formal_environment import EnvironmentAuthority

    document = EnvironmentAuthority.capture(ROOT).to_dict()
    document['unexpected'] = True
    with pytest.raises(FormalEnvironmentError, match='fields'):
        EnvironmentAuthority.from_dict(document)


def test_process_first_launcher_imports_only_stdlib_and_environment_authority():
    source = ROOT / 'tools/optimization/formal_stage_c_entrypoint.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert imports <= {
        '__future__', 'json', 'os', 'pathlib', 'sys',
        'mambapose_opt.formal_environment'}
    assert 'torch' not in source.read_text(encoding='utf-8')


def test_process_environment_sets_absent_values_and_rejects_conflicts():
    from mambapose_opt.formal_environment import EnvironmentAuthority

    authority = EnvironmentAuthority.capture(ROOT)
    applied = apply_required_process_environment(authority, {'PATH': '/bin'})
    assert applied['PATH'] == APPROVED_PATH
    assert applied['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert applied['CUDA_VISIBLE_DEVICES'] == '0'
    with pytest.raises(FormalEnvironmentError, match='conflicts'):
        apply_required_process_environment(
            authority, {'CUDA_VISIBLE_DEVICES': '1'})
    with pytest.raises(FormalEnvironmentError, match='forbidden'):
        apply_required_process_environment(
            authority, {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})


def test_capture_probe_removes_hostile_import_and_linker_environment(
        monkeypatch):
    from mambapose_opt.formal_environment import EnvironmentAuthority

    real_run = formal_environment.subprocess.run
    observed = []

    def observe(*args, **kwargs):
        observed.append(dict(kwargs['env']))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(formal_environment.subprocess, 'run', observe)
    monkeypatch.setenv('PATH', '/hostile/bin')
    for key in STRIPPED_AMBIENT:
        monkeypatch.setenv(key, '/hostile/value')
    EnvironmentAuthority.capture(ROOT)
    assert len(observed) == 1
    assert observed[0]['PATH'] == APPROVED_PATH
    assert all(key not in observed[0] for key in STRIPPED_AMBIENT)


def test_launcher_exec_environment_removes_hostile_ambient_values(
        tmp_path, monkeypatch):
    from mambapose_opt.formal_environment import EnvironmentAuthority
    import tools.optimization.formal_stage_c_entrypoint as launcher

    authority = EnvironmentAuthority.capture(ROOT)
    authority_path = tmp_path / 'environment-authority.json'
    authority_path.write_text(
        json.dumps(authority.to_dict(), sort_keys=True) + '\n',
        encoding='utf-8')
    monkeypatch.setattr(launcher, '_AUTHORITY', authority_path)
    monkeypatch.setattr(
        launcher, 'validate_environment_authority',
        lambda observed, root: None)
    monkeypatch.setenv('PATH', '/hostile/bin')
    for key in STRIPPED_AMBIENT:
        monkeypatch.setenv(key, '/hostile/value')
    monkeypatch.setattr(sys, 'argv', [
        'formal_stage_c_entrypoint.py', 'trace', '--cpu-only'])
    executed = {}

    class ExecIntercept(RuntimeError):
        pass

    def intercept(executable, arguments, environment):
        executed.update(
            executable=executable, arguments=arguments,
            environment=dict(environment))
        raise ExecIntercept

    monkeypatch.setattr(launcher.os, 'execve', intercept)
    with pytest.raises(ExecIntercept):
        launcher.main()
    assert executed['environment']['PATH'] == APPROVED_PATH
    assert all(key not in executed['environment'] for key in STRIPPED_AMBIENT)
    assert executed['arguments'][1:3] == ['-B', '-m']


def test_capture_rejects_unsafe_torch_weights_only_override(monkeypatch):
    from mambapose_opt.formal_environment import EnvironmentAuthority

    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    with pytest.raises(FormalEnvironmentError, match='forbidden'):
        EnvironmentAuthority.capture(ROOT)
