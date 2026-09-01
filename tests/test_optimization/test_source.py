from pathlib import Path
import json
import os
import subprocess

import pytest


def _commit_ignore_rules(repository: Path, rules: str) -> None:
    subprocess.run(['git', 'init', '-q'], cwd=repository, check=True)
    (repository / '.gitignore').write_text(rules)
    subprocess.run(['git', 'add', '.gitignore'], cwd=repository, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=repository, check=True)


def _commit_runtime_authority(repository: Path) -> None:
    subprocess.run(['git', 'init', '-q'], cwd=repository, check=True)
    (repository / '.gitignore').write_text('/work_dirs/\n')
    optimization = repository / 'optimization'
    optimization.mkdir()
    (optimization / 'candidates.json').write_text(json.dumps({
        'schema_version': 1,
        'candidates': [
            {
                'id': 'full-s-v1', 'route': 'baseline', 'seed': 0,
                'kind': 'float', 'features': {},
            },
            {
                'id': 'no-pif-s-v1', 'route': 'structural-pif', 'seed': 0,
                'kind': 'structural', 'features': {},
            },
            {
                'id': 'pwl-softplus-s-v1', 'route': 'ssm-quant-pwl',
                'seed': 0, 'kind': 'pwl',
                'features': {'numeric_kind': 'pwl'},
            },
        ],
    }))
    (optimization / 'formal_stage_c.json').write_text(json.dumps({
        'schema_version': 2,
        'runs': [
            {
                'output_root': (
                    'work_dirs/optimization/formal-stage-c/full-seed0'),
                'config': 'configs/optimization/formal_stage_c/full_seed0.py',
            },
            {
                'output_root': (
                    'work_dirs/optimization/formal-stage-c/no-pif-seed0'),
                'config': (
                    'configs/optimization/formal_stage_c/no_pif_seed0.py'),
            },
        ],
    }))
    (optimization / 'recovery-candidates.json').write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'w8a8-recovery', 'route': 'ssm-quant-pwl', 'seed': 0,
            'kind': 'fake-quant', 'features': {'numeric_kind': 'w8a8'},
        }],
    }))
    subprocess.run(
        ['git', 'add', '.gitignore', 'optimization/candidates.json',
         'optimization/formal_stage_c.json',
         'optimization/recovery-candidates.json'], cwd=repository, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=repository, check=True)


def test_clean_source_rejects_untracked_source_but_allows_ignored_runtime(
        tmp_path):
    from mambapose_opt.source import clean_git_commit

    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('work_dirs/\ndata/\n')
    (tmp_path / 'tracked.py').write_text('VALUE = 1\n')
    subprocess.run(
        ['git', 'add', '.gitignore', 'tracked.py'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=tmp_path, check=True)
    (tmp_path / 'work_dirs/runtime.json').parent.mkdir()
    (tmp_path / 'work_dirs/runtime.json').write_text('{}\n')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/asset.json').write_text('{}\n')

    assert len(clean_git_commit(tmp_path)) == 40

    source = tmp_path / 'mambapose_opt/untracked_source.py'
    source.parent.mkdir()
    source.write_text('VALUE = 2\n')
    with pytest.raises(RuntimeError, match='untracked|clean'):
        clean_git_commit(tmp_path)


@pytest.mark.parametrize('relative', [
    'pkg/injected.so',
    'pkg/__pycache__/injected.pyc',
    'pkg/ignored_source.py',
])
def test_clean_source_rejects_ignored_source_capable_files(
        tmp_path, relative):
    from mambapose_opt.source import clean_git_commit

    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('*.so\n*.pyc\nignored_source.py\n')
    subprocess.run(['git', 'add', '.gitignore'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', '-c', 'user.name=Fixture', '-c',
         'user.email=fixture@example.com', 'commit', '-qm', 'fixture'],
        cwd=tmp_path, check=True)
    injected = tmp_path / relative
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_bytes(b'injected')

    with pytest.raises(RuntimeError, match='ignored|source'):
        clean_git_commit(tmp_path)


@pytest.mark.parametrize('relative', [
    'settings.config',
    'run.script',
    'nested/scripts/tool',
    'nested/config/settings',
    'tensor.bin',
    '.venv-evil/payload',
    'dataevil/payload',
])
def test_clean_source_rejects_every_ignored_entry_outside_approved_roots(
        tmp_path, relative):
    from mambapose_opt.source import clean_git_commit

    _commit_ignore_rules(tmp_path, f'/{relative}\n')
    ignored = tmp_path / relative
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_bytes(b'ignored but load-bearing')

    with pytest.raises(RuntimeError, match='ignored|runtime|asset'):
        clean_git_commit(tmp_path)


def test_clean_source_rejects_ignored_symlink_to_external_python_package(
        tmp_path):
    from mambapose_opt.source import clean_git_commit

    _commit_ignore_rules(tmp_path, '/nested/package\n')
    external = tmp_path.parent / f'{tmp_path.name}-external-package'
    external.mkdir()
    (external / '__init__.py').write_text('VALUE = 1\n')
    link = tmp_path / 'nested/package'
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match='ignored|runtime|asset'):
        clean_git_commit(tmp_path)


@pytest.mark.parametrize('raw_name', [b'line\nbreak', b'non-utf8-\xff'])
def test_clean_source_rejects_nul_delimited_weird_ignored_names(
        tmp_path, raw_name):
    from mambapose_opt.source import clean_git_commit

    _commit_ignore_rules(tmp_path, '/*\n!/.gitignore\n')
    descriptor = os.open(
        os.fsencode(tmp_path) + b'/' + raw_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    os.close(descriptor)

    with pytest.raises(RuntimeError, match='ignored|runtime|asset'):
        clean_git_commit(tmp_path)


def test_clean_source_rejects_executable_drift_inside_asset_roots(tmp_path):
    from mambapose_opt.source import clean_git_commit

    approved = ('data', 'pretrained', 'work_dirs')
    _commit_ignore_rules(
        tmp_path, ''.join(f'/{root}/\n' for root in approved))
    for root in approved:
        payload = tmp_path / root / 'nested/settings.py'
        payload.parent.mkdir(parents=True)
        payload.write_text('external runtime or asset payload\n')

    with pytest.raises(RuntimeError, match='ignored|executable|source'):
        clean_git_commit(tmp_path)


def test_clean_source_allows_exact_shared_environment_symlinks(tmp_path):
    from mambapose_opt.source import clean_git_commit

    _commit_ignore_rules(tmp_path, '/.venv\n/data\n/pretrained\n')
    external = tmp_path.parent / f'{tmp_path.name}-assets'
    external.mkdir()
    for name in ('.venv', 'data', 'pretrained'):
        (tmp_path / name).symlink_to(external, target_is_directory=True)

    assert len(clean_git_commit(tmp_path)) == 40


@pytest.mark.parametrize('relative', [
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/'
     'convert/resolved-runtime.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/'
     'train/resolved-train.py'),
    ('work_dirs/optimization/ssm-quant-pwl/w8a8-recovery/0/'
     'train/resolved-train.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/'
     'recovery/resolved-binary-qk-recovery.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/'
     'evaluate/resolved-flip.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/'
     'evaluate/resolved-no-flip.py'),
    ('work_dirs/optimization/structural-pif/no-pif-s-v1/0/evaluate/'
     'mmpose-flip/resolved-flip.py'),
    ('work_dirs/optimization/structural-pif/no-pif-s-v1/0/evaluate/'
     'mmpose-flip/20260828_005047/vis_data/config.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/evaluate/'
     'mmpose-no-flip/tmplqkt_wwv.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/evaluate/'
     'mmpose-no-flip/20260830_230446/vis_data/config.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/recovery/'
     'binary_qk_s_v1.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/recovery/'
     '20260829_155406/vis_data/config.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/post-qat/seed-0/flip/'
     'mmpose/binary_qk_s_v1_deploy.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/post-qat/seed-0/flip/'
     'mmpose/20260829_235614/vis_data/config.py'),
    ('work_dirs/optimization/direct-binary-qk-scaled-s-v1/seed-0/no-flip/'
     'mmpose/binary_qk_s_v1_deploy.py'),
    ('work_dirs/optimization/formal-stage-c/full-seed0/full_seed0.py'),
    ('work_dirs/optimization/formal-stage-c/full-seed0/'
     '20260901_092211/vis_data/config.py'),
    ('work_dirs/optimization/formal-stage-c/no-pif-seed0/final-evaluation/'
     'no-flip/mmpose/no_pif_seed0_no-flip.py'),
    ('work_dirs/optimization/formal-stage-c/no-pif-seed0/final-evaluation/'
     'no-flip/mmpose/20260901_092006/vis_data/config.py'),
])
def test_clean_source_allows_only_canonical_generated_runtime_configs(
        tmp_path, relative):
    from mambapose_opt.source import clean_git_commit

    _commit_runtime_authority(tmp_path)
    generated = tmp_path / relative
    generated.parent.mkdir(parents=True)
    generated.write_text('model = dict()\n')

    assert len(clean_git_commit(tmp_path)) == 40


@pytest.mark.parametrize('relative', [
    'work_dirs/optimization/alternate/resolved-runtime.py',
    'work_dirs/optimization/candidate/convert/injected.py',
    'work_dirs/optimization/candidate/train/resolved-runtime.py',
    ('work_dirs/optimization/structural-pif/no-pif-s-v1/0/evaluate/'
     'mmpose-flip/resolved-no-flip.py'),
    ('work_dirs/optimization/structural-pif/no-pif-s-v1/0/evaluate/'
     'mmpose-flip/20260828-005047/vis_data/config.py'),
    ('work_dirs/optimization/ssm-quant-pwl/pwl-softplus-s-v1/0/train/'
     'mmpose-flip/tmp5wlnlazi.py'),
    ('work_dirs/optimization/unrelated/recovery/binary_qk_s_v1.py'),
    ('work_dirs/optimization/binary-qk-scaled-s-v1/post-qat/seed-0/flip/'
     'mmpose/arbitrary.py'),
    ('work_dirs/optimization/formal-stage-c/full-seed0/no_pif_seed0.py'),
    ('work_dirs/optimization/formal-stage-c/no-pif-seed0/final-evaluation/'
     'flip/mmpose/no_pif_seed1_flip.py'),
    ('work_dirs/optimization/runtime/20260901_092211/vis_data/config.py'),
    ('work_dirs/optimization/evil/candidate/0/evaluate/resolved-flip.py'),
    ('work_dirs/optimization/evil/recovery/'
     'resolved-binary-qk-recovery.py'),
    ('work_dirs/optimization/ssm-quant-pwl/not-in-manifest/999/evaluate/'
     'mmpose-flip/tmpattacker.py'),
    ('work_dirs/optimization/formal-stage-c/full-seed999/full_seed999.py'),
    ('work_dirs/optimization/direct-binary-qk-attacker/seed-0/flip/mmpose/'
     'binary_qk_s_v1_deploy.py'),
])
def test_clean_source_rejects_generated_config_names_outside_canonical_stage(
        tmp_path, relative):
    from mambapose_opt.source import clean_git_commit

    _commit_runtime_authority(tmp_path)
    generated = tmp_path / relative
    generated.parent.mkdir(parents=True)
    generated.write_text('model = dict()\n')

    with pytest.raises(RuntimeError, match='ignored|source'):
        clean_git_commit(tmp_path)


def test_clean_source_allows_exact_formal_prior_link_only(tmp_path):
    from mambapose_opt.source import clean_git_commit

    _commit_ignore_rules(tmp_path, '/work_dirs/\n')
    external = tmp_path.parent / f'{tmp_path.name}-prior-stage-b'
    external.mkdir()
    link = tmp_path / 'work_dirs/optimization/prior-stage-b'
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    assert len(clean_git_commit(tmp_path)) == 40
