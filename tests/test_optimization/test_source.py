from pathlib import Path
import subprocess

import pytest


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
    (tmp_path / 'data/asset.py').write_text('external asset payload\n')

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
