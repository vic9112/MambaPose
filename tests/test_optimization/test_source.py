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

    assert len(clean_git_commit(tmp_path)) == 40

    source = tmp_path / 'mambapose_opt/untracked_source.py'
    source.parent.mkdir()
    source.write_text('VALUE = 2\n')
    with pytest.raises(RuntimeError, match='untracked|clean'):
        clean_git_commit(tmp_path)
