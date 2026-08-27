"""Fail-closed source identity shared by formal optimization tools."""

from __future__ import annotations

from pathlib import Path
import subprocess


def clean_git_commit(repository_root: Path) -> str:
    """Return HEAD only when tracked and untracked source are both clean.

    Git-ignored runtime and external-data paths are intentionally absent from
    porcelain output and therefore do not weaken source cleanliness.
    """
    root = Path(repository_root).resolve(strict=True)
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=root, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError(
            'formal optimization requires a clean worktree with no untracked '
            'source files')
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
