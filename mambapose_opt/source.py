"""Fail-closed source identity shared by formal optimization tools."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


_ALLOWED_IGNORED_ROOTS = frozenset({
    '.pytest_cache', '.venv', 'data', 'pretrained', 'work_dirs'})
_SOURCE_CAPABLE_SUFFIXES = frozenset({
    '.bash', '.cfg', '.conf', '.ini', '.json', '.pth', '.py', '.pyc',
    '.pyo', '.sh', '.so', '.toml', '.yaml', '.yml', '.zsh'})


def _unsafe_ignored_paths(root: Path) -> tuple[str, ...]:
    ignored = subprocess.run(
        ['git', 'ls-files', '--others', '-i', '--exclude-standard', '-z'],
        cwd=root, check=True, capture_output=True).stdout
    unsafe: list[str] = []
    for raw in ignored.split(b'\0'):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if (
                not relative.parts
                or relative.is_absolute()
                or any(part in {'.', '..'} for part in relative.parts)):
            unsafe.append(os.fsdecode(raw))
            continue
        if relative.parts[0] in _ALLOWED_IGNORED_ROOTS:
            continue
        path = root / relative
        source_capable = relative.suffix.lower() in _SOURCE_CAPABLE_SUFFIXES
        try:
            executable = path.is_file() and bool(path.stat().st_mode & 0o111)
        except OSError:
            executable = True
        if source_capable or executable:
            unsafe.append(relative.as_posix())
    return tuple(sorted(unsafe))


def clean_git_commit(repository_root: Path) -> str:
    """Return HEAD only when tracked and untracked source are both clean.

    Only exact approved ignored runtime/asset roots are exempted. Ignored
    source-capable or executable files elsewhere fail closed.
    """
    root = Path(repository_root).resolve(strict=True)
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=root, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError(
            'formal optimization requires a clean worktree with no untracked '
            'source files')
    unsafe_ignored = _unsafe_ignored_paths(root)
    if unsafe_ignored:
        raise RuntimeError(
            'formal optimization rejects ignored source-capable files: '
            + ', '.join(unsafe_ignored))
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
