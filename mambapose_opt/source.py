"""Fail-closed source identity shared by formal optimization tools."""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import subprocess


_ALLOWED_IGNORED_ROOTS = frozenset({
    '.pytest_cache', '.venv', 'data', 'pretrained', 'work_dirs'})
_SOURCE_CAPABLE_SUFFIXES = frozenset({
    '.bash', '.c', '.cc', '.cpp', '.cu', '.cuh', '.dll', '.dylib', '.fish',
    '.h', '.hpp', '.ipynb', '.js', '.mjs', '.ps1', '.py', '.pyc', '.pyd',
    '.sh', '.so', '.toml', '.zsh',
})


class SourceIntegrityError(RuntimeError, ValueError):
    """Raised when ignored runtime state can influence executable behavior."""


def _approved_shared_link(root: Path, relative: Path) -> bool:
    """Allow only the checkout's explicit environment/asset link boundaries."""
    if relative.parts in {('.venv',), ('data',), ('pretrained',),
                          ('work_dirs', 'reproduction'),
                          ('work_dirs', 'optimization', 'prior-stage-b')}:
        return (root / relative).is_symlink()
    return False


def _approved_generated_config(relative: Path) -> bool:
    """Allow only canonical, authority-checked generated runtime configs."""
    if (
            len(relative.parts) < 4
            or relative.parts[:2] != ('work_dirs', 'optimization')):
        return False
    allowed_parent_by_name = {
        'resolved-flip.py': 'evaluate',
        'resolved-no-flip.py': 'evaluate',
        'resolved-runtime.py': 'convert',
        'resolved-train.py': 'train',
        'resolved-binary-qk-recovery.py': 'recovery',
    }
    return relative.parent.name == allowed_parent_by_name.get(relative.name)


def _ignored_entry_is_source_capable(root: Path, relative: Path) -> bool:
    lexical = root / relative
    if lexical.is_symlink():
        return not _approved_shared_link(root, relative)
    if _approved_generated_config(relative):
        return False
    if relative.suffix.lower() in _SOURCE_CAPABLE_SUFFIXES:
        return True
    try:
        return lexical.is_file() and bool(lexical.stat().st_mode & 0o111)
    except OSError:
        return True


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
            if _ignored_entry_is_source_capable(root, relative):
                unsafe.append(relative.as_posix())
            continue
        unsafe.append(relative.as_posix())
    return tuple(sorted(unsafe))


def clean_git_commit(repository_root: Path) -> str:
    """Return HEAD only when tracked and untracked source are both clean.

    Only exact approved ignored runtime/asset roots are exempted. Ignored
    Every ignored entry outside those component-exact roots fails closed.
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
        raise SourceIntegrityError(
            'formal optimization rejects noncanonical ignored source-capable, '
            'executable, or symlink entries: '
            + ', '.join(unsafe_ignored))
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def sha256_file(path: Path | str) -> str:
    """Hash one regular file without deserializing it."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'hash input must be a regular non-symlink file: {source}')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tracked_file_binding(
        repository_root: Path, path: Path | str, *, git_commit: str,
        ) -> dict[str, str]:
    """Bind a clean worktree file to the exact blob at ``git_commit``."""
    root = Path(repository_root).resolve(strict=True)
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError('tracked source path must be repository-relative')
    if not candidate.parts or any(part in {'.', '..'} for part in candidate.parts):
        raise ValueError('tracked source path must be safe and repository-relative')
    effective = root / candidate
    if effective.is_symlink() or not effective.is_file():
        raise ValueError('tracked source path must name a regular source file')
    try:
        blob = subprocess.run(
            ['git', 'show', f'{git_commit}:{candidate.as_posix()}'], cwd=root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('source path is not tracked at the clean commit') from error
    actual = effective.read_bytes()
    if actual != blob:
        raise ValueError('source file differs from its clean commit blob')
    return {
        'path': candidate.as_posix(),
        'sha256': hashlib.sha256(blob).hexdigest(),
    }
