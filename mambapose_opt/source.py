"""Fail-closed source identity shared by formal optimization tools."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
import subprocess


_ALLOWED_IGNORED_ROOTS = frozenset({
    '.pytest_cache', '.venv', 'data', 'pretrained', 'work_dirs'})
_SOURCE_CAPABLE_SUFFIXES = frozenset({
    '.bash', '.c', '.cc', '.cpp', '.cu', '.cuh', '.dll', '.dylib', '.fish',
    '.h', '.hpp', '.ipynb', '.js', '.mjs', '.ps1', '.py', '.pyc', '.pyd',
    '.sh', '.so', '.toml', '.zsh',
})
_MMPOSE_TIMESTAMP = re.compile(r'^\d{8}_\d{6}$')
_PWL_TEMP_CONFIG = re.compile(r'^tmp[a-z0-9_]+\.py$')
_LEGACY_BINARY_RECOVERY = 'binary-qk-scaled-s-v1'
_LEGACY_BINARY_EVALUATIONS = frozenset({
    'binary-qk-scaled-s-v1', 'direct-binary-qk-scaled-s-v1'})


class SourceIntegrityError(RuntimeError, ValueError):
    """Raised when ignored runtime state can influence executable behavior."""


def _approved_shared_link(root: Path, relative: Path) -> bool:
    """Allow only the checkout's explicit environment/asset link boundaries."""
    if relative.parts in {('.venv',), ('data',), ('pretrained',),
                          ('work_dirs', 'reproduction'),
                          ('work_dirs', 'optimization', 'prior-stage-b')}:
        return (root / relative).is_symlink()
    return False


def _candidate_authority(root: Path) -> dict[tuple[str, str, str], dict]:
    """Return exact candidate identities from tracked candidate manifests."""
    manifest_root = root / 'optimization'
    manifests = tuple(sorted(manifest_root.glob('*candidates.json')))
    if (
            not manifests
            or manifest_root / 'candidates.json' not in manifests
            or any(path.is_symlink() or not path.is_file()
                   for path in manifests)):
        return {}
    result: dict[tuple[str, str, str], dict] = {}
    for manifest in manifests:
        try:
            document = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        rows = document.get('candidates') \
            if isinstance(document, dict) else None
        if not isinstance(rows, list):
            return {}
        for row in rows:
            if (
                    not isinstance(row, dict)
                    or not isinstance(row.get('route'), str)
                    or not isinstance(row.get('id'), str)
                    or isinstance(row.get('seed'), bool)
                    or not isinstance(row.get('seed'), int)):
                return {}
            key = (row['route'], row['id'], str(row['seed']))
            if key in result:
                return {}
            result[key] = row
    return result


def _formal_config_authority(root: Path) -> dict[tuple[str, ...], str]:
    """Map each tracked formal output root to its source config basename."""
    try:
        document = json.loads(
            (root / 'optimization/formal_stage_c.json').read_text(
                encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    rows = document.get('runs') if isinstance(document, dict) else None
    if not isinstance(rows, list):
        return {}
    result: dict[tuple[str, ...], str] = {}
    for row in rows:
        if (
                not isinstance(row, dict)
                or not isinstance(row.get('output_root'), str)
                or not isinstance(row.get('config'), str)):
            return {}
        output = Path(row['output_root'])
        config = Path(row['config'])
        if (
                output.is_absolute()
                or output.parts[:3] != (
                    'work_dirs', 'optimization', 'formal-stage-c')
                or len(output.parts) != 4
                or config.is_absolute()
                or any(part in {'.', '..'} for part in config.parts)
                or not config.name.endswith('.py')):
            return {}
        if output.parts in result:
            return {}
        result[output.parts] = config.name
    return result


def _approved_generated_config(root: Path, relative: Path) -> bool:
    """Allow only exact controller/MMPose-generated config snapshots."""
    if (
            len(relative.parts) < 4
            or relative.parts[:2] != ('work_dirs', 'optimization')):
        return False
    candidates = _candidate_authority(root)
    allowed_parent_by_name = {
        'resolved-flip.py': 'evaluate',
        'resolved-no-flip.py': 'evaluate',
        'resolved-runtime.py': 'convert',
        'resolved-train.py': 'train',
        'resolved-binary-qk-recovery.py': 'recovery',
    }
    parts = relative.parts
    if len(parts) == 7:
        candidate = candidates.get(parts[2:5])
        if (
                candidate is not None
                and parts[5] == allowed_parent_by_name.get(parts[6])):
            return True

    if (
            parts == (
                'work_dirs', 'optimization', _LEGACY_BINARY_RECOVERY,
                'recovery', 'resolved-binary-qk-recovery.py')):
        return True

    # Stage-B evaluation: the controller materializes the first file and
    # MMPose copies it into a timestamped visualization directory.
    if (
            len(parts) >= 8
            and candidates.get(parts[2:5]) is not None
            and parts[5] == 'evaluate'
            and parts[6] in {'mmpose-flip', 'mmpose-no-flip'}):
        candidate = candidates[parts[2:5]]
        mode = parts[6].removeprefix('mmpose-')
        if len(parts) == 8 and parts[7] == f'resolved-{mode}.py':
            return True
        if (
                len(parts) == 8
                and candidate.get('kind') == 'pwl'
                and isinstance(candidate.get('features'), dict)
                and candidate['features'].get('numeric_kind') == 'pwl'
                and _PWL_TEMP_CONFIG.fullmatch(parts[7])):
            return True
        if (
                len(parts) == 10
                and _MMPOSE_TIMESTAMP.fullmatch(parts[7])
                and parts[8:] == ('vis_data', 'config.py')):
            return True

    # Binary recovery and the two measured direct/post-QAT evaluation layouts.
    if parts[2] == _LEGACY_BINARY_RECOVERY:
        if (
                len(parts) == 5
                and parts[3:] == ('recovery', 'binary_qk_s_v1.py')):
            return True
        if (
                len(parts) == 7
                and parts[3] == 'recovery'
                and _MMPOSE_TIMESTAMP.fullmatch(parts[4])
                and parts[5:] == ('vis_data', 'config.py')):
            return True
    if parts[2] in _LEGACY_BINARY_EVALUATIONS:
        direct = parts[2].startswith('direct-')
        binary_prefix = direct or (len(parts) > 3 and parts[3] == 'post-qat')
        seed_index = 3 if direct else 4
        if (
                binary_prefix
                and len(parts) > seed_index + 3
                and parts[seed_index] == 'seed-0'
                and parts[seed_index + 1] in {'flip', 'no-flip'}
                and parts[seed_index + 2] == 'mmpose'):
            tail = parts[seed_index + 3:]
            if tail == ('binary_qk_s_v1_deploy.py',):
                return True
            if (
                    len(tail) == 3
                    and _MMPOSE_TIMESTAMP.fullmatch(tail[0])
                    and tail[1:] == ('vis_data', 'config.py')):
                return True

    # Formal Stage C uses one seed-bound training snapshot plus MMPose's exact
    # training/final-evaluation copies.
    for output_parts, config_name in _formal_config_authority(root).items():
        if parts[:len(output_parts)] == output_parts:
            tail = parts[len(output_parts):]
            stem = Path(config_name).stem
            if tail == (config_name,):
                return True
            if (
                    len(tail) == 3
                    and _MMPOSE_TIMESTAMP.fullmatch(tail[0])
                    and tail[1:] == ('vis_data', 'config.py')):
                return True
            if (
                    len(tail) >= 4
                    and tail[0] == 'final-evaluation'
                    and tail[1] in {'flip', 'no-flip'}
                    and tail[2] == 'mmpose'):
                mode = tail[1]
                evaluation_tail = tail[3:]
                if evaluation_tail == (f'{stem}_{mode}.py',):
                    return True
                if (
                        len(evaluation_tail) == 3
                        and _MMPOSE_TIMESTAMP.fullmatch(evaluation_tail[0])
                        and evaluation_tail[1:] == ('vis_data', 'config.py')):
                    return True
    return False


def _ignored_entry_is_source_capable(root: Path, relative: Path) -> bool:
    lexical = root / relative
    if lexical.is_symlink():
        return not _approved_shared_link(root, relative)
    if _approved_generated_config(root, relative):
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

    Only exact approved ignored runtime/asset roots are exempted. Every ignored
    entry outside those component-exact roots fails closed.
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
