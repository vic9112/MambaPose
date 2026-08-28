"""Lexical path admission shared by PWL public artifacts and CLIs."""

from __future__ import annotations

from pathlib import Path


def canonical_path(
        value: object, *, label: str, allow_absolute: bool = False) -> Path:
    """Return a canonical POSIX path without first normalizing aliases."""
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'{label} path is invalid')
    absolute = value.startswith('/')
    if absolute and not allow_absolute:
        raise ValueError(f'{label} path must be repository-relative')
    components = value.split('/')
    if absolute:
        components = components[1:]
    if any(component in {'', '.', '..'} for component in components):
        raise ValueError(f'{label} path is unsafe')
    path = Path(value)
    if path.is_absolute() != absolute or path.as_posix() != value:
        raise ValueError(f'{label} path is not canonical')
    return path


def canonical_relative_path(value: object, *, label: str) -> Path:
    """Return a canonical POSIX relative path without normalizing aliases."""
    return canonical_path(value, label=label, allow_absolute=False)


__all__ = ['canonical_path', 'canonical_relative_path']
