"""Containment-checked optimization artifact paths shared by all CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


_ARTIFACT_PARTS = ('work_dirs', 'optimization')
_MESSAGE = 'output must be repository-relative under work_dirs/optimization'


def optimization_output_path(
        value: str, *, repository_root: Path) -> Path:
    """Resolve an output only when both artifact root and target stay in repo."""
    path = Path(value)
    if (
            path.is_absolute()
            or any(part in {'.', '..'} for part in path.parts)
            or path.parts[:2] != _ARTIFACT_PARTS):
        raise argparse.ArgumentTypeError(_MESSAGE)
    try:
        root = Path(repository_root).resolve(strict=False)
        lexical_artifact_root = root / Path(*_ARTIFACT_PARTS)
        artifact_root = lexical_artifact_root.resolve(strict=False)
        if lexical_artifact_root.absolute() != artifact_root:
            raise ValueError('optimization artifact root must not be symlinked')
        effective = (root / path).resolve(strict=False)
        artifact_root.relative_to(root)
        effective.relative_to(artifact_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(_MESSAGE) from error
    return effective
