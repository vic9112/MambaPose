"""Environment contracts for nested repository Python processes."""

from __future__ import annotations

import os
from pathlib import Path


def deterministic_child_environment(repository_root: Path) -> dict[str, str]:
    """Return a deterministic child environment with local imports first."""
    environment = os.environ.copy()
    environment.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
    repo_root = str(repository_root)
    pythonpath = environment.get('PYTHONPATH')
    path_entries = [] if not pythonpath else pythonpath.split(os.pathsep)
    environment['PYTHONPATH'] = os.pathsep.join([
        repo_root,
        *[entry for entry in path_entries if entry != repo_root],
    ])
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    })
    return environment
