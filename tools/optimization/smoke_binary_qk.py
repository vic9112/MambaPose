#!/usr/bin/env python3
"""Run one real full-model Binary Q/K Stage-A training smoke."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


_REQUIRED_ENVIRONMENT = {
    'PYTHONDONTWRITEBYTECODE': '1',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
}
if __name__ == '__main__' and any(
        os.environ.get(name) != value
        for name, value in _REQUIRED_ENVIRONMENT.items()):
    environment = os.environ.copy()
    environment.update(_REQUIRED_ENVIRONMENT)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _output_root(value: str) -> Path:
    path = Path(value)
    if (
            path.is_absolute()
            or any(part in {'.', '..'} for part in path.parts)
            or path.parts[:2] != ('work_dirs', 'optimization')
            or path.name != 'smoke-stage-a'):
        raise argparse.ArgumentTypeError(
            'output-root must be a repository-relative smoke-stage-a '
            'directory under work_dirs/optimization')
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument(
        '--manifest', type=Path,
        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output-root', type=_output_root, required=True)
    parser.add_argument('--device-index', type=int, required=True)
    args = parser.parse_args()
    if args.device_index < 0:
        parser.error('--device-index must be non-negative')
    from mambapose_opt.binary_smoke import run_binary_stage_a_smoke
    artifact = run_binary_stage_a_smoke(
        repository_root=REPOSITORY_ROOT, manifest_path=args.manifest,
        candidate_id=args.candidate, output_relative=args.output_root,
        device_index=args.device_index)
    print(artifact.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
