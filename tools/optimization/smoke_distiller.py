#!/usr/bin/env python3
"""Run one hash-bound real COCO batch through the MambaPose distiller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.artifacts import optimization_output_path


def _output_root(value: str) -> Path:
    path = Path(value)
    optimization_output_path(value, repository_root=REPOSITORY_ROOT)
    if path.parts[:3] != (
            'work_dirs', 'optimization', 'accuracy-first') or len(path.parts) < 4:
        raise argparse.ArgumentTypeError(
            'output must be a candidate directory under '
            'work_dirs/optimization/accuracy-first')
    return path


def _config(value: str) -> Path:
    path = Path(value)
    if (
            path.is_absolute()
            or path.parts[:3] != (
                'configs', 'optimization', 'accuracy_first')
            or any(part in {'.', '..'} for part in path.parts)):
        raise argparse.ArgumentTypeError(
            'config must be repository-relative under '
            'configs/optimization/accuracy_first')
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=_config, required=True)
    parser.add_argument('--output-root', type=_output_root, required=True)
    parser.add_argument('--device-index', type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device_index < 0:
        raise ValueError('--device-index must be non-negative')
    # Importing the execution harness does not import torch. Its production
    # model imports occur only after clean-source/data/hash preflight and lease.
    from mambapose_opt.distill_smoke import run_distill_smoke

    artifact = run_distill_smoke(
        repository_root=REPOSITORY_ROOT,
        config_path=args.config,
        output_relative=args.output_root,
        device_index=args.device_index)
    print(artifact.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
