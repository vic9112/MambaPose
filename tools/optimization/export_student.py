#!/usr/bin/env python3
"""Export a hash-validated distiller checkpoint as a plain student model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mmengine.config import Config

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.source import clean_git_commit
from mmpose.models.distillers.mambapose_heatmap_distiller import (
    MambaPoseHeatmapDistiller, export_student_checkpoint,
    load_hash_validated_checkpoint)
from mmpose.registry import MODELS
from mmpose.utils import register_all_modules

def _relative_existing_path(value: str, *, prefix: tuple[str, ...]) -> Path:
    path = Path(value)
    if (
            path.is_absolute()
            or path.parts[:len(prefix)] != prefix
            or any(part in {'.', '..'} for part in path.parts)):
        raise argparse.ArgumentTypeError(
            f'path must be repository-relative under {"/".join(prefix)}')
    resolved = (REPOSITORY_ROOT / path).resolve(strict=True)
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise argparse.ArgumentTypeError('path escapes repository root') from error
    if not resolved.is_file():
        raise argparse.ArgumentTypeError('path must name a file')
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    register_all_modules()
    config_path = _relative_existing_path(
        args.config, prefix=('configs', 'optimization', 'accuracy_first'))
    checkpoint_path = _relative_existing_path(
        args.checkpoint, prefix=('work_dirs', 'optimization'))
    output_path = optimization_output_path(
        args.output, repository_root=REPOSITORY_ROOT)
    clean_git_commit(REPOSITORY_ROOT)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite output: {args.output}')
    config = Config.fromfile(config_path)
    model = MODELS.build(config.model)
    if not isinstance(model, MambaPoseHeatmapDistiller):
        raise TypeError('config must build MambaPoseHeatmapDistiller')
    load_hash_validated_checkpoint(
        model,
        checkpoint_path,
        expected_sha256=args.checkpoint_sha256,
        strict=True)
    export_student_checkpoint(model, output_path)
    print(json.dumps({
        'schema_version': 1,
        'source_checkpoint': args.checkpoint,
        'source_checkpoint_sha256': args.checkpoint_sha256,
        'student_checkpoint': output_path.relative_to(REPOSITORY_ROOT).as_posix(),
        'student_checkpoint_sha256': _sha256(output_path),
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
