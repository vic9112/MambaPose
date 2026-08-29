#!/usr/bin/env python3
"""Prepare or launch the bounded scaled-Binary-Q/K S-V1 recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.binary_recovery import (
    DEPLOY_CONFIG_PATH, BoundBinaryRecoveryRunner,
    build_binary_recovery_readiness,
    export_and_publish_binary_recovery_completion,
    load_binary_recovery_completion, load_bound_config,
    load_training_resume_checkpoint,
    write_binary_recovery_launch_metadata)


def _selected_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob('best_*.pth'))
    if len(best) == 1:
        return best[0]
    final = work_dir / 'epoch_60.pth'
    if not best and final.is_file():
        return final
    raise ValueError(
        'recovery must produce exactly one best checkpoint or epoch_60.pth')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--work-dir', type=Path,
        default=REPOSITORY_ROOT
        / 'work_dirs/optimization/binary-qk-s-v1/recovery')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path, metavar='PATH')
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    if (args.resume is None) != (args.resume_sha256 is None):
        parser.error('--resume PATH and --resume-sha256 must be provided together')
    if args.prepare_only and args.resume is not None:
        parser.error('--prepare-only cannot resume training')

    work_dir = args.work_dir.resolve()
    readiness = build_binary_recovery_readiness(
        REPOSITORY_ROOT, work_dir)
    if args.prepare_only:
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return 0

    config = load_bound_config(
        REPOSITORY_ROOT, readiness['config'], label='recovery config')
    config.work_dir = str(work_dir)
    config.resume = False
    config.load_from = None
    try:
        completion = load_binary_recovery_completion(
            work_dir, repository_root=REPOSITORY_ROOT,
            deployment_binding=readiness['deployment']['config'])
    except ValueError:
        completion = None
    if completion is not None:
        print(json.dumps(completion.report, indent=2, sort_keys=True))
        return 0
    write_binary_recovery_launch_metadata(
        work_dir, config_text=config.pretty_text, readiness=readiness)

    runner = BoundBinaryRecoveryRunner.from_cfg(config)
    if args.resume is not None:
        resume_path = args.resume.resolve(strict=True)
        if resume_path.parent != work_dir:
            raise ValueError('resume checkpoint must be directly under work-dir')
        runner.bind_resume_checkpoint(load_training_resume_checkpoint(
            resume_path, args.resume_sha256,
            expected_config=config.pretty_text,
            expected_state_dict=runner.model.state_dict()))
    runner.train()

    completion = export_and_publish_binary_recovery_completion(
        _selected_checkpoint(work_dir),
        work_dir / 'binary_qk_s_v1_student.pth',
        deployment_config=REPOSITORY_ROOT / DEPLOY_CONFIG_PATH,
        repository_root=REPOSITORY_ROOT,
        deployment_binding=readiness['deployment']['config'])
    print(json.dumps(completion.report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
