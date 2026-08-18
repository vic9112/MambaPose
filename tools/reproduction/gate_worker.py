#!/usr/bin/env python3
"""Isolated one-process real-data train/test worker for admission gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from mmengine.config import Config
from mmengine.runner import Runner
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _common(config_path: Path, work_dir: Path) -> Config:
    config = Config.fromfile(config_path)
    config.work_dir = str(work_dir)
    config.launcher = 'none'
    config.randomness = dict(seed=0, deterministic=False)
    if 'preprocess_cfg' in config:
        config.model.setdefault(
            'data_preprocessor', config.get('preprocess_cfg', {}))
    return config


def _train(args: argparse.Namespace) -> dict[str, Any]:
    config = _common(args.config, args.work_dir)
    config.train_dataloader.batch_size = args.batch_size
    config.train_dataloader.num_workers = 0
    config.train_dataloader.persistent_workers = False
    config.train_dataloader.dataset.indices = args.batch_size
    config.train_cfg.max_epochs = args.epochs
    config.train_cfg.val_interval = args.epochs + 1
    config.val_cfg = None
    config.val_dataloader = None
    config.val_evaluator = None
    config.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1 if args.save_checkpoint else -1,
        max_keep_ckpts=2,
        save_last=args.save_checkpoint)
    config.optim_wrapper.pop('accumulative_counts', None)
    if args.resume is not None:
        config.resume = True
        config.load_from = str(args.resume)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = Runner.from_cfg(config)
    runner.train()
    checkpoints = sorted(
        args.work_dir.glob('epoch_*.pth'),
        key=lambda path: int(path.stem.split('_')[-1]))
    result = {
        'status': 'passed',
        'mode': 'train',
        'config': str(args.config),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'resumed_from': str(args.resume) if args.resume else None,
        'checkpoint': str(checkpoints[-1]) if checkpoints else None,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'device_total_bytes': torch.cuda.get_device_properties(0).total_memory,
    }
    if args.save_checkpoint and not checkpoints:
        raise RuntimeError('training gate did not create an epoch checkpoint')
    return result


def _test(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError('--checkpoint is required for test mode')
    config = _common(args.config, args.work_dir)
    config.test_dataloader.batch_size = args.batch_size
    config.test_dataloader.num_workers = 0
    config.test_dataloader.persistent_workers = False
    config.test_dataloader.dataset.indices = args.batch_size
    config.load_from = str(args.checkpoint)
    config.resume = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = Runner.from_cfg(config)
    metrics = runner.test()
    if not _finite(metrics):
        raise RuntimeError(f'test gate produced non-finite metrics: {metrics}')
    return {
        'status': 'passed',
        'mode': 'test',
        'config': str(args.config),
        'batch_size': args.batch_size,
        'checkpoint': str(args.checkpoint),
        'metrics': metrics,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'device_total_bytes': torch.cuda.get_device_properties(0).total_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'test'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    try:
        result = _train(args) if args.mode == 'train' else _test(args)
        result['started_at'] = started.isoformat()
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        _atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if 'out of memory' in str(error).lower():
            result = {
                'status': 'oom',
                'mode': args.mode,
                'config': str(args.config),
                'batch_size': args.batch_size,
                'error': f'{type(error).__name__}: {error}',
                'started_at': started.isoformat(),
                'finished_at': datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 42
        traceback.print_exc()
        return 78
    except Exception:
        traceback.print_exc()
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
