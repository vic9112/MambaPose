#!/usr/bin/env python3
"""Emit repeat sampler-order evidence from fresh child processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_schema import load_formal_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--repeat', type=int, default=2)
    arguments = parser.parse_args()
    if arguments.manifest != 'optimization/formal_stage_c.json':
        raise SystemExit('trace manifest path is not canonical')
    if arguments.repeat != 2 or not arguments.cpu_only:
        raise SystemExit('formal trace requires exactly two CPU-only processes')
    manifest = load_formal_manifest(
        ROOT / arguments.manifest, repository_root=ROOT)
    matches = tuple(
        spec for spec in manifest.runs if spec.run_id == arguments.run_id)
    if len(matches) != 1:
        raise SystemExit('formal trace run id is invalid')
    spec = matches[0]
    # Root determinism precedes config, registry, and dataset construction.
    from mambapose_opt.formal_determinism import configure_root_determinism
    configure_root_determinism(spec.seed)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS
    from mmpose.utils import register_all_modules
    register_all_modules(init_default_scope=False)
    config = Config.fromfile(ROOT / spec.config)
    init_default_scope(config.get('default_scope', 'mmpose'))
    dataset = DATASETS.build(config.train_dataloader.dataset)
    dataset_size = len(dataset)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('formal trace initialized CUDA before child replay')
    if dataset_size < manifest.data_authority['train_image_corpus'].image_count:
        raise SystemExit('formal train dataset is smaller than its image corpus')
    traces = []
    for _index in range(arguments.repeat):
        command = [
            sys.executable, '-B', '-m', 'mambapose_opt.formal_determinism',
            '--seed', str(spec.seed), '--epochs', str(arguments.epochs),
            '--dataset-size', str(dataset_size),
        ]
        result = subprocess.run(
            command, cwd=ROOT, env=dict(os.environ), check=True,
            text=True, capture_output=True)
        traces.append(json.loads(result.stdout))
    if traces[0]['pid'] == traces[1]['pid'] \
            or traces[0]['order_hashes'] != traces[1]['order_hashes'] \
            or any(trace['cuda_initialized'] for trace in traces):
        raise SystemExit('formal order trace did not reproduce')
    print(json.dumps({
        'schema_version': 1,
        'run_id': spec.run_id,
        'seed': spec.seed,
        'epochs': arguments.epochs,
        'dataset_size': dataset_size,
        'worker_cuda_initialized': torch.cuda.is_initialized(),
        'traces': traces,
    }, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
