#!/usr/bin/env python3
"""Print a read-only optimization campaign health observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_opt.observe import observe
from tools.optimization.run_campaign import _validated_gpu_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--campaign-root', type=Path,
        default=REPO_ROOT / 'work_dirs/optimization')
    parser.add_argument('--heartbeat-max-age', type=float, default=180.0)
    parser.add_argument('--gpu-lock-path', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        _, gpu_lock_path = _validated_gpu_lock(args.gpu_lock_path)
    except Exception as error:
        print(f'cannot derive canonical GPU lock: {error}', file=sys.stderr)
        return 1
    status = observe(
        args.campaign_root, args.heartbeat_max_age,
        gpu_lock_path=gpu_lock_path)
    print(json.dumps(status, indent=2, sort_keys=True))
    return (
        1 if args.check and status['health'] in {'failed', 'stalled'}
        else 0)


if __name__ == '__main__':
    raise SystemExit(main())
