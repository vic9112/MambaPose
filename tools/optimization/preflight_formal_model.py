#!/usr/bin/env python3
"""Run one authenticated full-model repeatability preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_training import (  # noqa: E402
    repeatability_result_to_dict,
    run_formal_model_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-seed', type=int, required=True)
    parser.add_argument('--role', choices=('baseline', 'no_pif'), required=True)
    parser.add_argument('--replay', type=int, choices=(1, 2), required=True)
    arguments = parser.parse_args()
    result = run_formal_model_preflight(arguments.pair_seed, arguments.role)
    stem = 'full' if arguments.role == 'baseline' else 'no-pif'
    destination = (
        ROOT / 'work_dirs/optimization/formal-stage-c'
        / f'{stem}-seed{arguments.pair_seed}' / 'preflight'
        / f'replay-{arguments.replay}.json')
    payload = (json.dumps(
        repeatability_result_to_dict(result), sort_keys=True,
        separators=(',', ':')) + '\n').encode('utf-8')
    from mambapose_opt.formal_training import write_immutable_artifact
    write_immutable_artifact(destination, payload, ROOT)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
