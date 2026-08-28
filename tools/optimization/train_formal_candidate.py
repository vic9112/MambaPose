#!/usr/bin/env python3
"""Train one authenticated formal Stage C candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_training import train_formal_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', required=True)
    parser.add_argument('--resume')
    arguments = parser.parse_args()
    result = train_formal_candidate(
        Path(arguments.init),
        None if arguments.resume is None else Path(arguments.resume))
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
