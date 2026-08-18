#!/usr/bin/env python3
"""CLI for the non-controlling MambaPose campaign observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_repro.observe import observe
from mambapose_repro.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument(
        '--campaign-dir', type=Path,
        default=REPO_ROOT / 'work_dirs/reproduction')
    args = parser.parse_args()
    manifest = load_manifest(REPO_ROOT / 'reproduction/manifest.json')
    status = observe(
        args.campaign_dir,
        expected_run_ids=(run.id for run in manifest.runs))
    print(json.dumps(status, indent=2, sort_keys=True))
    if args.check and status['health'] in {'stalled', 'failed'}:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
