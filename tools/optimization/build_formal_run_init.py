#!/usr/bin/env python3
"""Create all ten immutable formal run-init documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_environment import EnvironmentAuthority  # noqa: E402
from mambapose_opt.formal_schema import load_formal_manifest  # noqa: E402
from mambapose_opt.formal_training import (  # noqa: E402
    build_all_formal_run_inits,
    write_formal_run_init,
)


ENVIRONMENT = ROOT / (
    'work_dirs/optimization/formal-stage-c/environment-authority.json')
MANIFEST = ROOT / 'optimization/formal_stage_c.json'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    if ENVIRONMENT.is_symlink() or not ENVIRONMENT.is_file():
        raise SystemExit('capture the formal environment authority first')
    environment = EnvironmentAuthority.from_dict(
        json.loads(ENVIRONMENT.read_text(encoding='utf-8')))
    manifest = load_formal_manifest(MANIFEST, repository_root=ROOT)
    inits = build_all_formal_run_inits(manifest, environment, ROOT)
    authorities = []
    for init in inits:
        destination = ROOT / init.output_root / 'run-init.json'
        if arguments.check and not destination.is_file():
            raise SystemExit(f'formal run init is missing: {init.run_id}')
        authorities.append(write_formal_run_init(init, destination))
    print(json.dumps({
        'schema_version': 1,
        'count': len(authorities),
        'run_inits': [
            {'path': authority.path, 'sha256': authority.sha256}
            for authority in authorities],
    }, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
