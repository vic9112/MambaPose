#!/usr/bin/env python3
"""Create all ten immutable formal run-init documents."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_environment import EnvironmentAuthority  # noqa: E402
from mambapose_opt.formal_schema import (  # noqa: E402
    load_formal_manifest,
    load_formal_run_init,
)
from mambapose_opt.formal_training import (  # noqa: E402
    _canonical_json_bytes,
    _init_document,
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
    destinations = []
    for init in inits:
        destination = ROOT / init.output_root / 'run-init.json'
        if arguments.check and not destination.is_file():
            raise SystemExit(f'formal run init is missing: {init.run_id}')
        authorities.append(write_formal_run_init(init, destination))
        destinations.append(destination)
    for init, destination, authority in zip(
            inits, destinations, authorities):
        loaded = load_formal_run_init(destination, repository_root=ROOT)
        expected = replace(
            init, output_root_device=loaded.output_root_device,
            output_root_inode=loaded.output_root_inode)
        if loaded != expected:
            raise RuntimeError(
                f'formal run init public reload mismatch: {init.run_id}')
        relative = destination.relative_to(ROOT).as_posix()
        expected_payload = _canonical_json_bytes(_init_document(
            expected, output_identity=(
                loaded.output_root_device, loaded.output_root_inode)))
        expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
        if authority.path != relative \
                or authority.sha256 != expected_sha256:
            raise RuntimeError(
                f'formal run init public reload authority mismatch: '
                f'{init.run_id}')
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
