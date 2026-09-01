#!/usr/bin/env python3
"""Capture or check the canonical formal Stage C Python environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_environment import (  # noqa: E402
    EnvironmentAuthority,
    validate_environment_authority,
)


DESTINATION = Path(
    'work_dirs/optimization/formal-stage-c/environment-authority.json')


def _bytes(authority: EnvironmentAuthority) -> bytes:
    return (json.dumps(authority.to_dict(), sort_keys=True, separators=(',', ':'))
            + '\n').encode('utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    destination = ROOT / DESTINATION
    if arguments.check:
        if destination.is_symlink() or not destination.is_file():
            raise SystemExit('formal environment authority is missing')
        authority = EnvironmentAuthority.from_dict(
            json.loads(destination.read_text(encoding='utf-8')))
        validate_environment_authority(authority, ROOT)
        print(authority.inventory_sha256)
        return 0
    authority = EnvironmentAuthority.capture(ROOT)
    payload = _bytes(authority)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise SystemExit('immutable environment authority already differs')
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix='.environment-authority.', suffix='.tmp',
            dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    validate_environment_authority(authority, ROOT)
    print(authority.inventory_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
