#!/usr/bin/env python3
"""Standard-library-only process-first launcher for formal Stage C."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mambapose_opt.formal_environment import (  # noqa: E402
    EnvironmentAuthority,
    apply_required_process_environment,
    validate_environment_authority,
)


_WORKERS = {
    'build-init': 'tools.optimization.build_formal_run_init',
    'trace': 'tools.optimization.trace_formal_order',
    'preflight': 'tools.optimization.preflight_formal_model',
    'train': 'tools.optimization.train_formal_candidate',
}
_AUTHORITY = ROOT / (
    'work_dirs/optimization/formal-stage-c/environment-authority.json')


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _WORKERS:
        raise SystemExit('usage: formal_stage_c_entrypoint.py '
                         '{build-init,trace,preflight,train} '
                         '[worker arguments]')
    stage = sys.argv[1]
    if _AUTHORITY.exists():
        if _AUTHORITY.is_symlink() or not _AUTHORITY.is_file():
            raise SystemExit('formal environment authority path is unsafe')
        authority = EnvironmentAuthority.from_dict(
            json.loads(_AUTHORITY.read_text(encoding='utf-8')))
    elif stage == 'trace':
        # The trace establishes the environment document which is frozen before
        # run-init construction; model/training stages may never self-authorize.
        authority = EnvironmentAuthority.capture(ROOT)
    else:
        raise SystemExit('formal environment authority is required')
    validate_environment_authority(authority, ROOT)
    environment = apply_required_process_environment(authority)
    interpreter = ROOT / authority.interpreter_path
    if interpreter.resolve(strict=True) != Path(
            authority.interpreter_real_path):
        raise SystemExit('formal interpreter authority changed')
    module = _WORKERS[stage]
    arguments = [str(interpreter), '-B', '-m', module, *sys.argv[2:]]
    os.execve(str(interpreter), arguments, environment)
    return 70


if __name__ == '__main__':
    raise SystemExit(main())
