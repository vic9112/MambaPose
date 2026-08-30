#!/usr/bin/env python3
"""CPU-only structural audit for no-PIF plus tail-aware Softplus PWL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.checkpoints import (
    authorize_manifest_candidate, authorize_tracked_config,
    build_manifest_authorized_model)
from mambapose_opt.combined_candidate import (
    install_combined_cpu_smoke, load_combined_parent_authority)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(candidate_id: str, manifest_path: Path) -> dict:
    authorized = authorize_manifest_candidate(
        REPOSITORY_ROOT, manifest_path, candidate_id)
    candidate = authorized.candidate
    authority = load_combined_parent_authority(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=manifest_path)
    config_authority = authorize_tracked_config(
        REPOSITORY_ROOT, manifest_path, candidate)
    config = config_authority.load_config()
    model = build_manifest_authorized_model(
        REPOSITORY_ROOT, manifest_path, candidate,
        config_authority=config_authority, device='cpu')
    policy = config.numeric_optimization.pwl
    summary = install_combined_cpu_smoke(
        model, roles=tuple(policy.roles),
        function_name=policy.enabled_function,
        domain=tuple(policy.domain), segments=int(policy.segments),
        saturation=policy.saturation)
    config_authority.verify()
    return {
        'schema_version': 1,
        'artifact_kind': 'combined-candidate-cpu-structural-audit',
        'candidate_id': candidate.id,
        'source_git_commit': authorized.source['git_commit'],
        'config': {
            'path': candidate.config.as_posix(),
            'sha256': _sha256(authorized.config_path),
        },
        'checkpoint': {
            'path': candidate.checkpoint.as_posix(),
            'sha256': candidate.checkpoint_sha256,
        },
        'parents': authority['parents'],
        'comparator': authority['comparator'],
        'structure': summary,
        'claim_limits': authority['claim_limits'],
        'device': 'cpu',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--candidate', default='no-pif-pwl-softplus-s-v1')
    parser.add_argument(
        '--manifest', type=Path,
        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    args = parser.parse_args()
    try:
        value = audit(args.candidate, args.manifest)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
