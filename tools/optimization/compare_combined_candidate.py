#!/usr/bin/env python3
"""Build the direct full-S-V1 versus combined evaluation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


sys.dont_write_bytecode = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.combined_candidate import (
    load_combined_parent_authority, read_bound_workspace_artifact)
from mambapose_opt.combined_comparison import (
    build_combined_comparison, validate_combined_comparison_artifact)
from mambapose_opt.schema import load_candidate_manifest


def _candidate(manifest: Path, candidate_id: str):
    matches = tuple(
        item for item in load_candidate_manifest(manifest)
        if item.id == candidate_id)
    if len(matches) != 1:
        raise ValueError(f'combined candidate not found: {candidate_id}')
    candidate = matches[0]
    if candidate.features.get('numeric_kind') != 'pwl-combined':
        raise ValueError('compare tool accepts only the combined candidate')
    return candidate


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def compare(candidate_id: str, *, manifest_path: Path, output: Path) -> dict:
    candidate = _candidate(manifest_path, candidate_id)
    expected_output = (
        REPOSITORY_ROOT / 'work_dirs/optimization' / candidate.route /
        candidate.id / str(candidate.seed) / 'compare/compare.json')
    if output.absolute() != expected_output.absolute():
        raise ValueError('combined comparison output path is not canonical')
    authority = load_combined_parent_authority(
        candidate, repository_root=REPOSITORY_ROOT,
        manifest_path=manifest_path)
    full_reference = authority['comparator']['evaluation_artifact']
    full = read_bound_workspace_artifact(
        full_reference, checkout_root=REPOSITORY_ROOT)
    combined_path = output.parent.parent / 'evaluate/evaluate.json'
    payload = combined_path.read_bytes()
    combined_reference = {
        'path': combined_path.relative_to(REPOSITORY_ROOT).as_posix(),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }
    combined = read_bound_workspace_artifact(
        combined_reference, checkout_root=REPOSITORY_ROOT)
    value = build_combined_comparison(
        full, combined, full_reference=full_reference,
        combined_reference=combined_reference, candidate_id=candidate.id)
    _atomic_json(output, value)
    validate_combined_comparison_artifact(
        value, candidate=candidate, artifact_path=output,
        repository_root=REPOSITORY_ROOT, manifest_path=manifest_path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument(
        '--manifest', type=Path,
        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument(
        '--output', type=lambda value: optimization_output_path(
            value, repository_root=REPOSITORY_ROOT), required=True)
    args = parser.parse_args()
    compare(args.candidate_id, manifest_path=args.manifest, output=args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
