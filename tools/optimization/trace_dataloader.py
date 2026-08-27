#!/usr/bin/env python3
"""Repeat and record deterministic data-loader sample-order traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

from mambapose_opt.determinism import (
    build_determinism_record, deterministic_dataloader_config,
    repeated_order_hash)
from mambapose_opt.schema import CandidateSpec, load_candidate_manifest
from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.evaluation import resolve_project_asset_root
from mambapose_opt.source import clean_git_commit


def _output_path(value: str) -> Path:
    return optimization_output_path(value, repository_root=REPO_ROOT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _candidate(path: Path, identifier: str) -> CandidateSpec:
    for candidate in load_candidate_manifest(path):
        if candidate.id == identifier:
            return candidate
    raise ValueError(f'candidate not found: {identifier}')


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


def _git_commit() -> str:
    return clean_git_commit(REPO_ROOT)


def trace_candidate(
        candidate: CandidateSpec, output: Path, *, epochs: int) -> dict:
    """Validate frozen inputs before constructing any dataset and trace it."""
    commit = _git_commit()
    config_path = REPO_ROOT / candidate.config
    checkpoint_path = REPO_ROOT / candidate.checkpoint
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != candidate.checkpoint_sha256:
        raise ValueError(f'checkpoint sha256 mismatch for {candidate.id}')
    data_inventory = resolve_project_asset_root(REPO_ROOT) / 'data/inventory.json'
    config_sha256 = _sha256(config_path)
    data_inventory_sha256 = _sha256(data_inventory)
    config = Config.fromfile(config_path)
    loader = deterministic_dataloader_config(
        config.train_dataloader, seed=candidate.seed, worker_count=2)
    config.randomness = dict(seed=candidate.seed, deterministic=True)
    config.custom_imports = dict(
        imports=['mambapose_opt.determinism'], allow_failed_imports=False)
    config.train_dataloader = loader
    order_hashes = {
        epoch: repeated_order_hash(loader, seed=candidate.seed, epoch=epoch)
        for epoch in range(epochs)
    }
    record = build_determinism_record(
        seed=candidate.seed,
        worker_count=int(loader.num_workers),
        persistent_workers=bool(loader.persistent_workers),
        order_hashes=order_hashes,
        config_sha256=config_sha256,
        data_inventory_sha256=data_inventory_sha256,
        checkpoint_sha256=checkpoint_sha256,
        git_commit=commit,
    )
    envelope = {
        'schema_version': 1,
        'candidate_id': candidate.id,
        'trace': record,
        'repeat_preflight': True,
    }
    _atomic_json(output, envelope)
    return envelope


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--manifest', type=Path,
                        default=REPO_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', type=_output_path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')

    candidate = _candidate(args.manifest, args.candidate_id)
    trace_candidate(candidate, args.output, epochs=args.epochs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
