#!/usr/bin/env python3
"""Acquire and inventory all paper reproduction assets without prompts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_repro.download import (
    download_verified, extract_archive, sha256_file)
from mambapose_repro.preflight import load_sources


SOURCES_PATH = REPO_ROOT / 'reproduction/sources.json'
INVENTORY_PATH = REPO_ROOT / 'data/inventory.json'
PRETRAINED_INVENTORY_PATH = REPO_ROOT / 'pretrained/inventory.json'


def _source_status(source: dict) -> str:
    required = [REPO_ROOT / path for path in source['required_paths']]
    return 'already_present' if all(path.exists() for path in required) else 'missing'


def resolve_sources() -> list[dict]:
    return [{
        'id': source['id'],
        'url': source['url'],
        'source_class': source['source_class'],
        'license_status': source['license_status'],
        'auth': source['auth'],
        'status': _source_status(source),
    } for source in load_sources(SOURCES_PATH)]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def acquire_all() -> list[dict]:
    inventory: list[dict] = []
    for source in load_sources(SOURCES_PATH):
        print(f'asset={source["id"]} status={_source_status(source)}',
              flush=True)
        if source['archive'] == 'file':
            target = REPO_ROOT / source['target']
            result = download_verified(
                source['url'], target,
                expected_sha256=source['sha256'],
                expected_bytes=source['expected_bytes'])
        else:
            archive = REPO_ROOT / source['download_path']
            result = download_verified(
                source['url'], archive,
                expected_sha256=source['sha256'],
                expected_bytes=source['expected_bytes'])
            if _source_status(source) != 'already_present':
                print(f'asset={source["id"]} extracting={archive}', flush=True)
                extracted = extract_archive(
                    archive, REPO_ROOT / source['extract_to'])
                print(
                    f'asset={source["id"]} extracted_files={extracted}',
                    flush=True)
        if _source_status(source) != 'already_present':
            raise RuntimeError(
                f'asset {source["id"]} did not publish all required paths')
        inventory.append({
            'id': source['id'],
            'url': source['url'],
            'source_class': source['source_class'],
            'license_status': source['license_status'],
            'auth': source['auth'],
            'bytes': result.bytes,
            'sha256': result.sha256,
            'path': str(result.path.relative_to(REPO_ROOT)),
            'required_paths': source['required_paths'],
        })
    record = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'sources_manifest_sha256': sha256_file(SOURCES_PATH),
        'assets': inventory,
    }
    _atomic_json(INVENTORY_PATH, record)
    pretrained_assets = [
        asset for asset in inventory if asset['id'] == 'vmamba-tiny-pretrained'
    ]
    _atomic_json(PRETRAINED_INVENTORY_PATH, {
        'schema_version': 1,
        'created_at': record['created_at'],
        'assets': pretrained_assets,
    })
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--download-unattended', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.download_unattended:
        value = acquire_all()
    else:
        value = resolve_sources()
    rendered = json.dumps(value, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        _atomic_json(args.json, value)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
