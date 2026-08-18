import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_has_unique_stable_experiment_ids():
    from mambapose_repro.manifest import load_manifest

    manifest = load_manifest(ROOT / 'reproduction/manifest.json')
    ids = [run.id for run in manifest.runs]
    assert len(ids) == len(set(ids)) == 11
    assert ids[:5] == [
        'coco-s-v1', 'coco-s-v2', 'coco-b',
        'crowdpose-s-v1', 'crowdpose-s-v2'
    ]
    for run in manifest.runs:
        assert not run.config.is_absolute()
        assert not run.work_dir.is_absolute()
        assert '..' not in run.config.parts
        assert '..' not in run.work_dir.parts


def test_manifest_rejects_unknown_fields_and_duplicate_ids(tmp_path):
    from mambapose_repro.manifest import ManifestError, load_manifest

    data = json.loads((ROOT / 'reproduction/manifest.json').read_text())
    data['unexpected'] = True
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match='unknown fields'):
        load_manifest(bad)

    data.pop('unexpected')
    data['runs'].append(data['runs'][0])
    bad.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match='duplicate run id'):
        load_manifest(bad)


def test_manifest_rejects_secret_fields_and_parent_paths(tmp_path):
    from mambapose_repro.manifest import ManifestError, load_manifest

    data = json.loads((ROOT / 'reproduction/manifest.json').read_text())
    data['runs'][0]['token'] = 'secret'
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match='unknown fields'):
        load_manifest(bad)

    data['runs'][0].pop('token')
    data['runs'][0]['work_dir'] = '../escape'
    bad.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match='relative.*parent'):
        load_manifest(bad)


def test_atomic_state_and_append_only_events(tmp_path):
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    state = store.transition('coco-s-v1', 'running', attempt=1)
    persisted = json.loads((tmp_path / 'state.json').read_text())
    assert persisted['stage'] == 'running'
    assert persisted['generation'] == state['generation'] == 1
    assert persisted['runs']['coco-s-v1']['attempt'] == 1
    events = [json.loads(line) for line in (
        tmp_path / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['run_id'] == 'coco-s-v1'
    assert events[-1]['status'] == 'running'
    assert events[-1]['boot_id']


def test_atomic_state_preserves_previous_json_if_replace_fails(
        tmp_path, monkeypatch):
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    store.transition('coco-s-v1', 'pending', attempt=0)
    before = (tmp_path / 'state.json').read_bytes()

    def fail_replace(source, target):
        raise OSError('injected replace failure')

    monkeypatch.setattr(os, 'replace', fail_replace)
    with pytest.raises(OSError, match='injected'):
        store.transition('coco-s-v1', 'running', attempt=1)
    assert (tmp_path / 'state.json').read_bytes() == before
    assert json.loads(before)['stage'] == 'pending'


def test_result_records_are_fsynced_jsonl(tmp_path):
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    store.record_result('coco-s-v1', {'AP': 72.8})
    record = json.loads((tmp_path / 'results.jsonl').read_text())
    assert record['run_id'] == 'coco-s-v1'
    assert record['result'] == {'AP': 72.8}
    assert record['timestamp'].endswith('+00:00')
