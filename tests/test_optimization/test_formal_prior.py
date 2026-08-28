import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'tools/optimization/prepare_formal_prior_bundle.py'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tool():
    spec = importlib.util.spec_from_file_location('formal_prior_tool', TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', *args], cwd=root, text=True).strip()


def _fake_source(tmp_path):
    source = tmp_path / 'structural'
    source.mkdir()
    _git(source, 'init', '-q')
    _git(source, 'config', 'user.email', 'test@example.com')
    _git(source, 'config', 'user.name', 'Test')
    tracked = {
        'configs/optimization/structural/no_pif_pruned.py': b'model = {}\n',
        'configs/reproduction/ablations/coco_s_v1_no_pif.py': b'model = {}\n',
        'optimization/candidates.json': b'{"schema_version":1}\n',
        'optimization/coco_val2017_authority.json': b'{"schema_version":1}\n',
    }
    runtime = {
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/train/train.json':
            b'{"kind":"train"}\n',
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/train/runtime-metadata.json':
            b'{"kind":"runtime"}\n',
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/train/best_route2.pth':
            b'opaque checkpoint bytes',
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/profile/profile.json':
            b'{"kind":"profile"}\n',
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/evaluate/evaluate.json':
            b'{"kind":"evaluate"}\n',
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/latency/latency.json':
            b'{"kind":"latency"}\n',
    }
    for relative, payload in {**tracked, **runtime}.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(source, 'add', *tracked)
    _git(source, 'commit', '-qm', 'fixture')
    commit = _git(source, 'rev-parse', 'HEAD')
    _git(source, 'checkout', '--detach', '-q')
    audit = tmp_path / 'task-4-final-artifact-audit.md'
    audit.write_text('# approved\n')
    hashes = {
        key: _sha(source / relative)
        for key, relative in {
            'train': next(path for path in runtime if path.endswith('train.json')),
            'runtime_metadata': next(
                path for path in runtime if path.endswith('runtime-metadata.json')),
            'pruned_checkpoint': next(
                path for path in runtime if path.endswith('best_route2.pth')),
            'profile': next(path for path in runtime if path.endswith('profile.json')),
            'evaluate': next(path for path in runtime if path.endswith('evaluate.json')),
            'latency': next(path for path in runtime if path.endswith('latency.json')),
            'source_config':
                'configs/optimization/structural/no_pif_pruned.py',
            'candidate_manifest': 'optimization/candidates.json',
            'coco_authority': 'optimization/coco_val2017_authority.json',
            'parent_config':
                'configs/reproduction/ablations/coco_s_v1_no_pif.py',
        }.items()
    }
    hashes['independent_audit'] = _sha(audit)
    return source, audit, commit, hashes


def test_prepare_bundle_is_atomic_read_only_and_checkpoint_opaque(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    link = tmp_path / 'runtime/work_dirs/optimization/prior-stage-b'
    result = tool._prepare_prior_bundle(
        source_root=source,
        destination=destination,
        logical_link=link,
        audit_report=audit,
        expected_source_commit=commit,
        expected_hashes=hashes,
        unpruned_parent_sha256='1' * 64,
    )
    assert result.status == 'written'
    assert link.is_symlink()
    assert link.resolve() == destination.resolve()
    document = json.loads((destination / 'bundle.json').read_text())
    assert document['source_commit'] == commit
    assert document['unpruned_parent_checkpoint_sha256'] == '1' * 64
    assert document['entries']['pruned_checkpoint']['sha256'] == hashes[
        'pruned_checkpoint']
    assert (destination / 'artifacts/pruned-runtime.pth').read_bytes() == (
        b'opaque checkpoint bytes')
    for path in destination.rglob('*'):
        assert not (path.stat().st_mode & stat.S_IWUSR)
    assert not list(destination.parent.glob('.no-pif-seed0.*.tmp'))


def test_prepare_bundle_existing_exact_is_idempotent(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    link = tmp_path / 'runtime/work_dirs/optimization/prior-stage-b'
    kwargs = dict(
        source_root=source, destination=destination, logical_link=link,
        audit_report=audit, expected_source_commit=commit,
        expected_hashes=hashes, unpruned_parent_sha256='1' * 64)
    tool._prepare_prior_bundle(**kwargs)
    result = tool._prepare_prior_bundle(**kwargs)
    assert result.status == 'current'


def test_prepare_bundle_rejects_existing_mismatch(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    destination.mkdir(parents=True)
    (destination / 'foreign').write_text('do not overwrite')
    with pytest.raises(tool.PriorBundleError, match='existing bundle'):
        tool._prepare_prior_bundle(
            source_root=source, destination=destination,
            logical_link=tmp_path / 'runtime/prior-stage-b',
            audit_report=audit, expected_source_commit=commit,
            expected_hashes=hashes, unpruned_parent_sha256='1' * 64)
    assert (destination / 'foreign').read_text() == 'do not overwrite'


def test_prepare_bundle_rejects_dirty_or_wrong_source(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    (source / 'configs/optimization/structural/no_pif_pruned.py').write_text(
        'changed = True\n')
    with pytest.raises(tool.PriorBundleError, match='clean detached source'):
        tool._prepare_prior_bundle(
            source_root=source,
            destination=tmp_path / 'canonical/no-pif-seed0',
            logical_link=tmp_path / 'runtime/prior-stage-b',
            audit_report=audit, expected_source_commit=commit,
            expected_hashes=hashes, unpruned_parent_sha256='1' * 64)


def test_prepare_bundle_rejects_hash_drift_without_overwrite(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    hashes['pruned_checkpoint'] = '0' * 64
    destination = tmp_path / 'canonical/no-pif-seed0'
    with pytest.raises(tool.PriorBundleError, match='hash mismatch'):
        tool._prepare_prior_bundle(
            source_root=source, destination=destination,
            logical_link=tmp_path / 'runtime/prior-stage-b',
            audit_report=audit, expected_source_commit=commit,
            expected_hashes=hashes, unpruned_parent_sha256='1' * 64)
    assert not destination.exists()


def test_prepare_bundle_rejects_link_drift(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    link = tmp_path / 'runtime/work_dirs/optimization/prior-stage-b'
    link.parent.mkdir(parents=True)
    alternate = tmp_path / 'alternate'
    alternate.mkdir()
    link.symlink_to(alternate)
    with pytest.raises(tool.PriorBundleError, match='logical link'):
        tool._prepare_prior_bundle(
            source_root=source, destination=destination, logical_link=link,
            audit_report=audit, expected_source_commit=commit,
            expected_hashes=hashes, unpruned_parent_sha256='1' * 64)


def test_production_bundle_check_is_read_only_and_standard_library_only():
    manifest = ROOT / 'work_dirs/optimization/prior-stage-b/bundle.json'
    before = manifest.read_bytes()
    completed = subprocess.run(
        [sys.executable, '-B', str(TOOL), '--check', str(manifest)],
        cwd=ROOT, check=False, capture_output=True, text=True,
        env={'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
             'PATH': '/usr/bin:/bin'})
    assert completed.returncode == 0, completed.stderr
    assert manifest.read_bytes() == before
    output = json.loads(completed.stdout)
    assert output['status'] == 'current'
    assert output['torch_imported'] is False


def test_production_inventory_is_predeclared_and_not_caller_substitutable():
    tool = _load_tool()
    parameters = inspect.signature(tool.prepare_prior_bundle).parameters
    check_parameters = inspect.signature(tool.check_prior_bundle).parameters
    assert not parameters
    assert not check_parameters
    assert 'expected_hashes' not in parameters
    assert 'expected_source_commit' not in parameters
    assert tool.PRODUCTION_HASHES['train'].startswith('ea276713')
    assert tool.PRODUCTION_HASHES['coco_authority'].startswith('5d5945c3')
    assert tool.PRODUCTION_HASHES['parent_config'].startswith('706eaa31')
    with pytest.raises(TypeError):
        tool.PRODUCTION_HASHES['train'] = '0' * 64


def test_production_cli_has_no_authority_path_overrides():
    tool = _load_tool()
    with pytest.raises(SystemExit):
        tool._parse_args(['--source-root', '/tmp/alternate'])
    with pytest.raises(SystemExit):
        tool._parse_args(['--destination', '/tmp/alternate'])
    with pytest.raises(SystemExit):
        tool._parse_args(['--logical-link', '/tmp/alternate'])
    with pytest.raises(SystemExit):
        tool._parse_args(['--audit-report', '/tmp/alternate'])


def test_prepare_bundle_rejects_symlinked_logical_parent(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    outside = tmp_path / 'outside'
    outside.mkdir()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'work_dirs').symlink_to(outside, target_is_directory=True)
    link = runtime / 'work_dirs/optimization/prior-stage-b'
    with pytest.raises(tool.PriorBundleError, match='parent.*symlink'):
        tool._prepare_prior_bundle(
            source_root=source, destination=destination, logical_link=link,
            audit_report=audit, expected_source_commit=commit,
            expected_hashes=hashes, unpruned_parent_sha256='1' * 64)
    assert not destination.exists()
    assert not (outside / 'optimization/prior-stage-b').exists()


def test_prepare_bundle_rejects_symlinked_source_parent_before_mutation(
        tmp_path):
    tool = _load_tool()
    real_parent = tmp_path / 'real-source-parent'
    real_parent.mkdir()
    source, audit, commit, hashes = _fake_source(real_parent)
    alias_parent = tmp_path / 'source-parent-alias'
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    destination = tmp_path / 'canonical/no-pif-seed0'

    with pytest.raises(tool.PriorBundleError, match='source root.*symlink'):
        tool._prepare_prior_bundle(
            source_root=alias_parent / source.name,
            destination=destination,
            logical_link=tmp_path / 'runtime/prior-stage-b',
            audit_report=audit,
            expected_source_commit=commit,
            expected_hashes=hashes,
            unpruned_parent_sha256='1' * 64)

    assert not destination.exists()


def test_prepare_bundle_rejects_symlinked_audit_parent_before_mutation(
        tmp_path):
    tool = _load_tool()
    real_parent = tmp_path / 'real-audit-parent'
    real_parent.mkdir()
    source, audit, commit, hashes = _fake_source(real_parent)
    alias_parent = tmp_path / 'audit-parent-alias'
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    destination = tmp_path / 'canonical/no-pif-seed0'

    with pytest.raises(tool.PriorBundleError, match='audit report.*symlink'):
        tool._prepare_prior_bundle(
            source_root=source,
            destination=destination,
            logical_link=tmp_path / 'runtime/prior-stage-b',
            audit_report=alias_parent / audit.name,
            expected_source_commit=commit,
            expected_hashes=hashes,
            unpruned_parent_sha256='1' * 64)

    assert not destination.exists()


def test_existing_bundle_rejects_writable_root(tmp_path):
    tool = _load_tool()
    source, audit, commit, hashes = _fake_source(tmp_path)
    destination = tmp_path / 'canonical/no-pif-seed0'
    link = tmp_path / 'runtime/work_dirs/optimization/prior-stage-b'
    kwargs = dict(
        source_root=source, destination=destination, logical_link=link,
        audit_report=audit, expected_source_commit=commit,
        expected_hashes=hashes, unpruned_parent_sha256='1' * 64)
    tool._prepare_prior_bundle(**kwargs)
    destination.chmod(0o755)
    with pytest.raises(tool.PriorBundleError, match='read-only'):
        tool._check_prior_bundle(**kwargs)


def test_compound_initial_substitution_fails_before_bundle_manifest(tmp_path):
    tool = _load_tool()
    source = tool.DEFAULT_SOURCE_ROOT
    clone = tmp_path / 'semantic-clone'
    semantic_names = (
        'train', 'runtime_metadata', 'profile', 'evaluate', 'latency',
        'candidate_manifest')
    for name in semantic_names:
        relative = tool._SOURCE_ENTRIES[name][0]
        destination = clone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / relative).read_bytes())
    tool._validate_production_artifact_semantics(clone)

    old = tool.PRODUCTION_HASHES['pruned_checkpoint']
    replacement = '0' * 64
    for name in semantic_names:
        relative = tool._SOURCE_ENTRIES[name][0]
        path = clone / relative
        path.write_text(path.read_text().replace(old, replacement))
    with pytest.raises(tool.PriorBundleError, match='inconsistent|mismatch'):
        tool._validate_production_artifact_semantics(clone)
    assert not (tmp_path / 'bundle.json').exists()
